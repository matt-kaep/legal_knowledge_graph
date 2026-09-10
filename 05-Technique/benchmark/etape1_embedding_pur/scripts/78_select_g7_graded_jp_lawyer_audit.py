#!/usr/bin/env python3
"""Select and materialize the blind E016 lawyer audit sample."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
ETAPE1 = REPO / "05-Technique/benchmark/etape1_embedding_pur"
DEFAULT_OUT = (
    ETAPE1
    / "data/doctrine_v3plus_bench/G7-citation-JJ-cit1-sem025-knn5"
    / "eval_rich_retrievable_strict/E016-g7-graded-jp-v1"
)
DEFAULT_BENCH = (
    ETAPE1
    / "data/doctrine_v3plus_bench/eval_rich_retrievable_strict/bench_global.json"
)
LABELS = ("A", "B", "C", "D", "E", "non_jugeable")
BASE_ALLOCATION = {"A": 25, "B": 20, "C": 15, "D": 15, "E": 15, "non_jugeable": 10}


def _allocate(total: int, capacities: dict[str, int], desired: dict[str, float]) -> dict[str, int]:
    if total > sum(capacities.values()):
        raise ValueError(f"sample size {total} exceeds population {sum(capacities.values())}")
    raw_total = sum(desired.values()) or 1.0
    raw = {key: total * desired.get(key, 0.0) / raw_total for key in capacities}
    allocation = {key: min(capacities[key], int(raw[key])) for key in capacities}
    while sum(allocation.values()) < total:
        candidates = [key for key in capacities if allocation[key] < capacities[key]]
        if not candidates:
            raise ValueError("unable to allocate requested sample")
        key = max(
            candidates,
            key=lambda item: (raw[item] - allocation[item], desired.get(item, 0.0), -list(capacities).index(item)),
        )
        allocation[key] += 1
    return allocation


def _stable_token(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def select_sample(population: pd.DataFrame, *, sample_size: int = 100, seed: int = 42) -> pd.DataFrame:
    required = {"qid", "jp_id", "job_id", "rank", "classe", "justification", "exact_gold"}
    missing = required - set(population.columns)
    if missing:
        raise ValueError(f"graded detail missing columns: {sorted(missing)}")
    unknown = set(population["classe"].astype(str)) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown labels: {sorted(unknown)}")

    working = population.copy()
    if "duplicate_position" in working.columns:
        working = working.loc[~working["duplicate_position"].astype(bool)].copy()
    if working["job_id"].duplicated().any():
        raise ValueError("lawyer population must contain unique question-JP judgments")
    working["classe"] = working["classe"].astype(str)
    working["rank_bucket"] = working["rank"].astype(int).map(lambda rank: "1-3" if rank <= 3 else "4-10")
    working["exact_bucket"] = working["exact_gold"].astype(bool).map({True: "exact", False: "non_exact"})
    working["audit_stratum"] = (
        working["classe"] + "|" + working["rank_bucket"] + "|" + working["exact_bucket"]
    )
    label_capacities = {label: int((working["classe"] == label).sum()) for label in LABELS}
    desired = {label: BASE_ALLOCATION[label] * sample_size / 100 for label in LABELS}
    label_allocation = _allocate(sample_size, label_capacities, desired)

    selected_parts: list[pd.DataFrame] = []
    for label in LABELS:
        label_rows = working.loc[working["classe"] == label].copy()
        quota = label_allocation[label]
        if quota == 0:
            continue
        strata = sorted(label_rows["audit_stratum"].unique())
        capacities = {
            stratum: int((label_rows["audit_stratum"] == stratum).sum()) for stratum in strata
        }
        stratum_allocation = _allocate(
            quota,
            capacities,
            {stratum: float(capacities[stratum]) for stratum in strata},
        )
        for stratum in strata:
            take = stratum_allocation[stratum]
            if take == 0:
                continue
            rows = label_rows.loc[label_rows["audit_stratum"] == stratum].copy()
            rows["_sample_token"] = rows["job_id"].astype(str).map(
                lambda job_id: _stable_token(seed, job_id)
            )
            rows = rows.sort_values(["_sample_token", "job_id"], kind="stable").head(take)
            rows["population_stratum_size"] = capacities[stratum]
            rows["sample_stratum_size"] = take
            rows["inclusion_probability"] = take / capacities[stratum]
            rows["sampling_weight"] = capacities[stratum] / take
            selected_parts.append(rows)

    selected = pd.concat(selected_parts, ignore_index=True)
    selected["case_id"] = selected["job_id"].astype(str).map(
        lambda job_id: "E016-" + hashlib.sha256(f"lawyer|{seed}|{job_id}".encode("utf-8")).hexdigest()[:16]
    )
    order = {label: index for index, label in enumerate(LABELS)}
    selected["_label_order"] = selected["classe"].map(order)
    selected = selected.sort_values(["_label_order", "case_id"], kind="stable").drop(
        columns=["_label_order", "_sample_token"], errors="ignore"
    )
    if len(selected) != sample_size or selected["case_id"].duplicated().any():
        raise AssertionError("lawyer sample is not complete and unique")
    return selected.reset_index(drop=True)


def build_exports(
    selected: pd.DataFrame,
    *,
    questions: dict[str, dict],
    cards: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    blind_rows: list[dict] = []
    for row in selected.itertuples(index=False):
        qid = str(row.qid)
        jp_id = str(row.jp_id)
        if qid not in questions:
            raise ValueError(f"missing question {qid}")
        if jp_id not in cards:
            raise ValueError(f"missing frozen decision card {jp_id}")
        blind_rows.append(
            {
                "case_id": str(row.case_id),
                "question": str(questions[qid].get("enonce") or ""),
                "fiche_juridique": json.dumps(cards[jp_id], ensure_ascii=False, sort_keys=True),
                "classe_avocat": "",
                "justification_avocat": "",
            }
        )
    blind = pd.DataFrame(blind_rows)
    private_columns = [
        "case_id",
        "qid",
        "jp_id",
        "job_id",
        "rank",
        "classe",
        "justification",
        "exact_gold",
        "audit_stratum",
        "population_stratum_size",
        "sample_stratum_size",
        "inclusion_probability",
        "sampling_weight",
    ]
    private = selected[private_columns].copy()
    return blind, private


def _load_judilibre_loader():
    path = ETAPE1 / "scripts/72_materialize_g8_legal_audit.py"
    spec = importlib.util.spec_from_file_location("g8_full_text_loader_for_e016", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.load_judilibre_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--detail", type=Path)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--cards", type=Path)
    parser.add_argument("--corpus", type=Path, action="append", default=[])
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail_path = args.detail or args.out_dir / "graded_jp_detail.csv"
    cards_path = args.cards or args.out_dir / "decision_cards.json"
    detail = pd.read_csv(detail_path)
    bench = json.loads(args.bench.read_text(encoding="utf-8"))
    questions = {str(item["qid"]): item for item in bench["questions"]}
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    selected = select_sample(detail, sample_size=args.sample_size, seed=args.seed)
    blind, private = build_exports(selected, questions=questions, cards=cards)

    audit_dir = args.out_dir / "lawyer_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    blind.to_csv(audit_dir / "lawyer_audit_sample.csv", index=False)
    private.to_csv(audit_dir / "lawyer_audit_key.csv", index=False)

    if args.corpus:
        load_records = _load_judilibre_loader()
        records = load_records(args.corpus, set(private["jp_id"].astype(str)))
        with (audit_dir / "lawyer_evidence.jsonl").open("w", encoding="utf-8") as handle:
            for row in private.itertuples(index=False):
                handle.write(
                    json.dumps(
                        {
                            "case_id": row.case_id,
                            "qid": row.qid,
                            "jp_id": row.jp_id,
                            "decision": records[str(row.jp_id)],
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
    print(json.dumps({"sample_size": len(blind), "audit_dir": str(audit_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
