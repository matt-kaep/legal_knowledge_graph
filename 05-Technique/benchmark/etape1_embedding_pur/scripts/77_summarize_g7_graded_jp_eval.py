#!/usr/bin/env python3
"""Aggregate E016 G7 graded jurisprudence judgments with a fixed K denominator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        "/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph",
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


def _load_contract_module():
    path = ETAPE1 / "scripts/74_g7_graded_jp_contract.py"
    spec = importlib.util.spec_from_file_location("g7_graded_jp_contract_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract_module()


def load_responses(path: Path) -> list[dict]:
    if path.is_file():
        paths = [path]
    else:
        paths = sorted(path.glob("*.jsonl"))
        direct = path / "judge_responses.jsonl"
        if direct.exists() and direct not in paths:
            paths.append(direct)
    rows: list[dict] = []
    for current in paths:
        with current.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def select_terminal_responses(rows: list[dict]) -> dict[str, dict]:
    by_job: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("job_id"):
            by_job[str(row["job_id"])].append(row)
    selected: dict[str, dict] = {}
    for job_id, attempts in by_job.items():
        successful = [row for row in attempts if row.get("status") == "ok"]
        if successful:
            signatures = {
                json.dumps(row.get("response"), ensure_ascii=False, sort_keys=True)
                for row in successful
            }
            if len(signatures) > 1:
                raise ValueError(f"conflicting duplicate ok responses for job {job_id}")
            selected[job_id] = successful[-1]
        else:
            selected[job_id] = attempts[-1]
    return selected


def _validate_positions(positions: pd.DataFrame, *, k: int) -> None:
    required = {"qid", "rank", "jp_id", "job_id", "card_status"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"positions missing columns: {sorted(missing)}")
    expected = list(range(1, k + 1))
    for qid, group in positions.sort_values("rank").groupby("qid", sort=False):
        ranks = group["rank"].astype(int).tolist()
        if ranks != expected:
            raise ValueError(f"{qid}: fixed-K positions must contain ranks 1..{k}, found {ranks}")


def aggregate(
    positions: pd.DataFrame,
    responses: list[dict],
    questions: dict[str, dict],
    *,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    _validate_positions(positions, k=k)
    selected = select_terminal_responses(responses)
    details: list[dict] = []
    incomplete: list[str] = []
    for row in positions.sort_values(["qid", "rank"], kind="stable").itertuples(index=False):
        qid = str(row.qid)
        jp_id = str(row.jp_id)
        card_status = str(row.card_status)
        if card_status == "missing":
            label = "non_jugeable"
            justification = "Fiche Step1 absente; la position reste dans le dénominateur fixe."
            response_status = "missing_card"
        elif card_status == "available":
            response = selected.get(str(row.job_id))
            if response is None or response.get("status") != "ok":
                incomplete.append(str(row.job_id))
                continue
            payload = response.get("response") or {}
            valid, reason = CONTRACT.validate_judgment(payload)
            if not valid:
                incomplete.append(f"{row.job_id}:{reason}")
                continue
            label = str(payload["classe"])
            justification = str(payload["justification"])
            response_status = "ok"
        else:
            raise ValueError(f"unknown card_status={card_status!r} for {row.job_id}")

        question = questions.get(qid)
        if question is None:
            raise ValueError(f"missing benchmark question: {qid}")
        gold_ids = {str(value) for value in question.get("gold_jp_ids", [])}
        details.append(
            {
                "qid": qid,
                "rank": int(row.rank),
                "jp_id": jp_id,
                "job_id": str(row.job_id),
                "card_status": card_status,
                "response_status": response_status,
                "classe": label,
                "gain": CONTRACT.LABEL_GAIN[label],
                "justification": justification,
                "exact_gold": jp_id in gold_ids,
            }
        )
    if incomplete:
        raise ValueError(
            f"technical incompleteness: {len(incomplete)} card-present jobs lack a valid ok response; "
            f"examples={incomplete[:5]}"
        )

    detail = pd.DataFrame(details).sort_values(["qid", "rank"], kind="stable").reset_index(drop=True)
    per_rows: list[dict] = []
    for qid, group in detail.groupby("qid", sort=False):
        counts = Counter(group["classe"])
        per_rows.append(
            {
                "qid": qid,
                "score_gradue_at_10": float(group["gain"].sum() / k),
                "exact_hit_at_10": bool(group["exact_gold"].any()),
                **{f"count_{label}": int(counts[label]) for label in CONTRACT.VALID_LABELS},
                "non_jugeable_count": int(counts["non_jugeable"]),
            }
        )
    per_question = pd.DataFrame(per_rows)
    distribution = Counter(detail["classe"])
    summary = {
        "experiment_id": "E016",
        "status": "exploratory_internal_evaluation",
        "k": k,
        "n_questions": int(per_question["qid"].nunique()),
        "n_positions": int(len(detail)),
        "macro_score_gradue_at_10": float(per_question["score_gradue_at_10"].mean()),
        "exact_hit_at_10": float(per_question["exact_hit_at_10"].mean()),
        "non_jugeable_at_10": float(distribution["non_jugeable"] / len(detail)),
        "class_distribution": {
            label: int(distribution[label]) for label in CONTRACT.VALID_LABELS
        },
    }
    return detail, per_question, summary


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--positions", type=Path)
    parser.add_argument("--responses", type=Path)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positions_path = args.positions or args.out_dir / "rankings_topk.parquet"
    responses_path = args.responses or args.out_dir / "judge_responses.jsonl"
    positions = pd.read_parquet(positions_path)
    responses = load_responses(responses_path)
    bench_payload = json.loads(args.bench.read_text(encoding="utf-8"))
    questions = {str(item["qid"]): item for item in bench_payload["questions"]}
    detail, per_question, summary = aggregate(positions, responses, questions, k=args.k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = args.out_dir / "graded_jp_detail.csv"
    per_path = args.out_dir / "graded_jp_per_question.csv"
    summary_path = args.out_dir / "graded_jp_summary.json"
    detail.to_csv(detail_path, index=False)
    per_question.to_csv(per_path, index=False)
    summary["source_hashes"] = {
        "positions": file_sha256(positions_path),
        "responses": file_sha256(responses_path),
        "benchmark": file_sha256(args.bench),
    }
    summary["artifact_hashes"] = {
        "detail": file_sha256(detail_path),
        "per_question": file_sha256(per_path),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
