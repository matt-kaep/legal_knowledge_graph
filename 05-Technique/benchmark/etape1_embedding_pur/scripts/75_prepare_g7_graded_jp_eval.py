#!/usr/bin/env python3
"""Prepare frozen, blind jobs for the E016 graded G7 jurisprudence judge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
ETAPE1 = REPO / "05-Technique/benchmark/etape1_embedding_pur"
DATA = ETAPE1 / "data/doctrine_v3plus_bench"
G7 = "G7-citation-JJ-cit1-sem025-knn5"
EVAL_SPLIT = "eval_rich_retrievable_strict"
TRAIN_SPLIT = "train_augmented_retrievable_strict"
DEFAULT_METHOD = "LightGCN-trained_K2"
DEFAULT_MODEL = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"
DEFAULT_MODEL_REVISION = "519bdca117c8f10a9a578d1b70b5c0d54c59b7ba"
PROMPT_VERSION = "g7_graded_jp_judge_v1"
PROMPT_PATH = ETAPE1 / "prompts/g7_graded_jp_judge_v1.txt"
SCHEMA_PATH = ETAPE1 / "schemas/g7_graded_jp_judge_v1.json"
CARD_FIELDS = (
    "synthese_pour_avocat",
    "fondements_retenus",
    "cited_articles",
    "solution_resume",
    "arguments_parties",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_judge_contract(
    *,
    model_id: str,
    model_revision: str,
    prompt_path: Path = PROMPT_PATH,
    schema_path: Path = SCHEMA_PATH,
) -> dict[str, str]:
    return {
        "model_id": model_id,
        "model_revision": model_revision,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": sha256(prompt_path),
        "schema_sha256": sha256(schema_path),
    }


def validate_profile_paths(profile: str, rankings_path: Path, bench_path: Path) -> None:
    if profile not in {"evaluation", "calibration"}:
        raise ValueError(f"unknown profile: {profile}")
    combined = f"{rankings_path}\n{bench_path}"
    if profile == "calibration" and EVAL_SPLIT in combined:
        raise ValueError("calibration is train-only; internal eval inputs are forbidden")
    if profile == "calibration" and TRAIN_SPLIT not in str(bench_path):
        raise ValueError("calibration is train-only; benchmark must be the train split")


def select_question_ids(question_ids: set[str], *, limit: int | None, seed: int) -> set[str]:
    ordered = np.array(sorted(str(qid) for qid in question_ids), dtype=object)
    if limit is None or limit >= len(ordered):
        return set(ordered.tolist())
    if limit <= 0:
        raise ValueError("question limit must be positive")
    rng = np.random.default_rng(seed)
    indices = sorted(int(index) for index in rng.choice(len(ordered), size=limit, replace=False))
    return {str(ordered[index]) for index in indices}


def select_g7_positions(
    rankings: pd.DataFrame,
    *,
    question_ids: set[str],
    method: str,
    k: int,
) -> pd.DataFrame:
    required = {"qid", "method", "modality", "rank", "item_id"}
    missing_columns = required - set(rankings.columns)
    if missing_columns:
        raise ValueError(f"rankings missing columns: {sorted(missing_columns)}")
    if k <= 0:
        raise ValueError("K must be positive")
    selected = rankings.loc[
        rankings["qid"].astype(str).isin(question_ids)
        & (rankings["method"].astype(str) == method)
        & (rankings["modality"].astype(str) == "jp")
        & (rankings["rank"].astype(int) <= k),
        ["qid", "rank", "item_id"],
    ].copy()
    selected["qid"] = selected["qid"].astype(str)
    selected["rank"] = selected["rank"].astype(int)
    selected["jp_id"] = selected.pop("item_id").astype(str)
    selected = selected.sort_values(["qid", "rank"], kind="stable").reset_index(drop=True)

    found_qids = set(selected["qid"])
    if found_qids != set(question_ids):
        absent = sorted(set(question_ids) - found_qids)
        raise ValueError(f"rankings missing requested questions: {absent[:5]}")
    expected_ranks = list(range(1, k + 1))
    for qid, group in selected.groupby("qid", sort=False):
        ranks = group["rank"].tolist()
        if ranks != expected_ranks:
            raise ValueError(f"{qid}: expected ranks 1..{k}, found {ranks}")
    selected["duplicate_position"] = selected.duplicated(["qid", "jp_id"], keep="first")
    return selected


def _blind_card(card: dict) -> dict:
    return {field: card.get(field, [] if field in {"cited_articles", "arguments_parties"} else "") for field in CARD_FIELDS}


def build_blind_jobs(
    *,
    positions: pd.DataFrame,
    questions: dict[str, dict],
    cards: dict[str, dict],
    judge_contract: dict,
) -> tuple[list[dict], pd.DataFrame]:
    frozen = positions.copy()
    jobs: list[dict] = []
    statuses: list[str] = []
    job_ids: list[str] = []
    emitted_job_ids: set[str] = set()
    for row in frozen.itertuples(index=False):
        qid = str(row.qid)
        jp_id = str(row.jp_id)
        question = questions.get(qid)
        if question is None:
            raise ValueError(f"missing benchmark question: {qid}")
        job_id = hashlib.sha256(f"{qid}\x1f{jp_id}".encode("utf-8")).hexdigest()
        job_ids.append(job_id)
        card = cards.get(jp_id)
        if card is None:
            statuses.append("missing")
            continue
        statuses.append("available")
        if job_id in emitted_job_ids:
            continue
        emitted_job_ids.add(job_id)
        jobs.append(
            {
                "job_id": job_id,
                "qid": qid,
                "jp_id": jp_id,
                "question": str(question.get("enonce") or "").strip(),
                "decision_card": _blind_card(card),
                "judge_contract": dict(judge_contract),
            }
        )
    frozen["job_id"] = job_ids
    frozen["card_status"] = statuses
    return jobs, frozen


def load_questions(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("bench_global.json must contain a questions list")
    result = {str(item["qid"]): item for item in questions}
    if len(result) != len(questions):
        raise ValueError("duplicate qid in benchmark")
    return result


def _load_card_fetcher():
    path = ETAPE1 / "scripts/67_prepare_g8_llm_jp_link_jobs.py"
    spec = importlib.util.spec_from_file_location("g8_card_fetcher", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fetch_decision_cards


def write_outputs(
    *,
    out_dir: Path,
    positions: pd.DataFrame,
    jobs: list[dict],
    cards: dict[str, dict],
    manifest: dict,
    force: bool,
) -> None:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen campaign: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "rankings_topk.parquet"
    positions.to_parquet(positions_path, index=False)
    jobs_path = out_dir / "judge_jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(job, ensure_ascii=False, sort_keys=True) + "\n" for job in jobs),
        encoding="utf-8",
    )
    cards_path = out_dir / "decision_cards.json"
    cards_path.write_text(
        json.dumps(cards, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["artifacts"] = {
        "rankings_topk.parquet": sha256(positions_path),
        "judge_jobs.jsonl": sha256(jobs_path),
        "decision_cards.json": sha256(cards_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("evaluation", "calibration"), default="evaluation")
    parser.add_argument("--rankings", type=Path)
    parser.add_argument("--bench", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--method")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, str]:
    if args.profile == "evaluation":
        rankings = args.rankings or DATA / G7 / EVAL_SPLIT / "rankings.parquet"
        bench = args.bench or DATA / EVAL_SPLIT / "bench_global.json"
        out = args.out_dir or DATA / G7 / EVAL_SPLIT / "E016-g7-graded-jp-v1"
        method = args.method or DEFAULT_METHOD
    else:
        rankings = args.rankings or DATA / TRAIN_SPLIT / "rankings.parquet"
        bench = args.bench or DATA / TRAIN_SPLIT / "bench_global.json"
        out = args.out_dir or DATA / "calibration/E016-g7-graded-jp-v1"
        method = args.method or "B3-a"
    validate_profile_paths(args.profile, rankings, bench)
    return rankings, bench, out, method


def main() -> None:
    args = parse_args()
    rankings_path, bench_path, out_dir, method = resolve_paths(args)
    questions = load_questions(bench_path)
    rankings = pd.read_parquet(rankings_path)
    candidate_qids = set(questions) & set(
        rankings.loc[
            (rankings["method"].astype(str) == method)
            & (rankings["modality"].astype(str) == "jp"),
            "qid",
        ].astype(str)
    )
    limit = args.question_limit
    if args.profile == "calibration" and limit is None:
        raise ValueError("calibration requires --question-limit")
    selected_qids = select_question_ids(candidate_qids, limit=limit, seed=args.seed)
    positions = select_g7_positions(
        rankings, question_ids=selected_qids, method=method, k=args.k
    )

    fetch_cards = _load_card_fetcher()
    jp_ids = sorted(set(positions["jp_id"]))
    cards = fetch_cards(
        jp_ids,
        max_args=12,
        max_arg_chars=1_500,
        max_field_chars=4_000,
        batch_size=5_000,
    )
    judge_contract = make_judge_contract(
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    jobs, frozen_positions = build_blind_jobs(
        positions=positions,
        questions=questions,
        cards=cards,
        judge_contract=judge_contract,
    )
    manifest = {
        "experiment_id": "E016",
        "status": "exploratory_internal_evaluation",
        "profile": args.profile,
        "split": EVAL_SPLIT if args.profile == "evaluation" else TRAIN_SPLIT,
        "method": method,
        "k": args.k,
        "seed": args.seed,
        "question_limit": limit,
        "n_questions": int(frozen_positions["qid"].nunique()),
        "n_positions": int(len(frozen_positions)),
        "n_unique_jp": len(jp_ids),
        "n_cards_available": len(cards),
        "n_jobs": len(jobs),
        "n_duplicate_positions": int(frozen_positions["duplicate_position"].sum()),
        "n_missing_card_positions": int((frozen_positions["card_status"] == "missing").sum()),
        "judge_contract": judge_contract,
        "sources": {
            "rankings": {"path": str(rankings_path), "sha256": sha256(rankings_path)},
            "benchmark": {"path": str(bench_path), "sha256": sha256(bench_path)},
            "prompt": {"path": str(PROMPT_PATH), "sha256": sha256(PROMPT_PATH)},
            "schema": {"path": str(SCHEMA_PATH), "sha256": sha256(SCHEMA_PATH)},
            "preparation_script": {"path": str(Path(__file__)), "sha256": sha256(Path(__file__))},
        },
        "cards_sha256": stable_json_sha256(cards),
    }
    write_outputs(
        out_dir=out_dir,
        positions=frozen_positions,
        jobs=jobs,
        cards=cards,
        manifest=manifest,
        force=args.force,
    )
    print(json.dumps({"out_dir": str(out_dir), **{k: manifest[k] for k in ("n_questions", "n_positions", "n_jobs", "n_missing_card_positions")}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
