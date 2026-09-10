#!/usr/bin/env python3
"""Aggregate the completed E017 inter-graph graded-JP judgment campaign.

E017 compares frozen LightGCN replays for several graph/seed pairs.  Exact
retrieval metrics remain the benchmark metrics; the LLM-derived graded score is
reported separately as an exploratory diagnostic on the already consulted
internal evaluation split.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pandas as pd


CODE_REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).expanduser().resolve()
ETAPE1 = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DATA = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
DEFAULT_OUT = DATA / "E017-intergraph-graded-jp-v1"
DEFAULT_BENCH = DATA / "eval_rich_retrievable_strict/bench_global.json"


def _load_module(filename: str, name: str):
    path = ETAPE1 / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E016 = _load_module("77_summarize_g7_graded_jp_eval.py", "e016_graded_summary")
METRICS = _load_module("metrics.py", "e017_metrics")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_responses(path: Path) -> list[dict]:
    return E016.load_responses(path)


def _validate_positions(positions: pd.DataFrame, *, k: int) -> None:
    required = {"graph_id", "seed", "qid", "rank", "jp_id", "job_id", "card_status"}
    missing = required - set(positions.columns)
    if missing:
        raise ValueError(f"E017 positions missing columns: {sorted(missing)}")
    expected_ranks = list(range(1, k + 1))
    for (graph_id, seed, qid), group in positions.groupby(["graph_id", "seed", "qid"], sort=False):
        ranks = sorted(group["rank"].astype(int).tolist())
        if ranks != expected_ranks:
            raise ValueError(
                f"{graph_id}/seed={seed}/{qid}: expected ranks {expected_ranks}, found {ranks}"
            )


def _exact_metrics(
    positions: pd.DataFrame, questions: dict[str, dict], *, k: int
) -> dict[str, float]:
    per_question: list[dict[str, float]] = []
    for qid, group in positions.sort_values("rank", kind="stable").groupby("qid", sort=False):
        question = questions.get(str(qid))
        if question is None:
            raise ValueError(f"Missing benchmark question: {qid}")
        ranked = group["jp_id"].astype(str).tolist()
        gold = {str(item) for item in question.get("gold_jp_ids", [])}
        per_question.append(METRICS.all_metrics(ranked, gold, k))
    frame = pd.DataFrame(per_question)
    return {key: float(frame[key].mean()) for key in METRICS.METRIC_NAMES}


def summarize_graphs(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate run metrics with a sample standard deviation across seeds."""
    mean_columns = (
        "score_gradue_at_10",
        "exact_any_gold_at_10",
        "m1_recall_at_10",
        "m2_rank_at_10",
        "hit_at_10",
        "mrr_at_10",
        "ndcg_at_10",
    )
    std_columns = (
        "score_gradue_at_10",
        "m1_recall_at_10",
        "hit_at_10",
        "ndcg_at_10",
    )
    return (
        run_metrics.groupby("graph_id", sort=True)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{column}_mean": (column, "mean") for column in mean_columns},
            **{f"{column}_std": (column, "std") for column in std_columns},
        )
        .reset_index()
    )


def aggregate_e017(
    positions: pd.DataFrame,
    responses: list[dict],
    questions: dict[str, dict],
    *,
    k: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return run-level, graph-level, and per-question E017 metrics.

    The E016 implementation remains the sole owner of the A--E judgment
    contract. This wrapper only partitions fixed-K rankings by graph and seed.
    """
    _validate_positions(positions, k=k)
    run_rows: list[dict] = []
    per_question_frames: list[pd.DataFrame] = []
    for (graph_id, seed), group in positions.groupby(["graph_id", "seed"], sort=True):
        group = group.copy().sort_values(["qid", "rank"], kind="stable")
        detail, per_question, graded = E016.aggregate(group, responses, questions, k=k)
        exact = _exact_metrics(group, questions, k=k)
        run_rows.append(
            {
                "graph_id": str(graph_id),
                "seed": int(seed),
                "n_questions": int(per_question["qid"].nunique()),
                "n_positions": int(len(detail)),
                "score_gradue_at_10": graded["macro_score_gradue_at_10"],
                "exact_any_gold_at_10": graded["exact_hit_at_10"],
                "non_jugeable_at_10": graded["non_jugeable_at_10"],
                "duplicate_position_count": graded["duplicate_position_count"],
                "m1_recall_at_10": exact["m1"],
                "m2_rank_at_10": exact["m2"],
                "hit_at_10": exact["hit"],
                "mrr_at_10": exact["mrr"],
                "ndcg_at_10": exact["ndcg"],
                **{f"count_{label}": graded["class_distribution"][label] for label in E016.CONTRACT.VALID_LABELS},
            }
        )
        per_question = per_question.assign(graph_id=str(graph_id), seed=int(seed))
        per_question_frames.append(per_question)

    run_metrics = pd.DataFrame(run_rows).sort_values(["graph_id", "seed"], kind="stable")
    graph_metrics = summarize_graphs(run_metrics)
    per_question_metrics = pd.concat(per_question_frames, ignore_index=True)
    return run_metrics, graph_metrics, per_question_metrics


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
    responses = _load_responses(responses_path)
    bench_payload = json.loads(args.bench.read_text(encoding="utf-8"))
    questions = {str(item["qid"]): item for item in bench_payload["questions"]}
    run_metrics, graph_metrics, per_question = aggregate_e017(
        positions, responses, questions, k=args.k
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.out_dir / "e017_graph_seed_metrics.csv"
    graph_path = args.out_dir / "e017_graph_metrics.csv"
    per_question_path = args.out_dir / "e017_per_question_metrics.csv"
    run_metrics.to_csv(run_path, index=False)
    graph_metrics.to_csv(graph_path, index=False)
    per_question.to_csv(per_question_path, index=False)
    summary = {
        "experiment_id": "E017",
        "status": "exploratory_internal_evaluation",
        "k": args.k,
        "n_graph_seed_runs": int(len(run_metrics)),
        "n_graphs": int(run_metrics["graph_id"].nunique()),
        "n_questions_per_run": int(run_metrics["n_questions"].iloc[0]),
        "source_hashes": {
            "positions": _sha256(positions_path),
            "responses": _sha256(responses_path),
            "benchmark": _sha256(args.bench),
        },
        "artifact_hashes": {
            "graph_seed_metrics": _sha256(run_path),
            "graph_metrics": _sha256(graph_path),
            "per_question_metrics": _sha256(per_question_path),
        },
    }
    summary_path = args.out_dir / "e017_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
