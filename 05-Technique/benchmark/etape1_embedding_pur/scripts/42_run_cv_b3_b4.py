from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path(os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4])))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import graph_protocol  # noqa: E402


def _load_script_module(script_name: str, module_name: str):
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


baseline_eval = _load_script_module("26_eval_doctrine_v3plus_m1_m2.py", "eval_b3b4")


def refresh_results_surfaces() -> None:
    try:
        report = _load_script_module("49_build_intergraph_results_report.py", "intergraph_results_report")
        report.build_report(report.DEFAULT_OUT_DIR)
    except Exception as exc:  # pragma: no cover - best effort refresh
        print(f"[warn] inter-graph report refresh failed: {exc}")
    try:
        snippets = _load_script_module("50_build_week13_intergraph_snippets.py", "week13_intergraph_snippets")
        snippets.main([])
    except Exception as exc:  # pragma: no cover - best effort refresh
        print(f"[warn] week13 snippet refresh failed: {exc}")


def enforce_official_split(split: str) -> None:
    if split != graph_protocol.OFFICIAL_TRAIN_SPLIT:
        raise ValueError(
            "CV wrappers support only "
            f"split={graph_protocol.OFFICIAL_TRAIN_SPLIT}; got {split}"
        )


def validate_fold_assignments(df: pd.DataFrame, bench_qids: set[str]) -> pd.DataFrame:
    qids = df["qid"].astype(str)
    duplicate_qids = sorted(qids[qids.duplicated()].unique().tolist())
    if duplicate_qids:
        raise ValueError(f"duplicate qids in fold assignments: {duplicate_qids}")
    assigned_qids = set(qids.tolist())
    missing_qids = sorted(bench_qids - assigned_qids)
    extra_qids = sorted(assigned_qids - bench_qids)
    if missing_qids or extra_qids:
        raise ValueError(
            "fold assignments mismatch: "
            f"missing qids={missing_qids}; extra qids={extra_qids}"
        )
    return df


def load_fold_assignments(bench_dir: Path, bench_qids: set[str]) -> tuple[pd.DataFrame, dict]:
    df, metadata = graph_protocol.load_verified_grouped_fold_assignments(bench_dir)
    expected = set(range(graph_protocol.OFFICIAL_N_FOLDS))
    found = set(df["fold"].astype(int).unique().tolist())
    if found != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, got {sorted(found)}")
    return validate_fold_assignments(df, bench_qids), metadata


def _metric_columns(modality: str) -> dict[str, str]:
    if modality == "art":
        return {
            "article_hit_at_10": "hit_strict",
            "article_ndcg_at_10": "ndcg_strict",
            "article_mrr_at_10": "mrr_strict",
            "article_recall_at_10": "m1_strict",
        }
    if modality == "jp":
        return {
            "jp_hit_at_10": "hit_strict",
            "jp_ndcg_at_10": "ndcg_strict",
            "jp_mrr_at_10": "mrr_strict",
            "jp_recall_at_10": "m1_strict",
        }
    raise ValueError(f"Unsupported modality={modality}")


def summarize_grouped_cv_outputs(
    df: pd.DataFrame,
    modality: str,
    *,
    n_questions_benchmark: int,
    expected_qids_by_fold: dict[int, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["modality"].eq(modality)].copy()
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()
    fold_metrics, summary = graph_protocol.summarize_fold_metrics(
        sub,
        config_columns=["method", "k_in"],
        metric_columns=_metric_columns(modality),
        expected_qids_by_fold=expected_qids_by_fold,
    )
    fold_metrics["modality"] = modality
    summary["modality"] = modality
    summary["n_questions_benchmark"] = int(n_questions_benchmark)
    return fold_metrics, summary


def run_fold_subset(
    question_ids: set[str],
    bench_dir: Path,
    graph_version: str,
    ks_in: list[int] | None = None,
) -> pd.DataFrame:
    questions = graph_protocol.load_bench_questions(bench_dir)
    with tempfile.TemporaryDirectory(prefix="cv_b3b4_") as tmp_dir:
        out_dir = Path(tmp_dir)
        baseline_eval.eval_m1_m2(
            questions,
            out_dir,
            qid_filter=question_ids,
            ks_in=ks_in,
            question_cache_dir=bench_dir,
            graph_version=graph_version,
        )
        return pd.read_csv(out_dir / "eval_m1_m2.csv")


def summarize_cv_results(
    df: pd.DataFrame,
    modality: str,
    n_questions_benchmark: int | None = None,
) -> pd.DataFrame:
    sub = df[df["modality"] == modality].copy()
    if sub.empty:
        return pd.DataFrame()
    if n_questions_benchmark is None:
        n_questions_benchmark = int(sub["qid"].nunique())
    group_cols = ["method", "modality"]
    if "k_in" in sub.columns:
        group_cols.append("k_in")
    metric_cols = [
        col
        for col in sub.columns
        if col
        in {
            "hit",
            "ndcg",
            "mrr",
            "m1",
            "m2",
            "hit_strict",
            "ndcg_strict",
            "mrr_strict",
            "m1_strict",
            "m2_strict",
            "hit_ext",
            "ndcg_ext",
            "mrr_ext",
            "m1_ext",
            "m2_ext",
        }
    ]
    summary = (
        sub.groupby(group_cols, dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    coverage = (
        sub.groupby(group_cols, dropna=False)
        .agg(
            n_questions_covered=("qid", "nunique"),
            n_folds_covered=("fold", "nunique"),
        )
        .reset_index()
    )
    coverage["n_questions_benchmark"] = int(n_questions_benchmark)
    coverage["question_coverage"] = (
        coverage["n_questions_covered"] / coverage["n_questions_benchmark"]
    )
    coverage["fold_coverage"] = (
        coverage["n_folds_covered"] / graph_protocol.OFFICIAL_N_FOLDS
    )
    summary = (
        summary.merge(coverage, on=group_cols, how="left")
        .sort_values(group_cols, na_position="first")
        .reset_index(drop=True)
    )
    return summary


def select_champion(summary_df: pd.DataFrame, modality: str) -> dict:
    if summary_df.empty:
        raise ValueError(f"No CV results available for modality={modality}")
    if "eligible_champion" in summary_df:
        eligible = summary_df[summary_df["eligible_champion"].eq(True)].copy()
        if eligible.empty:
            raise ValueError(f"No eligible CV champion for modality={modality}")
        order = graph_protocol.champion_sort_columns(modality)
        return eligible.sort_values(
            [column for column, _ in order],
            ascending=[ascending for _, ascending in order],
            kind="stable",
        ).iloc[0].to_dict()
    return max(summary_df.to_dict(orient="records"), key=lambda row: graph_protocol.metric_rank_tuple(row, modality))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version", default="canonical")
    parser.add_argument("--split", default=graph_protocol.OFFICIAL_TRAIN_SPLIT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--k-in", type=int, action="append", dest="k_ins")
    parser.add_argument(
        "--direct-cosine-only",
        action="store_true",
        help="Keep only the graph-invariant B2-a article and B3-a JP controls.",
    )
    args = parser.parse_args(argv)

    enforce_official_split(args.split)
    bench_dir = graph_protocol.resolve_graph_bench_dir(args.graph_version, args.split)
    bench_questions = graph_protocol.load_bench_questions(bench_dir)
    bench_qids = {str(question["qid"]) for question in bench_questions}
    folds, fold_metadata = load_fold_assignments(bench_dir, bench_qids)
    expected_qids_by_fold = graph_protocol.expected_qids_by_fold(folds)
    rows = []
    for fold in sorted(folds["fold"].astype(int).unique()):
        qids = set(folds.loc[folds["fold"] == fold, "qid"].astype(str))
        fold_df = run_fold_subset(qids, bench_dir, args.graph_version, ks_in=args.k_ins)
        fold_df.insert(0, "fold", fold)
        rows.append(fold_df)

    raw_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if args.direct_cosine_only:
        raw_df = raw_df[
            (raw_df["modality"].eq("art") & raw_df["method"].eq("B2-a"))
            | (raw_df["modality"].eq("jp") & raw_df["method"].eq("B3-a"))
        ].copy()
    output_parts = [
        summarize_grouped_cv_outputs(
            raw_df, "art", n_questions_benchmark=len(bench_qids),
            expected_qids_by_fold=expected_qids_by_fold,
        ),
        summarize_grouped_cv_outputs(
            raw_df, "jp", n_questions_benchmark=len(bench_qids),
            expected_qids_by_fold=expected_qids_by_fold,
        ),
    ]
    fold_metrics_df = pd.concat(
        [fold_metrics for fold_metrics, _ in output_parts if not fold_metrics.empty],
        ignore_index=True,
    )
    summary_df = pd.concat(
        [summary for _, summary in output_parts if not summary.empty],
        ignore_index=True,
    )
    champions = {}
    if not summary_df.empty:
        for modality in ["art", "jp"]:
            sub = summary_df[summary_df["modality"] == modality]
            if not sub.empty:
                champions[modality] = select_champion(sub, modality)

    run_metadata = {
        key: fold_metadata[key]
        for key in ("protocol_version", "dataset_sha256", "fold_assignment_sha256")
    }
    for key, value in run_metadata.items():
        summary_df[key] = value
    for champion in champions.values():
        champion.update(run_metadata)

    out_dir = args.out_dir or (bench_dir / "_cv" / "b3_b4")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "raw.csv", index=False)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2))
    (out_dir / "champions.json").write_text(json.dumps(champions, ensure_ascii=False, indent=2))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
