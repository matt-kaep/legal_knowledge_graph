from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
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


def enforce_official_split(split: str) -> None:
    if split != graph_protocol.OFFICIAL_TRAIN_SPLIT:
        raise ValueError(
            "CV wrappers support only "
            f"split={graph_protocol.OFFICIAL_TRAIN_SPLIT}; got {split}"
        )


def load_fold_assignments() -> pd.DataFrame:
    fold_csv, _ = graph_protocol.resolve_shared_fold_paths()
    df = pd.read_csv(fold_csv)
    expected = set(range(graph_protocol.OFFICIAL_N_FOLDS))
    found = set(df["fold"].astype(int).unique().tolist())
    if found != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, got {sorted(found)}")
    return df


def run_fold_subset(
    question_ids: set[str],
    bench_dir: Path,
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
        )
        return pd.read_csv(out_dir / "eval_m1_m2.csv")


def summarize_cv_results(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    sub = df[df["modality"] == modality].copy()
    if sub.empty:
        return pd.DataFrame()
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
    records = summary_df.to_dict(orient="records")
    best = max(records, key=lambda row: graph_protocol.metric_rank_tuple(row, modality))
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version", default="canonical")
    parser.add_argument("--split", default=graph_protocol.OFFICIAL_TRAIN_SPLIT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--k-in", type=int, action="append", dest="k_ins")
    args = parser.parse_args(argv)

    enforce_official_split(args.split)
    bench_dir = graph_protocol.resolve_graph_bench_dir(args.graph_version, args.split)
    folds = load_fold_assignments()
    rows = []
    for fold in sorted(folds["fold"].astype(int).unique()):
        qids = set(folds.loc[folds["fold"] == fold, "qid"].astype(str))
        fold_df = run_fold_subset(qids, bench_dir, ks_in=args.k_ins)
        fold_df.insert(0, "fold", fold)
        rows.append(fold_df)

    raw_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_parts = [
        summarize_cv_results(raw_df, "art"),
        summarize_cv_results(raw_df, "jp"),
    ]
    summary_df = pd.concat(
        [part for part in summary_parts if not part.empty],
        ignore_index=True,
    )
    champions = {}
    if not summary_df.empty:
        for modality in ["art", "jp"]:
            sub = summary_df[summary_df["modality"] == modality]
            if not sub.empty:
                champions[modality] = select_champion(sub, modality)

    out_dir = args.out_dir or (bench_dir / "_cv" / "b3_b4")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "cv_results_raw.csv", index=False)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    (out_dir / "champions.json").write_text(json.dumps(champions, ensure_ascii=False, indent=2))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
