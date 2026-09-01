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


ppr_sweep = _load_script_module("25_ppr_kin_sweep.py", "ppr_sweep")


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
    graph_protocol.protocol_version_for_train_split(split)


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


def load_fold_assignments(
    bench_dir: Path,
    bench_qids: set[str],
    protocol_version: str = graph_protocol.PROTOCOL_VERSION,
) -> tuple[pd.DataFrame, dict]:
    df, metadata = graph_protocol.load_verified_grouped_fold_assignments(
        bench_dir, version=protocol_version
    )
    expected = set(range(graph_protocol.OFFICIAL_N_FOLDS))
    found = set(df["fold"].astype(int).unique().tolist())
    if found != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, got {sorted(found)}")
    return validate_fold_assignments(df, bench_qids), metadata


def run_fold_subset(
    question_ids: set[str],
    bench_dir: Path,
    graph_version: str,
    config_specs: list[str] | None = None,
    progress_path: Path | None = None,
    progress_label: str | None = None,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="cv_ppr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for filename in ["bench_global.json", "questions_emb.npy", "questions_ids.npy"]:
            src = bench_dir / filename
            if not src.exists():
                raise FileNotFoundError(f"Missing required bench artifact: {src}")
            dst = tmp_path / filename
            if src.suffix == ".json":
                dst.write_text(src.read_text())
            else:
                dst.write_bytes(src.read_bytes())
        ppr_sweep.main(
            tmp_path,
            config_specs=config_specs,
            qid_filter=question_ids,
            graph_version=graph_version,
            progress_path=progress_path,
            progress_label=progress_label,
        )
        return pd.read_csv(tmp_path / "ppr_kin_sweep_eval.csv")


def _metric_columns(modality: str) -> dict[str, str]:
    if modality == "art":
        return {
            "article_hit_at_10": "hit_strict_art",
            "article_ndcg_at_10": "ndcg_strict_art",
            "article_mrr_at_10": "mrr_strict_art",
            "article_recall_at_10": "m1_strict_art",
        }
    if modality == "jp":
        return {
            "jp_hit_at_10": "hit_jp",
            "jp_ndcg_at_10": "ndcg_jp",
            "jp_mrr_at_10": "mrr_jp",
            "jp_recall_at_10": "m1_jp",
        }
    raise ValueError(f"Unsupported modality: {modality}")


def summarize_cv_outputs(
    df: pd.DataFrame,
    modality: str,
    n_questions_benchmark: int | None = None,
    expected_qids_by_fold: dict[int, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    if n_questions_benchmark is None:
        n_questions_benchmark = int(df["qid"].nunique())
    config_columns = ["k_in", "seed_variant", "alpha"]
    metric_columns = _metric_columns(modality)
    fold_metrics, summary = graph_protocol.summarize_fold_metrics(
        df,
        config_columns=config_columns,
        metric_columns=metric_columns,
        expected_qids_by_fold=expected_qids_by_fold,
    )
    summary["n_questions_benchmark"] = int(n_questions_benchmark)
    if expected_qids_by_fold is None:
        summary["n_questions_covered"] = int(df["qid"].nunique())
        summary["question_coverage"] = 1.0
    summary = summary.sort_values(["k_in", "seed_variant", "alpha"]).reset_index(drop=True)
    summary.insert(0, "method", summary.apply(
        lambda row: f"PPR-sweep-k{int(row['k_in'])}-{row['seed_variant']}-a{float(row['alpha'])}",
        axis=1,
    ))
    summary.insert(1, "modality", modality)
    fold_metrics.insert(0, "method", fold_metrics.apply(
        lambda row: f"PPR-sweep-k{int(row['k_in'])}-{row['seed_variant']}-a{float(row['alpha'])}",
        axis=1,
    ))
    fold_metrics.insert(1, "modality", modality)
    return fold_metrics, summary


def summarize_cv_results(
    df: pd.DataFrame,
    modality: str,
    n_questions_benchmark: int | None = None,
) -> pd.DataFrame:
    return summarize_cv_outputs(df, modality, n_questions_benchmark)[1]


def build_paired_deltas(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    modality: str,
    expected_folds: int = graph_protocol.OFFICIAL_N_FOLDS,
) -> pd.DataFrame:
    config_columns = ["k_in", "seed_variant", "alpha"]
    metric_columns = [
        metric for metric in _metric_columns(modality).values()
        if metric in candidate.columns and metric in control.columns
    ]
    return graph_protocol.summarize_paired_fold_deltas(
        candidate,
        control,
        config_columns=config_columns,
        metric_columns=metric_columns,
        expected_folds=expected_folds,
    )


def select_champion(summary_df: pd.DataFrame, modality: str) -> dict:
    if summary_df.empty:
        raise ValueError(f"No CV results available for modality={modality}")
    if "eligible_champion" in summary_df:
        eligible = summary_df[summary_df["eligible_champion"]].copy()
        if eligible.empty:
            raise ValueError(f"No eligible CV champion for modality={modality}: missing fold coverage")
        sort_columns = graph_protocol.champion_sort_columns(modality)
        missing = [column for column, _ in sort_columns if column not in eligible.columns]
        if missing:
            raise KeyError(missing[0])
        return eligible.sort_values(
            [column for column, _ in sort_columns],
            ascending=[ascending for _, ascending in sort_columns],
            kind="stable",
        ).iloc[0].to_dict()
    records = summary_df.to_dict(orient="records")
    return max(records, key=lambda row: graph_protocol.metric_rank_tuple(row, modality))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version", default="canonical")
    parser.add_argument("--split", default=graph_protocol.OFFICIAL_TRAIN_SPLIT)
    parser.add_argument("--protocol-version")
    parser.add_argument("--bench-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--config", action="append")
    parser.add_argument("--control-fold-metrics", type=Path)
    parser.add_argument("--refresh-legacy-surfaces", action="store_true")
    args = parser.parse_args(argv)

    enforce_official_split(args.split)
    expected_protocol_version = graph_protocol.protocol_version_for_train_split(args.split)
    protocol_version = args.protocol_version or expected_protocol_version
    if protocol_version != expected_protocol_version:
        raise ValueError(
            f"split={args.split} requires protocol-version={expected_protocol_version}; got {protocol_version}"
        )
    bench_dir = args.bench_dir or graph_protocol.resolve_graph_bench_dir(args.graph_version, args.split)
    bench_questions = graph_protocol.load_bench_questions(bench_dir)
    bench_qids = {str(question["qid"]) for question in bench_questions}
    folds, fold_metadata = load_fold_assignments(bench_dir, bench_qids, protocol_version)
    expected_qids_by_fold = graph_protocol.expected_qids_by_fold(folds)
    rows = []
    out_dir = args.out_dir or (
        graph_protocol.cv_root(graph_protocol.BENCH_ROOT, protocol_version)
        / args.graph_version
        / "ppr"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.json"
    for fold in sorted(folds["fold"].astype(int).unique()):
        qids = set(folds.loc[folds["fold"] == fold, "qid"].astype(str))
        fold_df = run_fold_subset(
            qids,
            bench_dir,
            args.graph_version,
            config_specs=args.config,
            progress_path=progress_path,
            progress_label=f"fold {fold + 1}/{graph_protocol.OFFICIAL_N_FOLDS}",
        )
        fold_df.insert(0, "fold", fold)
        rows.append(fold_df)

    raw_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    output_parts = [
        summarize_cv_outputs(
            raw_df, "art", n_questions_benchmark=len(bench_qids), expected_qids_by_fold=expected_qids_by_fold
        ),
        summarize_cv_outputs(
            raw_df, "jp", n_questions_benchmark=len(bench_qids), expected_qids_by_fold=expected_qids_by_fold
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
        for key in ["protocol_version", "dataset_sha256", "fold_assignment_sha256"]
    }
    for key, value in run_metadata.items():
        summary_df[key] = value
    for champion in champions.values():
        champion.update(run_metadata)

    raw_df.to_csv(out_dir / "raw.csv", index=False)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2))
    (out_dir / "champions.json").write_text(json.dumps(champions, ensure_ascii=False, indent=2))
    if args.control_fold_metrics is not None:
        control_fold_metrics = pd.read_csv(args.control_fold_metrics)
        paired_parts = []
        for modality in ["art", "jp"]:
            candidate = fold_metrics_df[fold_metrics_df["modality"] == modality]
            control = control_fold_metrics[control_fold_metrics["modality"] == modality]
            paired_parts.append(
                build_paired_deltas(candidate, control, modality, expected_folds=len(expected_qids_by_fold))
            )
        pd.concat(paired_parts, ignore_index=True).to_csv(out_dir / "paired_deltas.csv", index=False)
    progress_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "graph_version": args.graph_version,
                "split": args.split,
                "folds_completed": graph_protocol.OFFICIAL_N_FOLDS,
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if args.refresh_legacy_surfaces:
        refresh_results_surfaces()
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
