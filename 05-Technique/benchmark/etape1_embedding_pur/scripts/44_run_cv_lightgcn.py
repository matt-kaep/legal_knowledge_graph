from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get(
    "LKG_REPO",
    str(Path(__file__).resolve().parents[4]),
))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import graph_protocol  # noqa: E402
import benchmark_labels  # noqa: E402


def _load_script_module(script_name: str, module_name: str):
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lightgcn = _load_script_module("32_lightgcn_strict.py", "lightgcn_strict")

CANDIDATE_COVERED_TRAIN_SPLIT = graph_protocol.CANDIDATE_COVERED_TRAIN_SPLIT
CANDIDATE_COVERED_PROTOCOL_VERSION = graph_protocol.CANDIDATE_COVERED_PROTOCOL_VERSION
EFFECTIVE_RETRIEVAL_TRAIN_SPLIT = graph_protocol.EFFECTIVE_RETRIEVAL_TRAIN_SPLIT
EFFECTIVE_RETRIEVAL_PROTOCOL_VERSION = graph_protocol.EFFECTIVE_RETRIEVAL_PROTOCOL_VERSION


def select_replay_epoch(history_df: pd.DataFrame, metric: str) -> int:
    """Select a fixed replay epoch from non-weighted means of validation folds."""
    required = {"fold", "epoch", metric}
    missing = required - set(history_df.columns)
    if missing:
        raise KeyError(f"Missing replay epoch columns: {sorted(missing)}")
    finite = history_df.dropna(subset=[metric]).copy()
    if finite.empty:
        raise ValueError(f"No finite validation history for metric={metric}")
    fold_epoch = finite.groupby(["fold", "epoch"], as_index=False)[metric].mean()
    epoch_scores = fold_epoch.groupby("epoch", as_index=False)[metric].mean()
    selected_epoch_index = int(
        epoch_scores.sort_values([metric, "epoch"], ascending=[False, True], kind="stable")
        .iloc[0]["epoch"]
    )
    # Histories are zero-based; the training CLI consumes an epoch count.
    return selected_epoch_index + 1


def selection_metric_for_target(target: str) -> str:
    normalized = {"art": "art", "article": "art", "jp": "jp"}.get(target)
    if normalized == "art":
        return "val_recall"
    if normalized == "jp":
        return "val_hit_jp"
    raise ValueError(f"Unsupported LightGCN selection target: {target}")


def attach_replay_epochs(
    champions: dict[str, dict], history_df: pd.DataFrame
) -> dict[str, dict]:
    """Attach fixed replay epochs using only each champion's matching CV history."""
    enriched: dict[str, dict] = {}
    match_columns = [
        "variant",
        "train_k",
        "seed",
        "lr",
        "epochs",
        "lambda_anchor",
        "negative_sampling_strategy",
        "graph_version",
        "selection_target",
    ]
    for target, champion in champions.items():
        row = dict(champion)
        if not str(row.get("variant", "")).startswith("trained_"):
            enriched[target] = row
            continue
        mask = pd.Series(True, index=history_df.index)
        for column in match_columns:
            if column not in history_df.columns or column not in row:
                raise KeyError(f"Missing champion history key: {column}")
            value = row[column]
            mask &= history_df[column].isna() if pd.isna(value) else history_df[column].eq(value)
        matching_history = history_df.loc[mask]
        if matching_history.empty:
            raise ValueError(f"No CV history matches LightGCN champion target={target}")
        metric = selection_metric_for_target(target)
        row["replay_epochs"] = select_replay_epoch(matching_history, metric)
        row["selected_epoch_index"] = row["replay_epochs"] - 1
        row["epoch_selection_metric"] = metric
        enriched[target] = row
    return enriched


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


def protocol_version_for_split(split: str) -> str:
    return graph_protocol.protocol_version_for_train_split(split)


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
    split: str = graph_protocol.OFFICIAL_TRAIN_SPLIT,
) -> tuple[pd.DataFrame, dict]:
    df, metadata = graph_protocol.load_verified_grouped_fold_assignments(
        bench_dir, split=split, version=protocol_version
    )
    expected = set(range(graph_protocol.OFFICIAL_N_FOLDS))
    found = set(df["fold"].astype(int).unique().tolist())
    if found != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, got {sorted(found)}")
    return validate_fold_assignments(df, bench_qids), metadata


def build_subset_bench(src_dir: Path, qids: set[str], dst_dir: Path) -> None:
    payload = json.loads((src_dir / "bench_global.json").read_text())
    questions = payload["questions"]
    ids = np.load(src_dir / "questions_ids.npy", allow_pickle=True)
    emb = np.load(src_dir / "questions_emb.npy")
    keep_mask = np.array([str(qid) in qids for qid in ids.tolist()], dtype=bool)
    filtered_questions = [q for q in questions if str(q["qid"]) in qids]
    filtered_qids = ids[keep_mask]
    filtered_emb = emb[keep_mask]
    if len(filtered_questions) != len(filtered_qids):
        qids_from_questions = {str(q["qid"]) for q in filtered_questions}
        qids_from_arrays = {str(qid) for qid in filtered_qids.tolist()}
        raise ValueError(
            "subset bench mismatch: "
            f"questions_only={sorted(qids_from_questions - qids_from_arrays)} "
            f"arrays_only={sorted(qids_from_arrays - qids_from_questions)}"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "bench_global.json").write_text(
        json.dumps({"questions": filtered_questions}, ensure_ascii=False, indent=2)
    )
    np.save(dst_dir / "questions_ids.npy", filtered_qids)
    np.save(dst_dir / "questions_emb.npy", filtered_emb)
    source_projection = src_dir / benchmark_labels.LIGHTGCN_PROJECTION_FILENAME
    if source_projection.exists():
        benchmark_labels.write_subset_lightgcn_article_positive_projection(
            src_dir,
            dst_dir,
            qids={str(qid) for qid in filtered_qids.tolist()},
        )


def run_lightgcn_config(
    train_bench_dir: Path,
    val_bench_dir: Path,
    *,
    graph_version: str,
    fold: int,
    train_k: int,
    seed: int,
    lr: float,
    epochs: int,
    lambda_anchor: float,
    negative_sampling_strategy: str,
    selection_target: str,
    include_baselines: bool,
    train_variant: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    suffix = (
        f"fold{fold}_{selection_target}_k{train_k}_s{seed}_lr{lr:g}_e{epochs}_la{lambda_anchor:g}_neg{negative_sampling_strategy}"
        .replace(".", "p")
    )
    args = [
        "--train-bench-dir",
        str(train_bench_dir),
        "--eval-bench-dir",
        str(val_bench_dir),
        "--graph-version",
        graph_version,
        "--train-k",
        str(train_k),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--epochs",
        str(epochs),
        "--lambda-anchor",
        str(lambda_anchor),
        "--negative-sampling-strategy",
        negative_sampling_strategy,
        "--selection-metric",
        selection_metric_for_target(selection_target),
        "--output-suffix",
        suffix,
    ]
    if not train_variant:
        args.append("--notrain")
    if not include_baselines:
        args.append("--trained-only")
    rc = lightgcn.main(args)
    if rc != 0:
        raise RuntimeError(f"LightGCN run failed for suffix={suffix} rc={rc}")
    raw_df = pd.read_csv(val_bench_dir / f"lightgcn_eval_{suffix}.csv")
    history_path = val_bench_dir / f"lightgcn_history_{suffix}.csv"
    history_df = pd.read_csv(history_path) if history_path.exists() else None
    return raw_df, history_df


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
    df = df.copy()
    if "selection_target" not in df.columns:
        df["selection_target"] = modality
    else:
        df = df[df["selection_target"].isin([modality, "shared"])].copy()
    if n_questions_benchmark is None:
        n_questions_benchmark = int(df["qid"].nunique())
    group_cols = [
        "variant",
        "train_k",
        "seed",
        "lr",
        "epochs",
        "lambda_anchor",
        "negative_sampling_strategy",
        "graph_version",
        "selection_target",
    ]
    fold_metrics, summary = graph_protocol.summarize_fold_metrics(
        df,
        config_columns=group_cols,
        metric_columns=_metric_columns(modality),
        expected_qids_by_fold=expected_qids_by_fold,
    )
    summary["n_questions_benchmark"] = int(n_questions_benchmark)
    if expected_qids_by_fold is None:
        summary["n_questions_covered"] = int(df["qid"].nunique())
        summary["question_coverage"] = 1.0
    summary.insert(
        0,
        "method",
        summary.apply(
            lambda row: (
                f"LightGCN-{row['variant']}"
                if row["variant"].startswith("untrained_")
                else (
                    f"LightGCN-{row['variant']}"
                    f"-s{int(row['seed'])}"
                    f"-lr{float(row['lr']):g}"
                    f"-e{int(row['epochs'])}"
                    f"-la{float(row['lambda_anchor']):g}"
                    f"-neg-{row['negative_sampling_strategy']}"
                )
            ),
            axis=1,
        ),
    )
    summary.insert(1, "modality", modality)
    fold_metrics.insert(
        0,
        "method",
        fold_metrics.apply(
            lambda row: (
                f"LightGCN-{row['variant']}"
                if row["variant"].startswith("untrained_")
                else (
                    f"LightGCN-{row['variant']}"
                    f"-s{int(row['seed'])}"
                    f"-lr{float(row['lr']):g}"
                    f"-e{int(row['epochs'])}"
                    f"-la{float(row['lambda_anchor']):g}"
                    f"-neg-{row['negative_sampling_strategy']}"
                )
            ),
            axis=1,
        ),
    )
    fold_metrics.insert(1, "modality", modality)
    return fold_metrics, summary.sort_values(["method"]).reset_index(drop=True)


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
    config_columns = [
        "variant", "train_k", "seed", "lr", "epochs", "lambda_anchor",
        "negative_sampling_strategy", "selection_target",
    ]
    metric_columns = [
        metric for metric in _metric_columns(modality).values()
        if metric in candidate.columns and metric in control.columns
    ]
    deltas = graph_protocol.summarize_paired_fold_deltas(
        candidate, control, config_columns, metric_columns, expected_folds
    )
    deltas["candidate_graph_version"] = candidate["graph_version"].iloc[0] if not candidate.empty else None
    deltas["control_graph_version"] = control["graph_version"].iloc[0] if not control.empty else None
    return deltas


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
    parser.add_argument(
        "--bench-dir",
        type=Path,
        help="Explicit train benchmark directory; used by sealed cluster campaigns.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--train-k", type=int, action="append", dest="train_ks")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--lr", type=float, action="append", dest="lrs")
    parser.add_argument("--epochs", type=int, action="append", dest="epochs_list")
    parser.add_argument(
        "--lambda-anchor",
        type=float,
        action="append",
        dest="lambda_anchors",
    )
    parser.add_argument("--control-fold-metrics", type=Path)
    parser.add_argument("--refresh-legacy-surfaces", action="store_true")
    parser.add_argument(
        "--selection-target",
        action="append",
        dest="selection_targets",
        choices=("art", "jp"),
        help="Limit trained runs to one selection target; repeat to request both.",
    )
    parser.add_argument(
        "--negative-sampling-strategy",
        action="append",
        dest="negative_sampling_strategies",
        help="Peut être répété: random, hard_negative_cosine_top20, hard_negative_cosine_top50.",
    )
    args = parser.parse_args(argv)

    enforce_official_split(args.split)
    expected_protocol_version = protocol_version_for_split(args.split)
    protocol_version = args.protocol_version or expected_protocol_version
    if protocol_version != expected_protocol_version:
        raise ValueError(
            f"split={args.split} requires protocol-version={expected_protocol_version}; got {protocol_version}"
        )
    bench_dir = args.bench_dir or graph_protocol.resolve_graph_bench_dir(
        args.graph_version, args.split
    )
    bench_questions = graph_protocol.load_bench_questions(bench_dir)
    bench_qids = {str(question["qid"]) for question in bench_questions}
    folds, fold_metadata = load_fold_assignments(
        bench_dir,
        bench_qids,
        split=args.split,
        protocol_version=protocol_version,
    )
    expected_qids_by_fold = graph_protocol.expected_qids_by_fold(folds)

    train_ks = args.train_ks or [2]
    seeds = args.seeds or [42]
    lrs = args.lrs or [1e-3]
    epochs_list = args.epochs_list or [30]
    lambda_anchors = args.lambda_anchors or [1.0]
    negative_sampling_strategies = args.negative_sampling_strategies or [lightgcn.NEGATIVE_RANDOM]
    selection_targets = args.selection_targets or ["art", "jp"]
    for strategy in negative_sampling_strategies:
        lightgcn.parse_negative_sampling_strategy(strategy)
    trained_configs = list(
        itertools.product(
            train_ks,
            seeds,
            lrs,
            epochs_list,
            lambda_anchors,
            negative_sampling_strategies,
        )
    )
    out_dir = args.out_dir or (
        graph_protocol.cv_root(graph_protocol.BENCH_ROOT, protocol_version)
        / args.graph_version
        / "lightgcn"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []

    for fold in sorted(folds["fold"].astype(int).unique()):
        val_qids = set(folds.loc[folds["fold"] == fold, "qid"].astype(str))
        train_qids = bench_qids - val_qids
        with tempfile.TemporaryDirectory(prefix=f"cv_lightgcn_fold{fold}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_dir = tmp_path / "train"
            val_dir = tmp_path / "val"
            build_subset_bench(bench_dir, train_qids, train_dir)
            build_subset_bench(bench_dir, val_qids, val_dir)

            baseline_df, _ = run_lightgcn_config(
                train_dir,
                val_dir,
                graph_version=args.graph_version,
                fold=fold,
                train_k=train_ks[0],
                seed=seeds[0],
                lr=lrs[0],
                epochs=epochs_list[0],
                lambda_anchor=lambda_anchors[0],
                negative_sampling_strategy=lightgcn.NEGATIVE_RANDOM,
                selection_target="art",
                include_baselines=True,
                train_variant=False,
            )
            baseline_df = baseline_df[baseline_df["variant"] != "cosine_raw"].copy()
            baseline_df.insert(0, "fold", fold)
            baseline_df["train_k"] = np.where(
                baseline_df["variant"].str.contains("_K"),
                baseline_df["variant"].str.rsplit("K", n=1).str[-1].astype(int),
                train_ks[0],
            )
            baseline_df["seed"] = np.nan
            baseline_df["lr"] = np.nan
            baseline_df["epochs"] = np.nan
            baseline_df["lambda_anchor"] = np.nan
            baseline_df["negative_sampling_strategy"] = lightgcn.NEGATIVE_RANDOM
            baseline_df["graph_version"] = args.graph_version
            baseline_df["selection_target"] = "shared"
            raw_parts.append(baseline_df)

            for train_k, seed, lr, epochs, lambda_anchor, negative_sampling_strategy in trained_configs:
                for selection_target in selection_targets:
                    trained_df, history_df = run_lightgcn_config(
                        train_dir,
                        val_dir,
                        graph_version=args.graph_version,
                        fold=fold,
                        train_k=train_k,
                        seed=seed,
                        lr=lr,
                        epochs=epochs,
                        lambda_anchor=lambda_anchor,
                        negative_sampling_strategy=negative_sampling_strategy,
                        selection_target=selection_target,
                        include_baselines=False,
                        train_variant=True,
                    )
                    trained_df.insert(0, "fold", fold)
                    trained_df["train_k"] = train_k
                    trained_df["seed"] = seed
                    trained_df["lr"] = lr
                    trained_df["epochs"] = epochs
                    trained_df["lambda_anchor"] = lambda_anchor
                    trained_df["negative_sampling_strategy"] = negative_sampling_strategy
                    trained_df["graph_version"] = args.graph_version
                    trained_df["selection_target"] = selection_target
                    raw_parts.append(trained_df)
                    if history_df is not None and not history_df.empty:
                        history_df.insert(0, "fold", fold)
                        history_df["train_k"] = train_k
                        history_df["seed"] = seed
                        history_df["lr"] = lr
                        history_df["epochs"] = epochs
                        history_df["lambda_anchor"] = lambda_anchor
                        history_df["negative_sampling_strategy"] = negative_sampling_strategy
                        history_df["selection_target"] = selection_target
                        history_parts.append(history_df)
                        history_suffix = (
                            f"fold{fold}_{selection_target}_k{train_k}_s{seed}_lr{lr:g}_e{epochs}_la{lambda_anchor:g}_neg{negative_sampling_strategy}"
                            .replace(".", "p")
                        )
                        history_df.to_csv(
                            out_dir / f"lightgcn_history_{history_suffix}.csv",
                            index=False,
                        )

    raw_df = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
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
    history_df = pd.concat(history_parts, ignore_index=True) if history_parts else pd.DataFrame()
    if not history_df.empty:
        champions = attach_replay_epochs(champions, history_df)
    for champion in champions.values():
        champion.update(run_metadata)

    raw_df.to_csv(out_dir / "raw.csv", index=False)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "run_metadata.json").write_text(json.dumps(run_metadata, ensure_ascii=False, indent=2))
    if not history_df.empty:
        history_df.to_csv(out_dir / "lightgcn_history_all.csv", index=False)
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
    if args.refresh_legacy_surfaces:
        refresh_results_surfaces()
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
