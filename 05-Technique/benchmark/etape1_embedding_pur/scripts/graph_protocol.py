from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

REPO = Path(os.environ.get(
    "LKG_REPO",
    str(Path(__file__).resolve().parents[4]),
))
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(REPO))).expanduser().resolve()
BENCH_ROOT = DATA_REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
OFFICIAL_TRAIN_SPLIT = "train_augmented_retrievable_strict"
NO_EVAL_OVERLAP_TRAIN_SPLIT = "train_augmented_retrievable_strict_no_eval_overlap_v1"
CANDIDATE_COVERED_TRAIN_SPLIT = "train_augmented_retrievable_strict_no_eval_overlap_candidate_covered_v2"
OFFICIAL_N_FOLDS = 5
SHARED_PROTOCOL_DIRNAME = "_protocol"
LEGACY_GRAPH_ALIASES = {"G0", "canonical"}
PROTOCOL_VERSION = "grouped_v2"
NO_EVAL_OVERLAP_PROTOCOL_VERSION = "grouped_v3_no_eval_overlap_v1"
CANDIDATE_COVERED_PROTOCOL_VERSION = "grouped_v4_no_eval_overlap_candidate_coverage_v2"
TRAIN_SPLIT_PROTOCOLS = {
    OFFICIAL_TRAIN_SPLIT: PROTOCOL_VERSION,
    NO_EVAL_OVERLAP_TRAIN_SPLIT: NO_EVAL_OVERLAP_PROTOCOL_VERSION,
    CANDIDATE_COVERED_TRAIN_SPLIT: CANDIDATE_COVERED_PROTOCOL_VERSION,
}
PRIMARY_METRICS = {"article": "recall_at_10", "jp": "hit_at_10"}
SECONDARY_METRICS = ("ndcg_at_10", "mrr_at_10")
T95_DF4 = 2.776


def protocol_root(bench_root: Path, version: str = PROTOCOL_VERSION) -> Path:
    return bench_root / SHARED_PROTOCOL_DIRNAME / version


def cv_root(bench_root: Path, version: str = PROTOCOL_VERSION) -> Path:
    return bench_root / f"_cv_{version}"


def final_root(bench_root: Path, version: str = PROTOCOL_VERSION) -> Path:
    return bench_root / f"_final_{version}"


def resolve_graph_bench_dir(graph_version: str, split: str) -> Path:
    graph_dir = BENCH_ROOT / graph_version / split
    if graph_dir.exists():
        return graph_dir
    legacy_dir = BENCH_ROOT / split
    if legacy_dir.exists() and (
        graph_version in LEGACY_GRAPH_ALIASES or split in TRAIN_SPLIT_PROTOCOLS
    ):
        return legacy_dir
    raise FileNotFoundError(
        "Missing bench directory for "
        f"graph_version={graph_version} split={split}. "
        f"Expected {graph_dir}"
    )


def resolve_official_train_bench_dir() -> Path:
    return BENCH_ROOT / OFFICIAL_TRAIN_SPLIT


def protocol_version_for_train_split(split: str) -> str:
    try:
        return TRAIN_SPLIT_PROTOCOLS[split]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported train split for shared CV: {split}; allowed={sorted(TRAIN_SPLIT_PROTOCOLS)}"
        ) from exc


def resolve_shared_protocol_dir(
    split: str = OFFICIAL_TRAIN_SPLIT,
    version: str | None = None,
) -> Path:
    if version is None:
        return BENCH_ROOT / SHARED_PROTOCOL_DIRNAME / split
    return protocol_root(BENCH_ROOT, version) / split


def resolve_shared_fold_paths(
    split: str = OFFICIAL_TRAIN_SPLIT,
    version: str | None = None,
) -> tuple[Path, Path]:
    protocol_dir = resolve_shared_protocol_dir(split, version)
    metadata_filename = "fold_metadata.json" if version is not None else "fold_assignments_meta.json"
    return protocol_dir / "fold_assignments.csv", protocol_dir / metadata_filename


def load_bench_questions(bench_dir: Path) -> list[dict]:
    payload = json.loads((bench_dir / "bench_global.json").read_text())
    return list(payload["questions"])


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_grouped_fold_assignments(
    bench_dir: Path,
    split: str = OFFICIAL_TRAIN_SPLIT,
    version: str = PROTOCOL_VERSION,
) -> tuple[pd.DataFrame, dict]:
    """Load grouped folds only after validating their recorded provenance."""
    fold_csv, metadata_path = resolve_shared_fold_paths(split, version)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing grouped fold metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("protocol_version") != version:
        raise ValueError(
            f"protocol_version mismatch: expected {version}, got {metadata.get('protocol_version')}"
        )
    dataset_sha256 = sha256_file(bench_dir / "bench_global.json")
    fold_assignment_sha256 = sha256_file(fold_csv)
    for key, actual in {
        "dataset_sha256": dataset_sha256,
        "fold_assignment_sha256": fold_assignment_sha256,
    }.items():
        if metadata.get(key) != actual:
            raise ValueError(f"{key} mismatch for grouped folds")
    return pd.read_csv(fold_csv), metadata


def expected_qids_by_fold(assignments: pd.DataFrame) -> dict[int, set[str]]:
    return {
        int(fold): set(group["qid"].astype(str).tolist())
        for fold, group in assignments.groupby("fold", sort=False)
    }


def summarize_fold_metrics(
    raw: pd.DataFrame,
    config_columns: Sequence[str],
    metric_columns: Mapping[str, str],
    expected_folds: int = OFFICIAL_N_FOLDS,
    expected_qids_by_fold: Mapping[int, set[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate question-level scores into fold means, then CV statistics."""
    required_columns = ["fold", *config_columns, *metric_columns.values()]
    if expected_qids_by_fold is not None:
        required_columns.append("qid")
    missing = [column for column in required_columns if column not in raw.columns]
    if missing:
        raise KeyError(f"Missing CV aggregation columns: {missing}")
    if raw.empty:
        return pd.DataFrame(columns=required_columns), pd.DataFrame()

    if expected_qids_by_fold is not None:
        identity_columns = [*config_columns, "fold", "qid"]
        duplicated = raw.duplicated(identity_columns, keep=False)
        if duplicated.any():
            sample = raw.loc[duplicated, identity_columns].head(5).to_dict(orient="records")
            raise ValueError(f"duplicate question rows within config/fold: {sample}")

    raw_metric_columns = list(metric_columns.values())
    fold_metrics = (
        raw.groupby([*config_columns, "fold"], dropna=False)[raw_metric_columns]
        .mean()
        .reset_index()
        .sort_values([*config_columns, "fold"], na_position="last")
        .reset_index(drop=True)
    )
    expected_fold_ids = (
        {int(fold) for fold in expected_qids_by_fold}
        if expected_qids_by_fold is not None
        else set(range(expected_folds))
    )
    expected_qids_by_fold = (
        {int(fold): {str(qid) for qid in qids} for fold, qids in expected_qids_by_fold.items()}
        if expected_qids_by_fold is not None
        else None
    )
    rows: list[dict] = []
    group_key: str | list[str] = config_columns[0] if len(config_columns) == 1 else list(config_columns)
    for values, group in fold_metrics.groupby(group_key, dropna=False, sort=False):
        values = (values,) if len(config_columns) == 1 else values
        row = dict(zip(config_columns, values, strict=True))
        found_fold_ids = {int(fold) for fold in group["fold"].tolist()}
        row["n_folds_covered"] = len(found_fold_ids)
        row["expected_folds"] = len(expected_fold_ids)
        row["fold_coverage"] = len(found_fold_ids) / len(expected_fold_ids)
        qid_coverage_complete = True
        if expected_qids_by_fold is not None:
            config_mask = pd.Series(True, index=raw.index)
            for column, value in zip(config_columns, values, strict=True):
                config_mask &= raw[column].isna() if pd.isna(value) else raw[column].eq(value)
            observed_qids_by_fold = {
                int(fold): set(
                    raw.loc[
                        (raw["fold"] == fold)
                        & config_mask,
                        "qid",
                    ].astype(str).tolist()
                )
                for fold in expected_fold_ids
            }
            qid_coverage_complete = all(
                observed_qids_by_fold[fold] == expected_qids_by_fold[fold]
                for fold in expected_fold_ids
            )
            expected_qids = set().union(*expected_qids_by_fold.values())
            observed_qids = set().union(*observed_qids_by_fold.values())
            row["n_questions_expected"] = len(expected_qids)
            row["n_questions_covered"] = len(observed_qids)
            row["question_coverage"] = len(observed_qids) / len(expected_qids)
        row["eligible_champion"] = found_fold_ids == expected_fold_ids and qid_coverage_complete
        for output_name, raw_metric in metric_columns.items():
            values = group[raw_metric].dropna()
            n_values = len(values)
            mean = float(values.mean()) if n_values else math.nan
            std = float(values.std(ddof=1)) if n_values > 1 else math.nan
            sem = std / math.sqrt(n_values) if n_values > 1 else math.nan
            row[f"{output_name}_mean"] = mean
            row[f"{output_name}_std"] = std
            row[f"{output_name}_ci95_low"] = mean - T95_DF4 * sem if n_values > 1 else math.nan
            row[f"{output_name}_ci95_high"] = mean + T95_DF4 * sem if n_values > 1 else math.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    return fold_metrics, summary


def summarize_paired_fold_delta(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    metric: str,
    expected_folds: int = OFFICIAL_N_FOLDS,
) -> dict[str, float | int | bool]:
    """Summarize candidate-control differences after matching fold identifiers."""
    for name, frame in (("candidate", candidate), ("control", control)):
        missing = {"fold", metric} - set(frame.columns)
        if missing:
            raise KeyError(f"{name} is missing columns: {sorted(missing)}")
        if frame["fold"].duplicated().any():
            raise ValueError(f"{name} contains duplicate fold identifiers")
    paired = candidate[["fold", metric]].merge(
        control[["fold", metric]], on="fold", how="inner", suffixes=("_candidate", "_control")
    )
    expected_fold_ids = set(range(expected_folds))
    paired_fold_ids = {int(fold) for fold in paired["fold"].tolist()}
    eligible = (
        {int(fold) for fold in candidate["fold"].tolist()} == expected_fold_ids
        and {int(fold) for fold in control["fold"].tolist()} == expected_fold_ids
        and paired_fold_ids == expected_fold_ids
    )
    deltas = paired[f"{metric}_candidate"] - paired[f"{metric}_control"]
    n_values = len(deltas)
    mean = float(deltas.mean()) if n_values else math.nan
    std = float(deltas.std(ddof=1)) if n_values > 1 else math.nan
    sem = std / math.sqrt(n_values) if n_values > 1 else math.nan
    return {
        "n_folds_paired": len(paired_fold_ids),
        "expected_folds": expected_folds,
        "eligible_comparison": eligible,
        f"{metric}_delta_mean": mean,
        f"{metric}_delta_std": std,
        f"{metric}_delta_ci95_low": mean - T95_DF4 * sem if n_values > 1 else math.nan,
        f"{metric}_delta_ci95_high": mean + T95_DF4 * sem if n_values > 1 else math.nan,
    }


def summarize_paired_fold_deltas(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
    config_columns: Sequence[str],
    metric_columns: Sequence[str],
    expected_folds: int = OFFICIAL_N_FOLDS,
) -> pd.DataFrame:
    """Materialize per-configuration paired deltas from fold means."""
    rows: list[dict] = []
    group_key: str | list[str] = config_columns[0] if len(config_columns) == 1 else list(config_columns)
    for values, candidate_group in candidate.groupby(group_key, dropna=False, sort=False):
        values = (values,) if len(config_columns) == 1 else values
        row_prefix = dict(zip(config_columns, values, strict=True))
        control_mask = pd.Series(True, index=control.index)
        for column, value in row_prefix.items():
            control_mask &= control[column].isna() if pd.isna(value) else control[column].eq(value)
        control_group = control.loc[control_mask]
        for metric in metric_columns:
            if metric not in candidate_group.columns or metric not in control_group.columns:
                continue
            delta = summarize_paired_fold_delta(
                candidate_group, control_group, metric, expected_folds=expected_folds
            )
            rows.append(
                {
                    **row_prefix,
                    "metric": metric,
                    "delta_mean": delta[f"{metric}_delta_mean"],
                    "delta_std": delta[f"{metric}_delta_std"],
                    "delta_ci95_low": delta[f"{metric}_delta_ci95_low"],
                    "delta_ci95_high": delta[f"{metric}_delta_ci95_high"],
                    **delta,
                }
            )
    return pd.DataFrame(rows)


def champion_sort_columns(target: str) -> list[tuple[str, bool]]:
    normalized_target = {"art": "article", "article": "article", "jp": "jp"}.get(target)
    if normalized_target is None:
        raise ValueError(f"Unsupported champion target: {target}")
    return [
        (f"{normalized_target}_{PRIMARY_METRICS[normalized_target]}_mean", False),
        *[(f"{normalized_target}_{metric}_mean", False) for metric in SECONDARY_METRICS],
    ]


def _metric_value(row: dict, names: list[str]) -> float:
    for name in names:
        if name in row:
            return float(row[name])
    raise KeyError(names[0])


def metric_rank_tuple(row: dict, modality: str) -> tuple[float, float, float, float, float]:
    is_jp = modality.lower() == "jp"
    metric_names = (
        {
            "hit": ["hit", "hit_jp", "hit_strict"],
            "ndcg": ["ndcg", "ndcg_jp", "ndcg_strict"],
            "mrr": ["mrr", "mrr_jp", "mrr_strict"],
            "m1": ["m1", "m1_jp", "m1_strict"],
            "m2": ["m2", "m2_jp", "m2_strict"],
        }
        if is_jp
        else {
            "hit": ["hit_strict", "hit_strict_art", "hit"],
            "ndcg": ["ndcg_strict", "ndcg_strict_art", "ndcg"],
            "mrr": ["mrr_strict", "mrr_strict_art", "mrr"],
            "m1": ["m1_strict", "m1_strict_art", "m1"],
            "m2": ["m2_strict", "m2_strict_art", "m2"],
        }
    )
    return (
        _metric_value(row, metric_names["hit"]),
        _metric_value(row, metric_names["ndcg"]),
        _metric_value(row, metric_names["mrr"]),
        _metric_value(row, metric_names["m1"]),
        _metric_value(row, metric_names["m2"]),
    )
