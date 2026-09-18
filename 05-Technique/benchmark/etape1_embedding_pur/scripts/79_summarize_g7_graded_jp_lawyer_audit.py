#!/usr/bin/env python3
"""Compute population-reweighted agreement for the E016 lawyer audit."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
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
DEFAULT_OUT = (
    ETAPE1
    / "data/doctrine_v3plus_bench/G7-citation-JJ-cit1-sem025-knn5"
    / "eval_rich_retrievable_strict/E016-g7-graded-jp-v1"
)


def _load_contract_module():
    path = ETAPE1 / "scripts/74_g7_graded_jp_contract.py"
    spec = importlib.util.spec_from_file_location("g7_graded_contract_lawyer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CONTRACT = _load_contract_module()
POSITIVE = {"A", "B"}


def _metric_values(frame: pd.DataFrame) -> dict[str, float | None]:
    weights = frame["sampling_weight"].astype(float).to_numpy()
    llm = frame["classe"].astype(str).to_numpy()
    lawyer = frame["classe_avocat"].astype(str).to_numpy()
    gain_llm = np.array([CONTRACT.LABEL_GAIN[label] for label in llm], dtype=float)
    gain_lawyer = np.array([CONTRACT.LABEL_GAIN[label] for label in lawyer], dtype=float)
    absolute_error = np.abs(gain_llm - gain_lawyer)
    mae = float(np.average(absolute_error, weights=weights))
    llm_positive = np.isin(llm, list(POSITIVE))
    lawyer_positive = np.isin(lawyer, list(POSITIVE))
    true_positive_weight = float(weights[llm_positive & lawyer_positive].sum())
    predicted_positive_weight = float(weights[llm_positive].sum())
    lawyer_positive_weight = float(weights[lawyer_positive].sum())
    return {
        "gain_agreement": 1.0 - mae,
        "mean_absolute_gain_error": mae,
        "positive_precision": (
            true_positive_weight / predicted_positive_weight if predicted_positive_weight else None
        ),
        "positive_recall": (
            true_positive_weight / lawyer_positive_weight if lawyer_positive_weight else None
        ),
    }


def _bootstrap_intervals(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, dict[str, float] | None]:
    if n_bootstrap <= 0:
        return {}
    rng = np.random.default_rng(seed)
    strata = [group.reset_index(drop=True) for _, group in frame.groupby("classe", sort=True)]
    samples: dict[str, list[float]] = {
        "gain_agreement": [],
        "mean_absolute_gain_error": [],
        "positive_precision": [],
        "positive_recall": [],
    }
    for _ in range(n_bootstrap):
        pieces = [
            group.iloc[rng.integers(0, len(group), size=len(group))]
            for group in strata
        ]
        values = _metric_values(pd.concat(pieces, ignore_index=True))
        for name, value in values.items():
            if value is not None:
                samples[name].append(float(value))
    intervals: dict[str, dict[str, float] | None] = {}
    for name, values in samples.items():
        if not values:
            intervals[name] = None
        else:
            low, high = np.percentile(np.asarray(values), [2.5, 97.5])
            intervals[name] = {"low": float(low), "high": float(high)}
    return intervals


def summarize_agreement(
    annotations: pd.DataFrame,
    private_key: pd.DataFrame,
    *,
    expected_count: int = 100,
    n_bootstrap: int = 2_000,
    seed: int = 42,
) -> dict:
    required_annotations = {"case_id", "classe_avocat", "justification_avocat"}
    required_key = {"case_id", "classe", "sampling_weight"}
    if required_annotations - set(annotations.columns) or required_key - set(private_key.columns):
        raise ValueError("lawyer audit files do not match the required columns")
    if annotations["case_id"].duplicated().any() or private_key["case_id"].duplicated().any():
        raise ValueError("duplicate case_id in lawyer audit")

    clean = annotations.copy()
    clean["classe_avocat"] = clean["classe_avocat"].fillna("").astype(str).str.strip()
    clean["justification_avocat"] = (
        clean["justification_avocat"].fillna("").astype(str).str.strip()
    )
    complete_ids = set(clean["case_id"].astype(str)) == set(private_key["case_id"].astype(str))
    valid_labels = clean["classe_avocat"].isin(CONTRACT.VALID_LABELS)
    filled_reasons = clean["justification_avocat"].ne("")
    complete = (
        len(clean) == expected_count
        and len(private_key) == expected_count
        and complete_ids
        and bool(valid_labels.all())
        and bool(filled_reasons.all())
    )
    if not complete:
        return {
            "experiment_id": "E016",
            "status": "incomplete_annotations",
            "expected_count": expected_count,
            "annotation_rows": int(len(clean)),
            "annotated_labels": int(valid_labels.sum()),
            "annotated_justifications": int(filled_reasons.sum()),
        }

    merged = private_key.merge(clean, on="case_id", how="inner", validate="one_to_one")
    if (merged["sampling_weight"].astype(float) <= 0).any():
        raise ValueError("sampling weights must be positive")
    metrics = _metric_values(merged)
    labels = list(CONTRACT.VALID_LABELS)
    confusion = {
        llm_label: {
            lawyer_label: float(
                merged.loc[
                    (merged["classe"] == llm_label)
                    & (merged["classe_avocat"] == lawyer_label),
                    "sampling_weight",
                ].sum()
            )
            for lawyer_label in labels
        }
        for llm_label in labels
    }
    agreement = float(metrics["gain_agreement"])
    precision = metrics["positive_precision"]
    passed = agreement >= 0.70 and precision is not None and float(precision) >= 0.85
    return {
        "experiment_id": "E016",
        "status": "complete",
        "n_annotations": int(len(merged)),
        **metrics,
        "weighted_confusion": confusion,
        "bootstrap_95_percentile": _bootstrap_intervals(
            merged, n_bootstrap=n_bootstrap, seed=seed
        ),
        "gate_thresholds": {"gain_agreement": 0.70, "positive_precision": 0.85},
        "gate_status": "passed" if passed else "failed",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit_dir = args.out_dir / "lawyer_audit"
    annotations_path = args.annotations or audit_dir / "lawyer_audit_sample.csv"
    key_path = args.private_key or audit_dir / "lawyer_audit_key.csv"
    report = summarize_agreement(
        pd.read_csv(annotations_path),
        pd.read_csv(key_path),
        expected_count=args.expected_count,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    report_path = audit_dir / "lawyer_agreement.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
