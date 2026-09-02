"""Invariant checks shared by the B1 campaign executed on the A3 snapshot."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd


REQUIRED_HISTORICAL_EXCLUSIONS = {"E017", "E021", "E022"}
REQUIRED_CANDIDATE_COUNTS = {"articles": 13236, "jurisprudence": 114851}


def _require_equal(name: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ValueError(f"{name} mismatch: expected={expected!r}, actual={actual!r}")


def validate_b1_against_a3(
    b1: Mapping[str, Any],
    a3: Mapping[str, Any],
    *,
    a3_sha256: str,
) -> None:
    """Reject any B1 contract that could differ from the sealed A3 inputs."""
    a3_ref = b1.get("a3", {})
    _require_equal("a3.manifest_id", a3_ref.get("manifest_id"), a3.get("manifest_id"))
    _require_equal("a3.sha256", a3_ref.get("sha256"), a3_sha256)

    train_a3 = a3["datasets"]["train"]
    train_b1 = b1["datasets"]["train"]
    for field in ("split", "questions", "sha256"):
        _require_equal(f"datasets.train.{field}", train_b1.get(field), train_a3.get(field))

    eval_a3 = a3["datasets"]["evaluation"]
    eval_b1 = b1["datasets"]["evaluation"]
    for field in ("split", "questions", "sha256"):
        _require_equal(f"datasets.evaluation.{field}", eval_b1.get(field), eval_a3.get(field))

    folds_a3 = a3["grouped_folds"]
    folds_b1 = b1["folds"]
    _require_equal("protocol_version", b1.get("protocol_version"), folds_a3.get("protocol_version"))
    _require_equal("folds.count", folds_b1.get("count"), 5)
    for field in ("seed", "sha256", "metadata_sha256"):
        _require_equal(f"folds.{field}", folds_b1.get(field), folds_a3.get(field))

    _require_equal("graphs", list(b1.get("graphs", [])), list(a3.get("main_protocol_graphs", [])))
    a3_candidates = a3["candidate_universes"]["retrieval_candidate_universe"]
    b1_candidates = b1.get("candidate_universe", {})
    for modality in ("articles", "jurisprudence"):
        expected_count = REQUIRED_CANDIDATE_COUNTS[modality]
        _require_equal(
            f"candidate_universe.{modality}.count",
            b1_candidates.get(modality, {}).get("count"),
            expected_count,
        )
        _require_equal(
            f"candidate_universe.{modality}.a3_count",
            a3_candidates[modality].get("unique_ids"),
            expected_count,
        )
        _require_equal(
            f"candidate_universe.{modality}.order_sha256",
            b1_candidates.get(modality, {}).get("order_sha256"),
            a3_candidates[modality].get("stable_unique_sequence_sha256"),
        )

    metrics = b1.get("metrics", {})
    _require_equal("metrics.primary", metrics.get("primary"), "normalized_hit_at_k")
    _require_equal(
        "metrics.formula", metrics.get("formula"), "dedup_intersection_over_min_gold_k"
    )
    _require_equal(
        "metrics.article_hit_equals_recall_at_10",
        metrics.get("article_hit_equals_recall_at_10"),
        True,
    )
    exclusions = set(map(str, b1.get("historical_experiments_excluded", [])))
    if not REQUIRED_HISTORICAL_EXCLUSIONS.issubset(exclusions):
        raise ValueError("historical_experiments_excluded must retain E017, E021 and E022")


def deduplicated_top_k(ranking: Iterable[object], k: int) -> list[str]:
    """Keep first occurrences among the first K returned positions only."""
    if k <= 0:
        raise ValueError("k must be positive")
    seen: set[str] = set()
    output: list[str] = []
    for item in list(ranking)[:k]:
        identifier = str(item)
        if identifier not in seen:
            seen.add(identifier)
            output.append(identifier)
    return output


def normalized_hit_at_k(ranking: Iterable[object], gold: Iterable[object], k: int) -> float:
    """Official B1 Hit@K: deduplicated coverage divided by min(|GT|, K)."""
    gold_ids = {str(item) for item in gold}
    if not gold_ids:
        raise ValueError("normalized Hit@K is undefined for an empty ground-truth set")
    denominator = min(len(gold_ids), k)
    return len(set(deduplicated_top_k(ranking, k)) & gold_ids) / float(denominator)


def recall_at_k(ranking: Iterable[object], gold: Iterable[object], k: int) -> float:
    """Standard Recall@K on the same deduplicated top-K ranking."""
    gold_ids = {str(item) for item in gold}
    if not gold_ids:
        raise ValueError("Recall@K is undefined for an empty ground-truth set")
    return len(set(deduplicated_top_k(ranking, k)) & gold_ids) / float(len(gold_ids))


def select_complete_cv_champions(
    rows: pd.DataFrame,
    *,
    expected_train_split: str,
) -> dict[str, dict[str, Any]]:
    """Choose one champion per task from complete train/CV rows only."""
    required = {
        "target",
        "graph_version",
        "primary_mean",
        "ndcg_at_10_mean",
        "mrr_at_10_mean",
        "n_folds_covered",
        "question_coverage",
        "dataset_split",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"CV summary is missing required column: {missing[0]}")
    if rows.empty:
        raise ValueError("CV summary is empty")
    if not rows["dataset_split"].astype(str).eq(expected_train_split).all():
        raise ValueError("CV summary contains a dataset_split outside the B1 train snapshot")

    champions: dict[str, dict[str, Any]] = {}
    for target in ("art", "jp"):
        candidates = rows.loc[
            rows["target"].astype(str).eq(target)
            & rows["n_folds_covered"].eq(5)
            & rows["question_coverage"].eq(1.0)
        ].copy()
        if candidates.empty:
            raise ValueError(f"No complete five-fold CV candidate for target={target}")
        candidates = candidates.sort_values(
            ["primary_mean", "ndcg_at_10_mean", "mrr_at_10_mean", "graph_version"],
            ascending=[False, False, False, True],
            kind="stable",
        )
        champions[target] = candidates.iloc[0].to_dict()
    return champions
