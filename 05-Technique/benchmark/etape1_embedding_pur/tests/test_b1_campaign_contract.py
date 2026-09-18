import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "b1_campaign_contract.py"


def _load_contract():
    spec = importlib.util.spec_from_file_location("b1_campaign_contract", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _a3_payload():
    return {
        "manifest_id": "checkpoint-a3-effective-retrieval-universe-2026-09-02",
        "datasets": {
            "train": {"split": "train-a3", "questions": 5578, "sha256": "train-sha"},
            "evaluation": {"split": "eval-a3", "questions": 754, "sha256": "eval-sha"},
        },
        "grouped_folds": {
            "protocol_version": "grouped-v5-a3",
            "seed": 42,
            "sha256": "fold-sha",
            "metadata_sha256": "fold-meta-sha",
        },
        "main_protocol_graphs": ["G1", "G6"],
        "candidate_universes": {
            "retrieval_candidate_universe": {
                "articles": {"unique_ids": 13236, "stable_unique_sequence_sha256": "articles-order"},
                "jurisprudence": {"unique_ids": 114851, "stable_unique_sequence_sha256": "jp-order"},
            }
        },
    }


def _b1_payload():
    return {
        "campaign_id": "b1-a3",
        "protocol_version": "grouped-v5-a3",
        "a3": {
            "manifest_id": "checkpoint-a3-effective-retrieval-universe-2026-09-02",
            "sha256": "a3-sha",
        },
        "datasets": {
            "train": {"split": "train-a3", "questions": 5578, "sha256": "train-sha"},
            "evaluation": {"split": "eval-a3", "questions": 754, "sha256": "eval-sha"},
        },
        "folds": {"count": 5, "seed": 42, "sha256": "fold-sha", "metadata_sha256": "fold-meta-sha"},
        "graphs": ["G1", "G6"],
        "candidate_universe": {
            "articles": {"count": 13236, "order_sha256": "articles-order"},
            "jurisprudence": {"count": 114851, "order_sha256": "jp-order"},
        },
        "metrics": {
            "primary": "normalized_hit_at_k",
            "formula": "dedup_intersection_over_min_gold_k",
            "article_hit_equals_recall_at_10": True,
        },
        "historical_experiments_excluded": ["E017", "E021", "E022"],
    }


def test_b1_contract_requires_exact_a3_identity_and_candidate_universe():
    contract = _load_contract()

    contract.validate_b1_against_a3(_b1_payload(), _a3_payload(), a3_sha256="a3-sha")

    payload = _b1_payload()
    payload["candidate_universe"]["articles"]["count"] = 13235
    with pytest.raises(ValueError, match="candidate_universe.articles.count"):
        contract.validate_b1_against_a3(payload, _a3_payload(), a3_sha256="a3-sha")


def test_normalized_hit_deduplicates_rankings_and_matches_recall_when_gold_at_most_k():
    contract = _load_contract()

    ranking = ["a", "a", "b", "c"]
    gold = {"a", "b", "z"}

    assert contract.normalized_hit_at_k(ranking, gold, 3) == pytest.approx(2 / 3)
    assert contract.recall_at_k(ranking, gold, 10) == pytest.approx(2 / 3)


def test_champion_selection_rejects_incomplete_cv_and_tie_breaks_on_ndcg_then_mrr():
    contract = _load_contract()
    rows = pd.DataFrame(
        [
            {
                "target": "art",
                "graph_version": "G1",
                "primary_mean": 0.50,
                "ndcg_at_10_mean": 0.40,
                "mrr_at_10_mean": 0.30,
                "n_folds_covered": 5,
                "question_coverage": 1.0,
                "dataset_split": "train-a3",
            },
            {
                "target": "art",
                "graph_version": "G6",
                "primary_mean": 0.99,
                "ndcg_at_10_mean": 0.99,
                "mrr_at_10_mean": 0.99,
                "n_folds_covered": 4,
                "question_coverage": 0.99,
                "dataset_split": "train-a3",
            },
            {
                "target": "jp",
                "graph_version": "G1",
                "primary_mean": 0.51,
                "ndcg_at_10_mean": 0.41,
                "mrr_at_10_mean": 0.30,
                "n_folds_covered": 5,
                "question_coverage": 1.0,
                "dataset_split": "train-a3",
            },
            {
                "target": "jp",
                "graph_version": "G6",
                "primary_mean": 0.51,
                "ndcg_at_10_mean": 0.41,
                "mrr_at_10_mean": 0.31,
                "n_folds_covered": 5,
                "question_coverage": 1.0,
                "dataset_split": "train-a3",
            },
        ]
    )

    selected = contract.select_complete_cv_champions(rows, expected_train_split="train-a3")

    assert selected["art"]["graph_version"] == "G1"
    assert selected["jp"]["graph_version"] == "G6"
