import pytest
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from metrics import hit_at_k, m2_rank, ndcg_at_k  # noqa: E402


def test_hit_at_k_scores_fraction_of_reachable_gt():
    ranked = ["a", "x", "b", "y", "z"]
    gt = {"a", "b", "c", "d"}

    assert hit_at_k(ranked, gt, k=10) == 0.5


def test_hit_at_k_is_capped_by_top_k_capacity():
    ranked = ["a", "b", "x", "y"]
    gt = {"a", "b", "c", "d"}

    assert hit_at_k(ranked, gt, k=2) == 1.0


def test_metrics_deduplicate_ranked_items_before_scoring():
    ranked = ["a", "a", "a", "x"]
    gt = {"a"}

    assert hit_at_k(ranked, gt, k=4) == 1.0
    assert ndcg_at_k(ranked, gt, k=4) == 1.0
    assert m2_rank(ranked, gt, k=4) == pytest.approx(1.0)
