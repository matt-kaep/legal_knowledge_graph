import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "97_build_b1_depth_curves.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("b1_depth_curves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_b1_depth_hit_uses_only_the_returned_top_k_positions():
    curves = _load_module()
    assert curves.hit_at_k(["a", "a", "b"], {"a", "b"}, 2) == pytest.approx(0.5)


def test_b1_depth_rejects_duplicate_or_outside_top100_candidates():
    curves = _load_module()
    questions = {"q1": {"articles_attendus": ["a"]}}
    rows = pd.DataFrame({
        "qid": ["q1"] * 100,
        "rank": list(range(1, 101)),
        "item_id": ["a"] * 2 + [f"x{i}" for i in range(98)],
    })
    with pytest.raises(ValueError, match="duplicate"):
        curves.score_ranking_group(rows, questions=questions, candidate_ids={"a", *[f"x{i}" for i in range(98)]}, target="articles")
