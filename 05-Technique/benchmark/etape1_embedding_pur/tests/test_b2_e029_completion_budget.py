import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "112_audit_b2_e029_completion_budget.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("b2_e029_completion_budget", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _job(qid: str, candidates: list[str]) -> dict:
    return {
        "family": "cosine",
        "modality": "article",
        "qid": qid,
        "k_in": len(candidates),
        "k_out": 2,
        "replay_seed": None,
        "candidates": [{"item_id": item} for item in candidates],
    }


def test_completion_audit_uses_a_conservative_bound_for_each_condition():
    auditor = _load_auditor()
    jobs = [_job("q1", ["a", "long-id"]), _job("q2", ["a", "long-id"])]

    report = auditor.audit_completion_budget(
        jobs,
        count_tokens=lambda text: len(text),
        expected_questions=2,
        max_output_tokens=40,
    )

    assert report["compatible_conditions"] == 1
    condition = report["conditions"][0]
    assert condition["max_conservative_completion_tokens"] > 0
    assert condition["compatible"] is True


def test_completion_audit_rejects_a_budget_below_the_observed_bound():
    auditor = _load_auditor()
    jobs = [_job("q1", ["very-long-candidate-id", "another-long-candidate-id"])]

    with pytest.raises(ValueError, match="completion budget overflow"):
        auditor.audit_completion_budget(
            jobs,
            count_tokens=lambda text: len(text),
            expected_questions=1,
            max_output_tokens=5,
        )
