from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "70_aggregate_e021_metrics.py"
SPEC = importlib.util.spec_from_file_location("e021_metrics", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_aggregation_keeps_same_qid_separate_by_family():
    questions = [{"qid": "q1", "gold_jp_ids": ["gold"]}]
    jobs = [
        {"family": "cosine_bge_m3", "qid": "q1"},
        {"family": "ppr", "qid": "q1"},
    ]
    responses = [
        {"family": "cosine_bge_m3", "qid": "q1", "status": "ok", "ranked_jp_ids": ["gold"]},
        {"family": "ppr", "qid": "q1", "status": "ok", "ranked_jp_ids": ["other"]},
    ]

    result = MODULE.aggregate_metrics(questions, jobs, responses, k=1)

    assert result["families"]["cosine_bge_m3"]["metrics"]["official_hit_at_10"] == 1.0
    assert result["families"]["ppr"]["metrics"]["official_hit_at_10"] == 0.0


def test_aggregation_reports_missing_questions_without_imputation():
    questions = [
        {"qid": "q1", "gold_jp_ids": ["gold"]},
        {"qid": "q2", "gold_jp_ids": ["gold-2"]},
    ]
    jobs = [{"family": "ppr", "qid": "q1"}, {"family": "ppr", "qid": "q2"}]
    responses = [
        {"family": "ppr", "qid": "q1", "status": "ok", "ranked_jp_ids": ["gold"]}
    ]

    result = MODULE.aggregate_metrics(questions, jobs, responses, k=1)
    family = result["families"]["ppr"]

    assert family["valid_responses"] == 1
    assert family["missing_questions"] == 1
    assert family["status"] == "incomplete_missing_responses"
    assert family["coverage"] == 0.5
