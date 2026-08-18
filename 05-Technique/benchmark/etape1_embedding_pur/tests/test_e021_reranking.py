from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "68_run_e021_reranking.py"
SPEC = importlib.util.spec_from_file_location("e021_reranking", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_parse_response_accepts_exact_distinct_candidates_from_pool():
    pool = ["jp-1", "jp-2", "jp-3"]
    parsed = MODULE.parse_ranked_ids(
        '{"ranked_jp_ids":["jp-3","jp-1"]}',
        pool,
        k_out=2,
    )

    assert parsed == ["jp-3", "jp-1"]


@pytest.mark.parametrize(
    "payload",
    [
        '{"ranked_jp_ids":["jp-1","jp-1"]}',
        '{"ranked_jp_ids":["jp-1","jp-9"]}',
        '{"ranked_jp_ids":["jp-1"]}',
        "```json\n{\"ranked_jp_ids\":[\"jp-1\",\"jp-2\"]}\n```",
    ],
)
def test_parse_response_rejects_invalid_or_non_strict_output(payload):
    with pytest.raises(MODULE.InvalidRerankerResponse):
        MODULE.parse_ranked_ids(payload, ["jp-1", "jp-2", "jp-3"], k_out=2)


def test_metrics_keep_official_hit_distinct_from_exact_any_diagnostic():
    questions = [
        {"qid": "q1", "gold_jp_ids": ["jp-a", "jp-b"]},
        {"qid": "q2", "gold_jp_ids": ["jp-c"]},
    ]
    responses = [
        {"qid": "q1", "ranked_jp_ids": ["jp-a", "jp-x"]},
        {"qid": "q2", "ranked_jp_ids": ["jp-y", "jp-z"]},
    ]

    result = MODULE.compute_metrics(questions, responses, k=2)

    assert result["official_hit_at_10"] == pytest.approx(0.25)
    assert result["exact_any_gold_at_10"] == pytest.approx(0.5)
    assert result["mrr_at_10"] == pytest.approx(0.5)
    assert result["ndcg_at_10"] == pytest.approx(0.3065735963827292)


def test_compute_input_sha256_changes_when_pool_changes():
    first_job = {
        "qid": "q1",
        "family": "ppr",
        "modality": "jp",
        "question": "question",
        "candidates": [{"item_id": "a", "text": "A"}, {"item_id": "b", "text": "B"}],
        "k_in": 2,
        "k_out": 2,
    }
    second_job = {**first_job, "candidates": [{"item_id": "a", "text": "A"}, {"item_id": "c", "text": "C"}]}
    first = MODULE.compute_input_sha256(first_job)
    second = MODULE.compute_input_sha256(second_job)

    assert first != second


def test_response_format_schema_freezes_exact_output_cardinality():
    response_format = MODULE.reranker_response_format(10)
    schema = response_format["json_schema"]["schema"]
    ids = schema["properties"]["ranked_jp_ids"]

    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    assert ids["minItems"] == 10
    assert ids["maxItems"] == 10
    assert ids["uniqueItems"] is True
