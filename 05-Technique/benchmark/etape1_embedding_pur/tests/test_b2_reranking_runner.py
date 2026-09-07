import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "102_run_b2_comparable_reranking.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b2_reranking_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reranker_accepts_only_pool_ids_and_completes_duplicate_slots():
    runner = _load_runner()

    parsed = runner.parse_ranked_ids(json.dumps({"ranked_ids": ["b", "b", "a"]}), {"a", "b", "c"}, 3)
    normalized, fallback = runner.normalize_ranked_ids(parsed, ["a", "b", "c"], 3)

    assert normalized == ["b", "a", "c"]
    assert fallback == ["c"]
    with pytest.raises(runner.InvalidRerankingResponse, match="supplied pool"):
        runner.parse_ranked_ids(json.dumps({"ranked_ids": ["outside", "a", "b"]}), {"a", "b", "c"}, 3)


def test_reranker_prompt_contains_only_one_real_pool_and_its_question():
    runner = _load_runner()
    job = {
        "family": "cosine",
        "modality": "article",
        "qid": "q1",
        "question": "Question ?",
        "k_in": 2,
        "k_out": 10,
        "candidates": [
            {"item_id": "a", "text": "Texte A", "source_rank": 1},
            {"item_id": "b", "text": "Texte B", "source_rank": 2},
        ],
    }

    prompt = runner.render_reranking_prompt("Instruction fixe.", job)

    assert "Question ?" in prompt
    assert "Texte A" in prompt and "Texte B" in prompt
    assert '"item_id": "a"' in prompt
    assert "PPR" not in prompt


def test_invalid_reranker_response_is_explicit_zero_not_pool_completion():
    runner = _load_runner()

    slots = runner.zero_slots(3, resolution="invalid_response")

    assert [slot["resolved_item_id"] for slot in slots] == [None, None, None]
    assert [slot["resolution"] for slot in slots] == ["invalid_response"] * 3
