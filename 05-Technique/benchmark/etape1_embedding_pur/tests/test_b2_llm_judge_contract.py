import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "b2_llm_as_judge_a3_preflight_v1.json"


def test_b2_judge_preflight_uses_shared_fixed_k_grading_contract():
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == "E030"
    assert payload["status"] == "preparation_only_no_model_call_no_frozen_lists"
    assert payload["execution_gate"]["model_calls_authorized"] is False
    assert payload["grading"]["labels"] == ["A", "B", "C", "D", "E", "non_jugeable"]
    assert payload["grading"]["gains"] == {
        "A": 1.0,
        "B": 0.5,
        "C": 0.0,
        "D": 0.0,
        "E": 0.0,
        "non_jugeable": 0.0,
    }
    assert payload["grading"]["k"] == 10
    assert payload["grading"]["duplicate_candidate_gain"] == 0.0
    assert set(payload["input_contract"]["modalities"]) == {"article", "jp"}
