import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "110_run_b2_e030_llm_judge.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b2_e030_judge_execution", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _job(position: int, candidate_id: str) -> dict:
    return {
        "job_id": f"job-{position}",
        "family": "ppr_reranked_kin50",
        "modality": "jp",
        "qid": "q1",
        "position": position,
        "question": "Quelle règle juridique s'applique ?",
        "candidate_id_internal": candidate_id,
        "document": {"synthese": "Une règle juridiquement pertinente."},
        "model_id": "model",
        "model_revision": "revision",
        "temperature": 0,
    }


def test_rendered_judge_prompt_exposes_only_question_and_visible_document():
    runner = _load_runner()
    job = _job(1, "JP-1")
    job["rank"] = 1
    job["graph"] = "G6"
    job["ground_truth"] = ["JP-gold"]

    prompt = runner.render_judge_prompt(
        "QUESTION: {question}\nFICHE: {document}", job
    )

    assert "Quelle règle juridique" in prompt
    assert "Une règle juridiquement pertinente" in prompt
    assert "JP-1" not in prompt
    assert "JP-gold" not in prompt
    assert "G6" not in prompt


def test_materialization_requires_all_terminal_responses(tmp_path):
    runner = _load_runner()
    jobs = [_job(position, f"JP-{position}") for position in range(1, 11)]
    responses = [
        {
            "job_id": "job-1",
            "status": "ok",
            "judgment": {"classe": "A", "justification": "règle directe"},
        }
    ]

    with pytest.raises(ValueError, match="terminal response coverage mismatch"):
        runner.materialize_judgments(jobs, responses, tmp_path / "materialized")


def test_materialization_keeps_fixed_k_and_zeroes_later_duplicate(tmp_path):
    runner = _load_runner()
    jobs = [_job(position, "JP-1" if position in {1, 3} else f"JP-{position}") for position in range(1, 11)]
    responses = []
    for position in range(1, 11):
        label = "A" if position in {1, 3} else ("B" if position == 2 else "E")
        responses.append(
            {
                "job_id": f"job-{position}",
                "status": "ok",
                "judgment": {"classe": label, "justification": "motif"},
            }
        )

    summary = runner.materialize_judgments(jobs, responses, tmp_path / "materialized")

    assert summary == [
        {
            "family": "ppr_reranked_kin50",
            "modality": "jp",
            "judge_score_at_10": 0.15,
            "questions": 1,
        }
    ]
    positions = (tmp_path / "materialized" / "per_position_judgments.jsonl").read_text().splitlines()
    third = json.loads(positions[2])
    assert third["effective_gain"] == 0.0


def test_run_is_append_only_and_skips_terminal_response(tmp_path, monkeypatch):
    runner = _load_runner()
    job = _job(1, "JP-1")
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return {"classe": "A", "justification": "règle directe"}

    monkeypatch.setattr(runner, "call_openai_compatible", fake_call)
    responses = tmp_path / "responses.jsonl"

    first = runner.run_judgments(
        jobs=[job],
        responses_path=responses,
        endpoint="http://local/v1",
        model_id="model",
        prompt_template="QUESTION: {question}\nFICHE: {document}",
        max_tokens=64,
        max_workers=1,
    )
    second = runner.run_judgments(
        jobs=[job],
        responses_path=responses,
        endpoint="http://local/v1",
        model_id="model",
        prompt_template="QUESTION: {question}\nFICHE: {document}",
        max_tokens=64,
        max_workers=1,
    )

    assert first == {"jobs": 1, "skipped": 0, "completed": 1}
    assert second == {"jobs": 1, "skipped": 1, "completed": 0}
    assert len(calls) == 1
    assert len(responses.read_text().splitlines()) == 1


def test_zero_slot_is_not_sent_to_the_judge_but_stays_in_fixed_k_score(tmp_path, monkeypatch):
    runner = _load_runner()
    jobs = [_job(position, f"JP-{position}") for position in range(1, 11)]
    jobs[0]["candidate_id_internal"] = None
    jobs[0]["document"] = None
    jobs[0]["zero_slot"] = True
    calls = []

    def fake_call(**kwargs):
        calls.append(kwargs)
        return {"classe": "A", "justification": "règle directe"}

    monkeypatch.setattr(runner, "call_openai_compatible", fake_call)
    responses_path = tmp_path / "responses.jsonl"
    runner.run_judgments(
        jobs=jobs,
        responses_path=responses_path,
        endpoint="http://local/v1",
        model_id="model",
        prompt_template="QUESTION: {question}\nFICHE: {document}",
        max_tokens=64,
        max_workers=1,
    )
    responses = [json.loads(line) for line in responses_path.read_text().splitlines()]
    summary = runner.materialize_judgments(jobs, responses, tmp_path / "materialized")

    assert len(calls) == 9
    assert responses[0]["validation_reason"] == "unresolved_zero_slot"
    assert summary[0]["judge_score_at_10"] == 0.9
