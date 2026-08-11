import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "76_run_g7_graded_jp_judge.py"
SPEC = importlib.util.spec_from_file_location("g7_graded_jp_runner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def blind_job(job_id="job-1"):
    return {
        "job_id": job_id,
        "qid": "q1",
        "jp_id": "jp1",
        "question": "Question ?",
        "decision_card": {"synthese_pour_avocat": "Synthèse de la décision."},
        "judge_contract": {
            "model_id": "model",
            "prompt_version": "v1",
            "prompt_sha256": "prompt-hash",
            "schema_sha256": "schema-hash",
        },
    }


def test_make_prompt_contains_only_question_and_card():
    prompt = MODULE.make_prompt("Q={question}\nJP={decision_card}", blind_job())
    assert "Question ?" in prompt
    assert "Synthèse de la décision." in prompt
    assert "rank" not in prompt
    assert "LightGCN" not in prompt


def test_load_done_retries_invalid_and_error_rows(tmp_path):
    responses = tmp_path / "responses.jsonl"
    responses.write_text(
        "".join(
            json.dumps({"job_id": job_id, "status": status}) + "\n"
            for job_id, status in (("ok", "ok"), ("invalid", "invalid"), ("error", "error"))
        ),
        encoding="utf-8",
    )
    assert MODULE.load_done(responses, retry_non_ok=False) == {"ok", "invalid", "error"}
    assert MODULE.load_done(responses, retry_non_ok=True) == {"ok"}


def test_mock_run_writes_one_valid_terminal_response_per_job(tmp_path):
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(
        "".join(json.dumps(blind_job(f"job-{index}"), ensure_ascii=False) + "\n" for index in range(3)),
        encoding="utf-8",
    )
    responses_path = tmp_path / "responses.jsonl"

    summary = MODULE.run_jobs(
        jobs_path=jobs_path,
        responses_path=responses_path,
        judge=MODULE.mock_judge,
        workers=2,
        retry_non_ok=False,
        limit=None,
    )

    rows = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
    assert summary == {"loaded": 3, "skipped": 0, "submitted": 3, "ok": 3, "invalid": 0, "error": 0}
    assert {row["job_id"] for row in rows} == {"job-0", "job-1", "job-2"}
    assert all(row["status"] == "ok" for row in rows)
    assert all(set(row["response"]) == {"classe", "justification"} for row in rows)


def test_verify_contract_rejects_jobs_from_another_prompt():
    expected = blind_job()["judge_contract"]
    changed = blind_job("other")
    changed["judge_contract"] = {**expected, "prompt_sha256": "changed"}
    with pytest.raises(ValueError, match="judge contract mismatch"):
        MODULE.verify_contracts([blind_job(), changed], expected)
