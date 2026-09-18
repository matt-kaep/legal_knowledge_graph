import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "121_run_b2_e030_judge_fail_closed_v3.py"


def test_http_failure_is_raised_not_serialized_as_a_judge_response(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("fail_closed", SCRIPT)
    runner = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(runner)
    job = {"job_id":"j1","family":"f","modality":"jp","qid":"q1","position":1,"question":"Q","candidate_id_internal":"x","document":{"synthese":"S"},"model_id":"m","model_revision":"r","temperature":0}
    jobs = tmp_path / "jobs.jsonl"; jobs.write_text(json.dumps(job)+"\n")
    prompt = tmp_path / "prompt.txt"; prompt.write_text("{question} {document}")
    monkeypatch.setattr(runner.base, "call_openai_compatible", lambda **_: (_ for _ in ()).throw(RuntimeError("HTTP Error 404")))
    with pytest.raises(RuntimeError, match="HTTP Error 404"):
        runner.run(jobs_path=jobs, responses_path=tmp_path / "responses.jsonl", endpoint="http://x/v1", frozen_model="m", served_model="m@r", prompt_path=prompt, max_tokens=1, max_workers=1)
    assert "\"status\":\"error\"" not in (tmp_path / "responses.jsonl").read_text()
