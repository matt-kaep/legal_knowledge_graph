from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "72_finalize_e021_resume.py"
BATCH_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sbatch_e021_reranking_resume.sh"
SPEC = importlib.util.spec_from_file_location("e021_resume_completion", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _write_jobs(path: Path) -> None:
    rows = [
        {"family": family, "qid": qid, "input_sha256": f"input-{family}-{qid}"}
        for family in ("cosine_bge_m3", "ppr", "lightgcn")
        for qid in ("q1", "q2")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_responses(path: Path) -> None:
    rows = [
        {
            "family": family,
            "qid": qid,
            "status": "ok",
            "input_sha256": f"input-{family}-{qid}",
        }
        for family in ("cosine_bge_m3", "ppr", "lightgcn")
        for qid in ("q1", "q2")
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _metrics(*, complete: bool) -> dict:
    return {
        "families": {
            family: {
                "expected_questions": 2,
                "valid_responses": 2 if complete or family != "lightgcn" else 1,
                "missing_questions": 0 if complete or family != "lightgcn" else 1,
                "coverage": 1.0 if complete or family != "lightgcn" else 0.5,
                "status": "complete" if complete or family != "lightgcn" else "incomplete_missing_responses",
            }
            for family in ("cosine_bge_m3", "ppr", "lightgcn")
        }
    }


def test_completion_record_is_complete_only_when_every_family_has_full_coverage(tmp_path: Path):
    jobs = tmp_path / "jobs.jsonl"
    responses = tmp_path / "responses.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_jobs(jobs)
    _write_responses(responses)
    metrics.write_text(json.dumps(_metrics(complete=True)), encoding="utf-8")

    record = MODULE.build_completion_record(
        jobs_path=jobs,
        responses_path=responses,
        metrics_path=metrics,
        expected_families=("cosine_bge_m3", "ppr", "lightgcn"),
        expected_questions_per_family=2,
    )

    assert record["status"] == "complete"
    assert record["latest_valid_response_keys"] == 6
    assert record["jobs_sha256"] == hashlib.sha256(jobs.read_bytes()).hexdigest()


def test_completion_record_keeps_an_incomplete_family_visible(tmp_path: Path):
    jobs = tmp_path / "jobs.jsonl"
    responses = tmp_path / "responses.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_jobs(jobs)
    _write_responses(responses)
    metrics.write_text(json.dumps(_metrics(complete=False)), encoding="utf-8")

    record = MODULE.build_completion_record(
        jobs_path=jobs,
        responses_path=responses,
        metrics_path=metrics,
        expected_families=("cosine_bge_m3", "ppr", "lightgcn"),
        expected_questions_per_family=2,
    )

    assert record["status"] == "incomplete"
    assert record["families"]["lightgcn"]["missing_questions"] == 1


def test_completion_record_rejects_a_response_for_a_different_input_contract(tmp_path: Path):
    jobs = tmp_path / "jobs.jsonl"
    responses = tmp_path / "responses.jsonl"
    metrics = tmp_path / "metrics.json"
    _write_jobs(jobs)
    _write_responses(responses)
    records = [json.loads(line) for line in responses.read_text(encoding="utf-8").splitlines()]
    records[-1]["input_sha256"] = "stale-input"
    responses.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    metrics.write_text(json.dumps(_metrics(complete=True)), encoding="utf-8")

    record = MODULE.build_completion_record(
        jobs_path=jobs,
        responses_path=responses,
        metrics_path=metrics,
        expected_families=("cosine_bge_m3", "ppr", "lightgcn"),
        expected_questions_per_family=2,
    )

    assert record["status"] == "incomplete"
    assert not record["families"]["lightgcn"]["checks"]["response_history_has_every_key"]


def test_e021_batch_serves_the_frozen_local_model_snapshot():
    source = BATCH_SCRIPT.read_text(encoding="utf-8")

    assert 'MODEL_SNAPSHOT="${E021_MODEL_SNAPSHOT:-$HOME/.cache/huggingface/hub/models--cyankiwi--gemma-4-26B-A4B-it-AWQ-4bit/snapshots/$REVISION}"' in source
    assert '"$VLLM_BIN" serve "$MODEL_SNAPSHOT"' in source
    assert '--served-model-name "$MODEL"' in source
