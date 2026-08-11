import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "80_manage_g7_graded_jp_campaign.py"
SPEC = importlib.util.spec_from_file_location("g7_graded_jp_campaign", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_campaign(tmp_path, *, positions=7540, questions=754, jobs=3, duplicates=0, responses=None, lawyer=None):
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": "E016",
                "status": "exploratory_internal_evaluation",
                "n_positions": positions,
                "n_questions": questions,
                "n_jobs": jobs,
                "n_missing_card_positions": positions - jobs,
                "n_duplicate_positions": duplicates,
            }
        ),
        encoding="utf-8",
    )
    if responses is not None:
        (tmp_path / "judge_responses.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in responses), encoding="utf-8"
        )
    if lawyer is not None:
        audit = tmp_path / "lawyer_audit"
        audit.mkdir()
        (audit / "lawyer_agreement.json").write_text(json.dumps(lawyer), encoding="utf-8")


def ok_response(index):
    return {"job_id": f"job-{index}", "status": "ok", "response": {"classe": "A"}}


def test_status_requires_7540_positions_and_no_open_technical_errors(tmp_path):
    write_campaign(
        tmp_path,
        responses=[ok_response(1), {"job_id": "job-2", "status": "error"}, ok_response(3)],
    )
    status = MODULE.campaign_status(tmp_path)
    assert status["preparation_gate"] == "passed"
    assert status["judge_gate"] == "blocked_technical_incompleteness"
    assert status["judge_ok_jobs"] == 2

    write_campaign(tmp_path, positions=7539, responses=[ok_response(i) for i in range(3)])
    assert MODULE.campaign_status(tmp_path)["preparation_gate"] == "failed"


def test_status_keeps_lawyer_gate_pending_until_complete_audit(tmp_path):
    write_campaign(tmp_path, responses=[ok_response(i) for i in range(3)])
    assert MODULE.campaign_status(tmp_path)["lawyer_gate"] == "pending"

    write_campaign(
        tmp_path,
        responses=[ok_response(i) for i in range(3)],
        lawyer={"status": "complete", "n_annotations": 100, "gate_status": "passed"},
    )
    assert MODULE.campaign_status(tmp_path)["lawyer_gate"] == "passed"


def test_status_never_calls_internal_eval_confirmatory(tmp_path):
    write_campaign(tmp_path, duplicates=53, responses=[ok_response(i) for i in range(3)])
    status = MODULE.campaign_status(tmp_path)
    assert status["scientific_status"] == "exploratory_internal_evaluation"
    assert status["duplicate_positions"] == 53
    assert "confirm" not in json.dumps(status).lower()
