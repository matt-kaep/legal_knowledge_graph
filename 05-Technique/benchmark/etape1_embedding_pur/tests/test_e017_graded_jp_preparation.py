import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "83_prepare_e017_graded_jp_eval.py"
SPEC = importlib.util.spec_from_file_location("e017_graded_jp_preparation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def ranking(*, duplicate: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "qid": "q1",
                "method": "LightGCN-trained_K2",
                "modality": "jp",
                "selection_target": "jp",
                "rank": 1,
                "item_id": "jp1",
            },
            {
                "qid": "q1",
                "method": "LightGCN-trained_K2",
                "modality": "jp",
                "selection_target": "jp",
                "rank": 2,
                "item_id": "jp1" if duplicate else "jp2",
            },
            {
                "qid": "q2",
                "method": "LightGCN-trained_K2",
                "modality": "jp",
                "selection_target": "jp",
                "rank": 1,
                "item_id": "jp1",
            },
            {
                "qid": "q2",
                "method": "LightGCN-trained_K2",
                "modality": "jp",
                "selection_target": "jp",
                "rank": 2,
                "item_id": "jp3",
            },
            {
                "qid": "q1",
                "method": "LightGCN-trained_K2",
                "modality": "art",
                "selection_target": "jp",
                "rank": 1,
                "item_id": "art1",
            },
            {
                "qid": "q1",
                "method": "LightGCN-trained_K2",
                "modality": "jp",
                "selection_target": "art",
                "rank": 1,
                "item_id": "jp-article-target",
            },
        ]
    )


def test_select_positions_keeps_each_run_and_marks_only_within_run_duplicates():
    first = MODULE.select_positions(
        ranking(duplicate=True), graph_id="G1", seed=42, k=2, expected_questions=2
    )
    second = MODULE.select_positions(
        ranking(), graph_id="G7", seed=43, k=2, expected_questions=2
    )
    positions = pd.concat([first, second], ignore_index=True)

    assert len(positions) == 8
    assert positions.groupby(["graph_id", "seed"]).size().to_dict() == {
        ("G1", 42): 4,
        ("G7", 43): 4,
    }
    assert first["duplicate_position"].tolist() == [False, True, False, False]
    assert not second["duplicate_position"].any()


def test_select_positions_rejects_an_incomplete_fixed_k_ranking():
    incomplete = ranking().query("not (qid == 'q2' and rank == 2)")
    with pytest.raises(ValueError, match="expected ranks 1..2"):
        MODULE.select_positions(
            incomplete, graph_id="G1", seed=42, k=2, expected_questions=2
        )


def test_build_unique_jobs_preserves_positions_but_emits_each_pair_once():
    positions = pd.concat(
        [
            MODULE.select_positions(
                ranking(), graph_id="G1", seed=42, k=2, expected_questions=2
            ),
            MODULE.select_positions(
                ranking(), graph_id="G7", seed=43, k=2, expected_questions=2
            ),
        ],
        ignore_index=True,
    )
    jobs, frozen = MODULE.build_unique_jobs(
        positions=positions,
        questions={"q1": {"enonce": "Question 1 ?"}, "q2": {"enonce": "Question 2 ?"}},
        cards={
            "jp1": {"solution_resume": "Solution 1"},
            "jp2": {"solution_resume": "Solution 2"},
            "jp3": {"solution_resume": "Solution 3"},
        },
        judge_contract={"model_id": "model", "model_revision": "rev"},
    )

    assert len(frozen) == 8
    assert len(jobs) == 4
    assert frozen["job_id"].nunique() == 4
    assert {(job["qid"], job["jp_id"]) for job in jobs} == {
        ("q1", "jp1"),
        ("q1", "jp2"),
        ("q2", "jp1"),
        ("q2", "jp3"),
    }


def exact_job() -> dict:
    return {
        "job_id": "job-1",
        "qid": "q1",
        "jp_id": "jp1",
        "question": "Question ?",
        "decision_card": {"solution_resume": "Solution"},
        "judge_contract": {
            "model_id": "model",
            "model_revision": "revision",
            "prompt_version": "prompt-v1",
            "prompt_sha256": "prompt-hash",
            "schema_sha256": "schema-hash",
        },
    }


def exact_response() -> dict:
    job = exact_job()
    return {
        "job_id": job["job_id"],
        "qid": job["qid"],
        "jp_id": job["jp_id"],
        "judge_contract": job["judge_contract"],
        "status": "ok",
        "response": {"classe": "A", "justification": "Raison juridique précise."},
    }


def test_exact_e016_identity_reuses_the_response():
    new_jobs, reused = MODULE.split_exact_cache(
        [exact_job()], [exact_job()], [exact_response()]
    )
    assert new_jobs == []
    assert reused == [exact_response()]


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("question", "Question modifiée ?"),
        ("decision_card", {"solution_resume": "Fiche modifiée"}),
        (
            "judge_contract",
            {
                "model_id": "model",
                "model_revision": "autre-revision",
                "prompt_version": "prompt-v1",
                "prompt_sha256": "prompt-hash",
                "schema_sha256": "schema-hash",
            },
        ),
    ],
)
def test_cache_is_not_reused_when_any_judgment_input_changes(field, replacement):
    current = exact_job()
    current[field] = replacement
    new_jobs, reused = MODULE.split_exact_cache(
        [current], [exact_job()], [exact_response()]
    )
    assert new_jobs == [current]
    assert reused == []


def test_cache_is_not_reused_for_a_non_ok_response():
    response = exact_response()
    response["status"] = "invalid"
    new_jobs, reused = MODULE.split_exact_cache(
        [exact_job()], [exact_job()], [response]
    )
    assert new_jobs == [exact_job()]
    assert reused == []


def test_write_outputs_does_not_seal_the_append_only_response_log(tmp_path):
    manifest = {"experiment_id": "E017"}
    MODULE.write_outputs(
        out_dir=tmp_path,
        positions=pd.DataFrame([{"qid": "q1", "jp_id": "jp1"}]),
        all_jobs=[exact_job()],
        new_jobs=[exact_job()],
        reused_responses=[exact_response()],
        cards={"jp1": {"solution_resume": "Solution"}},
        manifest=manifest,
        force=False,
    )

    frozen = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert "reused_responses.jsonl" in frozen["artifacts"]
    assert "judge_responses.jsonl" not in frozen["artifacts"]
