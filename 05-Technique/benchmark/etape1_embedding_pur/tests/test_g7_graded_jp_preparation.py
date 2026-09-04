import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "75_prepare_g7_graded_jp_eval.py"
SPEC = importlib.util.spec_from_file_location("g7_graded_jp_preparation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rankings():
    return pd.DataFrame(
        [
            {"qid": "q1", "method": "LightGCN-trained_K2", "modality": "jp", "rank": 2, "item_id": "jp2"},
            {"qid": "q1", "method": "LightGCN-trained_K2", "modality": "jp", "rank": 1, "item_id": "jp1"},
            {"qid": "q1", "method": "LightGCN-trained_K2", "modality": "art", "rank": 1, "item_id": "art1"},
            {"qid": "q2", "method": "other", "modality": "jp", "rank": 1, "item_id": "jp3"},
        ]
    )


def test_select_g7_positions_requires_one_to_k_per_question():
    selected = MODULE.select_g7_positions(
        rankings(), question_ids={"q1"}, method="LightGCN-trained_K2", k=2
    )
    assert selected[["qid", "rank", "jp_id"]].to_dict("records") == [
        {"qid": "q1", "rank": 1, "jp_id": "jp1"},
        {"qid": "q1", "rank": 2, "jp_id": "jp2"},
    ]


def test_select_g7_positions_rejects_incomplete_rank_sequence():
    with pytest.raises(ValueError, match="expected ranks 1..2"):
        MODULE.select_g7_positions(
            rankings().query("rank == 1"),
            question_ids={"q1"},
            method="LightGCN-trained_K2",
            k=2,
        )


def test_duplicate_jp_positions_are_kept_but_marked_as_wasted_slots():
    duplicated = rankings()
    duplicated.loc[duplicated["rank"] == 2, "item_id"] = "jp1"
    selected = MODULE.select_g7_positions(
        duplicated, question_ids={"q1"}, method="LightGCN-trained_K2", k=2
    )
    assert selected["duplicate_position"].tolist() == [False, True]

    jobs, frozen = MODULE.build_blind_jobs(
        positions=selected,
        questions={"q1": {"enonce": "Question ?", "gold_jp_ids": []}},
        cards={"jp1": {"solution_resume": "Solution"}},
        judge_contract={"model_id": "model"},
    )
    assert len(jobs) == 1
    assert frozen["job_id"].nunique() == 1


def test_build_blind_jobs_never_exposes_rank_method_gt_or_g8():
    jobs, positions = MODULE.build_blind_jobs(
        positions=pd.DataFrame([{"qid": "q1", "rank": 1, "jp_id": "jp1"}]),
        questions={"q1": {"enonce": "Question ?", "gold_jp_ids": ["gold"]}},
        cards={
            "jp1": {
                "jp_id": "jp1",
                "synthese_pour_avocat": "Synthèse",
                "solution_resume": "Solution",
                "secret_field": "doit disparaître",
            }
        },
        judge_contract={"model_id": "model", "prompt_sha256": "hash", "prompt_version": "v1"},
    )
    assert set(jobs[0]) == {
        "job_id",
        "qid",
        "jp_id",
        "question",
        "decision_card",
        "judge_contract",
    }
    serialized = json.dumps(jobs[0], ensure_ascii=False)
    assert "rank" not in serialized
    assert "gold" not in serialized
    assert "LightGCN" not in serialized
    assert "g8" not in serialized.lower()
    assert "secret_field" not in serialized
    assert positions.iloc[0]["card_status"] == "available"


def test_missing_card_keeps_position_without_creating_llm_job():
    jobs, positions = MODULE.build_blind_jobs(
        positions=pd.DataFrame([{"qid": "q1", "rank": 1, "jp_id": "jp-missing"}]),
        questions={"q1": {"enonce": "Question ?", "gold_jp_ids": ["gold"]}},
        cards={},
        judge_contract={"model_id": "model", "prompt_sha256": "hash", "prompt_version": "v1"},
    )
    assert jobs == []
    assert positions.iloc[0]["card_status"] == "missing"


def test_calibration_profile_rejects_internal_eval_inputs():
    with pytest.raises(ValueError, match="train-only"):
        MODULE.validate_profile_paths(
            "calibration",
            Path("data/eval_rich_retrievable_strict/rankings.parquet"),
            Path("data/train_augmented_retrievable_strict/bench_global.json"),
        )


def test_deterministic_question_sample_is_stable_and_sorted():
    first = MODULE.select_question_ids({"q3", "q1", "q2"}, limit=2, seed=42)
    second = MODULE.select_question_ids({"q2", "q3", "q1"}, limit=2, seed=42)
    assert first == second
    assert len(first) == 2


def test_existing_g8_card_fetcher_can_be_loaded_without_running_the_database():
    fetcher = MODULE._load_card_fetcher()
    assert callable(fetcher)
    assert fetcher.__name__ == "fetch_decision_cards"


def test_judge_contract_pins_model_revision_and_artifact_hashes(tmp_path):
    prompt = tmp_path / "prompt.txt"
    schema = tmp_path / "schema.json"
    prompt.write_text("prompt", encoding="utf-8")
    schema.write_text("{}", encoding="utf-8")
    contract = MODULE.make_judge_contract(
        model_id="model",
        model_revision="immutable-revision",
        prompt_path=prompt,
        schema_path=schema,
    )
    assert contract == {
        "model_id": "model",
        "model_revision": "immutable-revision",
        "prompt_version": "g7_graded_jp_judge_v1",
        "prompt_sha256": "cf07194ee232eb531e15f690000d19846dea69cf05504782658afcfacb9228a2",
        "schema_sha256": "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
    }
