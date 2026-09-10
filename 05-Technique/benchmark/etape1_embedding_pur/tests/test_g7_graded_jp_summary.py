import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "77_summarize_g7_graded_jp_eval.py"
SPEC = importlib.util.spec_from_file_location("g7_graded_jp_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def positions_10(card_status="available"):
    return pd.DataFrame(
        [
            {
                "qid": "q1",
                "rank": rank,
                "jp_id": f"jp{rank}",
                "job_id": f"job{rank}",
                "card_status": card_status,
            }
            for rank in range(1, 11)
        ]
    )


def responses_for(labels):
    return [
        {
            "job_id": f"job{rank}",
            "qid": "q1",
            "jp_id": f"jp{rank}",
            "status": "ok",
            "response": {
                "classe": label,
                "justification": f"La règle précise correspondant au cas {rank} motive ce classement.",
            },
        }
        for rank, label in enumerate(labels, start=1)
    ]


def questions(gold=None):
    return {"q1": {"enonce": "Question ?", "gold_jp_ids": gold or ["jp7"]}}


def test_aggregate_uses_fixed_ten_denominator():
    detail, per_question, summary = MODULE.aggregate(
        positions_10(),
        responses_for(["A"] * 4 + ["B"] * 2 + ["C"] * 4),
        questions(),
        k=10,
    )
    assert per_question.iloc[0]["score_gradue_at_10"] == 0.5
    assert len(detail) == 10
    assert summary["macro_score_gradue_at_10"] == 0.5


def test_missing_card_becomes_non_judgeable_and_keeps_denominator():
    positions = positions_10()
    positions.loc[0, "card_status"] = "missing"
    detail, per_question, summary = MODULE.aggregate(
        positions,
        responses_for(["C"] * 10)[1:],
        questions(),
        k=10,
    )
    assert detail.loc[detail["rank"] == 1, "classe"].item() == "non_jugeable"
    assert per_question.iloc[0]["non_jugeable_count"] == 1
    assert summary["class_distribution"]["non_jugeable"] == 1


def test_missing_or_non_ok_response_blocks_aggregation():
    with pytest.raises(ValueError, match="technical incompleteness"):
        MODULE.aggregate(positions_10(), responses_for(["A"] * 9), questions(), k=10)

    invalid = responses_for(["A"] * 10)
    invalid[-1]["status"] = "invalid"
    with pytest.raises(ValueError, match="technical incompleteness"):
        MODULE.aggregate(positions_10(), invalid, questions(), k=10)


def test_repeated_jp_position_keeps_k_but_cannot_earn_gain_twice():
    positions = positions_10()
    positions.loc[1, ["jp_id", "job_id"]] = ["jp1", "job1"]
    positions["duplicate_position"] = positions.duplicated(["qid", "jp_id"])
    responses = responses_for(["A"] + ["E"] * 9)
    detail, per_question, summary = MODULE.aggregate(positions, responses, questions(), k=10)
    duplicate = detail.loc[detail["rank"] == 2].iloc[0]
    assert duplicate["classe"] == "A"
    assert duplicate["gain"] == 1.0
    assert duplicate["effective_gain"] == 0.0
    assert per_question.iloc[0]["score_gradue_at_10"] == 0.1
    assert summary["duplicate_position_count"] == 1


def test_summary_keeps_exact_hit_separate_from_graded_score():
    _, per_question, summary = MODULE.aggregate(
        positions_10(), responses_for(["E"] * 10), questions(gold=["jp7"]), k=10
    )
    assert bool(per_question.iloc[0]["exact_hit_at_10"])
    assert per_question.iloc[0]["score_gradue_at_10"] == 0.0
    assert summary["exact_hit_at_10"] == 1.0
    assert summary["macro_score_gradue_at_10"] == 0.0


def test_conflicting_duplicate_responses_are_rejected():
    rows = responses_for(["A"] * 10)
    conflict = {**rows[0], "response": {"classe": "E", "justification": "Aucun rapport."}}
    with pytest.raises(ValueError, match="conflicting duplicate"):
        MODULE.aggregate(positions_10(), rows + [conflict], questions(), k=10)
