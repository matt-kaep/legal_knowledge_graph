from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/84_summarize_e017_intergraph_graded_jp.py"
SPEC = importlib.util.spec_from_file_location("e017_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def _response(job_id: str, label: str) -> dict:
    return {"job_id": job_id, "status": "ok", "response": {"classe": label, "justification": "test"}}


def test_aggregate_e017_keeps_exact_and_graded_metrics_separate() -> None:
    positions = pd.DataFrame(
        [
            {"graph_id": graph, "seed": seed, "qid": "q1", "rank": 1, "jp_id": "gold", "job_id": f"{graph}-{seed}-1", "card_status": "available"}
            for graph, seed in (("G1", 42), ("G1", 43), ("G7", 42), ("G7", 43))
        ]
        + [
            {"graph_id": graph, "seed": seed, "qid": "q1", "rank": 2, "jp_id": "other", "job_id": f"{graph}-{seed}-2", "card_status": "available"}
            for graph, seed in (("G1", 42), ("G1", 43), ("G7", 42), ("G7", 43))
        ]
    )
    responses = [
        _response(row.job_id, "A" if row.jp_id == "gold" else "E")
        for row in positions.itertuples(index=False)
    ]
    questions = {"q1": {"qid": "q1", "gold_jp_ids": ["gold"]}}

    by_seed, by_graph, per_question = MODULE.aggregate_e017(
        positions, responses, questions, k=2
    )

    assert len(by_seed) == 4
    assert set(by_graph["graph_id"]) == {"G1", "G7"}
    assert (by_seed["hit_at_10"] == 1.0).all()
    assert (by_seed["score_gradue_at_10"] == 0.5).all()
    assert len(per_question) == 4
