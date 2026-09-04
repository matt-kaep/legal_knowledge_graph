import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "84_summarize_e017_intergraph_graded_jp.py"
SPEC = importlib.util.spec_from_file_location("e017_intergraph_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def positions_for(graph_id, seed, labels, *, duplicate_second=False, jp_offset=0):
    rows = []
    for rank, label in enumerate(labels, start=1):
        jp_id = (
            f"jp{1 + jp_offset}"
            if duplicate_second and rank == 2
            else f"jp{rank + jp_offset}"
        )
        rows.append(
            {
                "graph_id": graph_id,
                "seed": seed,
                "qid": "q1",
                "rank": rank,
                "jp_id": jp_id,
                "job_id": f"q1-{jp_id}",
                "card_status": "available",
                "duplicate_position": duplicate_second and rank == 2,
                "expected_label": label,
            }
        )
    return rows


def response_rows(positions):
    by_job = {}
    for row in positions:
        by_job.setdefault(
            row["job_id"],
            {
                "job_id": row["job_id"],
                "qid": row["qid"],
                "jp_id": row["jp_id"],
                "status": "ok",
                "response": {
                    "classe": row["expected_label"],
                    "justification": "La règle et la solution décrites justifient précisément ce classement.",
                },
            },
        )
    return list(by_job.values())


def test_aggregate_runs_keeps_graph_seed_blocks_and_fixed_k_scores():
    rows = positions_for("G1", 42, ["A", "B"]) + positions_for(
        "G7", 42, ["E", "A"], jp_offset=10
    )
    positions = pd.DataFrame(rows).drop(columns="expected_label")
    per_run, _, per_question = MODULE.aggregate_e017(
        positions,
        response_rows(rows),
        {"q1": {"enonce": "Question ?", "gold_jp_ids": ["jp2"]}},
        k=2,
    )

    assert len(per_question) == 2
    scores = per_run.set_index(["graph_id", "seed"])["score_gradue_at_10"].to_dict()
    assert scores == {("G1", 42): 0.75, ("G7", 42): 0.5}


def test_duplicate_gain_is_zero_only_inside_its_own_ranking():
    rows = positions_for("G1", 42, ["A", "A"], duplicate_second=True)
    rows += positions_for("G1", 43, ["A", "A"])
    positions = pd.DataFrame(rows).drop(columns="expected_label")
    per_run, _, _ = MODULE.aggregate_e017(
        positions,
        response_rows(rows),
        {"q1": {"enonce": "Question ?", "gold_jp_ids": []}},
        k=2,
    )

    scores = per_run.set_index("seed")["score_gradue_at_10"].to_dict()
    assert scores == {42: 0.5, 43: 1.0}


def test_summarize_graphs_reports_mean_and_sample_std_across_seeds():
    per_run = pd.DataFrame(
        [
            {"graph_id": "G1", "seed": 42, "score_gradue_at_10": 0.2, "exact_any_gold_at_10": 0.1, "m1_recall_at_10": 0.1, "m2_rank_at_10": 0.1, "hit_at_10": 0.1, "mrr_at_10": 0.1, "ndcg_at_10": 0.1},
            {"graph_id": "G1", "seed": 43, "score_gradue_at_10": 0.4, "exact_any_gold_at_10": 0.3, "m1_recall_at_10": 0.3, "m2_rank_at_10": 0.3, "hit_at_10": 0.3, "mrr_at_10": 0.3, "ndcg_at_10": 0.3},
            {"graph_id": "G1", "seed": 44, "score_gradue_at_10": 0.6, "exact_any_gold_at_10": 0.5, "m1_recall_at_10": 0.5, "m2_rank_at_10": 0.5, "hit_at_10": 0.5, "mrr_at_10": 0.5, "ndcg_at_10": 0.5},
        ]
    )
    graph_summary = MODULE.summarize_graphs(per_run)
    row = graph_summary.iloc[0]

    assert row["n_seeds"] == 3
    assert abs(row["score_gradue_at_10_mean"] - 0.4) < 1e-12
    assert abs(row["score_gradue_at_10_std"] - 0.2) < 1e-12
    assert abs(row["exact_any_gold_at_10_mean"] - 0.3) < 1e-12
