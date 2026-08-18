from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "69_materialize_e021_pools.py"
SPEC = importlib.util.spec_from_file_location("e021_pools", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_materialize_pool_preserves_source_order_and_hash(tmp_path):
    ranking_path = tmp_path / "ranking.parquet"
    pd.DataFrame(
        [
            {"qid": "q1", "method": "PPR", "modality": "jp", "rank": rank, "item_id": f"jp-{rank}"}
            for rank in range(1, 21)
        ]
    ).to_parquet(ranking_path, index=False)
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(
        json.dumps({f"jp-{rank}": {"solution_resume": f"solution {rank}"} for rank in range(1, 21)}),
        encoding="utf-8",
    )
    output_path = tmp_path / "pool.jsonl"

    assert MODULE.materialize_pool(ranking_path, cards_path, output_path, "ppr") == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert [candidate["item_id"] for candidate in record["candidates"]] == [f"jp-{i}" for i in range(1, 21)]
    assert record["candidates"][0]["source_rank"] == 1
    assert record["source_ranking_sha256"] == MODULE.sha256_file(ranking_path)


def test_materialize_pool_uses_complete_summary_source_for_unjudged_candidates(tmp_path):
    ranking_path = tmp_path / "ranking.parquet"
    pd.DataFrame(
        [
            {"qid": "q1", "method": "PPR", "modality": "jp", "rank": rank, "item_id": f"jp-{rank}"}
            for rank in range(1, 21)
        ]
    ).to_parquet(ranking_path, index=False)
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(json.dumps({"jp-1": {"solution_resume": "judged card"}}), encoding="utf-8")
    summaries_path = tmp_path / "jp_summaries.parquet"
    pd.DataFrame(
        {"jp_id": [f"jp-{rank}" for rank in range(1, 21)], "synthese": [f"summary {rank}" for rank in range(1, 21)]}
    ).to_parquet(summaries_path, index=False)
    output_path = tmp_path / "pool.jsonl"

    assert MODULE.materialize_pool(
        ranking_path,
        cards_path,
        output_path,
        "ppr",
        summaries_path=summaries_path,
    ) == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert record["candidates"][0]["text"] == "solution: judged card"
    assert record["candidates"][1]["text"] == "summary 2"
    assert record["source_texts_sha256"] == MODULE.sha256_file(summaries_path)


def test_summary_source_rejects_conflicting_duplicate_ids(tmp_path):
    summaries_path = tmp_path / "jp_summaries.parquet"
    pd.DataFrame(
        {"jp_id": ["jp-1", "jp-1"], "synthese": ["summary A", "summary B"]}
    ).to_parquet(summaries_path, index=False)

    try:
        MODULE.load_summary_texts(summaries_path)
    except ValueError as error:
        assert "conflicting" in str(error)
    else:
        raise AssertionError("conflicting duplicate identifiers must be rejected")


def test_materialize_pool_skips_repeated_ranking_items_and_preserves_source_rank(tmp_path):
    ranking_path = tmp_path / "ranking.parquet"
    rows = []
    for rank in range(1, 22):
        item_number = rank if rank < 3 else rank - 1
        rows.append({"qid": "q1", "method": "PPR", "modality": "jp", "rank": rank, "item_id": f"jp-{item_number}"})
    pd.DataFrame(rows).to_parquet(ranking_path, index=False)
    cards_path = tmp_path / "cards.json"
    cards_path.write_text(
        json.dumps({f"jp-{rank}": {"solution_resume": f"solution {rank}"} for rank in range(1, 21)}),
        encoding="utf-8",
    )
    output_path = tmp_path / "pool.jsonl"

    assert MODULE.materialize_pool(ranking_path, cards_path, output_path, "ppr") == 1
    record = json.loads(output_path.read_text(encoding="utf-8"))

    assert [candidate["source_rank"] for candidate in record["candidates"]] == [1, 2, *range(4, 22)]
    assert record["source_rows_considered"] == 21
    assert record["duplicate_candidates_skipped"] == 1
