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
