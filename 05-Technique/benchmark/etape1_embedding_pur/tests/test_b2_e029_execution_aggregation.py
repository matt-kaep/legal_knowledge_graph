import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "107_aggregate_b2_e029_execution.py"


def _load_aggregator():
    spec = importlib.util.spec_from_file_location("b2_e029_execution_aggregation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _shard(root: Path, name: str, rows: list[dict[str, object]]) -> Path:
    shard = root / name
    materialized = shard / "materialized"
    materialized.mkdir(parents=True)
    metrics = materialized / "exact_metrics.csv"
    pd.DataFrame(rows).to_csv(metrics, index=False)
    (materialized / "materialization_receipt.json").write_text(json.dumps({"exact_metrics.csv": "placeholder"}), encoding="utf-8")
    return shard


def test_execution_aggregation_preserves_lightgcn_seeds_and_derives_their_mean(tmp_path):
    aggregator = _load_aggregator()
    base = {"family": "lightgcn_g6", "modality": "article", "k_in": 10, "hit_at_10": 0.2, "ndcg_at_10": 0.1, "mrr_at_10": 0.1, "recall_at_10": 0.2}
    first = _shard(tmp_path, "seed42", [{**base, "replay_seed": 42}])
    second = _shard(tmp_path, "seed43", [{**base, "replay_seed": 43, "hit_at_10": 0.4, "recall_at_10": 0.4}])
    output = tmp_path / "aggregate"

    receipt = aggregator.aggregate_execution([first, second], output, expected_conditions=2)

    per_seed = pd.read_csv(output / "per_seed_exact_metrics.csv")
    mean = pd.read_csv(output / "seed_mean_exact_metrics.csv")
    assert set(per_seed["replay_seed"].astype(str)) == {"42", "43"}
    assert mean.loc[0, "seed_count"] == 2
    assert mean.loc[0, "hit_at_10"] == 0.3
    assert receipt["conditions"] == 2
