from pathlib import Path
import importlib.util

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "graph_protocol.py"
)
spec = importlib.util.spec_from_file_location("graph_protocol", SCRIPT)
graph_protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(graph_protocol)


def test_metric_rank_tuple_orders_hit_then_ndcg_then_mrr():
    row = {
        "hit": 0.55,
        "ndcg": 0.41,
        "mrr": 0.39,
        "m1": 0.53,
        "m2": 0.44,
    }
    assert graph_protocol.metric_rank_tuple(row, "jp") == (0.55, 0.41, 0.39, 0.53, 0.44)


def test_resolve_graph_bench_dir_uses_graph_and_split():
    out = graph_protocol.resolve_graph_bench_dir("G2", "eval_rich_retrievable_strict")
    assert str(out).endswith("data/doctrine_v3plus_bench/G2/eval_rich_retrievable_strict")
