import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "104_characterize_a3_structural_graphs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("a3_structural_characterization", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_structural_record_distinguishes_raw_jp_rows_from_unique_decisions():
    module = _load_module()
    graph = sp.csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=np.int8))

    record = module.structural_record(
        graph_version="fixture",
        graph=graph,
        jp_ids=np.array(["d1", "d1"], dtype=object),
        article_ids=np.array(["a1"], dtype=object),
        edges_by_type={"citation_article_jp": 2},
        normalization="none",
        active_weights={"citation": 1.0},
    )

    assert record["jp_rows_raw"] == 2
    assert record["jp_decisions_unique"] == 1
    assert record["returnable_jp_candidates"] == 114851
    assert record["total_undirected_edges"] == 2
    assert record["connected_components"] == 1


def test_structural_export_writes_a_hash_manifest(tmp_path, monkeypatch):
    module = _load_module()
    monkeypatch.setattr(module, "GRAPHS", ("fixture",))
    monkeypatch.setattr(module, "characterize", lambda _graph: {"graph": "fixture", "jp_rows_raw": 2, "jp_decisions_unique": 1, "article_nodes": 1, "returnable_article_candidates": 13236, "returnable_jp_candidates": 114851, "edges_by_type": {"citation_article_jp": 2}, "total_undirected_edges": 2, "connected_components": 1, "largest_component_fraction": 1.0, "degree": {"mean": 1.0, "median": 1.0, "p90": 1.0, "p99": 1.0, "max": 1}, "normalization": "none", "active_weights": {"citation": 1.0}, "source_sha256": {"fixture": "hash"}})

    outputs = module.write_outputs(tmp_path / "out")

    assert outputs["manifest"].is_file()
