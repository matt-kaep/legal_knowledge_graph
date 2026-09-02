import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "98_characterize_bipartite_graphs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("b1_graph_characterization", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_b1_characterization_uses_shared_bipartite_definition(tmp_path):
    module = _load_module()
    path = tmp_path / "tiny.npz"
    np.savez(path, data=np.array([1, 1, 1]), indices=np.array([0, 1, 1]), indptr=np.array([0, 2, 3]), shape=np.array([2, 2]), jp_ids=np.array(["j1", "j2"], dtype=object), article_ids=np.array(["a1", "a2"], dtype=object), article_codes=np.array(["c1", "c2"], dtype=object))

    stats, codes = module.characterize("tiny", "explicit fixture", path)

    assert stats["nodes"] == {"jurisprudence": 2, "articles": 2, "total": 4}
    assert stats["edges_by_type"]["jp_article_citation"] == 3
    assert stats["connected_components"]["count"] == 1
    assert codes["article_nodes"].sum() == 2


def test_b1_characterization_requires_explicit_definition():
    module = _load_module()
    with pytest.raises(ValueError, match="NAME=EXPLICIT_DEFINITION"):
        module._parse_pair("only_a_path.npz")
