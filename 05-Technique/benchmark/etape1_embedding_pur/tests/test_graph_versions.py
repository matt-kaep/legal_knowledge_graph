from pathlib import Path
import sys

import numpy as np
import scipy.sparse as sp


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etape1 import graph_versions  # noqa: E402


def test_canonical_graph_version_accepts_g5_hybrid_names():
    assert graph_versions.canonical_graph_version("g5-citation-knn5") == "G5-citation-knn5"
    assert graph_versions.canonical_graph_version("G5-citation-knn10") == "G5-citation-knn10"


def test_canonical_graph_version_accepts_g7_weighted_typed_names():
    assert (
        graph_versions.canonical_graph_version("g7-citation-aa-cit1-sem025-knn5")
        == "G7-citation-AA-cit1-sem025-knn5"
    )
    assert (
        graph_versions.canonical_graph_version("G7-citation-JJ-cit025-sem1-knn5")
        == "G7-citation-JJ-cit025-sem1-knn5"
    )


def test_load_graph_variant_g2_composes_from_g1_masks(monkeypatch):
    graph = sp.csr_matrix(np.ones((3, 16), dtype=np.int8))
    jp_ids = np.array(["j1", "j2", "j3"], dtype=object)
    article_ids = np.array([f"a{i}" for i in range(16)], dtype=object)
    article_codes = np.array(["c"] * 16, dtype=object)

    monkeypatch.setattr(
        graph_versions,
        "load_canonical_graph_arrays",
        lambda: (graph, jp_ids, article_ids, article_codes),
    )
    monkeypatch.setattr(
        graph_versions,
        "_build_g1_article_keep_mask",
        lambda article_ids, deg_art: np.array([True] * 15 + [False]),
    )
    graph_versions.load_graph_variant.cache_clear()

    variant = graph_versions.load_graph_variant("G2")

    assert variant.graph_version == "G2"
    assert variant.article_ids.tolist() == ["a14"]
    assert variant.jp_ids.tolist() == ["j1", "j2", "j3"]
    assert variant.graph.shape == (3, 1)


def test_load_retrieval_view_filters_embeddings_to_variant_ids(monkeypatch):
    variant = graph_versions.GraphVariant(
        "G1",
        sp.csr_matrix(np.array([[1, 0], [0, 1]], dtype=np.int8)),
        np.array(["j2", "j4"], dtype=object),
        np.array(["a3", "a1"], dtype=object),
        np.array(["c", "c"], dtype=object),
    )
    monkeypatch.setattr(graph_versions, "load_graph_variant", lambda graph_version: variant)
    monkeypatch.setattr(
        graph_versions.config,
        "EMB_ARTICLES_ALL",
        Path("/tmp/emb_articles_all.npy"),
    )
    monkeypatch.setattr(
        graph_versions.config,
        "ARTICLES_ORDER_ALL",
        Path("/tmp/articles_order_all.npy"),
    )
    monkeypatch.setattr(
        graph_versions.config,
        "EMB_JP_SYNTHESE",
        Path("/tmp/emb_jp.npy"),
    )
    monkeypatch.setattr(
        graph_versions.config,
        "JP_SUMMARY_ORDER",
        Path("/tmp/jp_order.npy"),
    )

    art_emb = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    art_order = np.array(["a1", "a2", "a3"], dtype=object)
    jp_emb = np.array([[10.0], [20.0], [30.0], [40.0]], dtype=np.float32)
    jp_order = np.array(["j1", "j2", "j3", "j4"], dtype=object)

    def fake_np_load(path, *args, **kwargs):
        path_str = str(path)
        if path_str.endswith("emb_articles_all.npy"):
            return art_emb
        if path_str.endswith("articles_order_all.npy"):
            return art_order
        if path_str.endswith("emb_jp.npy"):
            return jp_emb
        if path_str.endswith("jp_order.npy"):
            return jp_order
        raise AssertionError(path)

    monkeypatch.setattr(graph_versions.np, "load", fake_np_load)
    monkeypatch.setattr(
        graph_versions,
        "load_effective_retrieval_candidate_universe",
        lambda: graph_versions.EffectiveRetrievalCandidateUniverse(
            article_ids=np.array(["a1", "a3"], dtype=str),
            article_embeddings=art_emb[[0, 2]],
            jp_ids=np.array(["j2", "j4"], dtype=str),
            jp_embeddings=jp_emb[[1, 3]],
        ),
    )
    graph_versions.load_retrieval_view.cache_clear()

    view = graph_versions.load_retrieval_view("G1")

    assert view.art_order.tolist() == ["a1", "a3"]
    assert view.p2col.tolist() == [1, 0]
    assert view.jp_order.tolist() == ["j2", "j4"]
    assert view.jp_to_row.tolist() == [0, 1]
