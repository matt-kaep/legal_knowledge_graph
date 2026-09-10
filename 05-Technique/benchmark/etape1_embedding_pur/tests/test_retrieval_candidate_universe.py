"""Regression tests for the A3 retrieval-candidate contract."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from etape1 import graph_versions, retrieval_candidate_universe


def _write_effective_representation_inputs(tmp_path: Path) -> None:
    np.save(tmp_path / "articles.npy", np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32))
    np.save(
        tmp_path / "articles_order.npy",
        np.asarray(["art-1", "art-representation-only", "art-2"], dtype=object),
    )
    np.save(tmp_path / "jp.npy", np.asarray([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32))
    np.save(
        tmp_path / "jp_order.npy",
        np.asarray(["jp-1", "jp-1", "jp-representation-only", "jp-2"], dtype=object),
    )
    reference = tmp_path / "retrieval-reference-graph"
    reference.mkdir()
    np.save(reference / "article_ids.npy", np.asarray(["art-1", "art-2", "art-aux"], dtype=object))
    np.save(reference / "jp_ids.npy", np.asarray(["jp-1", "jp-2", "jp-aux"], dtype=object))


def _patch_representation_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(graph_versions.config, "EMB_ARTICLES_ALL", tmp_path / "articles.npy")
    monkeypatch.setattr(graph_versions.config, "ARTICLES_ORDER_ALL", tmp_path / "articles_order.npy")
    monkeypatch.setattr(graph_versions.config, "EMB_JP_SYNTHESE", tmp_path / "jp.npy")
    monkeypatch.setattr(graph_versions.config, "JP_SUMMARY_ORDER", tmp_path / "jp_order.npy")
    monkeypatch.setattr(
        retrieval_candidate_universe,
        "RETRIEVAL_REFERENCE_GRAPH_DIR",
        tmp_path / "retrieval-reference-graph",
    )


def _variant(*, articles: list[str], decisions: list[str]) -> graph_versions.GraphVariant:
    return graph_versions.GraphVariant(
        graph_version="G-test",
        graph=sp.csr_matrix(np.ones((len(decisions), len(articles)), dtype=np.float32)),
        jp_ids=np.asarray(decisions, dtype=str),
        article_ids=np.asarray(articles, dtype=str),
        article_codes=np.asarray(articles, dtype=str),
    )


def test_effective_retrieval_universe_deduplicates_decisions_in_stable_order(tmp_path, monkeypatch):
    _write_effective_representation_inputs(tmp_path)
    _patch_representation_paths(monkeypatch, tmp_path)
    graph_versions.load_effective_retrieval_candidate_universe.cache_clear()

    universe = graph_versions.load_effective_retrieval_candidate_universe()

    assert universe.article_ids.tolist() == ["art-1", "art-2"]
    assert universe.jp_ids.tolist() == ["jp-1", "jp-2"]
    assert universe.article_count == 2
    assert universe.jp_count == 2


def test_retrieval_view_rejects_graph_missing_an_official_candidate(tmp_path, monkeypatch):
    _write_effective_representation_inputs(tmp_path)
    _patch_representation_paths(monkeypatch, tmp_path)
    graph_versions.load_effective_retrieval_candidate_universe.cache_clear()
    graph_versions.load_retrieval_view.cache_clear()
    monkeypatch.setattr(
        graph_versions,
        "load_graph_variant",
        lambda _graph: _variant(articles=["art-1"], decisions=["jp-1", "jp-2"]),
    )

    with pytest.raises(ValueError, match="missing official retrieval candidates"):
        graph_versions.load_retrieval_view("G-test")


def test_retrieval_view_preserves_the_same_official_order_for_each_graph(tmp_path, monkeypatch):
    _write_effective_representation_inputs(tmp_path)
    _patch_representation_paths(monkeypatch, tmp_path)
    graph_versions.load_effective_retrieval_candidate_universe.cache_clear()
    graph_versions.load_retrieval_view.cache_clear()
    monkeypatch.setattr(
        graph_versions,
        "load_graph_variant",
        lambda _graph: _variant(
            articles=["art-2", "art-1"],
            decisions=["jp-2", "jp-1"],
        ),
    )

    view = graph_versions.load_retrieval_view("G-test")

    assert view.art_order.tolist() == ["art-1", "art-2"]
    assert view.jp_order.tolist() == ["jp-1", "jp-2"]
