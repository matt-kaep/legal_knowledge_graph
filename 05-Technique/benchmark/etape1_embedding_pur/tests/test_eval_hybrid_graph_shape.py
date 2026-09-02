"""Regression tests for cosine evaluation on square hybrid graphs."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import scipy.sparse as sp


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "26_eval_doctrine_v3plus_m1_m2.py"
spec = importlib.util.spec_from_file_location("eval_hybrid_shape", SCRIPT)
eval_hybrid_shape = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(eval_hybrid_shape)


def test_hybrid_square_graph_extracts_the_same_jp_article_incidence_as_bipartite_graph():
    incidence = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    n_jp, n_articles = incidence.shape
    full = np.zeros((n_jp + n_articles, n_jp + n_articles), dtype=np.float32)
    full[:n_jp, n_jp:] = incidence
    full[n_jp:, :n_jp] = incidence.T

    actual = eval_hybrid_shape.jp_article_incidence(
        sp.csr_matrix(full), n_jp=n_jp, n_articles=n_articles
    )

    assert np.array_equal(actual.toarray(), incidence)


def test_bipartite_graph_keeps_its_original_incidence_block():
    incidence = sp.csr_matrix(np.asarray([[1.0], [2.0]], dtype=np.float32))

    actual = eval_hybrid_shape.jp_article_incidence(incidence, n_jp=2, n_articles=1)

    assert np.array_equal(actual.toarray(), incidence.toarray())
