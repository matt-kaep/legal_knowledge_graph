from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "109_materialize_a3_b1_paper_tables.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("a3_b1_tables", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_materialize_main_tables_preserves_exact_values_and_source_hash(tmp_path):
    tables = _load_module()
    source = tmp_path / "metrics.csv"
    source.write_text(
        "source,target,hit_at_10,hit_at_10_seed_std,ndcg_at_10,ndcg_at_10_seed_std,mrr_at_10,mrr_at_10_seed_std,seeds\n"
        "cosine,articles,0.12345678901234567,0,0.2,0,0.3,0,1\n"
        "ppr,articles,0.4,0,0.5,0,0.6,0,1\n"
        "lightgcn_g6,articles,0.7,0.01,0.8,0.02,0.9,0.03,3\n"
        "cosine,jurisprudence,0.11,0,0.21,0,0.31,0,1\n"
        "ppr,jurisprudence,0.41,0,0.51,0,0.61,0,1\n"
        "lightgcn_g6,jurisprudence,0.71,0.01,0.81,0.02,0.91,0.03,3\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"

    manifest = tables.materialize_tables(source, output)

    article = pd.read_csv(output / "main_table_articles.csv")
    jp = pd.read_csv(output / "main_table_jurisprudence.csv")
    assert article["method"].tolist() == ["cosine_BGE_M3", "PPR", "LightGCN"]
    assert jp["graph"].tolist() == ["G1", "G7-citation-AA-cit1-sem025-knn5", "G6-citation-AA-knn5"]
    assert article.loc[2, "hit_at_10"] == 0.7
    assert jp.loc[1, "mrr_at_10"] == 0.61
    assert "0.12345678901234567" in (output / "main_table_articles.csv").read_text(encoding="utf-8")
    assert manifest["source_metrics_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["source_metrics_path"] == "external_unbundled_source"
    assert manifest["source_rows"] == 6


def test_materialize_main_tables_rejects_an_incomplete_source(tmp_path):
    tables = _load_module()
    source = tmp_path / "metrics.csv"
    source.write_text(
        "source,target,hit_at_10,hit_at_10_seed_std,ndcg_at_10,ndcg_at_10_seed_std,mrr_at_10,mrr_at_10_seed_std,seeds\n"
        "cosine,articles,0.1,0,0.2,0,0.3,0,1\n",
        encoding="utf-8",
    )

    try:
        tables.materialize_tables(source, tmp_path / "out")
    except ValueError as exc:
        assert "Missing or duplicate source row" in str(exc)
    else:
        raise AssertionError("incomplete source must fail")
