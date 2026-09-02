import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "97_build_b1_depth_curves.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("b1_depth_curves", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_b1_depth_honors_configured_repo_when_run_from_a_shallow_staging_path(monkeypatch):
    staging_dir = Path(tempfile.mkdtemp(prefix="lkg-b1-depth-", dir="/tmp"))
    try:
        staged_script = staging_dir / "97_build_b1_depth_curves.py"
        staged_script.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        monkeypatch.setenv("LKG_REPO", str(staging_dir))
        monkeypatch.setenv("LKG_DATA_ROOT", str(staging_dir))
        spec = importlib.util.spec_from_file_location("staged_b1_depth_curves", staged_script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        assert module.CODE_REPO == staging_dir.resolve()
        assert module.DATA_REPO == staging_dir.resolve()
    finally:
        shutil.rmtree(staging_dir)


def test_b1_depth_hit_uses_only_the_returned_top_k_positions():
    curves = _load_module()
    assert curves.hit_at_k(["a", "a", "b"], {"a", "b"}, 2) == pytest.approx(0.5)


def test_b1_depth_rejects_duplicate_or_outside_top100_candidates():
    curves = _load_module()
    questions = {"q1": {"articles_attendus": ["a"]}}
    rows = pd.DataFrame({
        "qid": ["q1"] * 100,
        "rank": list(range(1, 101)),
        "item_id": ["a"] * 2 + [f"x{i}" for i in range(98)],
    })
    with pytest.raises(ValueError, match="duplicate"):
        curves.score_ranking_group(rows, questions=questions, candidate_ids={"a", *[f"x{i}" for i in range(98)]}, target="articles")


def test_b1_depth_exports_exact_ndcg_and_mrr_at_10_from_top100_rankings():
    curves = _load_module()
    questions = {"q1": {"articles_attendus": ["a", "b"]}}
    ranked = ["x0", "a"] + [f"x{i}" for i in range(1, 99)]
    rows = pd.DataFrame({
        "qid": ["q1"] * 100,
        "rank": list(range(1, 101)),
        "item_id": ranked,
    })

    scored = curves.score_ranking_group(
        rows,
        questions=questions,
        candidate_ids=set(ranked) | {"b"},
        target="articles",
    )
    at_10 = scored.loc[scored["k"].eq(10)].iloc[0]

    assert at_10["hit_at_k"] == pytest.approx(0.5)
    assert at_10["mrr_at_10"] == pytest.approx(0.5)
    assert at_10["ndcg_at_10"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_b1_depth_derivation_writes_versioned_exact_metrics_at_10(tmp_path, monkeypatch):
    curves = _load_module()
    monkeypatch.setattr(curves, "DATA_REPO", tmp_path)
    (tmp_path / "eval.json").write_text(json.dumps({"questions": [{
        "qid": "q1", "articles_attendus": ["a"], "gold_jp_ids": ["j"],
    }]}), encoding="utf-8")
    article_ids = ["a", *[f"a{i}" for i in range(1, 101)]]
    jp_ids = ["j", *[f"j{i}" for i in range(1, 101)]]
    np.save(tmp_path / "articles.npy", np.asarray(["representation_only_article", *article_ids], dtype=object))
    np.save(tmp_path / "jp.npy", np.asarray(["representation_only_jp", *jp_ids], dtype=object))
    np.save(tmp_path / "graph_articles.npy", np.asarray(article_ids, dtype=object))
    np.save(tmp_path / "graph_jp.npy", np.asarray(jp_ids, dtype=object))
    rows = []
    for modality, first, rest in (("art", "a", "a"), ("jp", "j", "j")):
        for rank in range(1, 101):
            rows.append({"qid": "q1", "modality": modality, "rank": rank, "item_id": first if rank == 2 else f"{rest}{rank}"})
    ranking_path = tmp_path / "rankings.parquet"
    pd.DataFrame(rows).to_parquet(ranking_path, index=False)
    payload = {
        "campaign_id": "test-b1",
        "datasets": {"evaluation": {"path": "eval.json", "questions": 1, "sha256": "test"}},
        "candidate_inputs": {
            "articles_order": {"path": "articles.npy"},
            "jurisprudence_order": {"path": "jp.npy"},
            "shared_article_ids": {"path": "graph_articles.npy"},
            "shared_jp_ids": {"path": "graph_jp.npy"},
        },
        "candidate_universe": {"articles": {"count": 101}, "jurisprudence": {"count": 101}},
    }

    outputs = curves.derive_curves(payload, {"cosine": ranking_path}, tmp_path / "validated")
    exported = pd.read_csv(outputs["metrics_at_10"])

    assert set(exported["target"]) == {"articles", "jurisprudence"}
    assert exported["hit_at_10"].tolist() == pytest.approx([1.0, 1.0])
    assert exported["ndcg_at_10"].tolist() == pytest.approx([1 / math.log2(3)] * 2)
    assert exported["mrr_at_10"].tolist() == pytest.approx([0.5, 0.5])
