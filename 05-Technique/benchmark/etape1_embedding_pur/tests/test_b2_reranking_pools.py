import importlib.util
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "101_materialize_b2_reranking_pools.py"
PREFLIGHT_MANIFEST = ROOT / "configs" / "b2_reranking_comparable_a3_preflight_v1.json"


def _load_materializer():
    spec = importlib.util.spec_from_file_location("b2_reranking_pools", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _rankings(modality: str, ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "qid": ["q1"] * len(ids),
        "method": ["frozen-method"] * len(ids),
        "modality": ["art" if modality == "article" else "jp"] * len(ids),
        "rank": list(range(1, len(ids) + 1)),
        "item_id": ids,
    })


def test_materializer_preserves_real_source_order_and_text(tmp_path):
    materializer = _load_materializer()
    ranking = tmp_path / "ranking.parquet"
    _rankings("article", ["a1", "a2", "a3"]).to_parquet(ranking, index=False)
    texts = tmp_path / "article_texts.parquet"
    pd.DataFrame({"pair_key": ["a1", "a2", "a3"], "texte": ["T1", "T2", "T3"]}).to_parquet(texts, index=False)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"qid": "q1", "enonce": "Question"}]}), encoding="utf-8")
    output = tmp_path / "pool.jsonl"

    count = materializer.materialize_pool(
        ranking_path=ranking,
        questions_path=questions,
        text_source_path=texts,
        output_path=output,
        family="cosine",
        modality="article",
        candidate_ids={"a1", "a2", "a3"},
        k_in=3,
    )

    assert count == 1
    row = json.loads(output.read_text(encoding="utf-8"))
    assert [candidate["item_id"] for candidate in row["candidates"]] == ["a1", "a2", "a3"]
    assert [candidate["source_rank"] for candidate in row["candidates"]] == [1, 2, 3]
    assert [candidate["text"] for candidate in row["candidates"]] == ["T1", "T2", "T3"]


def test_materializer_fails_for_missing_text_or_outside_a3_candidate(tmp_path):
    materializer = _load_materializer()
    ranking = tmp_path / "ranking.parquet"
    _rankings("jp", ["j1", "j2"]).to_parquet(ranking, index=False)
    texts = tmp_path / "jp_texts.parquet"
    pd.DataFrame({"jp_id": ["j1"], "synthese": ["T1"]}).to_parquet(texts, index=False)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [{"qid": "q1", "enonce": "Question"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing candidate text"):
        materializer.materialize_pool(
            ranking_path=ranking,
            questions_path=questions,
            text_source_path=texts,
            output_path=tmp_path / "missing.jsonl",
            family="ppr",
            modality="jp",
            candidate_ids={"j1", "j2"},
            k_in=2,
        )
    with pytest.raises(ValueError, match="outside A3"):
        materializer.materialize_pool(
            ranking_path=ranking,
            questions_path=questions,
            text_source_path=texts,
            output_path=tmp_path / "outside.jsonl",
            family="ppr",
            modality="jp",
            candidate_ids={"j1"},
            k_in=2,
        )


def test_materializer_requires_the_same_complete_question_set(tmp_path):
    materializer = _load_materializer()
    ranking = tmp_path / "ranking.parquet"
    _rankings("article", ["a1", "a2"]).to_parquet(ranking, index=False)
    texts = tmp_path / "article_texts.parquet"
    pd.DataFrame({"pair_key": ["a1", "a2"], "texte": ["T1", "T2"]}).to_parquet(texts, index=False)
    questions = tmp_path / "questions.json"
    questions.write_text(json.dumps({"questions": [
        {"qid": "q1", "enonce": "Question 1"},
        {"qid": "q2", "enonce": "Question 2"},
    ]}), encoding="utf-8")

    with pytest.raises(ValueError, match="question coverage"):
        materializer.materialize_pool(
            ranking_path=ranking,
            questions_path=questions,
            text_source_path=texts,
            output_path=tmp_path / "pool.jsonl",
            family="cosine",
            modality="article",
            candidate_ids={"a1", "a2"},
            k_in=2,
        )


def test_materializer_cli_accepts_lightgcn_as_a_frozen_ranking_family():
    materializer = _load_materializer()

    args = materializer.parse_args([
        "--ranking", "lightgcn_top100.parquet",
        "--questions", "questions.json",
        "--texts", "article_texts.parquet",
        "--output", "lightgcn_article_k10.jsonl",
        "--family", "lightgcn",
        "--modality", "article",
        "--a3-manifest", "a3.json",
        "--k-in", "10",
    ])

    assert args.family == "lightgcn"


def test_e029_preflight_is_explicitly_blocked_on_a_common_context_budget():
    payload = json.loads(PREFLIGHT_MANIFEST.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == "E029"
    assert payload["status"] == "blocked_requires_frozen_context_budget"
    assert payload["common_contract_if_unblocked"]["k_in"] == [50, 100]
    assert payload["common_contract_if_unblocked"]["k_out"] == 10
    assert payload["context_audit_characters_before_any_truncation"]["kin100"]["cosine_article"]["above_64000"] == 676
    assert payload["code_bundle"]["pool_materializer"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
