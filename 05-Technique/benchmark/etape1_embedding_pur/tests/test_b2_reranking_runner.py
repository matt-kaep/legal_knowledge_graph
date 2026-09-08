import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "102_run_b2_comparable_reranking.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b2_reranking_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_reranker_accepts_only_pool_ids_and_completes_duplicate_slots():
    runner = _load_runner()

    parsed = runner.parse_ranked_ids(json.dumps({"ranked_ids": ["b", "b", "a"]}), {"a", "b", "c"}, 3)
    normalized, fallback = runner.normalize_ranked_ids(parsed, ["a", "b", "c"], 3)

    assert normalized == ["b", "a", "c"]
    assert fallback == ["c"]
    with pytest.raises(runner.InvalidRerankingResponse, match="supplied pool"):
        runner.parse_ranked_ids(json.dumps({"ranked_ids": ["outside", "a", "b"]}), {"a", "b", "c"}, 3)


def test_reranker_prompt_contains_only_one_real_pool_and_its_question():
    runner = _load_runner()
    job = {
        "family": "cosine",
        "modality": "article",
        "qid": "q1",
        "question": "Question ?",
        "k_in": 2,
        "k_out": 10,
        "candidates": [
            {"item_id": "a", "text": "Texte A", "source_rank": 1},
            {"item_id": "b", "text": "Texte B", "source_rank": 2},
        ],
    }

    prompt = runner.render_reranking_prompt("Instruction fixe.", job)

    assert "Question ?" in prompt
    assert "Texte A" in prompt and "Texte B" in prompt
    assert '"item_id": "a"' in prompt
    assert "PPR" not in prompt


def test_invalid_reranker_response_is_explicit_zero_not_pool_completion():
    runner = _load_runner()

    slots = runner.zero_slots(3, resolution="invalid_response")

    assert [slot["resolved_item_id"] for slot in slots] == [None, None, None]
    assert [slot["resolution"] for slot in slots] == ["invalid_response"] * 3


def test_runner_can_execute_two_independent_jobs_concurrently_and_preserve_hashes(tmp_path, monkeypatch):
    runner = _load_runner()
    prompts = {"article": tmp_path / "article.txt", "jp": tmp_path / "jp.txt"}
    prompts["article"].write_text("Articles.", encoding="utf-8")
    prompts["jp"].write_text("JP.", encoding="utf-8")
    common = {
        "experiment_id": "E029", "family": "cosine", "modality": "article", "question": "Question",
        "k_in": 2, "k_out": 2, "replay_seed": None, "source_method": "cosine",
        "source_ranking_sha256": "ranking", "source_texts_sha256": "texts", "prompt_sha256": "prompt",
        "model_id": "model", "model_revision": "revision", "temperature": 0,
        "candidate_text_representation": {"source_field": "texte", "projection": "token_prefix", "tokenizer_id": "model", "tokenizer_revision": "revision", "token_cap": 192},
    }
    jobs = [
        {**common, "qid": "q1", "candidates": [{"item_id": "a1", "text": "A1"}, {"item_id": "a2", "text": "A2"}]},
        {**common, "qid": "q2", "candidates": [{"item_id": "b1", "text": "B1"}, {"item_id": "b2", "text": "B2"}]},
    ]
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text("".join(json.dumps(job) + "\n" for job in jobs), encoding="utf-8")
    responses = tmp_path / "responses.jsonl"
    monkeypatch.setattr(runner, "call_openai_compatible", lambda **kwargs: json.dumps({"ranked_ids": kwargs["pool_ids"]}))

    result = runner.run_jobs(
        jobs_path=jobs_path,
        responses_path=responses,
        endpoint="http://unused",
        model_id="model",
        prompts=prompts,
        max_workers=2,
    )

    rows = [json.loads(line) for line in responses.read_text(encoding="utf-8").splitlines()]
    assert result == {"jobs": 2, "skipped": 0, "completed": 2, "invalid": 0, "error": 0}
    assert {row["input_sha256"] for row in rows} == {runner.job_input_sha256(job) for job in jobs}


def test_jobs_keep_depth_and_replay_seed_as_distinct_frozen_conditions(tmp_path):
    runner = _load_runner()
    pool = tmp_path / "pools.jsonl"
    article_prompt = tmp_path / "article.txt"
    jp_prompt = tmp_path / "jp.txt"
    output = tmp_path / "jobs.jsonl"
    article_prompt.write_text("Articles.", encoding="utf-8")
    jp_prompt.write_text("Jurisprudence.", encoding="utf-8")
    common = {
        "experiment_id": "E029",
        "family": "lightgcn",
        "modality": "article",
        "qid": "q1",
        "question": "Question",
        "k_out": 2,
        "source_method": "LightGCN-trained_K2",
        "source_ranking_sha256": "ranking-sha",
        "source_texts_sha256": "texts-sha",
    }
    records = [
        {
            **common,
            "k_in": 2,
            "replay_seed": "42",
            "candidates": [{"item_id": "a1", "text": "A1"}, {"item_id": "a2", "text": "A2"}],
        },
        {
            **common,
            "k_in": 3,
            "replay_seed": "43",
            "candidates": [
                {"item_id": "a1", "text": "A1"},
                {"item_id": "a2", "text": "A2"},
                {"item_id": "a3", "text": "A3"},
            ],
        },
    ]
    pool.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    assert runner.prepare_jobs(
        pools=[pool],
        prompts={"article": article_prompt, "jp": jp_prompt},
        output_path=output,
        model_id="model",
        model_revision="revision",
    ) == 2

    jobs = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {(job["k_in"], job["replay_seed"]) for job in jobs} == {(2, "42"), (3, "43")}
    assert len({runner._key(job) for job in jobs}) == 2


def test_article_projection_is_token_bounded_and_jp_synthese_is_left_intact():
    runner = _load_runner()

    class Tokenizer:
        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            return list(range(len(text)))

        def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
            assert skip_special_tokens is True
            assert clean_up_tokenization_spaces is False
            return "|".join(map(str, ids))

    article, article_representation = runner.project_candidates_for_reranking(
        [{"item_id": "a1", "text": "abcdef", "source_rank": 1}],
        modality="article",
        article_token_cap=3,
        tokenizer=Tokenizer(),
        tokenizer_id="frozen/gemma",
        tokenizer_revision="abc",
    )
    jp, jp_representation = runner.project_candidates_for_reranking(
        [{"item_id": "j1", "text": "Synthèse intégrale", "source_rank": 1}],
        modality="jp",
        article_token_cap=3,
        tokenizer=Tokenizer(),
        tokenizer_id="frozen/gemma",
        tokenizer_revision="abc",
    )

    assert article == [{"item_id": "a1", "text": "0|1|2", "source_rank": 1}]
    assert article_representation == {
        "source_field": "texte",
        "projection": "token_prefix",
        "tokenizer_id": "frozen/gemma",
        "tokenizer_revision": "abc",
        "token_cap": 3,
    }
    assert jp == [{"item_id": "j1", "text": "Synthèse intégrale", "source_rank": 1}]
    assert jp_representation == {
        "source_field": "synthese",
        "projection": "complete_unmodified",
    }


def test_materialization_exports_seed_specific_metrics_before_the_seed_mean(tmp_path):
    runner = _load_runner()
    common = {
        "experiment_id": "E029",
        "family": "lightgcn",
        "modality": "article",
        "qid": "q1",
        "question": "Question",
        "k_in": 2,
        "k_out": 2,
        "source_method": "LightGCN-trained_K2",
        "source_ranking_sha256": "ranking-sha",
        "source_texts_sha256": "texts-sha",
        "prompt_sha256": "prompt-sha",
        "model_id": "model",
        "model_revision": "revision",
        "temperature": 0,
        "candidates": [{"item_id": "a1", "text": "A1"}, {"item_id": "a2", "text": "A2"}],
    }
    jobs = [{**common, "replay_seed": seed} for seed in ("42", "43")]
    responses = []
    for job in jobs:
        responses.append({
            "experiment_id": "E029",
            "family": "lightgcn",
            "modality": "article",
            "qid": "q1",
            "k_in": 2,
            "replay_seed": job["replay_seed"],
            "input_sha256": runner.job_input_sha256(job),
            "status": "ok",
            "slots": [
                {"rank": 1, "reference": "a1", "resolved_item_id": "a1", "resolution": "unique"},
                {"rank": 2, "reference": "a2", "resolved_item_id": "a2", "resolution": "unique"},
            ],
        })

    runner.materialize_exact_results(
        questions={"q1": {"articles_attendus": ["a1"]}},
        jobs=jobs,
        responses=responses,
        out_dir=tmp_path / "materialized",
    )

    by_seed = pd.read_csv(tmp_path / "materialized" / "exact_metrics.csv")
    seed_mean = pd.read_csv(tmp_path / "materialized" / "exact_metrics_seed_mean.csv")
    assert set(by_seed["replay_seed"].astype(str)) == {"42", "43"}
    assert seed_mean.loc[0, "seed_count"] == 2


def test_materialize_cli_does_not_require_run_only_prompt_arguments(tmp_path):
    runner = _load_runner()
    questions = tmp_path / "questions.json"
    jobs = tmp_path / "jobs.jsonl"
    responses = tmp_path / "responses.jsonl"
    out_dir = tmp_path / "materialized"
    questions.write_text(json.dumps({"questions": [{"qid": "q1", "articles_attendus": ["a1"]}]}), encoding="utf-8")
    job = {
        "experiment_id": "E029", "family": "cosine", "modality": "article", "qid": "q1",
        "question": "Question", "k_in": 1, "k_out": 1, "source_method": "cosine",
        "source_ranking_sha256": "ranking-sha", "source_texts_sha256": "texts-sha",
        "prompt_sha256": "prompt-sha", "model_id": "model", "model_revision": "revision",
        "temperature": 0, "candidates": [{"item_id": "a1", "text": "A1"}],
    }
    jobs.write_text(json.dumps(job) + "\n", encoding="utf-8")
    response = {
        "experiment_id": "E029", "family": "cosine", "modality": "article", "qid": "q1",
        "k_in": 1, "replay_seed": None, "input_sha256": runner.job_input_sha256(job), "status": "ok",
        "slots": [{"rank": 1, "reference": "a1", "resolved_item_id": "a1", "resolution": "unique"}],
    }
    responses.write_text(json.dumps(response) + "\n", encoding="utf-8")

    assert runner.main(["materialize", "--questions", str(questions), "--jobs", str(jobs), "--responses", str(responses), "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "materialization_receipt.json").exists()
