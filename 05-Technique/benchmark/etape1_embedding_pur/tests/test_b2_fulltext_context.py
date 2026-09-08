import importlib.util
import json
from pathlib import Path
import runpy
import sys
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "103_audit_b2_fulltext_context.py"


def _load_auditor():
    spec = importlib.util.spec_from_file_location("b2_fulltext_context", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_context_compatibility_reserves_completion_tokens():
    auditor = _load_auditor()

    report = auditor.audit_token_counts(
        [("q1", 16128), ("q2", 16000)],
        context_limit_tokens=16384,
        max_output_tokens=256,
    )

    assert report["input_token_budget"] == 16128
    assert report["compatible"] is True
    assert report["overflow_qids"] == []

    overflow = auditor.audit_token_counts(
        [("q1", 16129)],
        context_limit_tokens=16384,
        max_output_tokens=256,
    )

    assert overflow["compatible"] is False
    assert overflow["overflow_qids"] == ["q1"]
    assert overflow["max_total_tokens"] == 16385


def test_fulltext_audit_groups_conditions_and_retains_overflowing_question():
    auditor = _load_auditor()
    prompts_seen: list[str] = []

    def count_prompt_tokens(prompt: str) -> int:
        prompts_seen.append(prompt)
        return 16129 if "Question 2" in prompt else 16128

    jobs = [
        {
            "family": "cosine",
            "modality": "article",
            "qid": "q1",
            "question": "Question 1",
            "k_in": 10,
            "candidates": [
                {"item_id": f"a{rank}", "text": f"Texte intégral A {rank}", "source_rank": rank}
                for rank in range(1, 11)
            ],
        },
        {
            "family": "cosine",
            "modality": "article",
            "qid": "q2",
            "question": "Question 2",
            "k_in": 10,
            "candidates": [
                {"item_id": f"b{rank}", "text": f"Texte intégral B {rank}", "source_rank": rank}
                for rank in range(1, 11)
            ],
        },
    ]

    report = auditor.audit_fulltext_jobs(
        jobs,
        prompt_templates={"article": "Instruction Articles.", "jp": "Instruction JP."},
        count_prompt_tokens=count_prompt_tokens,
        context_limit_tokens=16384,
        max_output_tokens=256,
        expected_questions=2,
    )

    assert report["conditions"] == [{
        "family": "cosine",
        "modality": "article",
        "k_in": 10,
        "candidate_text_representation": {"source_field": "texte", "projection": "complete_unmodified"},
        "questions": 2,
        "context_limit_tokens": 16384,
        "max_output_tokens": 256,
        "input_token_budget": 16128,
        "max_input_tokens": 16129,
        "max_total_tokens": 16385,
        "overflow_qids": ["q2"],
        "compatible": False,
    }]
    assert all("Texte intégral" in prompt for prompt in prompts_seen)


def test_fulltext_audit_keeps_lightgcn_replay_seeds_as_distinct_conditions():
    auditor = _load_auditor()
    jobs = []
    for replay_seed in ("42", "43"):
        jobs.append({
            "family": "lightgcn_g6",
            "modality": "article",
            "qid": "q1",
            "question": "Question",
            "k_in": 10,
            "replay_seed": replay_seed,
            "candidates": [{"item_id": "a1", "text": "Article complet", "source_rank": 1}] * 10,
        })

    report = auditor.audit_fulltext_jobs(
        jobs,
        prompt_templates={"article": "Instruction Articles.", "jp": "Instruction JP."},
        count_prompt_tokens=lambda _prompt: 12,
        context_limit_tokens=16384,
        max_output_tokens=256,
        expected_questions=1,
    )

    assert [(row["replay_seed"], row["questions"]) for row in report["conditions"]] == [("42", 1), ("43", 1)]


def test_context_audit_records_the_materialized_article_projection_and_rejects_mixing():
    auditor = _load_auditor()
    base = {
        "family": "cosine",
        "modality": "article",
        "qid": "q1",
        "question": "Question",
        "k_in": 1,
        "candidates": [{"item_id": "a1", "text": "Prefixe gelé", "source_rank": 1}],
        "candidate_text_representation": {
            "source_field": "texte",
            "projection": "token_prefix",
            "tokenizer_id": "frozen/gemma",
            "tokenizer_revision": "abc",
            "token_cap": 192,
        },
    }

    report = auditor.audit_fulltext_jobs(
        [base],
        prompt_templates={"article": "Instruction Articles.", "jp": "Instruction JP."},
        count_prompt_tokens=lambda _prompt: 12,
        context_limit_tokens=16384,
        max_output_tokens=256,
        expected_questions=1,
    )

    assert report["conditions"][0]["candidate_text_representation"] == base["candidate_text_representation"]
    mixed = {**base, "candidate_text_representation": {**base["candidate_text_representation"], "token_cap": 191}}
    with pytest.raises(ValueError, match="representation differs"):
        auditor.audit_fulltext_jobs(
            [base, mixed],
            prompt_templates={"article": "Instruction Articles.", "jp": "Instruction JP."},
            count_prompt_tokens=lambda _prompt: 12,
            context_limit_tokens=16384,
            max_output_tokens=256,
            expected_questions=2,
        )


def test_chat_token_counter_uses_generation_template():
    auditor = _load_auditor()

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "Prompt complet"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return [7, 8, 9]

    assert auditor.count_chat_tokens(Tokenizer(), "Prompt complet") == 3


def test_chat_token_counter_reads_input_ids_from_batch_encoding_like_result():
    auditor = _load_auditor()

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return {"input_ids": list(range(57)), "attention_mask": [1] * 57}

    assert auditor.count_chat_tokens(Tokenizer(), "Prompt complet") == 57


def test_context_audit_uses_the_same_metadata_free_prompt_shape_as_the_reranker():
    auditor = _load_auditor()

    prompt = auditor.render_reranking_prompt("Instruction.", {
        "question": "Quelle règle ?",
        "candidates": [{"item_id": "a-1", "text": "Texte article", "source_rank": 7, "score": 0.99}],
    })

    assert prompt == (
        "Instruction.\n\nQuestion :\nQuelle règle ?\n\nRéférences à ordonner :\n"
        "[\n  {\n    \"item_id\": \"a-1\",\n    \"text\": \"Texte article\"\n  }\n]\n"
    )
    assert "source_rank" not in prompt
    assert "score" not in prompt


def test_fulltext_audit_report_is_immutable_and_records_completion_budget(tmp_path):
    auditor = _load_auditor()
    output = tmp_path / "fulltext_context_audit.json"
    report = {
        "schema_version": "b2-e029-fulltext-context-audit.v1",
        "model": {"id": "model", "revision": "revision"},
        "max_output_tokens": 256,
        "conditions": [],
    }

    auditor.write_immutable_report(output, report)

    assert json.loads(output.read_text(encoding="utf-8"))["max_output_tokens"] == 256
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        auditor.write_immutable_report(output, report)


def test_frozen_job_audit_records_hashed_inputs_and_model_contract(tmp_path):
    auditor = _load_auditor()
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps({
        "family": "ppr",
        "modality": "jp",
        "qid": "q1",
        "question": "Question complète",
        "k_in": 1,
        "candidates": [{"item_id": "j1", "text": "Décision entière", "source_rank": 1}],
    }) + "\n", encoding="utf-8")
    article_prompt = tmp_path / "article.txt"
    article_prompt.write_text("Articles.", encoding="utf-8")
    jp_prompt = tmp_path / "jp.txt"
    jp_prompt.write_text("Jurisprudence.", encoding="utf-8")

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return list(range(len(messages[0]["content"])))

    report = auditor.audit_frozen_job_files(
        [jobs_path],
        prompt_paths={"article": article_prompt, "jp": jp_prompt},
        tokenizer=Tokenizer(),
        model_id="model-id",
        model_revision="model-revision",
        context_limit_tokens=16384,
        max_output_tokens=256,
        expected_questions=1,
    )

    assert report["model"] == {"id": "model-id", "revision": "model-revision"}
    assert report["max_output_tokens"] == 256
    assert report["job_files"] == [{"path": str(jobs_path), "jobs": 1, "sha256": auditor.sha256(jobs_path)}]
    assert report["prompt_files"]["jp"]["sha256"] == auditor.sha256(jp_prompt)
    assert report["conditions"][0]["compatible"] is True


def test_cli_writes_fulltext_audit_with_the_frozen_completion_reserve(tmp_path, monkeypatch):
    auditor = _load_auditor()
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps({
        "family": "cosine", "modality": "article", "qid": "q1", "question": "Question",
        "k_in": 1, "candidates": [{"item_id": "a1", "text": "Article complet", "source_rank": 1}],
    }) + "\n", encoding="utf-8")
    article_prompt = tmp_path / "article.txt"
    article_prompt.write_text("Articles.", encoding="utf-8")
    jp_prompt = tmp_path / "jp.txt"
    jp_prompt.write_text("Jurisprudence.", encoding="utf-8")
    output = tmp_path / "audit.json"

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [0] * 12

    monkeypatch.setattr(auditor, "load_tokenizer", lambda model_id, revision, model_snapshot=None: Tokenizer())

    assert auditor.main([
        "--jobs", str(jobs_path),
        "--prompt-article", str(article_prompt),
        "--prompt-jp", str(jp_prompt),
        "--model-id", "model-id",
        "--model-revision", "model-revision",
        "--expected-questions", "1",
        "--output", str(output),
    ]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["max_output_tokens"] == 256
    assert payload["conditions"][0]["input_token_budget"] == 16128


def test_local_model_snapshot_avoids_unavailable_remote_revision(tmp_path, monkeypatch):
    auditor = _load_auditor()
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    calls = []

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            calls.append((source, kwargs))
            return "tokenizer"

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))

    assert auditor.load_tokenizer("remote/model", "obsolete-revision", model_snapshot=snapshot) == "tokenizer"
    assert calls == [(str(snapshot), {"local_files_only": True})]


def test_audit_records_tokenizer_files_from_the_local_model_snapshot(tmp_path):
    auditor = _load_auditor()
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps({
        "family": "cosine", "modality": "article", "qid": "q1", "question": "Question",
        "k_in": 1, "candidates": [{"item_id": "a1", "text": "Article entier", "source_rank": 1}],
    }) + "\n", encoding="utf-8")
    article_prompt = tmp_path / "article.txt"
    article_prompt.write_text("Articles.", encoding="utf-8")
    jp_prompt = tmp_path / "jp.txt"
    jp_prompt.write_text("Jurisprudence.", encoding="utf-8")
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text("config", encoding="utf-8")
    (snapshot / "chat_template.jinja").write_text("template", encoding="utf-8")

    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [0]

    report = auditor.audit_frozen_job_files(
        [jobs_path],
        prompt_paths={"article": article_prompt, "jp": jp_prompt},
        tokenizer=Tokenizer(),
        model_id="model-id",
        model_revision="4033",
        model_snapshot=snapshot,
        context_limit_tokens=16384,
        max_output_tokens=256,
        expected_questions=1,
    )

    assert report["model"]["local_snapshot"] == str(snapshot)
    assert report["model"]["tokenizer_files"]["tokenizer.json"] == auditor.sha256(snapshot / "tokenizer.json")


def test_script_entrypoint_runs_only_after_all_audit_definitions(tmp_path, monkeypatch):
    jobs_path = tmp_path / "jobs.jsonl"
    jobs_path.write_text(json.dumps({
        "family": "cosine", "modality": "article", "qid": "q1", "question": "Question",
        "k_in": 1, "candidates": [{"item_id": "a1", "text": "Article entier", "source_rank": 1}],
    }) + "\n", encoding="utf-8")
    article_prompt = tmp_path / "article.txt"
    article_prompt.write_text("Articles.", encoding="utf-8")
    jp_prompt = tmp_path / "jp.txt"
    jp_prompt.write_text("Jurisprudence.", encoding="utf-8")
    output = tmp_path / "audit.json"

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, **kwargs):
            return cls()

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return [0]

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=AutoTokenizer))
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT),
        "--jobs", str(jobs_path),
        "--prompt-article", str(article_prompt),
        "--prompt-jp", str(jp_prompt),
        "--model-id", "model-id",
        "--model-revision", "model-revision",
        "--expected-questions", "1",
        "--output", str(output),
    ])

    with pytest.raises(SystemExit) as exit_code:
        runpy.run_path(str(SCRIPT), run_name="__main__")

    assert exit_code.value.code == 0
    assert json.loads(output.read_text(encoding="utf-8"))["conditions"][0]["compatible"] is True
