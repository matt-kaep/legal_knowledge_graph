import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "100_run_b2_direct_llm.py"
MANIFEST = ROOT / "configs" / "b2_direct_llm_a3_v1.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b2_direct_llm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_direct_prompt_exposes_the_question_only():
    runner = _load_runner()

    prompt = runner.render_direct_prompt("Instruction fixe.", "Quelle règle est applicable ?")

    assert "Instruction fixe." in prompt
    assert "Quelle règle est applicable ?" in prompt
    assert "candidate" not in prompt.lower()
    assert "corpus" not in prompt.lower()


def test_article_resolution_is_deterministic_and_does_not_fuzzily_match():
    runner = _load_runner()
    resolver = runner.ArticleResolver(
        ["code_penal:121-2", "code_civil:1240"],
        {"code_penal": "Code pénal", "code_civil": "Code civil"},
    )

    assert resolver.resolve("Code pénal, article 121-2") == ("code_penal:121-2", "unique")
    assert resolver.resolve("code_penal:121-2") == ("code_penal:121-2", "unique")
    assert resolver.resolve("Code pénal, article 121-3") == (None, "unresolved")
    assert resolver.resolve("Une règle proche de l'article 121-2") == (None, "unresolved")


def test_article_resolution_rejects_an_ambiguous_canonical_reference():
    runner = _load_runner()
    resolver = runner.ArticleResolver(
        ["code_penal:121-2", "code_penal_bis:121-2"],
        {"code_penal": "Code pénal", "code_penal_bis": "Code pénal"},
    )

    assert resolver.resolve("Code pénal, article 121-2") == (None, "ambiguous")


def test_jurisprudence_accepts_only_an_exact_a3_identifier():
    runner = _load_runner()
    resolver = runner.ExactIdentifierResolver(["614024df7e15a6adfd695a23"])

    assert resolver.resolve("614024df7e15a6adfd695a23") == ("614024df7e15a6adfd695a23", "unique")
    assert resolver.resolve("Cass. crim., 1 janvier 2020") == (None, "unresolved")
    assert resolver.resolve(" 614024df7e15a6adfd695a23 ") == (None, "unresolved")


def test_response_keeps_unresolved_slots_at_their_original_ranks():
    runner = _load_runner()
    resolver = runner.ExactIdentifierResolver(["jp-1"])
    raw = json.dumps({"references": ["missing", "jp-1", "jp-1"]})

    slots = runner.resolve_response_slots(raw, resolver, k_out=10)

    assert [slot["resolved_item_id"] for slot in slots[:3]] == [None, "jp-1", None]
    assert [slot["resolution"] for slot in slots[:3]] == ["unresolved", "unique", "duplicate"]
    assert [slot["rank"] for slot in slots] == list(range(1, 11))


def test_response_rejects_noncontract_json_instead_of_repairing_it():
    runner = _load_runner()
    with pytest.raises(runner.InvalidDirectResponse, match="references"):
        runner.parse_direct_response('{"ranked_ids": ["a"]}', k_out=10)
    with pytest.raises(runner.InvalidDirectResponse, match="code fence"):
        runner.parse_direct_response('```json\n{"references": []}\n```', k_out=10)


def test_rankings_validate_resolved_ids_but_allow_explicit_zero_slots(tmp_path):
    runner = _load_runner()
    questions = [{"qid": "q1", "articles_attendus": ["a"], "gold_jp_ids": ["j"]}]
    responses = [{
        "qid": "q1",
        "modality": "article",
        "status": "ok",
        "slots": [
            {"rank": 1, "reference": "missing", "resolved_item_id": None, "resolution": "unresolved"},
            {"rank": 2, "reference": "a", "resolved_item_id": "a", "resolution": "unique"},
        ] + [
            {"rank": rank, "reference": None, "resolved_item_id": None, "resolution": "empty"}
            for rank in range(3, 11)
        ],
    }]

    rankings, metrics = runner.materialize_exact_results(
        questions=questions,
        responses=responses,
        candidate_ids={"a"},
        modality="article",
    )

    assert len(rankings) == 10
    assert rankings.loc[rankings["rank"].eq(1), "item_id"].isna().all()
    assert metrics["hit_at_10"] == pytest.approx(1.0)
    assert metrics["mrr_at_10"] == pytest.approx(0.5)


def test_b2_manifest_seals_question_only_contract_and_every_frozen_input():
    runner = _load_runner()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["experiment_id"] == "E027"
    assert payload["status"] == "frozen_authorized_waiting_for_g6_queue"
    assert payload["a3"]["sha256"] == "c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3"
    assert payload["model"] == {
        "id": "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit",
        "revision": "4033b16200f4152e55e100ea12dc388c537df622",
        "temperature": 0,
        "provider": "local_vllm_openai_compatible",
        "network_policy": "HF_HUB_OFFLINE=1; TRANSFORMERS_OFFLINE=1; no web or tool",
    }
    assert payload["model_visible_contract"]["inputs"] == ["fixed_modality_prompt", "question_text"]
    assert set(payload["model_visible_contract"]["forbidden_inputs"]) >= {
        "candidate_pool", "retrieval_rankings", "graph", "corpus", "web", "tools", "gold_labels"
    }
    assert payload["tasks"]["article"]["jobs"]["questions"] == 754
    assert payload["tasks"]["jp"]["jobs"]["questions"] == 754
    assert payload["candidate_universe"]["articles"]["count"] == 13236
    assert payload["candidate_universe"]["jurisprudence"]["count"] == 114851
    assert payload["code_bundle"]["direct_runner"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert runner.load_code_titles()["code_penal"] == "Code pénal"
