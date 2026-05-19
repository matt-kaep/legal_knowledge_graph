import json
import pytest
from analyzer.jp_analyzer import analyze_record, CircuitBreaker, RunConfig

PEN = "Droit pénal — fond"
SUB = "atteintes aux biens et appropriations frauduleuses"

def _good_payload():
    return json.dumps({
        "contexte": "Cass crim, vol.",
        "arguments_parties": [{"partie": "d", "argument": "a", "reponse_juge": "r"}],
        "fondements_retenus": "art 311-1 cp",
        "dispositif": "rejette",
        "attendu_cle": "z" * 250,
        "cited_articles": ["article 311-1 code pénal"],
        "solution_resume": "rejet",
        "dispositif_summary": "pourvoi rejeté",
        "synthese_pour_avocat": "y" * 300,
        "dispositif_nature": "REJETTE",
        "themes": [{"branche": PEN, "sous_branche": SUB}],
    })

class FakeClient:
    def __init__(self, behavior): self.behavior = behavior; self.calls = 0
    @property
    def chat(self):
        client = self
        class Comp:
            def create(inner, **kw):
                client.calls += 1
                b = client.behavior
                if isinstance(b, Exception): raise b
                msg = type("M", (), {"content": b})()
                ch = type("Ch", (), {"message": msg})()
                return type("R", (), {"choices": [ch],
                    "usage": type("U", (), {"prompt_tokens": 10,
                                            "completion_tokens": 20})()})()
        return type("Chat", (), {"completions": Comp()})()

CFG = RunConfig(model="gemma4-31B", threshold=10_000, max_attempts=3)

def test_ok_record_status_ok():
    rec = analyze_record({"id": "1", "number": "x", "juris": "CC",
                          "text": "arrêt..."}, FakeClient(_good_payload()), CFG)
    assert rec["status"] == "ok" and rec["failed"] is False
    assert rec["themes_valid"] is True and rec["prompt_variant"] == "cassation"
    assert rec["contexte"] == "Cass crim, vol."

def test_no_fulltext_is_terminal_no_call():
    fc = FakeClient(_good_payload())
    rec = analyze_record({"id": "2", "number": "", "juris": "CC", "text": ""}, fc, CFG)
    assert rec["status"] == "no_fulltext" and fc.calls == 0

def test_oversized_is_terminal_no_call():
    fc = FakeClient(_good_payload())
    big = {"id": "3", "number": "", "juris": "CA", "text": "x" * 50_000}
    rec = analyze_record(big, fc, CFG)
    assert rec["status"] == "oversized" and fc.calls == 0

def test_timeout_raises_retryable():
    fc = FakeClient(TimeoutError("vllm down"))
    with pytest.raises(Exception) as e:
        analyze_record({"id": "4", "number": "", "juris": "CC", "text": "a"}, fc, CFG)
    assert getattr(e.value, "error_class", None) == "retryable"

def test_irreparable_json_is_failed_terminal():
    rec = analyze_record({"id": "5", "number": "", "juris": "TJ", "text": "a"},
                         FakeClient("garbage <<<"), CFG)
    assert rec["status"] == "failed_terminal" and rec["error_class"] == "terminal"

def test_circuit_breaker_trips():
    cb = CircuitBreaker(window=4, max_fail_rate=0.5)
    for _ in range(2): cb.record(ok=True)
    cb.record(ok=False)
    assert cb.tripped() is False
    cb.record(ok=False)
    cb.record(ok=False)
    assert cb.tripped() is True

def test_empty_choices_is_retryable_not_uncaught():
    class EmptyChoicesClient:
        @property
        def chat(self):
            class Comp:
                def create(inner, **kw):
                    return type("R", (), {"choices": [],
                        "usage": type("U", (), {"prompt_tokens": 1,
                                                "completion_tokens": 0})()})()
            return type("Chat", (), {"completions": Comp()})()
    with pytest.raises(Exception) as e:
        analyze_record({"id": "9", "number": "", "juris": "CC", "text": "a"},
                       EmptyChoicesClient(), CFG)
    assert getattr(e.value, "error_class", None) == "retryable"
