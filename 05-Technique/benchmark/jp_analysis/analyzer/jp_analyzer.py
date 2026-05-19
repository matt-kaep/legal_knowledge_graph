"""Orchestrator core. I/O injected for testability. Spec §8/§9.
analyze_record is a PURE per-record function (no shared mutable state) so the
Task 11 driver can dispatch it across a bounded thread pool."""
import time
from collections import deque
from dataclasses import dataclass

from budget import is_oversized
from errors import classify_error, ErrorClass
from parsing import parse_model_json, ParseError
from schema import Step1Output, SCHEMA_VERSION
from themes_validation import canonicalize_themes
from prompts.step1.themes_taxonomy import TAXONOMY_VERSION
from prompts.step1.build_prompt import build_system_prompt

@dataclass
class RunConfig:
    model: str
    threshold: int
    max_attempts: int = 3
    max_tokens: int = 4000
    temperature: float = 0.1
    tokenizer: object = None

class RetryableError(RuntimeError):
    error_class = "retryable"

class CircuitBreaker:
    def __init__(self, window=500, max_fail_rate=0.2):
        self.events = deque(maxlen=window)
        self.max_fail_rate = max_fail_rate
    def record(self, ok: bool):
        self.events.append(0 if ok else 1)
    def tripped(self) -> bool:
        if len(self.events) < self.events.maxlen // 10 + 1:
            return False
        return (sum(self.events) / len(self.events)) > self.max_fail_rate

def _terminal(rec_id, number, juris, status, variant=None, err=None):
    return {"id": rec_id, "number": number, "juris": juris, "status": status,
            "failed": status != "ok", "themes_valid": None,
            "themes_taxonomy_version": TAXONOMY_VERSION,
            "schema_version": SCHEMA_VERSION, "model": None,
            "prompt_variant": variant, "tokens_in": None, "tokens_out": None,
            "duration_ms": None, "attempt_count": 1,
            "error_class": ("terminal" if status != "ok" else None),
            "error_message": err}

def analyze_record(row: dict, client, cfg: RunConfig) -> dict:
    rid, number, juris = row["id"], row.get("number", ""), row["juris"]
    text = row.get("text") or ""
    if not text.strip():
        return _terminal(rid, number, juris, "no_fulltext", err="empty fullText")
    if is_oversized(text, cfg.threshold, tokenizer=cfg.tokenizer):
        return _terminal(rid, number, juris, "oversized",
                         err="exceeds context budget")
    system, variant = build_system_prompt(juris)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": text}],
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            extra_body={"guided_json": Step1Output.model_json_schema()},
        )
    except Exception as exc:  # noqa: BLE001
        if classify_error(exc) == ErrorClass.RETRYABLE:
            re = RetryableError(str(exc)); re.error_class = "retryable"
            raise re from exc
        return _terminal(rid, number, juris, "failed_terminal", variant,
                         err=f"{type(exc).__name__}: {exc}")
    dur = int((time.time() - t0) * 1000)
    raw = resp.choices[0].message.content
    try:
        data = parse_model_json(raw)
        model_obj = Step1Output.model_validate(data)
    except (ParseError, ValueError) as exc:
        return _terminal(rid, number, juris, "failed_terminal", variant,
                         err=f"{type(exc).__name__}: {exc}")
    payload = model_obj.model_dump()
    clean, themes_valid, anomalies = canonicalize_themes(payload["themes"])
    payload["themes"] = clean
    rec = {"id": rid, "number": number, "juris": juris, "status": "ok",
           "failed": False, **payload, "themes_valid": themes_valid,
           "themes_taxonomy_version": TAXONOMY_VERSION,
           "schema_version": SCHEMA_VERSION, "model": cfg.model,
           "prompt_variant": variant,
           "tokens_in": getattr(resp.usage, "prompt_tokens", None),
           "tokens_out": getattr(resp.usage, "completion_tokens", None),
           "duration_ms": dur, "attempt_count": 1,
           "error_class": None, "error_message": None,
           "_anomalies": anomalies}
    return rec
