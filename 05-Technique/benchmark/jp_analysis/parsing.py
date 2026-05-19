"""Robust parsing of LLM JSON output: json.loads -> strip fence -> json-repair."""
import json
import re
from json_repair import repair_json

class ParseError(ValueError):
    pass

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)

def _strip_fence(s: str) -> str:
    return _FENCE.sub("", s).strip()

def parse_model_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    stripped = _strip_fence(raw)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    repaired = repair_json(stripped, return_objects=True)
    if isinstance(repaired, (dict, list)) and repaired != "":
        return repaired
    raise ParseError(f"irreparable JSON: {raw[:200]!r}")
