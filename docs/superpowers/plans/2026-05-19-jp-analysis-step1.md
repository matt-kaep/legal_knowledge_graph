# JP Analysis Step 1 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, mass-scale Python/vLLM pipeline that extracts a strict structured JSON (10 Hector fields + hierarchical `themes`) from each of the 1.12 M Judilibre decisions, using open-source `gemma4-31B-AWQ`.

**Architecture:** Pure functions per concern (taxonomy, schema, parsing, errors, themes validation, budget, ledger, prompt building), wired by one streaming orchestrator. The output JSONL shards are the single source of truth for idempotent resume. Errors are classified terminal vs retryable; a circuit breaker halts on infra degradation. Drop rule uses the live-verified model context budget.

**Tech Stack:** Python 3.12, pydantic v2, pyarrow (parquet streaming), `openai` client (vLLM OpenAI-compatible endpoint), `json-repair`, `rapidfuzz` (theme canonicalization), `transformers` tokenizer (near-threshold token counting), pytest.

**Spec:** `docs/superpowers/specs/2026-05-19-pipeline-step1-jp-analysis-design.md` (read it; this plan implements it 1:1, including §15 adversarial-review resolutions).

**Source of Hector prompt text:** the user-provided Hector "Step 1" dump (conversation 2026-05-19, sections §7a–§7e). The préambule/shared-block text in Tasks 8–9 is transcribed verbatim from that dump — it is the canonical IP and must not be paraphrased.

---

## File Structure

Base dir: `05-Technique/benchmark/jp_analysis/`

| File | Responsibility |
|------|----------------|
| `prompts/step1/themes_taxonomy.py` | Frozen 18-branch hierarchy, canonical `(branche, sous_branche)` PAIRS set, `TAXONOMY_VERSION`, `render_for_prompt()` |
| `prompts/step1/step1_shared.py` | `BLOC_FACTUEL_PARTAGE`, `BLOC_FORMAT_SORTIE_PARTAGE`, `BLOC_TAXONOMIE_THEMES` (verbatim Hector + taxonomy injection) |
| `prompts/step1/step1_cassation.py` | CC préambule (verbatim Hector §7a) |
| `prompts/step1/step1_cour_appel.py` | CA préambule (verbatim Hector §7b) |
| `prompts/step1/step1_tribunal.py` | TJ préambule (verbatim Hector §7c) |
| `prompts/step1/step1_routing.py` | `route(juris) -> (preambule, variant_name)` |
| `prompts/step1/build_prompt.py` | Assemble final system prompt (explicit string replacement, never `str.format()`) |
| `schema.py` | `Step1Output` pydantic v2 model + `json_schema()` for guided decoding |
| `parsing.py` | `parse_model_json()`: `json.loads` → strip fence → `json-repair` |
| `errors.py` | `classify_error()` → terminal vs retryable |
| `themes_validation.py` | `canonicalize_themes()` → normalized pairs + `themes_valid` + anomalies |
| `budget.py` | `verify_max_model_len()`, `prompt_overhead_tokens()`, `is_oversized()` (two-pass) |
| `ledger.py` | `derive_done_ids()`, `atomic_write_shard()`, `load_quarantine()` |
| `analyzer/jp_analyzer.py` | Streaming orchestrator: budget filter, batching, vLLM call, parse, validate, persist, resume, circuit breaker |
| `run_step1.py` | CLI: `--pilot[N] --juris --limit --resume --max-model-len --concurrency --out`; bounded thread-pool dispatch of `analyze_record` |
| `serve_vllm.sh` | Launch vLLM server for `gemma4-31B-AWQ` |
| `tests/` | pytest unit tests per module |

Outputs: `outputs/step1/<juris>/part-XXXXX.jsonl` (truth), `_quarantine.jsonl`, `_themes_anomalies.jsonl`, `_metrics.jsonl`.

---

## Task 0: Project skeleton & dependencies

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/__init__.py` (empty)
- Create: `05-Technique/benchmark/jp_analysis/prompts/__init__.py` (empty)
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/__init__.py` (empty)
- Create: `05-Technique/benchmark/jp_analysis/analyzer/__init__.py` (empty)
- Create: `05-Technique/benchmark/jp_analysis/tests/__init__.py` (empty)
- Create: `05-Technique/benchmark/jp_analysis/requirements.txt`
- Create: `05-Technique/benchmark/jp_analysis/tests/conftest.py`

- [ ] **Step 1: Create directories and empty `__init__.py` files**

```bash
cd 05-Technique/benchmark/jp_analysis
mkdir -p prompts/step1 analyzer tests
touch __init__.py prompts/__init__.py prompts/step1/__init__.py analyzer/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
pydantic>=2.6,<3
pyarrow>=15
openai>=1.30
json-repair>=0.25
rapidfuzz>=3.6
transformers>=4.44
pytest>=8
```

- [ ] **Step 3: Write `tests/conftest.py`** (makes the package importable in tests)

```python
import sys
from pathlib import Path

# jp_analysis/ on sys.path so `import schema`, `import prompts...` work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 4: Install deps and verify pytest collects**

Run: `pip install -r 05-Technique/benchmark/jp_analysis/requirements.txt && cd 05-Technique/benchmark/jp_analysis && python -m pytest -q`
Expected: `no tests ran` (exit 5) — collection works, no errors.

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis
git commit -m "chore(jp-analysis): project skeleton + deps"
```

---

## Task 1: `themes_taxonomy.py` — frozen taxonomy

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/themes_taxonomy.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_themes_taxonomy.py`

Source data: `docs/superpowers/specs/themes-taxonomy-jp.md` (18 `##` branches, `- ` sub-branches). The implementation transcribes that file's branches/sub-branches into a `TAXONOMY` dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_themes_taxonomy.py
from prompts.step1 import themes_taxonomy as tx

def test_18_branches_and_canonical_pairs():
    assert len(tx.TAXONOMY) == 18
    # every value is a non-empty list of sub-branches
    assert all(isinstance(v, list) and v for v in tx.TAXONOMY.values())
    # PAIRS is the flattened canonical set
    assert ("Droit immobilier, baux et construction",
            "baux commerciaux et indemnité d'éviction") in tx.PAIRS
    assert ("Droit des obligations et des contrats",
            "responsabilité contractuelle et dommages-intérêts") in tx.PAIRS
    # arbitrage: baux commerciaux NOT under sociétés
    assert ("Droit des sociétés et des affaires",
            "baux commerciaux et indemnité d'éviction") not in tx.PAIRS
    assert isinstance(tx.TAXONOMY_VERSION, str) and tx.TAXONOMY_VERSION

def test_render_for_prompt_lists_all_pairs():
    rendered = tx.render_for_prompt()
    for branche, subs in tx.TAXONOMY.items():
        assert branche in rendered
        for s in subs:
            assert s in rendered
    assert "Autre:" in rendered  # escape-hatch instruction present
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_themes_taxonomy.py -q`
Expected: FAIL (`ModuleNotFoundError: prompts.step1.themes_taxonomy`).

- [ ] **Step 3: Write `themes_taxonomy.py`**

Transcribe every branch (`##` heading) and sub-branch (`- ` line) from `docs/superpowers/specs/themes-taxonomy-jp.md` into the dict below. The skeleton shows the exact required structure and 3 representative branches; **fill all 18 branches with their full sub-branch lists from the canonical file** (do not abbreviate — the file is the source of truth):

```python
"""Taxonomie thèmes JP — figée. Source: docs/superpowers/specs/themes-taxonomy-jp.md
Toute modification = bump TAXONOMY_VERSION + revue _themes_anomalies."""

TAXONOMY_VERSION = "1.0.0"

TAXONOMY: dict[str, list[str]] = {
    "Droit pénal — fond": [
        "atteintes volontaires aux personnes",
        "atteintes involontaires aux personnes",
        "infractions sexuelles et atteintes aux mœurs",
        "harcèlements et violences intrafamiliales",
        "atteintes aux biens et appropriations frauduleuses",
        "atteintes à l'autorité de l'État et à l'ordre public",
        "terrorisme et association de malfaiteurs",
        "stupéfiants",
        "infractions de presse et d'expression",
        "responsabilité pénale et imputation",
        "peines, mesures de sûreté et confiscations",
        "mineurs délinquants",
    ],
    # ... transcribe the remaining 17 branches verbatim from the canonical file,
    # including the arbitrated changes:
    #   - "Droit des obligations et des contrats" CONTAINS
    #       "responsabilité contractuelle et dommages-intérêts"
    #   - "Responsabilité civile" does NOT contain it
    #   - "Droit immobilier, baux et construction" CONTAINS
    #       "baux commerciaux et indemnité d'éviction"
    #   - "Droit des sociétés et des affaires" has
    #       "fonds de commerce et opérations sur fonds" (NOT baux commerciaux)
}

PAIRS: frozenset[tuple[str, str]] = frozenset(
    (branche, sous) for branche, subs in TAXONOMY.items() for sous in subs
)

def render_for_prompt() -> str:
    lines = [
        "# Taxonomie des thèmes (OBLIGATOIRE)",
        "Choisis 1 à 4 paires (branche, sous_branche) STRICTEMENT dans la liste",
        "ci-dessous. Si aucune ne convient, et seulement dans ce cas, utilise",
        '`branche=\"Autre:<libellé court>\"` et `sous_branche=\"Autre:<libellé court>\"`.',
        "",
    ]
    for branche, subs in TAXONOMY.items():
        lines.append(f"## {branche}")
        lines.extend(f"- {s}" for s in subs)
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_themes_taxonomy.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/prompts/step1/themes_taxonomy.py 05-Technique/benchmark/jp_analysis/tests/test_themes_taxonomy.py
git commit -m "feat(jp-analysis): frozen themes taxonomy (18 branches)"
```

---

## Task 2: `schema.py` — Step1Output model

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/schema.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_schema.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schema.py
import pytest
from pydantic import ValidationError
from schema import Step1Output, SCHEMA_VERSION, json_schema

VALID = {
    "contexte": "Cour de cassation, chambre criminelle, pourvoi en matière de vol.",
    "arguments_parties": [{"partie": "demandeur", "argument": "a", "reponse_juge": "b"}],
    "fondements_retenus": "article 311-1 du code pénal",
    "dispositif": "rejette le pourvoi",
    "attendu_cle": "x" * 250,
    "cited_articles": ["article 311-1 code pénal"],
    "solution_resume": "rejet",
    "dispositif_summary": "pourvoi rejeté",
    "synthese_pour_avocat": "y" * 300,
    "dispositif_nature": "REJETTE",
    "themes": [{"branche": "Droit pénal — fond",
                "sous_branche": "atteintes aux biens et appropriations frauduleuses"}],
}

def test_valid_record_accepted():
    m = Step1Output.model_validate(VALID)
    assert m.dispositif_nature == "REJETTE"
    assert isinstance(SCHEMA_VERSION, str) and SCHEMA_VERSION

def test_missing_required_field_rejected():
    bad = dict(VALID); del bad["dispositif"]
    with pytest.raises(ValidationError):
        Step1Output.model_validate(bad)

def test_extra_key_rejected():
    bad = dict(VALID); bad["unexpected"] = 1
    with pytest.raises(ValidationError):
        Step1Output.model_validate(bad)

def test_json_schema_has_all_fields_and_no_additional_props():
    js = json_schema()
    assert js["additionalProperties"] is False
    for f in VALID:
        assert f in js["properties"]
    assert set(js["required"]) == set(VALID.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_schema.py -q`
Expected: FAIL (`ModuleNotFoundError: schema`).

- [ ] **Step 3: Write `schema.py`**

```python
"""Step1Output — schéma de sortie strict (10 champs Hector + themes)."""
from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = "1.0.0"

class ArgumentPartie(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partie: str
    argument: str
    reponse_juge: str

class Theme(BaseModel):
    model_config = ConfigDict(extra="forbid")
    branche: str
    sous_branche: str

class Step1Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contexte: str
    arguments_parties: list[ArgumentPartie]
    fondements_retenus: str
    dispositif: str
    attendu_cle: str
    cited_articles: list[str]
    solution_resume: str
    dispositif_summary: str
    synthese_pour_avocat: str
    dispositif_nature: str
    themes: list[Theme]

def json_schema() -> dict:
    """JSON Schema for vLLM guided decoding (strict, no extra keys)."""
    return Step1Output.model_json_schema()
```

Note: pydantic v2 with `extra="forbid"` emits `additionalProperties: false`. If `model_json_schema()` omits it at the top level, post-process: `js = Step1Output.model_json_schema(); js["additionalProperties"] = False; return js`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_schema.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/schema.py 05-Technique/benchmark/jp_analysis/tests/test_schema.py
git commit -m "feat(jp-analysis): Step1Output strict schema"
```

---

## Task 3: `parsing.py` — robust model-output parsing

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/parsing.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_parsing.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_parsing.py
import pytest
from parsing import parse_model_json, ParseError

def test_plain_json():
    assert parse_model_json('{"a": 1}') == {"a": 1}

def test_fenced_block_stripped():
    assert parse_model_json('```json\n{"a": 1}\n```') == {"a": 1}

def test_broken_json_repaired():
    # trailing comma + missing brace, json-repair fixes it
    assert parse_model_json('{"a": 1, "b": [1,2,],}') == {"a": 1, "b": [1, 2]}

def test_irreparable_raises():
    with pytest.raises(ParseError):
        parse_model_json("not json at all <<<")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_parsing.py -q`
Expected: FAIL (`ModuleNotFoundError: parsing`).

- [ ] **Step 3: Write `parsing.py`**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_parsing.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/parsing.py 05-Technique/benchmark/jp_analysis/tests/test_parsing.py
git commit -m "feat(jp-analysis): robust JSON parsing"
```

---

## Task 4: `errors.py` — terminal vs retryable classification (finding #2)

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/errors.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_errors.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_errors.py
from errors import classify_error, ErrorClass
from parsing import ParseError

def test_timeout_is_retryable():
    assert classify_error(TimeoutError("vllm timeout")) == ErrorClass.RETRYABLE

def test_connection_is_retryable():
    assert classify_error(ConnectionError("conn refused")) == ErrorClass.RETRYABLE

def test_http_5xx_is_retryable():
    class E(Exception):
        status_code = 503
    assert classify_error(E("server error")) == ErrorClass.RETRYABLE

def test_http_400_is_terminal():
    class E(Exception):
        status_code = 400
    assert classify_error(E("bad request / context overflow")) == ErrorClass.TERMINAL

def test_parse_error_is_terminal():
    assert classify_error(ParseError("irreparable")) == ErrorClass.TERMINAL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_errors.py -q`
Expected: FAIL (`ModuleNotFoundError: errors`).

- [ ] **Step 3: Write `errors.py`**

```python
"""Classify exceptions into terminal (write failed_terminal record, never retry)
vs retryable (quarantine, retry next run). Resolves adversarial finding #2."""
from enum import Enum
from parsing import ParseError

class ErrorClass(str, Enum):
    TERMINAL = "terminal"
    RETRYABLE = "retryable"

def classify_error(exc: Exception) -> ErrorClass:
    # Parse/validation/content errors: data is bad, retrying won't help.
    if isinstance(exc, (ParseError, ValueError)):
        return ErrorClass.TERMINAL
    # HTTP status if present (openai.APIStatusError exposes .status_code)
    status = getattr(exc, "status_code", None)
    if status is not None:
        if 500 <= int(status) <= 599:
            return ErrorClass.RETRYABLE
        return ErrorClass.TERMINAL  # 4xx incl. 400 context overflow
    # Infra: timeouts / connection drops are retryable.
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ErrorClass.RETRYABLE
    # Unknown -> retryable (safer: re-tried, never silently lost).
    return ErrorClass.RETRYABLE
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_errors.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/errors.py 05-Technique/benchmark/jp_analysis/tests/test_errors.py
git commit -m "feat(jp-analysis): terminal vs retryable error classification"
```

---

## Task 5: `themes_validation.py` — canonicalization (finding #1)

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/themes_validation.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_themes_validation.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_themes_validation.py
from themes_validation import canonicalize_themes

PEN = "Droit pénal — fond"
SUB = "atteintes aux biens et appropriations frauduleuses"

def test_exact_pair_accepted():
    pairs, valid, anomalies = canonicalize_themes([{"branche": PEN, "sous_branche": SUB}])
    assert pairs == [{"branche": PEN, "sous_branche": SUB}]
    assert valid is True and anomalies == []

def test_accent_case_variant_canonicalized():
    pairs, valid, anomalies = canonicalize_themes(
        [{"branche": "droit penal - fond", "sous_branche": SUB.upper()}])
    assert pairs == [{"branche": PEN, "sous_branche": SUB}]
    assert valid is True and anomalies == []

def test_well_formed_autre_accepted():
    t = [{"branche": "Autre:droit minier", "sous_branche": "Autre:redevances"}]
    pairs, valid, anomalies = canonicalize_themes(t)
    assert pairs == t and valid is True and anomalies == []

def test_incoherent_pair_dropped_and_flagged():
    t = [{"branche": "Droit du travail", "sous_branche": "stupéfiants"}]
    pairs, valid, anomalies = canonicalize_themes(t)
    assert pairs == []           # bad pair removed
    assert valid is False
    assert anomalies and anomalies[0]["raw"] == t[0]

def test_mixed_keeps_good_drops_bad():
    t = [{"branche": PEN, "sous_branche": SUB},
         {"branche": "xxx", "sous_branche": "yyy"}]
    pairs, valid, anomalies = canonicalize_themes(t)
    assert {"branche": PEN, "sous_branche": SUB} in pairs
    assert len(pairs) == 1 and valid is False and len(anomalies) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_themes_validation.py -q`
Expected: FAIL (`ModuleNotFoundError: themes_validation`).

- [ ] **Step 3: Write `themes_validation.py`**

```python
"""Canonicalize/validate themes against the frozen taxonomy. Resolves finding #1.
Invalid pairs are DROPPED + flagged (not a record-level failure): see spec §3.2."""
import re
import unicodedata
from rapidfuzz import process, fuzz
from prompts.step1.themes_taxonomy import PAIRS

_AUTRE = re.compile(r"^Autre:[\w \-'’/().]{2,40}$")
_FUZZ_MIN = 92  # high threshold: canonicalize only near-identical variants

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[\s\-—–]+", " ", s).strip().lower()
    return s

# normalized canonical lookup: normed (branche|sous) -> exact canonical pair
_LOOKUP = {(_norm(b), _norm(s)): (b, s) for (b, s) in PAIRS}
_NORM_KEYS = list(_LOOKUP.keys())

def _match_canonical(b: str, s: str):
    key = (_norm(b), _norm(s))
    if key in _LOOKUP:
        return _LOOKUP[key]
    joined = f"{key[0]} || {key[1]}"
    choices = {f"{kb} || {ks}": (kb, ks) for (kb, ks) in _NORM_KEYS}
    hit = process.extractOne(joined, choices.keys(), scorer=fuzz.ratio,
                             score_cutoff=_FUZZ_MIN)
    if hit:
        return _LOOKUP[choices[hit[0]]]
    return None

def canonicalize_themes(themes: list[dict]):
    """Returns (clean_pairs, themes_valid, anomalies)."""
    clean, anomalies = [], []
    for t in themes or []:
        b, s = (t or {}).get("branche", ""), (t or {}).get("sous_branche", "")
        if _AUTRE.match(b or "") and _AUTRE.match(s or ""):
            clean.append({"branche": b, "sous_branche": s})
            continue
        m = _match_canonical(b, s)
        if m:
            clean.append({"branche": m[0], "sous_branche": m[1]})
        else:
            anomalies.append({"raw": t, "reason": "no_canonical_match"})
    return clean, len(anomalies) == 0, anomalies
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_themes_validation.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/themes_validation.py 05-Technique/benchmark/jp_analysis/tests/test_themes_validation.py
git commit -m "feat(jp-analysis): themes canonicalization + anomaly flagging"
```

---

## Task 6: `budget.py` — live context budget (finding #4)

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/budget.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_budget.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_budget.py
import pytest
from budget import compute_threshold, is_oversized, BudgetError

def test_max_model_len_required():
    with pytest.raises(BudgetError):
        compute_threshold(max_model_len=None, overhead_tokens=2000, max_tokens=4000)

def test_threshold_formula():
    # 32768 - 4000 - 2000 - 512 margin = 26256
    assert compute_threshold(32768, overhead_tokens=2000, max_tokens=4000) == 26256

def test_coarse_oversized_far_above():
    thr = 26256
    # /3.0 estimate; 3.0*thr chars -> exactly thr; far above => oversized, no tokenizer
    assert is_oversized("x" * (3 * thr + 10_000), thr, tokenizer=None) is True

def test_coarse_clearly_below_not_oversized():
    thr = 26256
    assert is_oversized("x" * 1000, thr, tokenizer=None) is False

def test_near_threshold_uses_tokenizer():
    thr = 100
    # within ±20% band -> tokenizer decides. Fake tokenizer: 1 token per char.
    class T:
        def encode(self, s): return list(s)
    text = "x" * 105  # est /3 = 35 (below coarse) but real tokens=105 (> thr)
    assert is_oversized(text, thr, tokenizer=T(), band=0.5) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_budget.py -q`
Expected: FAIL (`ModuleNotFoundError: budget`).

- [ ] **Step 3: Write `budget.py`**

```python
"""Context budget. max_model_len is REQUIRED and must be verified against the
live vLLM /v1/models at startup. Near-threshold records counted with the real
tokenizer. Resolves adversarial finding #4."""

class BudgetError(RuntimeError):
    pass

_MARGIN = 512
_CHARS_PER_TOK = 3.0

def compute_threshold(max_model_len, overhead_tokens: int, max_tokens: int) -> int:
    if not max_model_len:
        raise BudgetError("max_model_len is required (verify against /v1/models)")
    thr = int(max_model_len) - int(max_tokens) - int(overhead_tokens) - _MARGIN
    if thr <= 0:
        raise BudgetError(f"non-positive input budget: {thr}")
    return thr

def is_oversized(full_text: str, threshold: int, tokenizer=None,
                 band: float = 0.2) -> bool:
    n = len(full_text or "")
    est = n / _CHARS_PER_TOK
    lo, hi = threshold * (1 - band), threshold * (1 + band)
    if est < lo:
        return False
    if est > hi:
        return True
    if tokenizer is None:
        return est > threshold  # no tokenizer: fall back to estimate
    return len(tokenizer.encode(full_text)) > threshold

def verify_max_model_len(client, expected: int) -> int:
    """Query the live vLLM server; abort if it disagrees with `expected`."""
    models = client.models.list()
    served = getattr(models.data[0], "max_model_len", None)
    if served is not None and int(served) != int(expected):
        raise BudgetError(
            f"server max_model_len={served} != expected {expected}")
    return int(expected)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_budget.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/budget.py 05-Technique/benchmark/jp_analysis/tests/test_budget.py
git commit -m "feat(jp-analysis): live-verified context budget + two-pass oversize"
```

---

## Task 7: `ledger.py` — atomic output-derived resume (finding #3)

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/ledger.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_ledger.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ledger.py
import json
from pathlib import Path
from ledger import atomic_write_shard, derive_done_ids, append_jsonl

def test_atomic_write_and_derive(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    recs = [{"id": "a", "status": "ok"}, {"id": "b", "status": "oversized"}]
    atomic_write_shard(out / "CC" / "part-00000.jsonl", recs)
    # one record per id, both statuses count as done
    assert derive_done_ids(out) == {"a", "b"}

def test_no_partial_file_on_crash(tmp_path: Path, monkeypatch):
    out = tmp_path / "outputs" / "step1"
    target = out / "CC" / "part-00001.jsonl"
    import ledger
    def boom(*a, **k): raise OSError("disk full")
    monkeypatch.setattr(ledger.os, "replace", boom)
    try:
        atomic_write_shard(target, [{"id": "x", "status": "ok"}])
    except OSError:
        pass
    assert not target.exists()              # no partial/corrupt shard
    assert not list((out / "CC").glob("*.tmp*")) if (out / "CC").exists() else True

def test_append_jsonl_roundtrip(tmp_path: Path):
    f = tmp_path / "q.jsonl"
    append_jsonl(f, {"id": "z", "attempt_count": 1})
    append_jsonl(f, {"id": "z2", "attempt_count": 2})
    rows = [json.loads(l) for l in f.read_text().splitlines()]
    assert rows == [{"id": "z", "attempt_count": 1}, {"id": "z2", "attempt_count": 2}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_ledger.py -q`
Expected: FAIL (`ModuleNotFoundError: ledger`).

- [ ] **Step 3: Write `ledger.py`**

```python
"""Idempotent ledger. Source of truth = committed JSONL shards (atomic write).
derive_done_ids scans shards at startup. Resolves adversarial finding #3."""
import json
import os
import tempfile
from pathlib import Path

def atomic_write_shard(path: Path, records: list[dict]) -> None:
    """Write all records to a shard via temp file + fsync + atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)          # atomic on POSIX
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

def derive_done_ids(out_root: Path) -> set[str]:
    """Done = any id with a terminal record in any <juris>/part-*.jsonl."""
    out_root = Path(out_root)
    done: set[str] = set()
    if not out_root.exists():
        return done
    for shard in out_root.glob("*/part-*.jsonl"):
        for line in shard.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done

def append_jsonl(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def load_quarantine(path: Path) -> dict[str, int]:
    """id -> max attempt_count seen (so reruns increment, not reset)."""
    path = Path(path)
    q: dict[str, int] = {}
    if not path.exists():
        return q
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            q[r["id"]] = max(q.get(r["id"], 0), int(r.get("attempt_count", 1)))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return q
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_ledger.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/ledger.py 05-Technique/benchmark/jp_analysis/tests/test_ledger.py
git commit -m "feat(jp-analysis): atomic output-derived resume ledger"
```

---

## Task 8: Prompt préambules + shared blocks (verbatim Hector)

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/step1_shared.py`
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/step1_cassation.py`
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/step1_cour_appel.py`
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/step1_tribunal.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_prompt_blocks.py`

> **Verbatim transcription required.** The full text of `PREAMBULE_CASSATION`
> (Hector §7a), `PREAMBULE_COUR_APPEL` (§7b), `PREAMBULE_TRIBUNAL` (§7c),
> `BLOC_FACTUEL_PARTAGE` (§7d), `BLOC_FORMAT_SORTIE_PARTAGE` (§7e) must be
> copied **word-for-word** from the user-provided Hector dump (conversation
> 2026-05-19). Do not paraphrase, summarize, or "improve" it — invariant D2
> (verbatim prompt) depends on this. Keep the strict `attendu_cle` verbatim
> wording from §7e exactly as given.

- [ ] **Step 1: Write the failing test (structural invariants)**

```python
# tests/test_prompt_blocks.py
from prompts.step1 import step1_shared as sh
from prompts.step1.step1_cassation import PREAMBULE_CASSATION
from prompts.step1.step1_cour_appel import PREAMBULE_COUR_APPEL
from prompts.step1.step1_tribunal import PREAMBULE_TRIBUNAL

def test_preambules_are_distinct_and_nonempty():
    ps = [PREAMBULE_CASSATION, PREAMBULE_COUR_APPEL, PREAMBULE_TRIBUNAL]
    assert all(len(p) > 400 for p in ps)
    assert len(set(ps)) == 3

def test_preambule_markers():
    assert "Cour de cassation" in PREAMBULE_CASSATION
    assert "Cour d'appel" in PREAMBULE_COUR_APPEL
    assert "tribunal de première instance" in PREAMBULE_TRIBUNAL.lower() \
        or "première instance" in PREAMBULE_TRIBUNAL.lower()

def test_shared_blocks_present():
    assert "préservation factuelle" in sh.BLOC_FACTUEL_PARTAGE.lower()
    # verbatim invariant for attendu_cle kept exactly (D2)
    assert "Reproduction littérale" in sh.BLOC_FORMAT_SORTIE_PARTAGE
    assert "synthese_pour_avocat" in sh.BLOC_FORMAT_SORTIE_PARTAGE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_prompt_blocks.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write the four prompt modules**

`step1_cassation.py`:
```python
PREAMBULE_CASSATION = r"""<<< transcribe Hector §7a verbatim from the
2026-05-19 dump: "Tu es un juriste expert ... # Spécificités Cassation /
Conseil d'État ... synthese_pour_avocat — registre Cassation ..." >>>"""
```
`step1_cour_appel.py`:
```python
PREAMBULE_COUR_APPEL = r"""<<< transcribe Hector §7b verbatim >>>"""
```
`step1_tribunal.py`:
```python
PREAMBULE_TRIBUNAL = r"""<<< transcribe Hector §7c verbatim >>>"""
```
`step1_shared.py`:
```python
from prompts.step1.themes_taxonomy import render_for_prompt

BLOC_FACTUEL_PARTAGE = r"""<<< transcribe Hector §7d verbatim >>>"""
BLOC_FORMAT_SORTIE_PARTAGE = r"""<<< transcribe Hector §7e verbatim >>>"""
# Taxonomy block appended after the output-format block (spec §6).
BLOC_TAXONOMIE_THEMES = render_for_prompt()
```
Use Python raw triple-quoted strings (`r"""..."""`). If the dump text contains a literal `"""`, escape by closing/concatenating string literals. Do not alter punctuation, accents, or casing.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_prompt_blocks.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/prompts/step1/step1_shared.py 05-Technique/benchmark/jp_analysis/prompts/step1/step1_cassation.py 05-Technique/benchmark/jp_analysis/prompts/step1/step1_cour_appel.py 05-Technique/benchmark/jp_analysis/prompts/step1/step1_tribunal.py 05-Technique/benchmark/jp_analysis/tests/test_prompt_blocks.py
git commit -m "feat(jp-analysis): verbatim Hector préambules + shared blocks"
```

---

## Task 9: `step1_routing.py` + `build_prompt.py`

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/step1_routing.py`
- Create: `05-Technique/benchmark/jp_analysis/prompts/step1/build_prompt.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_routing_build.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_routing_build.py
import pytest
from prompts.step1.step1_routing import route
from prompts.step1.build_prompt import build_system_prompt

def test_routing():
    assert route("CC")[1] == "cassation"
    assert route("CA")[1] == "cour_appel"
    assert route("TJ")[1] == "tribunal"

def test_routing_unknown_raises():
    with pytest.raises(ValueError):
        route("XX")

def test_build_contains_all_blocks_and_no_format_keyerror():
    # JSON braces in taxonomy/schema must NOT break assembly (lesson #18859)
    sys_prompt, variant = build_system_prompt("CA")
    assert variant == "cour_appel"
    assert "Cour d'appel" in sys_prompt
    assert "# Règles" in sys_prompt
    assert "préservation factuelle" in sys_prompt.lower()
    assert "Taxonomie des thèmes" in sys_prompt
    # an accidental str.format() would raise on the JSON braces; assert braces survive
    assert "{" in sys_prompt and "}" in sys_prompt

def test_build_all_three_variants_have_taxonomy():
    for j in ("CC", "CA", "TJ"):
        sp, _ = build_system_prompt(j)
        assert "Autre:" in sp
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_routing_build.py -q`
Expected: FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Write both modules**

`step1_routing.py`:
```python
from prompts.step1.step1_cassation import PREAMBULE_CASSATION
from prompts.step1.step1_cour_appel import PREAMBULE_COUR_APPEL
from prompts.step1.step1_tribunal import PREAMBULE_TRIBUNAL

_ROUTES = {
    "CC": (PREAMBULE_CASSATION, "cassation"),
    "CA": (PREAMBULE_COUR_APPEL, "cour_appel"),
    "TJ": (PREAMBULE_TRIBUNAL, "tribunal"),
}

def route(juris: str):
    try:
        return _ROUTES[juris]
    except KeyError:
        raise ValueError(f"unknown juris {juris!r} (expected CC|CA|TJ)")
```

`build_prompt.py`:
```python
from prompts.step1.step1_routing import route
from prompts.step1 import step1_shared as sh

def build_system_prompt(juris: str):
    """Assemble by concatenation ONLY — never str.format()/%/f-string on the
    block bodies (they contain literal JSON braces). Lesson: doctrine_qgen #18859."""
    preambule, variant = route(juris)
    system = (
        preambule
        + "\n\n# Règles\n\n"
        + sh.BLOC_FACTUEL_PARTAGE
        + "\n\n"
        + sh.BLOC_FORMAT_SORTIE_PARTAGE
        + "\n\n"
        + sh.BLOC_TAXONOMIE_THEMES
    )
    return system, variant
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_routing_build.py -q`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/prompts/step1/step1_routing.py 05-Technique/benchmark/jp_analysis/prompts/step1/build_prompt.py 05-Technique/benchmark/jp_analysis/tests/test_routing_build.py
git commit -m "feat(jp-analysis): juris routing + prompt assembly"
```

---

## Task 10: `analyzer/jp_analyzer.py` — orchestrator core (unit-tested with a fake client)

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/analyzer/jp_analyzer.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_analyzer.py`

The orchestrator's I/O (parquet stream, vLLM HTTP) is injected so it is unit-testable. `analyze_record` handles one decision; `CircuitBreaker` tracks retryable failure rate; `run_stream` drives an iterable of input dicts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyzer.py
import json
from pathlib import Path
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
    class _C:  # chat.completions.create shim
        pass
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

def test_timeout_raises_retryable(monkeypatch):
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
    cb.record(ok=False); 
    assert cb.tripped() is False
    cb.record(ok=False)         # 2/4 = 0.5 -> not > 0.5
    cb.record(ok=False)         # 3/4 within last 4 -> tripped
    assert cb.tripped() is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_analyzer.py -q`
Expected: FAIL (`ModuleNotFoundError: analyzer.jp_analyzer`).

- [ ] **Step 3: Write `analyzer/jp_analyzer.py`**

```python
"""Orchestrator core. I/O injected for testability. Spec §8/§9."""
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_analyzer.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/analyzer/jp_analyzer.py 05-Technique/benchmark/jp_analysis/tests/test_analyzer.py
git commit -m "feat(jp-analysis): orchestrator core (analyze_record + circuit breaker)"
```

---

## Task 11: `run_step1.py` — streaming CLI driver

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/run_step1.py`
- Test: `05-Technique/benchmark/jp_analysis/tests/test_run_step1.py`

`run_step1` wires parquet streaming + analyzer + ledger + quarantine. The parquet reader and client are injected via params defaulting to real implementations, so the test uses an in-memory iterable.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_step1.py
import json
from pathlib import Path
from run_step1 import run
from analyzer.jp_analyzer import RunConfig
from tests.test_analyzer import FakeClient, _good_payload

def _rows():
    return [
        {"id": "a", "number": "1", "juris": "CC", "text": "arrêt a"},
        {"id": "b", "number": "2", "juris": "CA", "text": "arrêt b"},
        {"id": "c", "number": "3", "juris": "TJ", "text": ""},  # no_fulltext
    ]

def test_run_writes_shards_and_is_resumable(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    run(_rows(), FakeClient(_good_payload()), cfg, out_root=out)
    ids = set()
    for shard in out.glob("*/part-*.jsonl"):
        for line in shard.read_text().splitlines():
            ids.add(json.loads(line)["id"])
    assert ids == {"a", "b", "c"}
    # second run: all done -> no duplicate records
    before = sorted(p.read_text() for p in out.glob("*/part-*.jsonl"))
    run(_rows(), FakeClient(_good_payload()), cfg, out_root=out)
    after = sorted(p.read_text() for p in out.glob("*/part-*.jsonl"))
    assert before == after  # idempotent

def test_retryable_goes_to_quarantine_not_shard(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    run([{"id": "z", "number": "", "juris": "CC", "text": "a"}],
        FakeClient(TimeoutError("down")), cfg, out_root=out)
    assert not list(out.glob("*/part-*.jsonl"))            # no terminal record
    q = (out / "_quarantine.jsonl").read_text().strip()
    assert json.loads(q)["id"] == "z"

def test_concurrent_run_processes_every_id_exactly_once(tmp_path: Path):
    out = tmp_path / "outputs" / "step1"
    cfg = RunConfig(model="gemma4-31B", threshold=10_000)
    rows = [{"id": f"r{i}", "number": str(i), "juris": "CC", "text": "x"}
            for i in range(40)]
    run(rows, FakeClient(_good_payload()), cfg, out_root=out, concurrency=8)
    ids = [json.loads(l)["id"]
           for shard in out.glob("*/part-*.jsonl")
           for l in shard.read_text().splitlines()]
    assert sorted(ids) == sorted(r["id"] for r in rows)   # no loss
    assert len(ids) == len(set(ids))                       # no duplicate
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_run_step1.py -q`
Expected: FAIL (`ModuleNotFoundError: run_step1`).

- [ ] **Step 3: Write `run_step1.py`**

```python
"""Streaming driver + CLI. Spec §7/§8. Parquet & client injected for tests."""
import argparse
import itertools
from pathlib import Path

from analyzer.jp_analyzer import analyze_record, CircuitBreaker, RunConfig, RetryableError
from ledger import (atomic_write_shard, derive_done_ids, append_jsonl,
                    load_quarantine)

PARQUET = "05-Technique/benchmark/baseline_b2/jp_index.parquet"

def _iter_parquet(path, limit=None, juris=None):
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    seen = 0
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["id", "number", "juris", "text"])
        for r in tbl.to_pylist():
            if juris and r["juris"] != juris:
                continue
            yield r
            seen += 1
            if limit and seen >= limit:
                return

def run(rows, client, cfg: RunConfig, out_root: Path, shard_size=500,
        concurrency: int = 16):
    """Stream rows, dispatch analyze_record across a bounded thread pool
    (vLLM continuous-batches server-side; client concurrency is the throughput
    lever). analyze_record is a pure per-record function -> thread-safe. ALL
    result handling (quarantine, buckets, flush, circuit breaker) runs in this
    coordinator thread only, so shard atomicity (§9) is preserved.
    concurrency=1 == sequential."""
    import concurrent.futures as cf
    out_root = Path(out_root)
    done = derive_done_ids(out_root)
    quarantine = load_quarantine(out_root / "_quarantine.jsonl")
    cb = CircuitBreaker()
    buckets: dict[str, list] = {}
    counters: dict[str, int] = {}

    def flush(juris):
        recs = buckets.get(juris)
        if not recs:
            return
        n = counters.get(juris, 0)
        atomic_write_shard(out_root / juris / f"part-{n:05d}.jsonl",
                            [{k: v for k, v in r.items() if k != "_anomalies"}
                             for r in recs])
        for r in recs:
            for a in r.get("_anomalies", []):
                append_jsonl(out_root / "_themes_anomalies.jsonl",
                             {"id": r["id"], **a})
            append_jsonl(out_root / "_metrics.jsonl",
                         {k: r.get(k) for k in ("id", "status", "error_class",
                          "attempt_count", "tokens_in", "tokens_out",
                          "duration_ms", "model")})
        counters[juris] = n + 1
        buckets[juris] = []

    def handle(row, result_exc):
        """Coordinator-thread post-processing of one finished record."""
        rec, exc = result_exc
        if isinstance(exc, RetryableError):
            attempts = quarantine.get(row["id"], 0) + 1
            cb.record(ok=False)
            if attempts >= cfg.max_attempts:
                rec = {"id": row["id"], "number": row.get("number", ""),
                       "juris": row["juris"], "status": "failed_terminal",
                       "failed": True, "error_class": "retryable",
                       "error_message": str(exc), "attempt_count": attempts}
            else:
                append_jsonl(out_root / "_quarantine.jsonl",
                             {"id": row["id"], "attempt_count": attempts,
                              "error_message": str(exc)})
                return cb.tripped()
        else:
            cb.record(ok=(rec["status"] == "ok"))
        buckets.setdefault(row["juris"], []).append(rec)
        if len(buckets[row["juris"]]) >= shard_size:
            flush(row["juris"])
        return cb.tripped()

    def work(row):
        try:
            return analyze_record(row, client, cfg), None
        except RetryableError as exc:
            return None, exc

    pending = (r for r in rows if r["id"] not in done)
    with cf.ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        inflight: dict = {}
        exhausted = False
        while True:
            while not exhausted and len(inflight) < max(1, concurrency):
                nxt = next(pending, None)
                if nxt is None:
                    exhausted = True
                    break
                inflight[pool.submit(work, nxt)] = nxt
            if not inflight:
                break
            done_fut, _ = cf.wait(inflight, return_when=cf.FIRST_COMPLETED)
            for fut in done_fut:
                row = inflight.pop(fut)
                if handle(row, fut.result()):
                    for j in list(buckets):
                        flush(j)
                    raise RuntimeError(
                        "circuit breaker tripped — pausing run "
                        "(retryable failure rate too high; infra degraded)")
    for j in list(buckets):
        flush(j)

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--pilot", type=int, default=0,
                   help="N stratified records (CC/CA/TJ) smoke run")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--juris", choices=["CC", "CA", "TJ"], default=None)
    p.add_argument("--max-model-len", type=int, required=True)
    p.add_argument("--model", default="gemma4-31B")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--out", default="outputs/step1")
    p.add_argument("--parquet", default=PARQUET)
    p.add_argument("--concurrency", type=int, default=16,
                   help="in-flight vLLM requests (1 = sequential)")
    args = p.parse_args(argv)

    from openai import OpenAI
    from transformers import AutoTokenizer
    from budget import compute_threshold, verify_max_model_len
    from prompts.step1.build_prompt import build_system_prompt

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    verify_max_model_len(client, args.max_model_len)
    tok = AutoTokenizer.from_pretrained(args.model) if False else None  # set real id at deploy
    overhead = max(len(build_system_prompt(j)[0]) // 3 for j in ("CC", "CA", "TJ"))
    threshold = compute_threshold(args.max_model_len, overhead, 4000)
    cfg = RunConfig(model=args.model, threshold=threshold, tokenizer=tok)

    if args.pilot:
        per = max(1, args.pilot // 3)
        rows = itertools.chain(
            _iter_parquet(args.parquet, per, "CC"),
            _iter_parquet(args.parquet, per, "CA"),
            _iter_parquet(args.parquet, per, "TJ"))
    else:
        rows = _iter_parquet(args.parquet, args.limit, args.juris)
    run(rows, client, cfg, Path(args.out), concurrency=args.concurrency)

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest tests/test_run_step1.py -q`
Expected: PASS (2 tests).

- [ ] **Step 5: Run the full suite**

Run: `cd 05-Technique/benchmark/jp_analysis && python -m pytest -q`
Expected: PASS (all tests, ~35).

- [ ] **Step 6: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/run_step1.py 05-Technique/benchmark/jp_analysis/tests/test_run_step1.py
git commit -m "feat(jp-analysis): streaming CLI driver + resume/quarantine"
```

---

## Task 12: `serve_vllm.sh` — vLLM server launcher

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/serve_vllm.sh`

- [ ] **Step 1: Write `serve_vllm.sh`** (mirrors `run_all_models.py` patterns; explicit `--max-model-len`)

```bash
#!/usr/bin/env bash
set -euo pipefail
# gemma4-31B-AWQ via vLLM OpenAI-compatible server.
# max-model-len MUST be explicit (adversarial finding #4) — never inherit a default.
MODEL_ID="${MODEL_ID:-QuantTrio/gemma-4-31B-it-AWQ}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
PORT="${PORT:-8000}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --max-model-len "$MAX_MODEL_LEN" \
  --port "$PORT" \
  --guided-decoding-backend xgrammar \
  --gpu-memory-utilization 0.92
```

- [ ] **Step 2: Make executable & syntax-check**

Run: `chmod +x 05-Technique/benchmark/jp_analysis/serve_vllm.sh && bash -n 05-Technique/benchmark/jp_analysis/serve_vllm.sh`
Expected: no output, exit 0.

- [ ] **Step 3: Commit**

```bash
git add 05-Technique/benchmark/jp_analysis/serve_vllm.sh
git commit -m "feat(jp-analysis): vLLM server launcher (explicit max-model-len)"
```

---

## Task 13: Pilot run (30 JP) — integration gate

**Files:**
- Create: `05-Technique/benchmark/jp_analysis/PILOT.md` (run log + qualitative review)

This task is **executed on the cluster** (not in CI). It validates R1–R4 from spec §12.

- [ ] **Step 1: Prerequisite — cluster env (spec R1)**
  Start `serve_vllm.sh` on the L40S node. Confirm one test completion succeeds (env not regressed: transformers has gemma4 arch, NumPy pinned). If it fails, STOP and report — do not proceed (memory `cluster-user-env-fragile`).

- [ ] **Step 2: Run the pilot**

Run: `cd 05-Technique/benchmark/jp_analysis && python run_step1.py --pilot 30 --max-model-len 32768 --out outputs/step1_pilot`
Expected: 30 records across `outputs/step1_pilot/{CC,CA,TJ}/part-*.jsonl`, exit 0.

- [ ] **Step 3: Qualitative review → `PILOT.md`**
  Record: % `status==ok`, % `themes_valid`, `_themes_anomalies` list (taxonomy gaps), `attendu_cle` verbatim fidelity (spot-check vs source text — R2), guided-decoding success (R4), oversize/backlog counts.

- [ ] **Step 3b: Parallelization benchmark (spec §10)**
  Re-run the same 30-JP sample at `--concurrency` ∈ {1, 8, 16, 32} against the same vLLM server (use a throwaway `--out` per level so shards don't collide; the resume ledger would otherwise skip already-done ids):
  ```
  for C in 1 8 16 32; do
    rm -rf outputs/step1_bench_$C
    /usr/bin/time -p python run_step1.py --pilot 30 --max-model-len 32768 \
      --concurrency $C --out outputs/step1_bench_$C
  done
  ```
  Record in `PILOT.md` a table: concurrency → wall time, records/min, latence p50/p95 (from `_metrics.jsonl` `duration_ms`), error rate. Identify the knee where throughput saturates (vLLM server becomes the bottleneck). Pick the full-run `--concurrency` from this and recompute the 1.12 M GPU-hours estimate (R3).

- [ ] **Step 4: Human gate (spec §10)**
  Present `PILOT.md` to the user. Do NOT launch the full 1.12 M run without explicit approval. Taxonomy adjustments from anomalies → bump `TAXONOMY_VERSION`, re-pilot if material.

- [ ] **Step 5: Commit the pilot log**

```bash
git add 05-Technique/benchmark/jp_analysis/PILOT.md
git commit -m "docs(jp-analysis): pilot 30 JP run log + qualitative review"
```

---

## Self-Review

**1. Spec coverage:**
- §2 D1–D8 decisions → Tasks 1–12. D2 verbatim prompt → Task 8 (verbatim transcription, no gate in analyzer Task 10). D3 guided decoding → Task 10 (`extra_body.guided_json`). D6 gemma → Task 12. D7 pilot 30 → Task 13. ✓
- §3.2 themes validation (finding #1) → Task 5. §3.3 record metadata/status → Task 10 `_terminal`/`rec`. ✓
- §4 live budget (finding #4) → Task 6 + Task 11 `verify_max_model_len`/`compute_threshold`. ✓
- §5 taxonomy → Task 1. §6 routing/assembly → Tasks 8–9. ✓
- §7 file structure → all tasks; §8 flow incl. client-side concurrency (`--concurrency`, bounded thread pool, coordinator-thread result handling) → Tasks 10–11; §9 terminal/retryable + circuit breaker + atomic ledger (findings #2,#3) → Tasks 4, 7, 10, 11; §10 pilot parallelization benchmark → Task 13 Step 3b. ✓
- §11 tests → every task is TDD. §12 R1–R4 → Task 13 steps. ✓

**2. Placeholder scan:** The only intentional `<<< transcribe ... >>>` markers are in Task 8, with an explicit verbatim-source pointer and structural-invariant tests — this is a copy-from-source instruction, not undefined behavior. All logic modules have complete code.

**3. Type consistency:** `RunConfig`, `analyze_record`, `CircuitBreaker`, `RetryableError` consistent across Tasks 10–11. `canonicalize_themes` returns `(clean, valid, anomalies)` used identically in Task 5 tests and Task 10. `derive_done_ids`/`atomic_write_shard`/`append_jsonl`/`load_quarantine` signatures consistent Tasks 7↔11. `route()` returns `(preambule, variant)` consistent Tasks 9↔10. `build_system_prompt()` returns `(system, variant)` consistent Tasks 9↔10↔11.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-19-jp-analysis-step1.md`.
