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

# normalized canonical lookup: normed (branche|sous) -> exact canonical pair.
# PAIRS is a frozenset (unordered): a FUTURE taxonomy edit that introduces a
# normalization collision (two raw pairs normalizing to the same key) would
# resolve order-dependently here — surface it explicitly if PAIRS ever grows.
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
