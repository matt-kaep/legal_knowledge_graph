"""Conversion pair_key (forme compacte v5) ↔ formats `num` candidats côté LEGI.

Le format pair_key v5 :
  <code_slug>:<article>
  article : préfixe optionnel L|R|D|A|E collé, suffixes latins minuscule,
            lettre finale majuscule (ex. `L743-7`, `1649quinquiesB`, `L122-1bis`).

LEGI peut stocker le num avec :
  - point après la lettre de partie : "L. 743-7"
  - espaces autour des suffixes : "1649 quinquies B"
  - tout collé : "L743-7"
On renvoie tous les candidats raisonnables, le caller essaye dans l'ordre.
"""
from __future__ import annotations
import re

_LATIN_SUFFIXES = ("bis", "ter", "quater", "quinquies", "sexies", "septies",
                   "octies", "novies", "decies", "undecies", "duodecies")
_PART_LETTERS = ("L", "R", "D", "A", "E")

_PAIR_KEY_RE = re.compile(r"^\s*([a-z0-9_]+):([^\s]+)\s*$")


def parse_pair_key(pk: str) -> tuple[str, str]:
    """Parse `<code_slug>:<article_num>`. Lève ValueError si malformé."""
    m = _PAIR_KEY_RE.match(pk)
    if not m:
        raise ValueError(f"pair_key malformé : {pk!r}")
    return m.group(1), m.group(2)


def _split_part_letter(num: str) -> tuple[str | None, str]:
    """`L743-7` → ('L', '743-7'). `222-23` → (None, '222-23')."""
    if num and num[0] in _PART_LETTERS and len(num) > 1 and not num[1].isalpha():
        return num[0], num[1:]
    return None, num


def _split_latin_suffix(rest: str) -> tuple[str, str | None, str | None]:
    """`1649quinquiesB` → ('1649', 'quinquies', 'B'). `122-1bis` → ('122-1', 'bis', None)."""
    for suf in _LATIN_SUFFIXES:
        if suf in rest:
            i = rest.lower().index(suf)
            head = rest[:i]
            after = rest[i + len(suf):]
            tail = after if after else None
            return head, suf, tail
    m = re.match(r"^(.*?)([A-Z])$", rest)
    if m and len(m.group(1)) > 0:
        return m.group(1), None, m.group(2)
    return rest, None, None


def legi_num_candidates(compact: str) -> list[str]:
    """Renvoie une liste ordonnée de variantes plausibles côté LEGI."""
    part, rest = _split_part_letter(compact)
    head, suf, tail = _split_latin_suffix(rest)

    cands: list[str] = []

    def emit(*pieces: str) -> None:
        s = "".join(p for p in pieces if p)
        if s and s not in cands:
            cands.append(s)

    cands.append(compact)

    for part_fmt in ([part, f"{part} ", f"{part}. "] if part else [""]):
        suf_variants = [f"{suf}", f" {suf}", f" {suf} "] if suf else [""]
        tail_variants = [tail, f" {tail}", f"{tail}"] if tail else [""]
        for sv in suf_variants:
            for tv in tail_variants:
                emit(part_fmt, head, sv, tv)

    return cands
