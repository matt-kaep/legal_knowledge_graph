"""Résolution pair_key → texte LEGI + diagnostic de couverture.

Stratégie : pour chaque pair_key, tente chaque candidat de `legi_num_candidates`
contre la SQLite, premier match gagnant. Renvoie un dict structuré exploitable
pour produire articles_penal.parquet et articles_coverage.json.
"""
from __future__ import annotations
import sqlite3
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypedDict
from .normalize import parse_pair_key, legi_num_candidates


class ResolveEntry(TypedDict):
    code_slug: str
    code_titre: str
    num_compact: str
    matched_num: str | None
    texte: str | None


def resolve_pair_keys(
    legi_db: Path,
    pair_keys: Iterable[str],
    code_map: Mapping[str, str],
) -> dict[str, ResolveEntry]:
    """Renvoie {pair_key: ResolveEntry} pour chaque pair_key fourni."""
    out: dict[str, ResolveEntry] = {}
    sql = """
    SELECT a.num, a.bloc_textuel
    FROM articles a
    JOIN sommaires s ON s.element = a.id
    JOIN textes_versions tv ON tv.id = s.cid_parent
    WHERE tv.titre_court = ? AND a.num = ? AND a.etat = 'VIGUEUR'
    LIMIT 1
    """
    with sqlite3.connect(legi_db) as cx:
        for pk in pair_keys:
            slug, compact = parse_pair_key(pk)
            titre = code_map.get(slug)
            entry: ResolveEntry = {
                "code_slug":   slug,
                "code_titre":  titre or "",
                "num_compact": compact,
                "matched_num": None,
                "texte":       None,
            }
            if titre is None:
                out[pk] = entry
                continue
            for cand in legi_num_candidates(compact):
                row = cx.execute(sql, (titre, cand)).fetchone()
                if row is not None:
                    entry["matched_num"] = row[0]
                    entry["texte"] = row[1]
                    break
            out[pk] = entry
    return out


def coverage_report(
    resolved: dict[str, ResolveEntry],
    gold_pair_keys: set[str],
) -> dict:
    """Compte global et sur le sous-ensemble gold."""
    n_total = len(resolved)
    n_resolved = sum(1 for e in resolved.values() if e["texte"] is not None)
    gold_in_set = gold_pair_keys & set(resolved.keys())
    n_gold_total = len(gold_in_set)
    n_gold_resolved = sum(1 for pk in gold_in_set if resolved[pk]["texte"] is not None)
    return {
        "n_total":              n_total,
        "n_resolved":           n_resolved,
        "resolution_rate":      n_resolved / max(n_total, 1),
        "n_gold_total":         n_gold_total,
        "n_gold_resolved":      n_gold_resolved,
        "gold_resolution_rate": n_gold_resolved / max(n_gold_total, 1),
        "missed_examples":      [pk for pk, e in resolved.items() if e["texte"] is None][:20],
    }
