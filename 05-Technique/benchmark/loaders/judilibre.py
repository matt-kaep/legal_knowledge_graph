"""
Loader pour les fichiers JSONL Judilibre (données locales).

Deux versions disponibles :
- database-judilibre/         : 34 champs (texte, zones, visa, rapprochements…)
- database-judilibre-enrichie/ : +3 champs (articles, codes, code_article_pairs)

Volume total : ~1,13 M décisions (CC 553k, CA 431k, TJ 142k)
Format : JSONL (une ligne JSON par décision)

IMPORTANT : fichiers de 1.7 à 7.7 GB — le chargement est en streaming.

Usage :
    from loaders.judilibre import iter_decisions, load_decisions, search_decisions

    # Itérer en streaming (mémoire constante)
    for dec in iter_decisions("CC", enriched=True, max_rows=100):
        print(dec["number"], dec["solution"])

    # Charger N décisions en DataFrame
    df = load_decisions("CC", enriched=True, n=500)

    # Chercher des décisions par mots-clés dans le texte
    results = search_decisions("CC", keywords=["légitime défense"], max_results=10)
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, Literal

import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


# ── Chemins ─────────────────────────────────────────────────────────────────

_BASE_DIR = Path(__file__).resolve().parent.parent
_JUDILIBRE_DIR = _BASE_DIR / "database-judilibre"
_JUDILIBRE_ENRICHED_DIR = _BASE_DIR / "database-judilibre-enrichie"

JuridictionKey = Literal["CC", "CA", "TJ"]

_FILENAMES: dict[JuridictionKey, str] = {
    "CC": "Cour de cassation",
    "CA": "Cours d'appel",
    "TJ": "Tribunal judiciaire",
}


def _resolve_path(juridiction: JuridictionKey, enriched: bool = True) -> Path:
    """Résout le chemin vers le fichier JSONL."""
    base = _JUDILIBRE_ENRICHED_DIR if enriched else _JUDILIBRE_DIR
    path = base / _FILENAMES[juridiction]
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            f"Vérifiez que les données Judilibre sont dans {base}"
        )
    return path


# ── Extraction des zones ────────────────────────────────────────────────────


def extract_zones(record: dict) -> dict[str, str | None]:
    """Extrait les sections structurées d'une décision via les offsets `zones`.

    Retourne un dict avec les clés :
        introduction, expose (faits), moyens, motivations (motifs),
        dispositif, annexes
    Chaque valeur est le texte extrait ou None si absent.
    """
    text = record.get("text", "")
    zones = record.get("zones")
    if not zones or not isinstance(zones, dict):
        return {
            "introduction": None,
            "expose": None,
            "moyens": None,
            "motivations": None,
            "dispositif": None,
            "annexes": None,
        }
    result = {}
    for zone_name in ("introduction", "expose", "moyens", "motivations", "dispositif", "annexes"):
        ranges = zones.get(zone_name)
        if ranges and isinstance(ranges, list) and len(ranges) > 0:
            # Concatène si plusieurs plages (rare mais possible)
            parts = []
            for r in ranges:
                start = r.get("start", 0)
                end = r.get("end", 0)
                parts.append(text[start:end])
            result[zone_name] = "\n".join(parts).strip()
        else:
            result[zone_name] = None
    return result


# ── Mapping vers le schéma Pydantic ─────────────────────────────────────────

# Import lazy pour éviter les dépendances circulaires
_JURIDICTION_MAP = {
    "Cour de cassation": "CC",
    "Cour d'appel": "CA",
    "Tribunal judiciaire": "TJ",
    "Tribunal de commerce": "TC",
    "Conseil de prud'hommes": "CPH",
    "Tribunal administratif": "TA",
    "Cour administrative d'appel": "CAA",
    "Conseil d'État": "CE",
}


def to_decision(record: dict) -> "Decision":
    """Convertit un enregistrement JSONL en objet Decision (schema.py)."""
    from schema import Decision, DecisionStructure, Juridiction

    jur_str = _JURIDICTION_MAP.get(record.get("jurisdiction", ""), "AUTRE")
    zones = extract_zones(record)

    structure = None
    if any(zones.get(k) for k in ("expose", "moyens", "motivations", "dispositif")):
        structure = DecisionStructure(
            faits=zones.get("expose"),
            moyens=zones.get("moyens"),
            motifs=zones.get("motivations"),
            dispositif=zones.get("dispositif"),
        )

    return Decision(
        id=record.get("id", ""),
        juridiction=Juridiction(jur_str),
        chambre=record.get("chamber"),
        date=date.fromisoformat(record["decision_date"]),
        numero=record.get("number"),
        full_text=record.get("text", ""),
        structure=structure,
        is_synthetic=False,
    )


# ── Itération streaming ────────────────────────────────────────────────────


def iter_decisions(
    juridiction: JuridictionKey,
    *,
    enriched: bool = True,
    max_rows: int | None = None,
    filter_fn: Callable[[dict], bool] | None = None,
    date_min: str | None = None,
    date_max: str | None = None,
    solution: str | None = None,
    chamber: str | None = None,
) -> Iterator[dict]:
    """Itère en streaming sur les décisions d'un fichier JSONL.

    Parameters
    ----------
    juridiction : "CC", "CA" ou "TJ"
    enriched : utiliser la version enrichie (avec articles/codes)
    max_rows : nombre max de résultats (None = tout)
    filter_fn : fonction de filtre personnalisée
    date_min, date_max : filtrage par date (format "YYYY-MM-DD")
    solution : filtre sur le champ solution (ex: "Cassation")
    chamber : filtre sur la chambre (match partiel, case-insensitive)
    """
    path = _resolve_path(juridiction, enriched)
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and count >= max_rows:
                return
            record = json.loads(line)
            # Filtres rapides
            if date_min and record.get("decision_date", "") < date_min:
                continue
            if date_max and record.get("decision_date", "") > date_max:
                continue
            if solution and solution.lower() not in (record.get("solution") or "").lower():
                continue
            if chamber and chamber.lower() not in (record.get("chamber") or "").lower():
                continue
            if filter_fn and not filter_fn(record):
                continue
            count += 1
            yield record


def load_decisions(
    juridiction: JuridictionKey,
    *,
    enriched: bool = True,
    n: int = 100,
    **kwargs,
) -> pd.DataFrame:
    """Charge N décisions en DataFrame pandas."""
    rows = list(iter_decisions(juridiction, enriched=enriched, max_rows=n, **kwargs))
    return pd.DataFrame(rows)


# ── Recherche par mots-clés ─────────────────────────────────────────────────


def search_decisions(
    juridiction: JuridictionKey,
    *,
    keywords: list[str],
    enriched: bool = True,
    max_results: int = 10,
    date_min: str | None = None,
    date_max: str | None = None,
    chamber: str | None = None,
    case_sensitive: bool = False,
) -> list[dict]:
    """Recherche des décisions contenant tous les mots-clés dans le texte.

    Les mots-clés doivent TOUS être présents (AND logique).
    """
    if not case_sensitive:
        keywords = [k.lower() for k in keywords]

    def matches(record: dict) -> bool:
        text = record.get("text", "")
        if not case_sensitive:
            text = text.lower()
        return all(kw in text for kw in keywords)

    return list(
        iter_decisions(
            juridiction,
            enriched=enriched,
            max_rows=max_results,
            filter_fn=matches,
            date_min=date_min,
            date_max=date_max,
            chamber=chamber,
        )
    )


# ── Description ─────────────────────────────────────────────────────────────


def describe(juridiction: JuridictionKey, enriched: bool = True, sample: int = 200) -> None:
    """Affiche un panorama du fichier Judilibre pour une juridiction."""
    if not _RICH_AVAILABLE:
        raise ImportError("Le module 'rich' est requis pour describe(). Installez-le : pip install rich")
    console = Console()
    label = "enrichie" if enriched else "base"
    console.rule(f"[bold cyan]Judilibre — {_FILENAMES[juridiction]} ({label})[/]")

    path = _resolve_path(juridiction, enriched)
    size_gb = path.stat().st_size / (1024**3)
    console.print(f"[bold]Fichier :[/] {path.name} ({size_gb:.1f} GB)")

    # Stats sur un échantillon
    solutions: dict[str, int] = {}
    chambers: dict[str, int] = {}
    dates: list[str] = []
    has_zones = 0
    total = 0

    for rec in iter_decisions(juridiction, enriched=enriched, max_rows=sample):
        total += 1
        sol = rec.get("solution") or "(vide)"
        solutions[sol] = solutions.get(sol, 0) + 1
        ch = rec.get("chamber") or "(vide)"
        chambers[ch] = chambers.get(ch, 0) + 1
        dates.append(rec.get("decision_date", ""))
        zones = rec.get("zones")
        if zones and isinstance(zones, dict) and any(v for v in zones.values() if v is not None):
            has_zones += 1

    console.print(f"[bold]Échantillon :[/] {total} décisions")
    console.print(f"[bold]Zones structurées :[/] {has_zones}/{total}")
    if dates:
        console.print(f"[bold]Plage dates :[/] {min(dates)} → {max(dates)}")

    # Solutions
    table = Table(title="Distribution des solutions")
    table.add_column("Solution")
    table.add_column("Nb", justify="right")
    for sol, cnt in sorted(solutions.items(), key=lambda x: -x[1])[:15]:
        table.add_row(sol, str(cnt))
    console.print(table)

    # Chambres
    table = Table(title="Distribution des chambres")
    table.add_column("Chambre")
    table.add_column("Nb", justify="right")
    for ch, cnt in sorted(chambers.items(), key=lambda x: -x[1])[:15]:
        table.add_row(ch, str(cnt))
    console.print(table)

    # Premier enregistrement
    console.rule("Premier enregistrement")
    for rec in iter_decisions(juridiction, enriched=enriched, max_rows=1):
        for k, v in rec.items():
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + " [...tronqué...]"
            console.print(f"[bold]{k} :[/] {v_str}")


if __name__ == "__main__":
    describe("CC", enriched=True, sample=200)
