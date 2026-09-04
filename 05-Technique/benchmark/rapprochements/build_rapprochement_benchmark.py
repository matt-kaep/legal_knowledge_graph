"""Benchmark JP→rapprochements (MVP α, non typé).

Input  : database-judilibre-enrichie/Cour de cassation (JSONL, ~553k records)
Output : data/rapprochements/benchmark-rapp-v1.json

Pipeline (3 passes streaming sur un même fichier, pas de graphe persisté) :
  Pass 1 — scan + filtrage : garder tout arrêt avec >=3 rapprochements parsables
           (pourvoi extractible par regex dans le title).
  Pass 2 — index inverse pourvoi_normalise -> id Judilibre (construit à partir du
           champ `numbers`), puis résolution des rapprochements.
  Pass 3 — stratification par chambre, export JSON final.

Filtres retenus (décisions brainstorming 2026-04-20) :
  - Juridiction = Cour de cassation uniquement
  - >=3 rapprochements dont le numéro de pourvoi est parsable
  - Pas de filtre publication (on garde publiés ET non publiés)
  - Stratification par chambre dans l'export (pas de cap)
  - Typage de la relation non implémenté (variante α)

Format du benchmark : cf. bloc `build_question` en bas.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

# ── Chemins ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
INPUT_PATH = ROOT / "database-judilibre-enrichie" / "Cour de cassation"
OUTPUT_DIR = ROOT / "data" / "rapprochements"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "benchmark-rapp-v1.json"
STATS_MD = OUTPUT_DIR / "stats-v1.md"

MIN_PARSABLE_RAPP = 3

# ── Parsers ─────────────────────────────────────────────────────────────────
# Dans les titles on voit :
#   "Crim., 18 janvier 2011, pourvoi n° 10-87.525, Bull. crim. 2011, n° 7 (cassation)."
#   "Ass. plén., 22 décembre 2023, pourvoi n° <a href=\"65855660673fa80008f8d98d\">20-20.648</a>, Bull. (...)."
# Certains très vieux titles n'ont pas de pourvoi lisible (ex. "Chambre sociale,
# 1956-06-29, Bulletin 1956, IV, n° 604, p. 451") — on les exclut.
POURVOI_RE = re.compile(r"pourvoi\s*n°?\s*(?:<a[^>]*>)?(\d{2}[\-\.]\d{2}[\-\.]?\d+)", re.IGNORECASE)
HREF_RE = re.compile(r'<a\s+href="([a-f0-9]{24})"', re.IGNORECASE)
CHAMBER_PREFIX_RE = re.compile(
    r"^\s*((?:Ass\.?\s*pl[eé]n\.?|Ch\.?\s*mixte|Civ\.?\s*\d+\s*re?|1re\s*Civ\.?|2e\s*Civ\.?|3e\s*Civ\.?|"
    r"Civ\.?\s*\d+e|Com\.?|Soc\.?|Crim\.?|Ch\.?\s*commerciale|Ch\.?\s*sociale|Ch\.?\s*criminelle|"
    r"Chambre\s+\w+|Premi[eè]re\s+chambre\s+\w+|Deuxi[eè]me\s+chambre\s+\w+|Troisi[eè]me\s+chambre\s+\w+))",
    re.IGNORECASE,
)


def normalize_pourvoi(raw: str) -> str:
    """`10-87.525` / `10.87.525` / `10-87-525` -> `10-87525` (clé de matching)."""
    if not raw:
        return ""
    s = raw.strip().replace(".", "").replace(" ", "")
    # normaliser en un seul tiret après les 2 premiers chiffres
    m = re.match(r"^(\d{2})[\-]?(\d+)$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    return s


def parse_chamber_from_title(title: str) -> str | None:
    m = CHAMBER_PREFIX_RE.match(title or "")
    return m.group(1).strip().rstrip(".,") if m else None


def parse_rapprochement(title: str) -> dict | None:
    """Retourne dict parsable {pourvoi_norm, pourvoi_raw, chamber, href_id, raw_title}
    ou None si pas de pourvoi extractible."""
    if not title:
        return None
    m = POURVOI_RE.search(title)
    if not m:
        return None
    pourvoi_raw = m.group(1)
    pourvoi_norm = normalize_pourvoi(pourvoi_raw)
    href_m = HREF_RE.search(title)
    return {
        "pourvoi_raw": pourvoi_raw,
        "pourvoi_norm": pourvoi_norm,
        "chamber_hint": parse_chamber_from_title(title),
        "href_id": href_m.group(1) if href_m else None,
        "raw_title": title,
    }


# ── Canonicalisation chambre ────────────────────────────────────────────────
CHAMBER_CANON_PATTERNS = [
    (re.compile(r"premi[eè]re\s+chambre\s+civile", re.I), "Civ. 1re"),
    (re.compile(r"deuxi[eè]me\s+chambre\s+civile", re.I), "Civ. 2e"),
    (re.compile(r"troisi[eè]me\s+chambre\s+civile", re.I), "Civ. 3e"),
    (re.compile(r"chambre\s+commerciale", re.I), "Com."),
    (re.compile(r"chambre\s+sociale", re.I), "Soc."),
    (re.compile(r"chambre\s+criminelle", re.I), "Crim."),
    (re.compile(r"chambres?\s+r[eé]unies", re.I), "Ass. plén."),
    (re.compile(r"assembl[eé]e\s+pl[eé]ni[eè]re", re.I), "Ass. plén."),
    (re.compile(r"chambre\s+mixte", re.I), "Ch. mixte"),
]


def canon_chamber(raw: str | None) -> str:
    if not raw:
        return "(inconnue)"
    for pat, canon in CHAMBER_CANON_PATTERNS:
        if pat.search(raw):
            return canon
    return raw


# ── Passes ──────────────────────────────────────────────────────────────────

def pass1_filter(input_path: Path) -> list[dict]:
    """Retourne la liste des candidats (dict minimal) qui passent le filtre MIN_PARSABLE_RAPP."""
    candidates: list[dict] = []
    stats = {
        "total": 0,
        "with_any_rapp": 0,
        "with_enough_parsable": 0,
        "rapp_total_seen": 0,
        "rapp_parsable_seen": 0,
        "rapp_with_href_id": 0,
    }
    t0 = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            stats["total"] += 1
            rapps = rec.get("rapprochements") or []
            if not rapps:
                continue
            stats["with_any_rapp"] += 1
            parsed_rapps = []
            for r in rapps:
                stats["rapp_total_seen"] += 1
                p = parse_rapprochement(r.get("title"))
                if p is not None:
                    stats["rapp_parsable_seen"] += 1
                    if p["href_id"]:
                        stats["rapp_with_href_id"] += 1
                    parsed_rapps.append(p)
            if len(parsed_rapps) < MIN_PARSABLE_RAPP:
                continue
            stats["with_enough_parsable"] += 1
            candidates.append({
                "id": rec.get("id"),
                "ecli": rec.get("ecli"),
                "numbers": rec.get("numbers") or [],
                "chamber": rec.get("chamber"),
                "decision_date": rec.get("decision_date"),
                "solution": rec.get("solution"),
                "publication": rec.get("publication"),
                "particularInterest": rec.get("particularInterest"),
                "summary": rec.get("summary"),
                "codes": rec.get("codes") or [],
                "code_article_pairs": rec.get("code_article_pairs") or [],
                "text": rec.get("text"),
                "parsed_rapprochements": parsed_rapps,
                "n_rapp_brut": len(rapps),
            })
            if stats["total"] % 100_000 == 0:
                print(f"  pass1 — {stats['total']:,} records, {stats['with_enough_parsable']:,} candidats "
                      f"({time.time()-t0:.1f}s)")
    print(f"\n[pass1] {stats['total']:,} records ; "
          f"{stats['with_any_rapp']:,} avec rapp ; "
          f"{stats['with_enough_parsable']:,} candidats.")
    print(f"[pass1] rapprochements totaux vus {stats['rapp_total_seen']:,}, "
          f"parsables {stats['rapp_parsable_seen']:,} "
          f"({stats['rapp_parsable_seen']/max(1,stats['rapp_total_seen'])*100:.1f}%), "
          f"avec href_id {stats['rapp_with_href_id']:,}")
    return candidates, stats


def pass2_build_index(input_path: Path) -> dict[str, dict]:
    """Re-scan pour construire l'index pourvoi_normalisé -> {id, ecli, date, chamber}."""
    index: dict[str, dict] = {}
    collisions = 0
    total = 0
    t0 = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            for num in (rec.get("numbers") or []):
                key = normalize_pourvoi(num)
                if not key:
                    continue
                if key in index:
                    collisions += 1
                    continue  # premier gagnant
                index[key] = {
                    "id": rec.get("id"),
                    "ecli": rec.get("ecli"),
                    "decision_date": rec.get("decision_date"),
                    "chamber": rec.get("chamber"),
                    "solution": rec.get("solution"),
                }
            if total % 100_000 == 0:
                print(f"  pass2 — indexé {total:,} ({time.time()-t0:.1f}s)")
    print(f"[pass2] index construit : {len(index):,} pourvois uniques, {collisions:,} collisions ignorées.")
    return index


def pass3_stratify_and_export(candidates: list[dict],
                              index: dict[str, dict],
                              global_stats: dict) -> dict:
    """Résout chaque rapp via l'index, regroupe par chambre, exporte le JSON."""
    questions = []
    by_chamber: dict[str, int] = Counter()
    resolved_counter = Counter()
    for i, c in enumerate(candidates, 1):
        gt = []
        for p in c["parsed_rapprochements"]:
            resolved = index.get(p["pourvoi_norm"])
            if p["href_id"]:
                resolved_counter["href"] += 1
                resolved_id = p["href_id"]
                if resolved and not resolved.get("id"):
                    resolved["id"] = resolved_id
                elif not resolved:
                    resolved = {"id": resolved_id}
            if resolved:
                resolved_counter["matched" if not p["href_id"] else "matched_and_href"] += 1
            else:
                resolved_counter["unmatched"] += 1
            gt.append({
                "pourvoi": p["pourvoi_norm"],
                "pourvoi_raw": p["pourvoi_raw"],
                "chamber_hint": p["chamber_hint"],
                "raw_title": p["raw_title"],
                "resolved": resolved,
            })
        chamber_canon = canon_chamber(c["chamber"])
        by_chamber[chamber_canon] += 1
        questions.append({
            "id": f"rapp-Q{i:05d}",
            "decision": {
                "id": c["id"],
                "ecli": c["ecli"],
                "pourvoi": c["numbers"][0] if c["numbers"] else None,
                "pourvois_all": c["numbers"],
                "chamber_raw": c["chamber"],
                "chamber": chamber_canon,
                "decision_date": c["decision_date"],
                "solution": c["solution"],
                "publication": c["publication"],
                "particularInterest": c["particularInterest"],
                "summary": c["summary"],
                "codes": c["codes"],
                "code_article_pairs": c["code_article_pairs"],
                "text": c["text"],
            },
            "ground_truth": {
                "rapprochements": gt,
                "n_parsable": len(gt),
                "n_total_brut": c["n_rapp_brut"],
            },
        })

    # n_total_brut <- depuis candidates (déjà stocké)
    payload = {
        "version": "v1-2026-04-20",
        "variant": "α — non typé",
        "source": "database-judilibre-enrichie/Cour de cassation",
        "filters": {
            "min_parsable_rapp": MIN_PARSABLE_RAPP,
            "publication_filter": None,
            "chamber_filter": None,
        },
        "stats": {
            **global_stats,
            "questions_total": len(questions),
            "by_chamber": dict(by_chamber),
            "resolution": dict(resolved_counter),
        },
        "questions": questions,
    }
    return payload


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    print(f"Input  : {INPUT_PATH}")
    print(f"Output : {OUTPUT_JSON}")
    assert INPUT_PATH.exists(), f"Fichier introuvable : {INPUT_PATH}"

    print("\n── Pass 1 : scan + filtrage ──────────────────────────────────────")
    candidates, stats = pass1_filter(INPUT_PATH)

    print("\n── Pass 2 : index inverse pourvoi -> id ─────────────────────────")
    index = pass2_build_index(INPUT_PATH)

    print("\n── Pass 3 : résolution + stratification + export ────────────────")
    payload = pass3_stratify_and_export(candidates, index, stats)

    print(f"\nQuestions produites : {len(payload['questions']):,}")
    print(f"Par chambre :")
    for ch, n in sorted(payload["stats"]["by_chamber"].items(), key=lambda x: -x[1]):
        print(f"  {ch:30s} {n:>6}")
    print(f"Résolution rapprochements : {payload['stats']['resolution']}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    size_mo = OUTPUT_JSON.stat().st_size / 1e6
    print(f"\n✅ Export : {OUTPUT_JSON} ({size_mo:.1f} Mo)")

    # stats.md
    lines = [
        "# Benchmark rapprochements v1 — stats",
        "",
        f"- Source : `{payload['source']}`",
        f"- Filtre : `>= {MIN_PARSABLE_RAPP}` rapprochements parsables, toutes chambres",
        f"- Questions : **{len(payload['questions']):,}**",
        "",
        "## Par chambre",
        "",
        "| Chambre | N questions |",
        "|---|---:|",
    ]
    for ch, n in sorted(payload["stats"]["by_chamber"].items(), key=lambda x: -x[1]):
        lines.append(f"| {ch} | {n} |")
    lines += ["", "## Pipeline", "", "```", json.dumps(payload["stats"], indent=2, ensure_ascii=False), "```"]
    STATS_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Stats  : {STATS_MD}")


if __name__ == "__main__":
    main()
