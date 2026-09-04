"""
Export/import des cas M6 gold standard en CSV (séparateur ;, listes avec |).

Format conçu pour être facilement ouvert dans Excel / Google Sheets.

Produit 2 fichiers liés par l'ID du cas :
    - cases.csv   : contexte client + gold annotation (7 dimensions)
    - decisions.csv : décision de justice associée

Usage :
    python export_csv.py                         # exporte dans data/m6/
    python export_csv.py --outdir mon_dossier    # exporte dans mon_dossier/
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from m6_mvp.gold_cases import get_all_cases
from schema import M6TestCase


_SEPARATOR = ";"
_LIST_SEPARATOR = "|"


def _join_list(items: list[str]) -> str:
    """Sérialise une liste en chaîne séparée par |."""
    return _LIST_SEPARATOR.join(items)


def export_cases_csv(cases: list[M6TestCase], outdir: Path) -> Path:
    """Exporte les cas (contexte + gold) en CSV."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "cases.csv"
    fieldnames = [
        "id",
        "specialisation",
        "difficulty",
        "case_summary",
        "client_position",
        "key_facts",
        "question",
        "decision_id",
        "gold_camp_in_decision",
        "gold_sens_arret",
        "gold_is_favorable",
        "gold_dispositif_summary",
        "gold_relevance",
        "gold_principles_extracted",
        "gold_transfer_reasoning",
        "notes",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=_SEPARATOR, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for c in cases:
            writer.writerow({
                "id": c.id,
                "specialisation": c.specialisation.value,
                "difficulty": c.difficulty.value,
                "case_summary": c.case.case_summary,
                "client_position": c.case.client_position.value,
                "key_facts": _join_list(c.case.key_facts),
                "question": c.question,
                "decision_id": c.decision.id,
                "gold_camp_in_decision": c.gold.camp_in_decision,
                "gold_sens_arret": c.gold.sens_arret.value,
                "gold_is_favorable": str(c.gold.is_favorable),
                "gold_dispositif_summary": c.gold.dispositif_summary,
                "gold_relevance": f"{c.gold.relevance:.2f}",
                "gold_principles_extracted": _join_list(c.gold.principles_extracted),
                "gold_transfer_reasoning": c.gold.transfer_reasoning,
                "notes": c.notes or "",
            })
    return path


def export_decisions_csv(cases: list[M6TestCase], outdir: Path) -> Path:
    """Exporte les décisions associées en CSV."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "decisions.csv"
    fieldnames = [
        "case_id",
        "decision_id",
        "juridiction",
        "chambre",
        "date",
        "numero",
        "is_synthetic",
        "full_text",
        "structure_faits",
        "structure_moyens",
        "structure_motifs",
        "structure_dispositif",
    ]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=_SEPARATOR, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for c in cases:
            d = c.decision
            s = d.structure
            writer.writerow({
                "case_id": c.id,
                "decision_id": d.id,
                "juridiction": d.juridiction.value,
                "chambre": d.chambre or "",
                "date": str(d.date),
                "numero": d.numero or "",
                "is_synthetic": str(d.is_synthetic),
                "full_text": d.full_text,
                "structure_faits": s.faits if s else "",
                "structure_moyens": s.moyens if s else "",
                "structure_motifs": s.motifs if s else "",
                "structure_dispositif": s.dispositif if s else "",
            })
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export des cas M6 en CSV")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).parent / "data" / "m6",
        help="Répertoire de sortie (défaut: data/m6/)",
    )
    args = parser.parse_args(argv)

    cases = get_all_cases()
    cases_path = export_cases_csv(cases, args.outdir)
    decisions_path = export_decisions_csv(cases, args.outdir)

    print(f"Exporté {len(cases)} cas :")
    print(f"  {cases_path} ({cases_path.stat().st_size / 1024:.1f} KB)")
    print(f"  {decisions_path} ({decisions_path.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
