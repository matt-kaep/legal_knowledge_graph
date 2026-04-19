"""
Runner d'affichage des 5 cas gold M6.

Usage :
    cd 05-Technique/benchmark
    python -m m6_mvp.display_mvp
    python -m m6_mvp.display_mvp --id M6-SYNTH-001    # un cas précis
    python -m m6_mvp.display_mvp --summary             # une ligne par cas
"""

import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from m6_mvp.gold_cases import get_all_cases
from schema import M6TestCase, test_case_summary_line


console = Console()


def print_summary_table() -> None:
    """Tableau une ligne par cas."""
    table = Table(title="M6 MVP — 5 cas gold standard")
    table.add_column("ID", style="cyan")
    table.add_column("Spécialisation")
    table.add_column("Difficulté")
    table.add_column("Juridiction")
    table.add_column("Sens")
    table.add_column("Favorable", justify="center")
    table.add_column("Relevance", justify="right")
    for c in get_all_cases():
        fav = "✅" if c.gold.is_favorable else "❌"
        table.add_row(
            c.id,
            c.specialisation.value,
            c.difficulty.value,
            c.decision.juridiction.value,
            c.gold.sens_arret.value,
            fav,
            f"{c.gold.relevance:.2f}",
        )
    console.print(table)


def print_case_detail(c: M6TestCase) -> None:
    """Affichage détaillé d'un cas."""
    console.rule(f"[bold cyan]{c.id} — {c.specialisation.value}[/] ({c.difficulty.value})")

    # Contexte du dossier
    console.print(Panel(
        f"[bold]Résumé :[/] {c.case.case_summary}\n"
        f"[bold]Position client :[/] {c.case.client_position.value}\n"
        f"[bold]Faits clés :[/] {', '.join(c.case.key_facts)}\n"
        f"[bold]Question :[/] {c.question}",
        title="📁 Dossier client",
        border_style="blue",
    ))

    # Décision
    structure_text = ""
    if c.decision.structure:
        s = c.decision.structure
        structure_text = (
            f"\n[bold]Faits :[/] {s.faits}\n"
            f"[bold]Moyens :[/] {s.moyens}\n"
            f"[bold]Motifs :[/] {s.motifs}\n"
            f"[bold]Dispositif :[/] {s.dispositif}"
        )
    console.print(Panel(
        f"[bold]ID :[/] {c.decision.id}\n"
        f"[bold]Juridiction :[/] {c.decision.juridiction.value} — {c.decision.chambre or '—'}\n"
        f"[bold]Date :[/] {c.decision.date}\n"
        f"[bold]Numéro :[/] {c.decision.numero or '—'}\n"
        f"[bold]Synthétique :[/] {'oui' if c.decision.is_synthetic else 'non'}"
        + structure_text,
        title="⚖️  Décision fournie",
        border_style="yellow",
    ))

    # Gold annotation
    gold_text = (
        f"[bold]1. camp_in_decision :[/] {c.gold.camp_in_decision}\n"
        f"[bold]2. sens_arret :[/] {c.gold.sens_arret.value}\n"
        f"[bold]3. is_favorable :[/] {c.gold.is_favorable}\n"
        f"[bold]4. dispositif_summary :[/]\n  {c.gold.dispositif_summary}\n"
        f"[bold]5. relevance :[/] {c.gold.relevance:.2f}\n"
        f"[bold]6. principles_extracted :[/]\n"
        + "\n".join(f"  - {p}" for p in c.gold.principles_extracted)
        + f"\n[bold]7. transfer_reasoning :[/]\n  {c.gold.transfer_reasoning}"
    )
    console.print(Panel(gold_text, title="🎯 Gold annotation (7 dimensions)", border_style="green"))

    if c.notes:
        console.print(Panel(c.notes, title="📝 Notes méthodologiques", border_style="magenta"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Affichage des cas gold M6")
    parser.add_argument("--id", help="Afficher un cas spécifique par ID", default=None)
    parser.add_argument("--summary", action="store_true", help="Résumé une ligne par cas")
    args = parser.parse_args(argv)

    if args.summary:
        print_summary_table()
        return 0

    if args.id:
        matches = [c for c in ALL_CASES if c.id == args.id]
        if not matches:
            console.print(f"[red]Aucun cas trouvé avec l'ID {args.id}[/]")
            return 1
        print_case_detail(matches[0])
        return 0

    # Par défaut : tableau résumé + tous les cas détaillés
    print_summary_table()
    for c in get_all_cases():
        print_case_detail(c)
        console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
