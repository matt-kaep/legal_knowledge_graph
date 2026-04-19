"""
Loader pour Les-Audits-Affaires (LegMLAI, juin 2025).

Dataset : legmlai/les-audits-affaires
Taille : 2670 cas, 9 codes juridiques FR
Licence : cf. HF (à vérifier — usage académique a priori OK)

Requiert `huggingface-cli login` pour la première requête.

Usage :
    from loaders.les_audits import load_les_audits, describe
    df = load_les_audits()
    describe(df)
"""

from __future__ import annotations

import pandas as pd
from rich.console import Console
from rich.table import Table


DATASET_URL = (
    "hf://datasets/legmlai/les-audits-affaires/data/train-00000-of-00001.parquet"
)


def load_les_audits() -> pd.DataFrame:
    """Charge le dataset Les-Audits-Affaires en DataFrame pandas."""
    return pd.read_parquet(DATASET_URL)


def describe(df: pd.DataFrame) -> None:
    """Affiche un panorama du dataset."""
    console = Console()
    console.rule("[bold cyan]Les-Audits-Affaires — aperçu[/]")

    console.print(f"[bold]Lignes :[/] {len(df):,}")
    console.print(f"[bold]Colonnes :[/] {list(df.columns)}")
    console.print()

    # Si 'category' ou colonne de domaine présente, on distribue
    category_col = None
    for candidate in ("category", "domain", "code", "law_code", "juridical_domain"):
        if candidate in df.columns:
            category_col = candidate
            break

    if category_col:
        counts = df[category_col].value_counts()
        table = Table(title=f"Distribution par {category_col}")
        table.add_column(category_col)
        table.add_column("Nb cas", justify="right")
        for val, cnt in counts.items():
            table.add_row(str(val), str(cnt))
        console.print(table)

    # Premier exemple
    console.rule("[bold cyan]Premier cas (exemple)[/]")
    first = df.iloc[0].to_dict()
    for k, v in first.items():
        v_str = str(v)
        if len(v_str) > 400:
            v_str = v_str[:400] + " [...tronqué...]"
        console.print(f"[bold]{k} :[/] {v_str}")


if __name__ == "__main__":
    df = load_les_audits()
    describe(df)
