"""
Loader pour BSARD (Belgian Statutory Article Retrieval Dataset, Louis & Spanakis 2022).

Dataset HF : maastrichtlawtech/bsard
Taille : 22 600+ articles, ~1 100 questions annotées par 6 juristes
Langue : français (belge)
Licence : CC BY-NC-SA 4.0

Deux manières d'accéder au dataset :
1. Via HF datasets (officiel, plus riche) — requiert `datasets` et parfois login
2. Via l'API datasets-server HTTP (rapide, sans login, limité à 100 lignes par appel)

Ce loader supporte les deux.

Usage :
    from loaders.bsard import load_corpus_rows, load_questions_rows
    articles = load_corpus_rows(n=100)
    questions = load_questions_rows(split="test", n=100)
"""

from __future__ import annotations

from typing import Literal

import requests
import pandas as pd
from rich.console import Console


DATASET = "maastrichtlawtech/bsard"
DATASETS_SERVER = "https://datasets-server.huggingface.co/rows"


def _fetch_rows(
    config: str,
    split: str,
    offset: int = 0,
    length: int = 100,
) -> list[dict]:
    """Appel direct à l'API datasets-server HF."""
    params = {
        "dataset": DATASET,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    }
    resp = requests.get(DATASETS_SERVER, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return [row["row"] for row in payload.get("rows", [])]


def load_corpus_rows(n: int = 100, offset: int = 0) -> pd.DataFrame:
    """Charge `n` articles du corpus BSARD (via datasets-server)."""
    rows = _fetch_rows("corpus", "corpus", offset=offset, length=n)
    return pd.DataFrame(rows)


def load_questions_rows(
    split: Literal["train", "test"] = "test",
    n: int = 100,
    offset: int = 0,
) -> pd.DataFrame:
    """Charge `n` questions (train ou test) du dataset BSARD."""
    # Le dataset BSARD utilise la config 'questions' pour les questions
    rows = _fetch_rows("questions", split, offset=offset, length=n)
    return pd.DataFrame(rows)


def describe(corpus_df: pd.DataFrame, questions_df: pd.DataFrame) -> None:
    """Affiche un panorama des deux tables."""
    console = Console()
    console.rule("[bold cyan]BSARD — corpus d'articles[/]")
    console.print(f"[bold]Lignes chargées :[/] {len(corpus_df):,}")
    console.print(f"[bold]Colonnes :[/] {list(corpus_df.columns)}")
    if len(corpus_df) > 0:
        first = corpus_df.iloc[0].to_dict()
        console.rule("Premier article")
        for k, v in first.items():
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + " [...tronqué...]"
            console.print(f"[bold]{k} :[/] {v_str}")

    console.rule("[bold cyan]BSARD — questions[/]")
    console.print(f"[bold]Lignes chargées :[/] {len(questions_df):,}")
    console.print(f"[bold]Colonnes :[/] {list(questions_df.columns)}")
    if len(questions_df) > 0:
        first = questions_df.iloc[0].to_dict()
        console.rule("Première question")
        for k, v in first.items():
            v_str = str(v)
            if len(v_str) > 300:
                v_str = v_str[:300] + " [...tronqué...]"
            console.print(f"[bold]{k} :[/] {v_str}")


if __name__ == "__main__":
    corpus = load_corpus_rows(n=100)
    try:
        questions = load_questions_rows(split="test", n=100)
    except requests.HTTPError as e:
        questions = pd.DataFrame()
        print(f"[warn] Questions split inaccessible : {e}")
    describe(corpus, questions)
