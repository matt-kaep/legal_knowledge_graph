"""Wrapper minimal sur la SQLite produite par legi.py.

Schéma legi.py (cf. https://github.com/Legilibre/legi.py) :
  articles(id PRIMARY KEY, section, num, etat, date_debut, date_fin, ...)
  textes_versions(id, titre, ...)
  sommaires + tables structurelles pour relier articles ↔ code

Une seule fonction : `load_code_articles(legi_db, code_titre)` renvoie un DataFrame
[id, num, texte, etat] pour tous les articles d'un code.
"""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd


def load_code_articles(legi_db: Path, code_titre: str) -> pd.DataFrame:
    """Charge tous les articles 'VIGUEUR' du code `code_titre` (ex. 'Code pénal').

    Retour : DataFrame avec colonnes id, num, texte, etat.
    """
    with sqlite3.connect(legi_db) as cx:
        q = """
        SELECT a.id, a.num, a.bloc_textuel AS texte, a.etat
        FROM articles a
        JOIN sommaires s ON s.element = a.id
        JOIN textes_versions tv ON tv.id = s.cid_parent
        WHERE tv.titre_court = ? AND a.etat = 'VIGUEUR'
        """
        df = pd.read_sql_query(q, cx, params=(code_titre,))
    return df


def count_articles(legi_db: Path) -> int:
    with sqlite3.connect(legi_db) as cx:
        return cx.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
