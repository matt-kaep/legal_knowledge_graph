"""Artefacts d'alignement embedding-rows ↔ nœuds graphe.

Convention :
  emb_articles.npy[i] est l'embedding du nœud article de colonne
  pairkey_to_graphcol[i] dans le graphe ;
  articles_order[i] = article_ids[pairkey_to_graphcol[i]].

Idem pour les JP : jp_to_graphrow / jp_order.
"""
from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import pandas as pd


def build_articles_linkage(
    article_ids: np.ndarray,
    article_codes: np.ndarray,
    resolved_pair_keys: Iterable[str],
    penal_codes: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Filtre les colonnes du graphe : (a) codes pénaux, (b) pair_keys résolus.
    Préserve l'ordre original de `article_ids`.

    Retour : (articles_order, pairkey_to_graphcol).
    """
    penal_set = set(penal_codes)
    resolved = set(resolved_pair_keys)
    cols_keep = [i for i, (pk, c) in enumerate(zip(article_ids, article_codes))
                 if c in penal_set and pk in resolved]
    p2col = np.array(cols_keep, dtype=np.int32)
    order = article_ids[p2col].astype(object)
    return order, p2col


def build_jp_linkage(
    jp_ids: np.ndarray,
    jp_index_df: pd.DataFrame,
    text_col: str = "text",
    juris_filter: str | None = "CC",
) -> tuple[np.ndarray, np.ndarray]:
    """Filtre les lignes JP : garde celles avec `text_col` non-vide et,
    optionnellement, dont `juris == juris_filter`. Préserve l'ordre original
    de `jp_ids`.

    Le bundle pénal préparé le 5 mai ne contient que `text` (pas `summary`).
    On filtre sur juris=CC pour rester dans l'esprit du design Johnny
    (~95 % de couverture summary = les CC).

    Retour : (jp_order, jp_to_graphrow).
    """
    keepers: set = set()
    for row in jp_index_df.itertuples():
        if juris_filter is not None and getattr(row, "juris", None) != juris_filter:
            continue
        txt = getattr(row, text_col, None)
        if isinstance(txt, str) and txt.strip():
            keepers.add(row.id)
    rows_keep = [i for i, jpid in enumerate(jp_ids) if jpid in keepers]
    j2row = np.array(rows_keep, dtype=np.int32)
    order = jp_ids[j2row].astype(object)
    return order, j2row
