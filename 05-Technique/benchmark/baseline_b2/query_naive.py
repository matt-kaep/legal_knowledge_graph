#!/usr/bin/env python3
"""Pipeline de requête naïf pour la baseline B2.

Pour une question donnée :
  1. Embed "query: <question>" avec multilingual-e5-base
  2. Cosine similarity = dot product (embeddings L2-normalisés)
  3. Top-K JP par similarité
  4. Agrégation des articles : comptage de fréquence parmi les K JP voisines
  5. Renvoie parsed_canon compatible eval_rubric.evaluate

Interface :
  from query_naive import NaiveRetriever
  r = NaiveRetriever()
  canon = r.query(question_text, k=5, min_freq=1)
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from scipy.sparse import csr_matrix

HERE = Path(__file__).parent.resolve()
GRAPH_NPZ = HERE.parent / "graphs_v5" / "graph_bipartite.npz"
EMB_FILE  = HERE / "jp_embeddings.npy"
IDS_FILE  = HERE / "jp_order.npy"
PARQUET   = HERE / "jp_index.parquet"

PREFIX_Q  = "query: "


class NaiveRetriever:
    """Charge les ressources en mémoire une seule fois, réutilisable pour N requêtes."""

    def __init__(self) -> None:
        print("Chargement des embeddings…", flush=True)
        self._emb: np.ndarray = np.load(EMB_FILE)
        self._emb_ids: np.ndarray = np.load(IDS_FILE, allow_pickle=True)

        print("Chargement du graphe…", flush=True)
        data = np.load(GRAPH_NPZ, allow_pickle=True)
        jp_ids_graph = data["jp_ids"]
        article_ids  = data["article_ids"]
        mat = csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"]),
        )

        # Index rapide jp_id → position dans chaque structure
        emb_id2pos   = {uid: i for i, uid in enumerate(self._emb_ids)}
        graph_id2pos = {uid: i for i, uid in enumerate(jp_ids_graph)}

        # Intersection : JP présentes dans le graphe ET ayant un embedding
        common_ids = [uid for uid in jp_ids_graph if uid in emb_id2pos]
        print(f"JP utilisables (graphe ∩ embeddings) : {len(common_ids)}", flush=True)

        graph_rows = [graph_id2pos[uid] for uid in common_ids]
        emb_rows   = [emb_id2pos[uid]   for uid in common_ids]

        self._mat_sub  = mat[graph_rows, :]                        # (N_common × N_article)
        self._sub_emb  = self._emb[emb_rows, :]                    # (N_common × 768)
        self._sub_ids  = np.array(common_ids)                      # (N_common,)
        self._article_ids = article_ids

        # Mapping jp_id → number (pourvoi) pour le scoring
        df = pq.read_table(PARQUET, columns=["id", "number"]).to_pandas()
        self._jp_id2pourvoi: dict[str, str] = dict(zip(df["id"], df["number"]))

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("intfloat/multilingual-e5-base")
        print("✓ NaiveRetriever prêt", flush=True)

    def query(self, question: str, k: int = 5, min_freq: int = 1) -> dict:
        """Renvoie parsed_canon compatible eval_rubric.evaluate.

        min_freq : article retenu si cité par au moins min_freq des K JP voisines.
        """
        q_vec = self._model.encode(
            [PREFIX_Q + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        scores = self._sub_emb @ q_vec
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        jp_ids_topk = self._sub_ids[top_k_idx]
        jp_scores   = scores[top_k_idx]

        article_counts = np.zeros(self._article_ids.shape[0], dtype=np.int32)
        for idx in top_k_idx:
            row = self._mat_sub.getrow(idx)
            article_counts[row.indices] += 1

        retained_idx = np.where(article_counts >= min_freq)[0]
        retained_art = self._article_ids[retained_idx]

        articles = [{"pair_key": pk} for pk in retained_art]
        jurisprudences = [
            {"pourvoi": self._jp_id2pourvoi.get(uid, ""), "score": float(s)}
            for uid, s in zip(jp_ids_topk, jp_scores)
            if self._jp_id2pourvoi.get(uid, "")
        ]

        return {
            "articles":       articles,
            "jurisprudences": jurisprudences,
            "arguments":      [],
            "_meta": {
                "k":          k,
                "min_freq":   min_freq,
                "n_articles": len(articles),
                "n_jp":       len(jurisprudences),
            },
        }
