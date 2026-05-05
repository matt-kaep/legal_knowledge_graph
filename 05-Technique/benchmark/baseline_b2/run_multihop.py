#!/usr/bin/env python3
"""A.3 — Multi-hop retrieval [LOCAL].

Pipeline 2-hop : Question → top-K_entry JP (embedding) → leurs articles
              → JP qui citent ≥ M de ces articles → leurs articles → output

Permet de répondre : "le graphe rattrape-t-il les défaillances de l'embedding ?"
Si une JP GT cite des articles communs avec le top-10 entry, elle remonte
au 2-hop même si elle est très loin dans le ranking embedding.

Aussi mesure :
  - n_gt_jp_in_expanded : combien de pourvois GT sont dans l'expansion 2-hop
  - JP score sur le pool {entry + expansion} (pas que entry)

Tourne en local en quelques secondes (utilise embeddings déjà calculés).

Usage (depuis penal_bundle/) :
    python run_multihop.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix


HERE       = Path(__file__).parent.resolve()
PARQUET    = HERE / "jp_index_penal.parquet"
GRAPH_NPZ  = HERE / "graph_penal.npz"
RUBRICS    = HERE / "rubrics_penal.json"
EMB_DIR    = HERE / "embeddings"
RESULTS    = HERE / "Results"
PER_Q_DIR  = RESULTS / "per_question_multihop"
PER_Q_DIR.mkdir(parents=True, exist_ok=True)

EMB_FILE   = EMB_DIR / "jp_embeddings_e5-base.npy"
IDS_FILE   = EMB_DIR / "jp_order_e5-base.npy"
EMB_DIM    = 768
MODEL_NAME = "intfloat/multilingual-e5-base"
PREFIX_Q   = "query: "

CSV_PATH   = RESULTS / "comparison_multihop.csv"
CSV_HEADER = [
    "config", "k_entry", "hop2_M", "hop2_top_n", "max_jp_returned",
    "S_retrieval_mean", "S_bar_art_mean", "S_bar_jp_mean",
    "art_core_mean", "art_expected_mean", "art_expert_mean",
    "jp_core_mean",  "jp_expected_mean",  "jp_expert_mean",
    "n_articles_mean", "n_jp_returned_mean", "n_jp_expanded_mean",
    "n_gt_jp_in_expanded_mean",
]

POURVOI_RE = re.compile(r"\b(\d{2})[\s\-]*(\d{2})[.\-]?(\d{3})\b")

sys.path.insert(0, str(HERE))
from eval_rubric import evaluate


def extract_pourvoi(text: str) -> str | None:
    if not text:
        return None
    m = POURVOI_RE.search(text)
    return f"{m.group(1)}-{m.group(2)}.{m.group(3)}" if m else None


def gt_pourvois(question: dict) -> set[str]:
    """Set des pourvois GT extractibles via la regex (toutes strates)."""
    rubric = question.get("rubric") or {}
    out = set()
    for strate in ("core", "expected", "expert"):
        for item in rubric.get(strate) or []:
            p = extract_pourvoi(item.get("linked_jp", ""))
            if p:
                out.add(p)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Retriever multi-hop
# ═══════════════════════════════════════════════════════════════════════

class MultiHopRetriever:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        print("Chargement graphe…", flush=True)
        g = np.load(GRAPH_NPZ, allow_pickle=True)
        graph_jp_ids = g["jp_ids"]
        article_ids  = g["article_ids"]
        mat = csr_matrix(
            (g["data"], g["indices"], g["indptr"]),
            shape=tuple(g["shape"]),
        )

        print("Chargement embeddings…", flush=True)
        emb_ids = np.load(IDS_FILE, allow_pickle=True)
        n_total = len(emb_ids)
        emb = np.array(np.memmap(EMB_FILE, dtype=np.float32, mode="r",
                                  shape=(n_total, EMB_DIM)))

        emb_id2pos   = {uid: i for i, uid in enumerate(emb_ids)}
        graph_id2pos = {uid: i for i, uid in enumerate(graph_jp_ids)}
        common = [uid for uid in graph_jp_ids if uid in emb_id2pos]
        print(f"JP utilisables : {len(common)}", flush=True)

        graph_rows = [graph_id2pos[uid] for uid in common]
        emb_rows   = [emb_id2pos[uid]   for uid in common]

        self._mat        = mat[graph_rows, :].tocsr()
        self._mat_T      = self._mat.T.tocsr()
        self._sub_emb    = emb[emb_rows, :]
        self._sub_ids    = np.array(common)
        self._article_ids = article_ids

        df = pq.read_table(PARQUET, columns=["id", "number"]).to_pandas()
        self._jp_id2pourvoi = dict(zip(df["id"], df["number"]))

        print(f"Chargement {MODEL_NAME}…", flush=True)
        self._model = SentenceTransformer(MODEL_NAME)
        print("✓ Retriever prêt\n", flush=True)

    def query(self, question: str,
              k_entry: int = 10,
              hop2_M: int | None = None,
              hop2_top_n: int | None = None,
              max_jp_returned: int = 10,
              ) -> tuple[dict, dict]:
        """Renvoie (parsed_canon, debug_info).

        Si hop2_M est None : retrieval 1-hop pur.
        Sinon : 2-hop avec filtre overlap >= hop2_M et cap top-N.

        max_jp_returned : combien de JP au max dans parsed_canon["jurisprudences"]
            (les K entry + les top expandus par overlap).
        """
        # 1) Embed + top-K entry
        q_vec = self._model.encode(
            [PREFIX_Q + question],
            normalize_embeddings=True, convert_to_numpy=True,
        )[0]
        scores = self._sub_emb @ q_vec
        top_k_idx = np.argpartition(scores, -k_entry)[-k_entry:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        entry_jp_idx = top_k_idx

        # 2) Articles 1-hop
        article_counts_entry = np.zeros(len(self._article_ids), dtype=np.int32)
        for idx in entry_jp_idx:
            article_counts_entry[self._mat.getrow(idx).indices] += 1
        entry_articles = np.where(article_counts_entry >= 1)[0]

        if hop2_M is None:
            # Pur 1-hop
            article_idx = entry_articles
            expanded_jp_idx = np.array([], dtype=np.int32)
            jp_overlap = None
        else:
            # 3) 2-hop : JP citant >= M articles entry, hors entry
            jp_overlap = np.zeros(self._mat.shape[0], dtype=np.int32)
            for art_idx in entry_articles:
                jp_overlap[self._mat_T.getrow(art_idx).indices] += 1
            mask = jp_overlap >= hop2_M
            mask[entry_jp_idx] = False
            expanded_jp_idx = np.where(mask)[0]

            # 4) Cap top-N par overlap
            if hop2_top_n is not None and len(expanded_jp_idx) > hop2_top_n:
                top = np.argsort(jp_overlap[expanded_jp_idx])[-hop2_top_n:][::-1]
                expanded_jp_idx = expanded_jp_idx[top]

            # 5) Articles : union de (entry + expansion)
            all_jp_idx = np.concatenate([entry_jp_idx, expanded_jp_idx])
            article_freq = np.zeros(len(self._article_ids), dtype=np.int32)
            for idx in all_jp_idx:
                article_freq[self._mat.getrow(idx).indices] += 1
            article_idx = np.where(article_freq >= 1)[0]

        # 6) JP retournées : entry + top expandus, capés à max_jp_returned
        if hop2_M is None:
            ranked_jp_idx = entry_jp_idx[:max_jp_returned]
        else:
            # Trier expandus par overlap décroissant
            if len(expanded_jp_idx) > 0:
                exp_sorted = expanded_jp_idx[
                    np.argsort(jp_overlap[expanded_jp_idx])[::-1]
                ]
            else:
                exp_sorted = expanded_jp_idx
            # Concat entry + expandus, cap
            combined = np.concatenate([entry_jp_idx, exp_sorted])[:max_jp_returned]
            ranked_jp_idx = combined

        # 7) Format parsed_canon
        retained_art = self._article_ids[article_idx]
        articles = [{"pair_key": pk} for pk in retained_art]

        jurisprudences = []
        for idx in ranked_jp_idx:
            uid = self._sub_ids[idx]
            pourvoi = self._jp_id2pourvoi.get(uid, "")
            if pourvoi:
                jurisprudences.append({
                    "pourvoi": pourvoi,
                    "score":   float(scores[idx]),
                    "source":  "entry" if idx in set(entry_jp_idx.tolist()) else "expanded",
                })

        canon = {
            "articles":       articles,
            "jurisprudences": jurisprudences,
            "arguments":      [],
            "_meta": {
                "k_entry":         k_entry,
                "hop2_M":          hop2_M,
                "hop2_top_n":      hop2_top_n,
                "max_jp_returned": max_jp_returned,
                "n_articles":      len(articles),
                "n_jp_returned":   len(jurisprudences),
                "n_jp_expanded":   int(len(expanded_jp_idx)),
            },
        }

        debug = {
            "expanded_jp_pourvois": [
                self._jp_id2pourvoi.get(self._sub_ids[i], "")
                for i in expanded_jp_idx
            ],
        }

        return canon, debug


# ═══════════════════════════════════════════════════════════════════════
# Configurations à tester (toutes avec K_entry=10)
# ═══════════════════════════════════════════════════════════════════════

CONFIGS = [
    # (name, hop2_M, hop2_top_n, max_jp_returned)
    ("1hop",                None, None,  10),    # baseline 1-hop
    ("2hop_M2",             2,    None,  10),
    ("2hop_M3",             3,    None,  10),
    ("2hop_M5",             5,    None,  10),
    ("2hop_M2_top200",      2,    200,   10),
    ("2hop_M2_jp50",        2,    None,  50),    # idem M2 mais retourne 50 JP au scorer
    ("2hop_M2_jp200",       2,    None,  200),
    ("2hop_M2_jp1000",      2,    None,  1000),  # gros pool
]


def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def main():
    questions = json.loads(RUBRICS.read_text(encoding="utf-8"))["questions"]
    print(f"Questions : {len(questions)}\n", flush=True)

    retriever = MultiHopRetriever()

    rows = []
    for cfg_name, m, top_n, max_jp in CONFIGS:
        per_q = []
        n_gt_recovered_list = []
        t0 = time.time()
        for q in questions:
            gt_p = gt_pourvois(q)
            canon, debug = retriever.query(
                q["question"],
                k_entry=10, hop2_M=m, hop2_top_n=top_n,
                max_jp_returned=max_jp,
            )
            scores = evaluate(canon, q)

            # Combien de pourvois GT dans l'expansion (avant cap max_jp_returned)
            n_in_expanded = len(set(debug["expanded_jp_pourvois"]) & gt_p)

            record = {
                "qid":      q["id"],
                "config":   cfg_name,
                "canon":    canon,
                "scores":   scores,
                "gt_pourvois":          sorted(gt_p),
                "n_gt_in_expansion":    n_in_expanded,
                "n_total_expansion":    len(debug["expanded_jp_pourvois"]),
            }
            (PER_Q_DIR / f"{q['id']}__{cfg_name}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            per_q.append(record)
            n_gt_recovered_list.append(n_in_expanded)
        elapsed = time.time() - t0

        means = {
            "S_retrieval":  _mean([r["scores"]["regime"]["retrieval"]                   for r in per_q]),
            "S_bar_art":    _mean([r["scores"]["articles"]["S_bar"]                     for r in per_q]),
            "S_bar_jp":     _mean([r["scores"]["jurisprudences"]["S_bar"]               for r in per_q]),
            "art_core":     _mean([r["scores"]["articles"]["per_strate"]["core"]            for r in per_q]),
            "art_expected": _mean([r["scores"]["articles"]["per_strate"]["expected"]        for r in per_q]),
            "art_expert":   _mean([r["scores"]["articles"]["per_strate"]["expert"]          for r in per_q]),
            "jp_core":      _mean([r["scores"]["jurisprudences"]["per_strate"]["core"]      for r in per_q]),
            "jp_expected":  _mean([r["scores"]["jurisprudences"]["per_strate"]["expected"]  for r in per_q]),
            "jp_expert":    _mean([r["scores"]["jurisprudences"]["per_strate"]["expert"]    for r in per_q]),
            "n_articles":     _mean([r["canon"]["_meta"]["n_articles"]      for r in per_q]),
            "n_jp_returned":  _mean([r["canon"]["_meta"]["n_jp_returned"]   for r in per_q]),
            "n_jp_expanded":  _mean([r["canon"]["_meta"]["n_jp_expanded"]   for r in per_q]),
            "n_gt_in_expanded": _mean(n_gt_recovered_list),
        }

        rows.append({
            "config":            cfg_name,
            "k_entry":           10,
            "hop2_M":            m if m else "",
            "hop2_top_n":        top_n if top_n else "",
            "max_jp_returned":   max_jp,
            "S_retrieval_mean":  means["S_retrieval"],
            "S_bar_art_mean":    means["S_bar_art"],
            "S_bar_jp_mean":     means["S_bar_jp"],
            "art_core_mean":     means["art_core"],
            "art_expected_mean": means["art_expected"],
            "art_expert_mean":   means["art_expert"],
            "jp_core_mean":      means["jp_core"],
            "jp_expected_mean":  means["jp_expected"],
            "jp_expert_mean":    means["jp_expert"],
            "n_articles_mean":     means["n_articles"],
            "n_jp_returned_mean":  means["n_jp_returned"],
            "n_jp_expanded_mean":  means["n_jp_expanded"],
            "n_gt_jp_in_expanded_mean": means["n_gt_in_expanded"],
        })

        print(f"[{cfg_name:<22s}] "
              f"S̄_art={means['S_bar_art']:.3f}  "
              f"S̄_jp={means['S_bar_jp']:.3f}  "
              f"n_art={means['n_articles']:.0f}  "
              f"n_exp={means['n_jp_expanded']:.0f}  "
              f"GT_in_exp={means['n_gt_in_expanded']:.2f}  "
              f"({elapsed:.1f}s)", flush=True)

    # Écrire CSV
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ Résultats → {CSV_PATH}")

    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        cols = ["config", "S_bar_art_mean", "S_bar_jp_mean",
                "art_core_mean", "art_expected_mean", "art_expert_mean",
                "n_articles_mean", "n_jp_expanded_mean",
                "n_gt_jp_in_expanded_mean"]
        print("\n" + df[cols].to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
