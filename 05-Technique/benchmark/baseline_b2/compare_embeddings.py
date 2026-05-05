#!/usr/bin/env python3
"""[Phase C — LOCAL] Compare plusieurs modèles d'embedding × techniques de pooling.

Pour chaque modèle (e5-base / e5-large / bge-m3 / camembert-large) ET pour
chaque technique (mean_pool / max_pool sur chunks / first_chunk), évalue :

  - Rang médian des pourvois GT dans le ranking → qualité intrinsèque
  - Recall top-10 / top-100 / top-1000 GT
  - Diversité inter-modèles : Jaccard top-10 entre paires de modèles
  - Score CRFPA S̄_art (1-hop union K=10) → comparaison au benchmark

Suppose les embeddings rapatriés du cluster dans embeddings/ :
  mean_<alias>.npy, chunks_<alias>.npy, chunk_to_jp_<alias>.npy,
  jp_order_<alias>.npy, chunk_offsets_<alias>.npy

Usage (depuis penal_bundle/) :
    python compare_embeddings.py
    python compare_embeddings.py --techniques mean_pool max_pool
    python compare_embeddings.py --models e5-base e5-large
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import combinations
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
RESULTS.mkdir(exist_ok=True)

CSV_OUT    = RESULTS / "phaseC_embeddings_comparison.csv"
JACCARD_OUT = RESULTS / "phaseC_jaccard_topK.csv"

POURVOI_RE = re.compile(r"\b(\d{2})[\s\-]*(\d{2})[.\-]?(\d{3})\b")

MODEL_DIMS = {
    "e5-base":         768,
    "e5-large":        1024,
    "bge-m3":          1024,
    "camembert-large": 1024,
}

PREFIX_QUERY = "query: "

sys.path.insert(0, str(HERE))
from eval_rubric import evaluate


def extract_pourvoi(text: str) -> str | None:
    if not text:
        return None
    m = POURVOI_RE.search(text)
    return f"{m.group(1)}-{m.group(2)}.{m.group(3)}" if m else None


# ═══════════════════════════════════════════════════════════════════════
# Loaders
# ═══════════════════════════════════════════════════════════════════════

def load_model_artifacts(alias: str):
    """Charge les artefacts d'un modèle. Renvoie dict ou None si manquants."""
    dim = MODEL_DIMS[alias]
    files = {
        "mean":     EMB_DIR / f"mean_{alias}.npy",
        "chunks":   EMB_DIR / f"chunks_{alias}.npy",
        "c2jp":     EMB_DIR / f"chunk_to_jp_{alias}.npy",
        "ids":      EMB_DIR / f"jp_order_{alias}.npy",
        "offsets":  EMB_DIR / f"chunk_offsets_{alias}.npy",
    }
    if not all(f.exists() for f in files.values()):
        missing = [n for n, f in files.items() if not f.exists()]
        print(f"  ⚠ {alias} : fichiers manquants {missing}")
        return None

    ids = np.load(files["ids"], allow_pickle=True)
    n_jp = len(ids)
    offsets = np.load(files["offsets"])
    n_chunks_total = int(offsets[-1])

    return {
        "alias":     alias,
        "dim":       dim,
        "ids":       ids,
        "mean":      np.memmap(files["mean"], dtype=np.float32, mode="r",
                                shape=(n_jp, dim)),
        "chunks":    np.memmap(files["chunks"], dtype=np.float32, mode="r",
                                shape=(n_chunks_total, dim)),
        "c2jp":      np.memmap(files["c2jp"], dtype=np.int32, mode="r",
                                shape=(n_chunks_total,)),
        "offsets":   offsets,
        "n_jp":      n_jp,
        "n_chunks":  n_chunks_total,
    }


def load_pourvoi_index():
    """parquet jp_id → number + parquet position → jp_id."""
    df = pq.read_table(PARQUET, columns=["id", "number"]).to_pandas()
    pourvoi2positions: dict[str, list[int]] = {}
    for i, num in enumerate(df["number"]):
        if num:
            pourvoi2positions.setdefault(num, []).append(i)
    return df["id"].to_numpy(), pourvoi2positions


# ═══════════════════════════════════════════════════════════════════════
# Techniques de pooling : produisent des scores par JP
# ═══════════════════════════════════════════════════════════════════════

def score_mean_pool(art: dict, q_vec: np.ndarray) -> np.ndarray:
    """Cosine similarity avec le vecteur moyen de chaque JP."""
    return art["mean"] @ q_vec    # (N_jp,)


def score_max_pool(art: dict, q_vec: np.ndarray) -> np.ndarray:
    """Pour chaque JP : max de cosine similarity sur ses chunks (chunk retrieval).

    Implémentation : score chaque chunk, puis pour chaque JP prendre le max
    via les offsets cumulés.
    """
    chunk_scores = art["chunks"] @ q_vec           # (N_chunks,)
    n_jp = art["n_jp"]
    offsets = art["offsets"]
    jp_scores = np.full(n_jp, -1.0, dtype=np.float32)
    for i in range(n_jp):
        s, e = int(offsets[i]), int(offsets[i+1])
        if e > s:
            jp_scores[i] = chunk_scores[s:e].max()
    return jp_scores


def score_first_chunk(art: dict, q_vec: np.ndarray) -> np.ndarray:
    """Score = similarité avec le PREMIER chunk seulement (proxy 'que le début')."""
    n_jp = art["n_jp"]
    offsets = art["offsets"]
    first_chunks_idx = offsets[:-1]                 # offset du 1er chunk de chaque doc
    first_embs = np.array(art["chunks"][first_chunks_idx])  # (N_jp, dim)
    return first_embs @ q_vec


SCORERS = {
    "mean_pool":    score_mean_pool,
    "max_pool":     score_max_pool,
    "first_chunk":  score_first_chunk,
}


# ═══════════════════════════════════════════════════════════════════════
# Évaluation par modèle × technique
# ═══════════════════════════════════════════════════════════════════════

def evaluate_model_technique(art: dict, technique: str,
                              questions: list[dict],
                              parquet_ids, pourvoi2pos,
                              graph: csr_matrix, article_ids,
                              graph_id2subpos: dict,
                              ) -> dict:
    """Pour un (modèle, technique) :
      - Embed chaque question
      - Score les JP via la technique
      - Rang des GT, recall@K, S̄_art via 1-hop union K=10
    """
    from sentence_transformers import SentenceTransformer
    # Charger le modèle pour embedder les queries
    hf_id = {
        "e5-base":         "intfloat/multilingual-e5-base",
        "e5-large":        "intfloat/multilingual-e5-large",
        "bge-m3":          "BAAI/bge-m3",
        "camembert-large": "dangvantuan/sentence-camembert-large",
    }[art["alias"]]
    model = SentenceTransformer(hf_id)

    # Helper pourvoi → emb_pos via parquet→graph→sub
    emb_id2pos = {uid: i for i, uid in enumerate(art["ids"])}
    def pourvoi_to_emb(pourvoi):
        positions = []
        for parquet_p in pourvoi2pos.get(pourvoi, []):
            uid = parquet_ids[parquet_p]
            if uid in emb_id2pos:
                positions.append(emb_id2pos[uid])
        return positions

    scorer = SCORERS[technique]

    rank_records = []
    s_art_list, jacc_topK = [], []
    article_recalls_core, article_recalls_exp, article_recalls_expert = [], [], []
    topK_jp_per_q: dict[str, list[str]] = {}

    for q in questions:
        qid = q["id"]
        q_vec = model.encode([PREFIX_QUERY + q["question"]],
                              normalize_embeddings=True, convert_to_numpy=True)[0]

        scores = scorer(art, q_vec)

        # Rang des GT
        order_desc = np.argsort(-scores)
        rank_of = np.empty(art["n_jp"], dtype=np.int32)
        rank_of[order_desc] = np.arange(art["n_jp"])

        rubric = q.get("rubric") or {}
        for strate in ("core", "expected", "expert"):
            for item in rubric.get(strate) or []:
                p = extract_pourvoi(item.get("linked_jp", ""))
                if not p:
                    continue
                positions = pourvoi_to_emb(p)
                if not positions:
                    continue
                best_rank = min(rank_of[pos] for pos in positions) + 1
                rank_records.append({
                    "qid": qid, "strate": strate, "pourvoi": p,
                    "rank": int(best_rank),
                })

        # S̄_art via 1-hop union K=10
        K = 10
        top_k_idx = np.argpartition(scores, -K)[-K:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]
        topK_jp_per_q[qid] = [str(art["ids"][i]) for i in top_k_idx]

        # Pour le scoring article : top-K JP → leurs articles (1-hop) → union
        # Mapping emb_pos → graph row : besoin du graph_id2subpos qu'on a passé
        # mais ici les top_k_idx sont déjà indices dans art["ids"] = mêmes que graph rows alignés ?
        # On suppose que les indices alignent (cas standard dans les bundles cohérents)
        # Sinon on fait le lookup explicite :
        article_counts = np.zeros(len(article_ids), dtype=np.int32)
        for idx in top_k_idx:
            uid = art["ids"][idx]
            graph_pos = graph_id2subpos.get(uid)
            if graph_pos is not None:
                article_counts[graph[graph_pos].indices] += 1
        retained_art_idx = np.where(article_counts >= 1)[0]
        retained_art = article_ids[retained_art_idx]

        # Build canon for eval_rubric
        canon = {
            "articles":       [{"pair_key": pk} for pk in retained_art],
            "jurisprudences": [],  # pas évalué ici (on sait que c'est nul)
            "arguments":      [],
            "_meta": {},
        }
        s = evaluate(canon, q)
        s_art_list.append(s["articles"]["S_bar"])
        article_recalls_core.append(s["articles"]["per_strate"]["core"])
        article_recalls_exp.append(s["articles"]["per_strate"]["expected"])
        article_recalls_expert.append(s["articles"]["per_strate"]["expert"])

    # Stats globales
    valid_ranks = [r["rank"] for r in rank_records]
    out = {
        "alias":     art["alias"],
        "technique": technique,
        "n_gt":      len(rank_records),
        "rank_median": int(np.median(valid_ranks)) if valid_ranks else None,
        "rank_mean":   int(np.mean(valid_ranks))   if valid_ranks else None,
        "recall_top10":   sum(1 for r in valid_ranks if r <= 10)   / max(len(valid_ranks),1),
        "recall_top100":  sum(1 for r in valid_ranks if r <= 100)  / max(len(valid_ranks),1),
        "recall_top1k":   sum(1 for r in valid_ranks if r <= 1000) / max(len(valid_ranks),1),
        "recall_top10k":  sum(1 for r in valid_ranks if r <= 10000)/ max(len(valid_ranks),1),
        "S_bar_art_mean":   round(float(np.mean([v for v in s_art_list if v is not None])), 4) if s_art_list else None,
        "art_core_mean":    round(float(np.mean([v for v in article_recalls_core if v is not None])), 4),
        "art_expected_mean":round(float(np.mean([v for v in article_recalls_exp if v is not None])), 4),
        "art_expert_mean":  round(float(np.mean([v for v in article_recalls_expert if v is not None])), 4),
    }
    return out, topK_jp_per_q


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_DIMS),
                    help=f"Modèles à comparer (défaut tous : {list(MODEL_DIMS)})")
    ap.add_argument("--techniques", nargs="+", default=list(SCORERS),
                    help=f"Techniques (défaut tous : {list(SCORERS)})")
    args = ap.parse_args()

    # Charger graphe
    print("Chargement graphe pénal…")
    g = np.load(GRAPH_NPZ, allow_pickle=True)
    graph_jp_ids = g["jp_ids"]
    article_ids  = g["article_ids"]
    graph = csr_matrix((g["data"], g["indices"], g["indptr"]), shape=tuple(g["shape"]))
    graph_id2pos = {uid: i for i, uid in enumerate(graph_jp_ids)}

    # Charger pourvoi index
    parquet_ids, pourvoi2pos = load_pourvoi_index()

    # Charger questions
    questions = json.loads(RUBRICS.read_text(encoding="utf-8"))["questions"]
    print(f"Questions : {len(questions)}\n")

    rows = []
    topK_per_model: dict[str, dict[str, list[str]]] = {}  # alias → qid → topK jp_ids

    for alias in args.models:
        if alias not in MODEL_DIMS:
            print(f"⚠ Skip {alias} (alias inconnu)")
            continue
        art = load_model_artifacts(alias)
        if art is None:
            continue

        print(f"\n{'='*60}\n[{alias}]  dim={art['dim']}  n_jp={art['n_jp']}  "
              f"n_chunks={art['n_chunks']}\n{'='*60}")

        for technique in args.techniques:
            print(f"  ── {technique}")
            out, topK = evaluate_model_technique(
                art, technique, questions,
                parquet_ids, pourvoi2pos,
                graph, article_ids, graph_id2pos,
            )
            rows.append(out)
            print(f"    rank median = {out['rank_median']}, "
                  f"top10={out['recall_top10']:.0%}, top100={out['recall_top100']:.0%}, "
                  f"top1k={out['recall_top1k']:.0%}, "
                  f"S̄_art={out['S_bar_art_mean']}")
            if technique == "mean_pool":
                topK_per_model[alias] = topK

    # Écrire CSV principal
    if rows:
        with open(CSV_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n✓ Comparaison → {CSV_OUT}")

    # Jaccard top-10 entre modèles (sur mean_pool)
    if len(topK_per_model) >= 2:
        jacc_rows = []
        for a, b in combinations(sorted(topK_per_model), 2):
            jacc_per_q = []
            for qid in topK_per_model[a]:
                set_a = set(topK_per_model[a][qid])
                set_b = set(topK_per_model[b][qid])
                if set_a or set_b:
                    j = len(set_a & set_b) / len(set_a | set_b)
                    jacc_per_q.append(j)
            jacc_rows.append({
                "model_a": a, "model_b": b,
                "jaccard_mean": round(float(np.mean(jacc_per_q)), 3) if jacc_per_q else None,
                "jaccard_min":  round(float(np.min(jacc_per_q)), 3)  if jacc_per_q else None,
                "jaccard_max":  round(float(np.max(jacc_per_q)), 3)  if jacc_per_q else None,
            })
        with open(JACCARD_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(jacc_rows[0].keys()))
            w.writeheader()
            w.writerows(jacc_rows)
        print(f"✓ Jaccard top-10 inter-modèles → {JACCARD_OUT}")
        print()
        try:
            import pandas as pd
            print(pd.read_csv(JACCARD_OUT).to_string(index=False))
        except ImportError:
            pass

    # Affichage final
    try:
        import pandas as pd
        df = pd.read_csv(CSV_OUT)
        print()
        cols = ["alias", "technique", "rank_median", "recall_top10",
                "recall_top100", "recall_top1k", "S_bar_art_mean"]
        print(df[cols].to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
