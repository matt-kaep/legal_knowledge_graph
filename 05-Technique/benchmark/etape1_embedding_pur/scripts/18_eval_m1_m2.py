"""Éval M1 (recall@K) + M2 (rang moyen normalisé) sur la cohorte 977.

Cohorte = bench_article ∩ bench_jp depuis bench_global.json (970 doctrine_qgen + 7 CRFPA).

Baselines évaluées :
  - B2-a : top-cosinus sur articles
  - B3-a : top-cosinus sur JP
  - B3-e : articles via JP (re-rank cosinus du B_art)
  - B4-a : cross-modal articles UNION (A_art ∪ B_art)
  - B4-c : cross-modal JP UNION (A_jp ∪ B_jp)
  - B4-d : cross-modal JP INTERSECTION (A_jp ∩ B_jp)
  - B4-e : cross-modal JP RRF (Reciprocal Rank Fusion B3-a + B3-b)
  - B4-f : cross-modal JP citation-weighted (union triée par # top-articles citants)

K fixé à 10 pour articles et JP.
K_in (pour B3-e, B4-a, B4-c) : 10, 20, 50.

Ground truth (= « GT ») :
  - strict     = articles_attendus (depuis bench_global)
  - étendu     = articles_attendus_etendu (depuis bench_global, déjà précalculé via graphe v5)

Métriques :
  - M1 = recall@K = |GT ∩ R[:K]| / |GT|
  - M2 = rang moyen normalisé : f(x) = (x - (K+1)) / ((b'+1)/2 - (K+1))
         avec b' = min(|GT|, K) (clip), x = rang moyen des GT (cap à K+1)
         f(K+1)=0 (pire), f((b'+1)/2)=1 (meilleur)
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
sys.path.insert(0, str(Path(__file__).parent))
from etape1 import config  # noqa: E402
import metrics as M  # noqa: E402

BENCH_PATH = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench/bench_global.json"
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
OUT_DIR.mkdir(exist_ok=True, parents=True)

Q_EMB_CACHE = OUT_DIR / "questions_977_emb.npy"
Q_IDS_CACHE = OUT_DIR / "questions_977_ids.npy"

# Cache existant du script 13 (1707 questions doctrine_qgen)
OLD_Q_EMB = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen/questions_emb.npy"
OLD_Q_IDS = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen/questions_ids.npy"

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

K_ART = 10
K_JP = 10
KS_IN = [10, 20, 50]


# ────────────────────────────────────────────────────────────────────────
# Chargement cohorte 977
# ────────────────────────────────────────────────────────────────────────
def load_cohort_977() -> list[dict]:
    d = json.loads(BENCH_PATH.read_text())
    out = []
    for q in d["questions"]:
        arts = q.get("articles_attendus") or []
        pourvois = q.get("pourvois_cc") or []
        n_jp_res = q.get("n_jp_resolues", 0)
        # Critère cohorte : ≥1 article ET ≥1 JP CC effectivement résolue dans Judilibre
        # (cohérent avec bench_jp construit par 17_build_global_bench.py)
        if not arts or not pourvois or n_jp_res < 1:
            continue
        arts_ext = q.get("articles_attendus_etendu") or arts
        out.append({
            "id": q["qid"],
            "source": q.get("source"),
            "enonce": q["enonce"],
            "gt_strict": set(arts),
            "gt_ext": set(arts_ext),
            "pourvois": set(pourvois),
        })
    return out


def build_pourvoi_map() -> dict[str, list[str]]:
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def encode(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if torch.backends.mps.is_available()
           else "cpu")
    print(f"  device : {dev}")
    m = SentenceTransformer(config.MODEL_ID, device=dev)
    m.max_seq_length = config.BATCH_MAX_LEN
    return m.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────
# Métriques (factorisées dans metrics.py — chantier 1 Week-10)
# ────────────────────────────────────────────────────────────────────────
# Panel : M1, M2, Hit@K, MRR@K, NDCG@K sur strict ET étendu (10 colonnes).
# Pour côté JP, gt_strict = gt_ext = gold_jp ⇒ cols strict/ext identiques.
both_metrics = M.panel_strict_ext  # alias rétrocompat


def ranking_rows(qid, method, k_in, modality, ranked, k):
    """Lignes (qid, method, k_in, modality, rank, item_id) pour le top-k.

    Sert à dumper rankings.parquet : M3 (LLM-judge, script 23) doit juger
    EXACTEMENT les rankings que M1/M2 ont mesurés ici — pas un recalcul
    susceptible de diverger.
    """
    return [
        {"qid": qid, "method": method, "k_in": k_in, "modality": modality,
         "rank": r + 1, "item_id": str(item)}
        for r, item in enumerate(ranked[:k])
    ]


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────
def main() -> int:
    t0 = time.time()
    print("══ Chargement embeddings + graphe ─────────────────────────")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    jp_emb = np.load(config.EMB_JP_SYNTHESE)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    jp_to_row = np.load(config.JP_SUMMARY_TO_GRAPHROW)

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]

    pk_to_emb_idx = {pk: i for i, pk in enumerate(art_order)}
    jpid_to_emb_idx = {jid: i for i, jid in enumerate(jp_order)}
    pool_articles = set(art_order.tolist())   # univers retrievable côté articles
    pool_jp = set(jp_order.tolist())          # univers retrievable côté JP
    pourvoi_map = build_pourvoi_map()
    print(f"  art_emb {art_emb.shape}  jp_emb {jp_emb.shape}  G {G.shape}")
    print(f"  pool articles : {len(pool_articles)}  pool JP : {len(pool_jp)}")

    print("\n══ Chargement cohorte 977 ─────────────────────────────────")
    questions = load_cohort_977()
    print(f"  cohorte : {len(questions)} questions")
    by_src = {}
    for q in questions:
        by_src[q["source"]] = by_src.get(q["source"], 0) + 1
    print(f"  par source : {by_src}")

    qids = [q["id"] for q in questions]

    # Stratégie : réutiliser le cache embeddings du script 13 (1707 q doctrine_qgen).
    # Les questions manquantes (CRFPA + nouvelles doctrine) sont droppées avec un
    # avertissement explicite — torch >= 2.4 requis pour les ré-encoder, à régler
    # dans un second temps.
    use_cache = False
    if Q_EMB_CACHE.exists() and Q_IDS_CACHE.exists():
        cached_ids = np.load(Q_IDS_CACHE, allow_pickle=True).tolist()
        if cached_ids == qids:
            print("  → cache local 977 HIT")
            Q = np.load(Q_EMB_CACHE)
            use_cache = True
    if not use_cache and OLD_Q_EMB.exists() and OLD_Q_IDS.exists():
        old_ids = np.load(OLD_Q_IDS, allow_pickle=True).tolist()
        old_emb = np.load(OLD_Q_EMB)
        old_idx_by_id = {qid: i for i, qid in enumerate(old_ids)}
        kept_indices, dropped = [], []
        for q in questions:
            i = old_idx_by_id.get(q["id"])
            if i is None:
                dropped.append(q["id"])
            else:
                kept_indices.append((q, i))
        if dropped:
            print(f"  ⚠ {len(dropped)} questions sans embedding (dropped) — "
                  f"ex : {dropped[:3]}")
        questions = [q for q, _ in kept_indices]
        Q = np.asarray([old_emb[i] for _, i in kept_indices])
        np.save(Q_EMB_CACHE, Q)
        np.save(Q_IDS_CACHE, np.array([q["id"] for q in questions]))
        use_cache = True
    if not use_cache:
        print("\n══ Encoding questions ─────────────────────────────────────")
        enonces = [q["enonce"] for q in questions]
        Q = encode(enonces)
        np.save(Q_EMB_CACHE, Q)
        np.save(Q_IDS_CACHE, np.array(qids))
    print(f"  Q : {Q.shape}  ({len(questions)} questions)  (t={time.time()-t0:.1f}s)")

    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T
    sim_jp = Q @ jp_emb.T
    art_rank = np.argsort(-sim_art, axis=1)
    jp_rank = np.argsort(-sim_jp, axis=1)
    print(f"  argsort done  (t={time.time()-t0:.1f}s)")

    rows = []
    rankings = []  # dump (qid, method, k_in, modality, rank, item_id) -> rankings.parquet
    print("\n══ Boucle d'éval ──────────────────────────────────────────")
    for qi, q in enumerate(questions):
        if qi % 200 == 0:
            print(f"  q {qi}/{len(questions)} (t={time.time()-t0:.1f}s)")

        # GT filtrée au pool indexé (pas de plafonnement artificiel par items hors base)
        gt_s = q["gt_strict"] & pool_articles
        gt_e = q["gt_ext"] & pool_articles
        gold_jp_all = {jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
        gold_jp = gold_jp_all & pool_jp
        gold_jp_ext = gold_jp  # pas d'extension JP

        # Skip si GT vide après filtre pool (rare mais possible)
        if not gt_s and not gold_jp:
            continue

        # ─── B2-a articles open
        top_emb_idx = art_rank[qi, :K_ART]
        ranked_pks = list(art_order[top_emb_idx])
        rows.append({
            "qid": q["id"], "method": "B2-a",
            "k_in": None, "k": K_ART, "modality": "art",
            **both_metrics(ranked_pks, gt_s, gt_e, K_ART),
        })
        rankings.extend(ranking_rows(q["id"], "B2-a", None, "art", ranked_pks, K_ART))

        # ─── B3-a JP direct
        if gold_jp:
            top_emb_idx = jp_rank[qi, :K_JP]
            ranked_jp = list(jp_order[top_emb_idx])
            rows.append({
                "qid": q["id"], "method": "B3-a",
                "k_in": None, "k": K_JP, "modality": "jp",
                **both_metrics(ranked_jp, gold_jp, gold_jp_ext, K_JP),
            })
            rankings.extend(ranking_rows(q["id"], "B3-a", None, "jp", ranked_jp, K_JP))

        # ─── Sweep K_in pour les variantes cross/graph
        for k_in in KS_IN:
            top_art_emb_idx = art_rank[qi, :k_in]
            top_art_cols = p2col[top_art_emb_idx]
            top_art_pks = set(art_order[top_art_emb_idx].tolist())
            top_jp_emb_idx = jp_rank[qi, :k_in]
            top_jp_ids = set(jp_order[top_jp_emb_idx].tolist())
            top_jp_rows = jp_to_row[top_jp_emb_idx]

            sub = G[:, top_art_cols]
            jp_count_arr = np.asarray((sub != 0).sum(axis=1)).ravel()
            A_jp_ids = set(jp_ids_graph[jp_count_arr >= 1].tolist())
            # mapping JP -> # de top-K_in articles qui la citent (pour B4-f)
            jp_citation_count = {
                jid: int(jp_count_arr[i])
                for i, jid in enumerate(jp_ids_graph)
                if jp_count_arr[i] >= 1
            }

            sub2 = G[top_jp_rows, :]
            art_count = np.asarray((sub2 != 0).sum(axis=0)).ravel()
            B_art_cols = np.where(art_count >= 1)[0]
            B_art_pks = set(article_ids_graph[B_art_cols].tolist())

            # ─── B3-e : articles via JP (re-rank cosinus B_art)
            emb_idx_S = [pk_to_emb_idx[pk] for pk in B_art_pks if pk in pk_to_emb_idx]
            if emb_idx_S:
                arr = np.asarray(emb_idx_S, dtype=np.int64)
                order = np.argsort(-sim_art[qi, arr])
                ranked = list(art_order[arr[order]])
            else:
                ranked = []
            rows.append({
                "qid": q["id"], "method": "B3-e",
                "k_in": k_in, "k": K_ART, "modality": "art",
                **both_metrics(ranked, gt_s, gt_e, K_ART),
            })
            rankings.extend(ranking_rows(q["id"], "B3-e", k_in, "art", ranked, K_ART))

            # ─── B4-a : cross-modal articles UNION
            S_pks = top_art_pks | B_art_pks
            emb_idx_S = [pk_to_emb_idx[pk] for pk in S_pks if pk in pk_to_emb_idx]
            if emb_idx_S:
                arr = np.asarray(emb_idx_S, dtype=np.int64)
                order = np.argsort(-sim_art[qi, arr])
                ranked = list(art_order[arr[order]])
            else:
                ranked = []
            rows.append({
                "qid": q["id"], "method": "B4-a",
                "k_in": k_in, "k": K_ART, "modality": "art",
                **both_metrics(ranked, gt_s, gt_e, K_ART),
            })

            # ─── B4-c : cross-modal JP UNION
            if gold_jp:
                S_jp = A_jp_ids | top_jp_ids
                emb_idx_S = [jpid_to_emb_idx[j] for j in S_jp if j in jpid_to_emb_idx]
                if emb_idx_S:
                    arr = np.asarray(emb_idx_S, dtype=np.int64)
                    order = np.argsort(-sim_jp[qi, arr])
                    ranked = list(jp_order[arr[order]])
                else:
                    ranked = []
                rows.append({
                    "qid": q["id"], "method": "B4-c",
                    "k_in": k_in, "k": K_JP, "modality": "jp",
                    **both_metrics(ranked, gold_jp, gold_jp_ext, K_JP),
                })

            # ─── B4-d : cross-modal JP INTERSECTION (= B3-a ∩ B3-b)
            # JP confirmées par les deux modalités : cosine direct (top-K_in)
            # ET citées par au moins un top-K_in article (graphe).
            if gold_jp:
                S_jp = A_jp_ids & top_jp_ids
                emb_idx_S = [jpid_to_emb_idx[j] for j in S_jp if j in jpid_to_emb_idx]
                if emb_idx_S:
                    arr = np.asarray(emb_idx_S, dtype=np.int64)
                    order = np.argsort(-sim_jp[qi, arr])
                    ranked = list(jp_order[arr[order]])
                else:
                    ranked = []
                rows.append({
                    "qid": q["id"], "method": "B4-d",
                    "k_in": k_in, "k": K_JP, "modality": "jp",
                    "n_inter": len(S_jp),
                    **both_metrics(ranked, gold_jp, gold_jp_ext, K_JP),
                })
                rankings.extend(ranking_rows(q["id"], "B4-d", k_in, "jp", ranked, K_JP))

            # ─── B4-e : cross-modal JP RRF (Reciprocal Rank Fusion)
            # score(jp) = 1/(K_RRF + rank_cosine) + 1/(K_RRF + rank_graph)
            # rank_cosine : position dans top-K_in B3-a (sinon +∞)
            # rank_graph  : position dans A_jp triée par citation_count desc (sinon +∞)
            # On considère l'union des candidats et on récompense la confirmation par les deux.
            if gold_jp:
                K_RRF = 60  # constante standard (Cormack 2009)
                rank_cos = {jp_order[top_jp_emb_idx[r]]: r + 1 for r in range(k_in)}
                # ranking graphe : JPs A_jp triées par citation_count desc
                a_sorted = sorted(A_jp_ids, key=lambda j: (-jp_citation_count.get(j, 0), j))
                rank_graph = {j: r + 1 for r, j in enumerate(a_sorted)}

                S_jp = A_jp_ids | top_jp_ids
                def rrf_score(j):
                    s = 0.0
                    if j in rank_cos:   s += 1.0 / (K_RRF + rank_cos[j])
                    if j in rank_graph: s += 1.0 / (K_RRF + rank_graph[j])
                    return s
                # Trier par RRF desc ; tiebreak = sim_jp cosine
                jp_with_emb = [j for j in S_jp if j in jpid_to_emb_idx]
                jp_with_emb.sort(
                    key=lambda j: (-rrf_score(j), -float(sim_jp[qi, jpid_to_emb_idx[j]]))
                )
                ranked = jp_with_emb[:K_JP]
                rows.append({
                    "qid": q["id"], "method": "B4-e",
                    "k_in": k_in, "k": K_JP, "modality": "jp",
                    **both_metrics(ranked, gold_jp, gold_jp_ext, K_JP),
                })
                rankings.extend(ranking_rows(q["id"], "B4-e", k_in, "jp", ranked, K_JP))

            # ─── B4-f : cross-modal JP citation-weighted
            # « les plus intersectées » : union triée par # de top-K_in articles
            # qui citent chaque JP (0 si absente du graphe candidat).
            # Tiebreak : cosine similarity (= B3-a).
            if gold_jp:
                S_jp = A_jp_ids | top_jp_ids
                jp_with_emb = [j for j in S_jp if j in jpid_to_emb_idx]
                jp_with_emb.sort(key=lambda j: (
                    -jp_citation_count.get(j, 0),
                    -float(sim_jp[qi, jpid_to_emb_idx[j]]),
                ))
                ranked = jp_with_emb[:K_JP]
                rows.append({
                    "qid": q["id"], "method": "B4-f",
                    "k_in": k_in, "k": K_JP, "modality": "jp",
                    **both_metrics(ranked, gold_jp, gold_jp_ext, K_JP),
                })
                rankings.extend(ranking_rows(q["id"], "B4-f", k_in, "jp", ranked, K_JP))

    print(f"  fin boucle (t={time.time()-t0:.1f}s)")

    # ────────────────────────────────────────────────────────────────
    # Agrégation
    # ────────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "eval_m1_m2.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}  ({len(df)} lignes)")

    # Dump rankings pour M3 (LLM-judge, script 23). On écrase la part "18" du
    # parquet (méthodes B*) en préservant celles d'autres producteurs (PPR/20).
    rk = pd.DataFrame(rankings)
    rank_path = OUT_DIR / "rankings.parquet"
    if rank_path.exists():
        prev = pd.read_parquet(rank_path)
        prev = prev[~prev["method"].isin(rk["method"].unique())]
        rk = pd.concat([prev, rk], ignore_index=True)
    rk.to_parquet(rank_path, index=False)
    print(f"✓ {rank_path}  ({len(rk)} lignes rankings, "
          f"méthodes={sorted(rk['method'].unique())})")

    print("\n══ Agrégats ───────────────────────────────────────────────")
    summary = {}
    cols_metrics = [f"{m}_{r}" for m in M.METRIC_NAMES for r in ("strict", "ext")]
    # Print compact : M1_s / Hit_s / MRR_s / NDCG_s / M2_s côté strict
    # (M2 historiquement à droite). Étendu : juste M1/Hit/NDCG (MRR moins parlant multi-GT).
    print(f"\n  {'méthode':<8s} {'modal':<5s} {'k_in':>4s} {'n':>4s} | "
          f"{'M1_s':>6s} {'Hit_s':>6s} {'MRR_s':>6s} {'NDCG_s':>6s} {'M2_s':>6s} | "
          f"{'M1_e':>6s} {'Hit_e':>6s} {'NDCG_e':>6s}")
    print("  " + "─" * 100)
    for (method, modality, k_in), sub in df.groupby(
        ["method", "modality", "k_in"], dropna=False
    ):
        n = len(sub)
        means = {c: float(sub[c].mean()) for c in cols_metrics}
        kin_disp = str(int(k_in)) if pd.notna(k_in) else "-"
        key = f"{method}|{modality}|kin={kin_disp}"
        summary[key] = {"n_q": n, **means}
        print(f"  {method:<8s} {modality:<5s} {kin_disp:>4s} {n:>4d} | "
              f"{means['m1_strict']:>6.3f} {means['hit_strict']:>6.3f} "
              f"{means['mrr_strict']:>6.3f} {means['ndcg_strict']:>6.3f} "
              f"{means['m2_strict']:>6.3f} | "
              f"{means['m1_ext']:>6.3f} {means['hit_ext']:>6.3f} "
              f"{means['ndcg_ext']:>6.3f}")

    (OUT_DIR / "eval_m1_m2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n✓ {OUT_DIR}/eval_m1_m2_summary.json")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
