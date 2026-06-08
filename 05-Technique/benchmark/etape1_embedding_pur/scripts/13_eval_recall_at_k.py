"""Éval unifiée recall@K + precision@K — définition Johnny du 29/05.

Pour CHAQUE méthode (B2-a, B2-b, B3-a, B3-b, B3-e, B4-a, B4-b, B4-c, B4-d) :

  1. Construire un candidate set S = M(question)
  2. Calculer Z_i = cosine(X_question, X_S_i) pour i ∈ S
  3. rank(Z, K) = top-K candidats par similarité cosinus décroissante
  4. recall@K  = |gold ∩ rank(Z,K)| / |gold|
     precision@K = |gold ∩ rank(Z,K)| / K

Cas |S| < K : rank(Z,K) = S, donc recall plafonné à recall@|S| et
precision = |gold ∩ S| / K (pénalise les méthodes à faible |S|).

K final :
  - articles : K ∈ {10, 20}
  - JP       : K ∈ {5, 10}

K_in (pour les variantes cross / via_graph) : sweep {10, 20, 50}

Sorties :
  data/doctrine_qgen/recall_at_k.csv          (long format)
  data/doctrine_qgen/recall_at_k_summary.json (agrégats moyens)
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

from etape1 import config  # noqa: E402

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

CORPUS_PATH = REPO / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_strict_gemma4-26B-A4B.json"
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen"
OUT_DIR.mkdir(exist_ok=True, parents=True)

Q_EMB_CACHE = OUT_DIR / "questions_emb.npy"
Q_IDS_CACHE = OUT_DIR / "questions_ids.npy"

FILTERED_CODES_PENAL_STRICT = set(config.PENAL_CODES.keys())

KS_ARTICLES = [10, 20]
KS_JP = [5, 10]
KS_IN = [10, 20, 50]


def load_doctrine_qgen() -> list[dict]:
    d = json.loads(CORPUS_PATH.read_text())
    out = []
    for q in d["questions"]:
        oblig = {
            f'{a["code_slug"]}:{a["article_num"]}'
            for a in q.get("articles_attendus", [])
        }
        pourvois = {
            p
            for j in q.get("jp_attendues", [])
            if (p := (j.get("pourvoi") or "").strip()) and _POURVOI_RE.fullmatch(p)
        }
        out.append({
            "id": q["qid"],
            "doc_id": q.get("doc_id"),
            "enonce": q["enonce"],
            "oblig": oblig,
            "pourvois": pourvois,
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


def rerank_by_cosine(candidate_ids: list, id_to_emb_idx: dict, sim_q_row: np.ndarray) -> list:
    """Re-rank une liste d'IDs par similarité cosinus décroissante avec la question."""
    if not candidate_ids:
        return []
    emb_idx = [id_to_emb_idx[c] for c in candidate_ids if c in id_to_emb_idx]
    if not emb_idx:
        return []
    emb_idx_arr = np.asarray(emb_idx, dtype=np.int64)
    sims = sim_q_row[emb_idx_arr]
    order = np.argsort(-sims)
    return emb_idx_arr[order].tolist()  # liste d'indices dans le pool d'emb


def recall_precision_at_k(ranked_ids: list, gold: set, k: int) -> tuple[float, float, int]:
    if not gold:
        return float("nan"), float("nan"), 0
    top = ranked_ids[:k]
    hits = len(set(top) & gold)
    return hits / len(gold), hits / k, hits


def rp_pair(ranked_ids: list, gold_strict: set, gold_extended: set, k: int) -> dict:
    """Renvoie recall/precision pour gold strict ET gold étendu en un seul passage."""
    rs, ps, _ = recall_precision_at_k(ranked_ids, gold_strict, k)
    re_, pe_, _ = recall_precision_at_k(ranked_ids, gold_extended, k)
    return {"recall": rs, "precision": ps, "recall_ext": re_, "precision_ext": pe_}


def main() -> int:
    t0 = time.time()
    print("══ Chargement ─────────────────────────────────────────────")
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
    full_codes = z["article_codes"]
    art_codes = full_codes[p2col]

    pourvoi_map = build_pourvoi_map()
    # Pour l'extension du gold articles : jpid -> row dans G
    jpid_to_row_graph = {j: i for i, j in enumerate(jp_ids_graph)}
    print(f"  art_emb {art_emb.shape}  jp_emb {jp_emb.shape}  G {G.shape}")

    pk_to_emb_idx = {pk: i for i, pk in enumerate(art_order)}
    jpid_to_emb_idx = {jid: i for i, jid in enumerate(jp_order)}

    questions = load_doctrine_qgen()
    print(f"  questions : {len(questions)}")

    enonces = [q["enonce"] for q in questions]
    qids = [q["id"] for q in questions]
    use_cache = False
    if Q_EMB_CACHE.exists() and Q_IDS_CACHE.exists():
        cached_ids = np.load(Q_IDS_CACHE, allow_pickle=True).tolist()
        if cached_ids == qids:
            print("  → cache embeddings questions HIT")
            Q = np.load(Q_EMB_CACHE)
            use_cache = True
    if not use_cache:
        print("\n══ Encoding questions ─────────────────────────────────────")
        Q = encode(enonces)
        np.save(Q_EMB_CACHE, Q)
        np.save(Q_IDS_CACHE, np.array(qids))
    print(f"  Q : {Q.shape}  (t={time.time()-t0:.1f}s)")

    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T
    sim_jp = Q @ jp_emb.T
    print(f"  sim_art {sim_art.shape}  sim_jp {sim_jp.shape}")
    print(f"  t={time.time()-t0:.1f}s")

    print("  argsort sim_art + sim_jp …")
    art_rank = np.argsort(-sim_art, axis=1)
    jp_rank = np.argsort(-sim_jp, axis=1)
    print(f"  t={time.time()-t0:.1f}s")

    mask_strict = np.array([c in FILTERED_CODES_PENAL_STRICT for c in art_codes])
    print(f"  pool strict pénal : {mask_strict.sum()} articles")

    rows = []
    print("\n══ Boucle d'éval ──────────────────────────────────────────")
    K_ART_MAX = max(KS_ARTICLES)
    K_JP_MAX = max(KS_JP)
    for qi, q in enumerate(questions):
        if qi % 200 == 0:
            print(f"  q {qi}/{len(questions)} (t={time.time()-t0:.1f}s)")

        oblig = q["oblig"]
        gold_jp = {jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
        # Gold étendu côté articles : oblig ∪ (articles cités par les JP gold via graphe)
        oblig_ext = set(oblig)
        for jid in gold_jp:
            if jid in jpid_to_row_graph:
                row = jpid_to_row_graph[jid]
                cols = G[row].indices
                for col in cols:
                    oblig_ext.add(article_ids_graph[int(col)])
        # Gold JP : pas d'extension (l'extension serait circulaire pour B3-b)
        gold_jp_ext = gold_jp

        # ─── B2-a articles open
        if oblig:
            top_emb_idx = art_rank[qi, :K_ART_MAX]
            ranked_pks = list(art_order[top_emb_idx])
            for K in KS_ARTICLES:
                _m = rp_pair(ranked_pks, oblig, oblig_ext, K)
                rows.append({
                    "question_id": q["id"], "method": "B2-a_articles_open",
                    "k_in": None, "k_final": K, "n_S": len(art_order),
                    **_m,
                })

        # ─── B2-b articles strict
        if oblig:
            mask_local = mask_strict[art_rank[qi]]
            local_idx = art_rank[qi][mask_local][:K_ART_MAX]
            ranked_pks_strict = list(art_order[local_idx])
            for K in KS_ARTICLES:
                _m = rp_pair(ranked_pks_strict, oblig, oblig_ext, K)
                rows.append({
                    "question_id": q["id"], "method": "B2-b_articles_strict",
                    "k_in": None, "k_final": K, "n_S": int(mask_strict.sum()),
                    **_m,
                })

        # ─── B3-a JP direct
        if gold_jp:
            top_emb_idx = jp_rank[qi, :K_JP_MAX]
            ranked_jp = list(jp_order[top_emb_idx])
            for K in KS_JP:
                _m = rp_pair(ranked_jp, gold_jp, gold_jp_ext, K)
                rows.append({
                    "question_id": q["id"], "method": "B3-a_jp_direct",
                    "k_in": None, "k_final": K, "n_S": len(jp_order),
                    **_m,
                })

        # ─── Variantes graphe et cross — sweep K_in
        for k_in in KS_IN:
            top_art_emb_idx = art_rank[qi, :k_in]
            top_art_cols = p2col[top_art_emb_idx]
            top_art_pks = set(art_order[top_art_emb_idx].tolist())
            top_jp_emb_idx = jp_rank[qi, :k_in]
            top_jp_ids = set(jp_order[top_jp_emb_idx].tolist())
            top_jp_rows = jp_to_row[top_jp_emb_idx]

            # JP voisines des top-K_in articles (= A_jp pour cross)
            sub = G[:, top_art_cols]
            jp_count = np.asarray((sub != 0).sum(axis=1)).ravel()
            A_jp_ids = set(jp_ids_graph[jp_count >= 1].tolist())
            # Articles cités par top-K_in JP (= B_art pour cross)
            sub2 = G[top_jp_rows, :]
            art_count = np.asarray((sub2 != 0).sum(axis=0)).ravel()
            B_art_cols = np.where(art_count >= 1)[0]
            B_art_pks = set(article_ids_graph[B_art_cols].tolist())

            # ─── B3-b JP via_graph : re-rank A_jp par cosinus
            if gold_jp:
                emb_idx_S = [jpid_to_emb_idx[j] for j in A_jp_ids if j in jpid_to_emb_idx]
                if emb_idx_S:
                    arr = np.asarray(emb_idx_S, dtype=np.int64)
                    order = np.argsort(-sim_jp[qi, arr])
                    ranked = list(jp_order[arr[order]])
                else:
                    ranked = []
                for K in KS_JP:
                    _m = rp_pair(ranked, gold_jp, gold_jp_ext, K)
                    rows.append({
                        "question_id": q["id"], "method": "B3-b_jp_via_graph",
                        "k_in": k_in, "k_final": K, "n_S": len(ranked),
                        **_m,
                    })

            # ─── B3-e Articles via JP : re-rank B_art par cosinus
            if oblig:
                emb_idx_S = [pk_to_emb_idx[pk] for pk in B_art_pks if pk in pk_to_emb_idx]
                if emb_idx_S:
                    arr = np.asarray(emb_idx_S, dtype=np.int64)
                    order = np.argsort(-sim_art[qi, arr])
                    ranked = list(art_order[arr[order]])
                else:
                    ranked = []
                for K in KS_ARTICLES:
                    _m = rp_pair(ranked, oblig, oblig_ext, K)
                    rows.append({
                        "question_id": q["id"], "method": "B3-e_art_via_jp",
                        "k_in": k_in, "k_final": K, "n_S": len(ranked),
                        **_m,
                    })

            # ─── B4-a/b cross-modal articles
            if oblig:
                A_art_pks = top_art_pks
                for variant_name, S_pks in (
                    ("B4-a_cross_art_union", A_art_pks | B_art_pks),
                    ("B4-b_cross_art_inter", A_art_pks & B_art_pks),
                ):
                    emb_idx_S = [pk_to_emb_idx[pk] for pk in S_pks if pk in pk_to_emb_idx]
                    if emb_idx_S:
                        arr = np.asarray(emb_idx_S, dtype=np.int64)
                        order = np.argsort(-sim_art[qi, arr])
                        ranked = list(art_order[arr[order]])
                    else:
                        ranked = []
                    for K in KS_ARTICLES:
                        _m = rp_pair(ranked, oblig, oblig_ext, K)
                        rows.append({
                            "question_id": q["id"], "method": variant_name,
                            "k_in": k_in, "k_final": K, "n_S": len(ranked),
                            **_m,
                        })

            # ─── B4-c/d cross-modal JP
            if gold_jp:
                B_jp_ids = top_jp_ids
                for variant_name, S_ids in (
                    ("B4-c_cross_jp_union", A_jp_ids | B_jp_ids),
                    ("B4-d_cross_jp_inter", A_jp_ids & B_jp_ids),
                ):
                    emb_idx_S = [jpid_to_emb_idx[j] for j in S_ids if j in jpid_to_emb_idx]
                    if emb_idx_S:
                        arr = np.asarray(emb_idx_S, dtype=np.int64)
                        order = np.argsort(-sim_jp[qi, arr])
                        ranked = list(jp_order[arr[order]])
                    else:
                        ranked = []
                    for K in KS_JP:
                        _m = rp_pair(ranked, gold_jp, gold_jp_ext, K)
                        rows.append({
                            "question_id": q["id"], "method": variant_name,
                            "k_in": k_in, "k_final": K, "n_S": len(ranked),
                            **_m,
                        })

    print(f"  fin boucle (t={time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "recall_at_k.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}  ({len(df)} lignes)")

    print("\n══ Agrégats ───────────────────────────────────────────────")
    summary = {}
    print(f"\n  {'method':<28s} {'kin':>4s} {'K':>3s} {'n_q':>5s} {'mean_S':>7s} "
          f"{'rec':>6s} {'prec':>6s} {'rec_x':>6s} {'pr_x':>6s} {'r≥.5':>7s}")
    for (method, k_in, k_final), sub in df.groupby(["method", "k_in", "k_final"], dropna=False):
        n_q = len(sub)
        mean_S = sub["n_S"].mean()
        mean_r = sub["recall"].mean()
        mean_p = sub["precision"].mean()
        mean_rx = sub["recall_ext"].mean() if "recall_ext" in sub.columns else float("nan")
        mean_px = sub["precision_ext"].mean() if "precision_ext" in sub.columns else float("nan")
        n_pass = int((sub["recall"] >= 0.5).sum())
        pct = 100 * n_pass / max(n_q, 1)
        key = f"{method}|kin={k_in}|K={k_final}"
        summary[key] = {
            "n_q": n_q, "mean_S": float(mean_S),
            "mean_recall": float(mean_r), "mean_precision": float(mean_p),
            "mean_recall_ext": float(mean_rx), "mean_precision_ext": float(mean_px),
            "n_pass_50": n_pass, "pct_pass_50": pct,
        }
        kin_disp = str(int(k_in)) if pd.notna(k_in) else "-"
        print(f"  {method:<28s} {kin_disp:>4s} {k_final:>3d} {n_q:>5d} {mean_S:>7.1f} "
              f"{mean_r:>6.3f} {mean_p:>6.3f} {mean_rx:>6.3f} {mean_px:>6.3f} "
              f"{n_pass:>3d}({pct:>2.0f}%)")

    (OUT_DIR / "recall_at_k_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n✓ {OUT_DIR}/recall_at_k_summary.json")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
