"""PPR naïf sur graphe Art↔JP bipartite, eval M1/M2 sur cohorte 977.

Version naïve = pas de Q-node ajouté, pas de normalisation symétrique.
Seed = top-K cosine articles + JPs (poids = sim cosinus). Power iteration.

Graphe : G de shape (n_jp=118 112, n_art=87 821), bipartite.
On construit la matrice symétrique block-bipartite :
    G_full = [[0,   G],
              [G^T, 0]]                                # (n_jp + n_art)^2
Row-normalize → P. Indices : [0..n_jp-1] = JP, [n_jp..n_jp+n_art-1] = articles.

Variantes :
  - seed = top-K_in cosine articles + JPs
  - α ∈ {0.5, 0.7, 0.85}
  - K_in = 10

Sorties : top-10 articles, top-10 JPs après PPR (filtrés au pool BGE-M3).
Comparaison vs B3-e (art) et B4-e (JP).
"""
from __future__ import annotations
import json, re, sys, time
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

BENCH = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench/bench_global.json"
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
Q_EMB_CACHE = OUT_DIR / "questions_977_emb.npy"
Q_IDS_CACHE = OUT_DIR / "questions_977_ids.npy"
_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

K_OUT = 10           # top-K final pour articles + JP
K_IN = 10            # nb seeds cosine
ALPHAS = [0.5, 0.7, 0.85, 0.95]   # 0.95 = champion étendu (handoff M3)
N_ITER = 20
TOL = 1e-7

# Méthodes PPR dont on dumpe les rankings pour M3 (script 23) : row-norm
# uniquement (sym collapse sur cosine, droppée), α champions {0.85, 0.95}.
M3_DUMP_PPR = {("row", 0.85), ("row", 0.95)}


def load_cohort_977() -> list[dict]:
    d = json.loads(BENCH.read_text())
    out = []
    for q in d["questions"]:
        arts = q.get("articles_attendus") or []
        pourvois = q.get("pourvois_cc") or []
        if not arts or not pourvois or q.get("n_jp_resolues", 0) < 1:
            continue
        out.append({
            "id": q["qid"],
            "gt_strict": set(arts),
            "gt_ext": set(q.get("articles_attendus_etendu") or arts),
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


# Métriques factorisées dans metrics.py (chantier 1 Week-10) — alias rétrocompat
m1_recall = M.m1_recall
m2_rank   = M.m2_rank


def ranking_rows(qid, method, k_in, modality, ranked, k):
    """Lignes (qid, method, k_in, modality, rank, item_id) pour le top-k.

    Même schéma que script 18 → rankings.parquet commun consommé par M3.
    """
    return [
        {"qid": qid, "method": method, "k_in": k_in, "modality": modality,
         "rank": r + 1, "item_id": str(item)}
        for r, item in enumerate(ranked[:k])
    ]


def row_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    """Row-stochastic : chaque ligne somme à 1 (ou reste à 0). PageRank standard."""
    rs = np.asarray(M.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ M


def sym_normalize(M: sp.csr_matrix) -> sp.csr_matrix:
    """Normalisation symétrique : D^(-1/2) M D^(-1/2). Style LightGCN/GCN.
    Pénalise les arêtes vers les hubs proportionnellement à √degré.
    Plus row-stochastic après ⇒ propagation = diffusion d'énergie, pas marche aléatoire.
    """
    d = np.asarray(M.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D = sp.diags(d_inv_sqrt)
    return D @ M @ D


def ppr_power_iteration(P: sp.csr_matrix, s: np.ndarray, alpha: float,
                        n_iter: int = N_ITER, tol: float = TOL,
                        symmetric: bool = False) -> tuple[np.ndarray, int]:
    """Itère r ← α P^? r + (1-α) s.
    - row-stochastic : PageRank standard, r ← α P^T r + (1-α) s
    - symétrique     : P = P^T, r ← α P r + (1-α) s
    """
    Mat = P if symmetric else P.T.tocsr()
    r = s.copy()
    for t in range(n_iter):
        r_new = alpha * (Mat @ r) + (1 - alpha) * s
        if np.abs(r_new - r).sum() < tol:
            return r_new, t + 1
        r = r_new
    return r, n_iter


def main() -> int:
    t0 = time.time()
    print("══ Chargement graphe + embeddings ─────────────────────────")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    jp_emb = np.load(config.EMB_JP_SYNTHESE)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    n_jp, n_art = G.shape
    print(f"  G {G.shape}  nnz {G.nnz:,}")

    # Index helpers
    jpid_to_graphrow = {jid: i for i, jid in enumerate(jp_ids_graph)}
    artid_to_graphcol = {aid: i for i, aid in enumerate(article_ids_graph)}
    pool_articles_set = set(art_order.tolist())
    pool_jp_set = set(jp_order.tolist())
    pk_to_emb_idx = {pk: i for i, pk in enumerate(art_order)}
    jpid_to_emb_idx = {jid: i for i, jid in enumerate(jp_order)}

    # Construction graphe symétrique block-bipartite
    # N_total = n_jp + n_art ; indices [0..n_jp-1] = JP, [n_jp..n_jp+n_art-1] = articles
    print("\n══ Block-bipartite + row-normalize ────────────────────────")
    N_total = n_jp + n_art
    # data = G ∈ (n_jp, n_art)
    G_full = sp.bmat(
        [[None, G], [G.T, None]], format="csr"
    )
    print(f"  G_full {G_full.shape}  nnz {G_full.nnz:,}")
    P_row = row_normalize(G_full)
    P_sym = sym_normalize(G_full)
    print(f"  P_row + P_sym ready  (t={time.time()-t0:.1f}s)")

    # Pool indices dans le graphe (pour ranking)
    art_pool_graph_idx = np.array(
        [n_jp + artid_to_graphcol[pk] for pk in art_order if pk in artid_to_graphcol],
        dtype=np.int64
    )
    art_pool_pks = np.array(
        [pk for pk in art_order if pk in artid_to_graphcol]
    )
    jp_pool_graph_idx = np.array(
        [jpid_to_graphrow[j] for j in jp_order if j in jpid_to_graphrow],
        dtype=np.int64
    )
    jp_pool_ids = np.array(
        [j for j in jp_order if j in jpid_to_graphrow]
    )
    print(f"  pool art dans graphe : {len(art_pool_graph_idx):,}")
    print(f"  pool JP  dans graphe : {len(jp_pool_graph_idx):,}")

    # Cohorte + cache embeddings questions
    print("\n══ Chargement cohorte ─────────────────────────────────────")
    questions = load_cohort_977()
    print(f"  cohorte brute : {len(questions)} questions")
    pourvoi_map = build_pourvoi_map()

    Q_emb = np.load(Q_EMB_CACHE)
    cached_qids = np.load(Q_IDS_CACHE, allow_pickle=True).tolist()
    qid_set = set(cached_qids)
    questions = [q for q in questions if q["id"] in qid_set]
    qid_to_emb = {qid: i for i, qid in enumerate(cached_qids)}
    Q = np.asarray([Q_emb[qid_to_emb[q["id"]]] for q in questions])
    print(f"  questions évaluées : {len(questions)}")

    # Cosine sim → pour seeds top-K_in
    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T
    sim_jp = Q @ jp_emb.T
    print(f"  sim_art {sim_art.shape}  sim_jp {sim_jp.shape}  (t={time.time()-t0:.1f}s)")

    # ─────────────────────────────────────────────────────────────
    # Boucle d'éval
    # ─────────────────────────────────────────────────────────────
    print("\n══ Boucle PPR ─────────────────────────────────────────────")
    rows = []
    rankings = []  # dump rankings PPR champions -> rankings.parquet (M3)
    for qi, q in enumerate(questions):
        if qi % 100 == 0:
            print(f"  q {qi}/{len(questions)}  (t={time.time()-t0:.1f}s)")

        gt_s = q["gt_strict"] & pool_articles_set
        gt_e = q["gt_ext"] & pool_articles_set
        gold_jp = ({jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
                   & pool_jp_set)
        if not gt_s and not gold_jp:
            continue

        # Top-K_in cosine seeds
        top_art_emb_idx = np.argpartition(-sim_art[qi], K_IN)[:K_IN]
        top_art_pks = art_order[top_art_emb_idx]
        art_sim_values = sim_art[qi, top_art_emb_idx]
        top_jp_emb_idx = np.argpartition(-sim_jp[qi], K_IN)[:K_IN]
        top_jp_ids = jp_order[top_jp_emb_idx]
        jp_sim_values = sim_jp[qi, top_jp_emb_idx]

        # Construction seed s (taille N_total) : poids = sim (clippé positif)
        s = np.zeros(N_total)
        for pk, sim in zip(top_art_pks, art_sim_values):
            col = artid_to_graphcol.get(pk)
            if col is not None:
                s[n_jp + col] = max(float(sim), 0.0)
        for jid, sim in zip(top_jp_ids, jp_sim_values):
            row = jpid_to_graphrow.get(jid)
            if row is not None:
                s[row] = max(float(sim), 0.0)
        ssum = s.sum()
        if ssum == 0:
            continue
        s /= ssum

        for norm_name, P_mat, is_sym in [("row", P_row, False), ("sym", P_sym, True)]:
            for alpha in ALPHAS:
                r, n_used = ppr_power_iteration(P_mat, s, alpha, symmetric=is_sym)

                r_art_pool = r[art_pool_graph_idx]
                top_art_idx = np.argsort(-r_art_pool)[:K_OUT]
                ranked_art = list(art_pool_pks[top_art_idx])

                r_jp_pool = r[jp_pool_graph_idx]
                top_jp_idx = np.argsort(-r_jp_pool)[:K_OUT]
                ranked_jp = list(jp_pool_ids[top_jp_idx])

                if (norm_name, alpha) in M3_DUMP_PPR:
                    mname = f"PPR-{norm_name}-a{alpha}"
                    rankings.extend(
                        ranking_rows(q["id"], mname, K_IN, "art", ranked_art, K_OUT))
                    rankings.extend(
                        ranking_rows(q["id"], mname, K_IN, "jp", ranked_jp, K_OUT))

                # Panel complet : 5 métriques × {strict, ext} côté art = 10 cols art,
                # + 5 cols côté JP (pas de strict/ext). Suffixés _art / _jp.
                art_metrics = M.panel_strict_ext(ranked_art, gt_s, gt_e, K_OUT)
                art_metrics = {f"{k}_art": v for k, v in art_metrics.items()}
                jp_metrics  = {f"{k}_jp": v for k, v in
                               M.all_metrics(ranked_jp, gold_jp, K_OUT).items()}
                rows.append({
                    "qid": q["id"], "norm": norm_name, "alpha": alpha,
                    "n_iter_used": n_used,
                    **art_metrics,
                    **jp_metrics,
                })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "ppr_naive_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}  ({len(df)} lignes)")

    # Dump rankings PPR champions vers le parquet commun (M3, script 23).
    # On préserve les méthodes des autres producteurs (B* de 18) et n'écrase
    # que les lignes PPR-*.
    rk = pd.DataFrame(rankings)
    rank_path = OUT_DIR / "rankings.parquet"
    if rank_path.exists():
        prev = pd.read_parquet(rank_path)
        prev = prev[~prev["method"].isin(rk["method"].unique())]
        rk = pd.concat([prev, rk], ignore_index=True)
    rk.to_parquet(rank_path, index=False)
    print(f"✓ {rank_path}  ({len(rk)} lignes rankings, "
          f"méthodes={sorted(rk['method'].unique())})")

    # Agrégats par (norm, alpha)
    print("\n══ Agrégats par (norm, α) ─────────────────────────────────")
    cols = (
        [f"{m}_strict_art" for m in M.METRIC_NAMES] +
        [f"{m}_ext_art"    for m in M.METRIC_NAMES] +
        [f"{m}_jp"         for m in M.METRIC_NAMES]
    )
    # Affichage compact : M1/Hit/NDCG côté art × {strict, ext} + JP (M1/Hit/MRR/NDCG)
    print(f"  {'norm':>5s} {'α':>5s} {'iter':>5s} | "
          f"{'M1s_a':>6s} {'Hits_a':>6s} {'NDCs_a':>6s} | "
          f"{'M1e_a':>6s} {'Hite_a':>6s} {'NDCe_a':>6s} | "
          f"{'M1_jp':>6s} {'Hit_jp':>6s} {'MRR_jp':>6s} {'NDC_jp':>6s}")
    print("  " + "─" * 110)
    summary = {}
    for (norm, alpha), sub in df.groupby(["norm", "alpha"]):
        means = {c: float(sub[c].mean()) for c in cols}
        iters = float(sub["n_iter_used"].mean())
        summary[f"{norm}|alpha={alpha}"] = {"n_iter_avg": iters, **means}
        print(f"  {norm:>5s} {alpha:>5.2f} {iters:>5.1f} | "
              f"{means['m1_strict_art']:>6.3f} {means['hit_strict_art']:>6.3f} "
              f"{means['ndcg_strict_art']:>6.3f} | "
              f"{means['m1_ext_art']:>6.3f} {means['hit_ext_art']:>6.3f} "
              f"{means['ndcg_ext_art']:>6.3f} | "
              f"{means['m1_jp']:>6.3f} {means['hit_jp']:>6.3f} "
              f"{means['mrr_jp']:>6.3f} {means['ndcg_jp']:>6.3f}")

    (OUT_DIR / "ppr_naive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n✓ {OUT_DIR}/ppr_naive_summary.json")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
