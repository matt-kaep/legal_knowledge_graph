"""Sweep PPR sur K_IN × SEED_VARIANT × ALPHA (norme row uniquement).

Tâche #27 (Week-10). On reprend la structure du script 20_ppr_naive.py et on
étend le sweep selon trois dimensions :
  - K_IN ∈ {5, 10, 20, 50}
  - SEED_VARIANT ∈ {"art_only", "jp_only", "both"}
  - ALPHA ∈ {0.5, 0.7, 0.85, 0.95}

→ 48 variantes / question × 971 q.

Sym-norm déjà confirmé collapsé sur cosine (cf. décisions Week-9), donc on
n'évalue ici que row-norm. Power iteration identique au script 20 :
20 itérations max, tol 1e-7, r ← α P^T r + (1-α) s.

Sortie :
  - data/global_bench/ppr_kin_sweep_eval.csv   (1 ligne / qid / variante)
  - data/global_bench/ppr_kin_sweep_summary.json (agrégats par triplet)
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

K_OUT = 10
K_INS = [5, 10, 20, 50]
SEED_VARIANTS = ["art_only", "jp_only", "both"]
ALPHAS = [0.5, 0.7, 0.85, 0.95]
N_ITER = 20
TOL = 1e-7

OUT_CSV = OUT_DIR / "ppr_kin_sweep_eval.csv"
OUT_SUMMARY = OUT_DIR / "ppr_kin_sweep_summary.json"


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


def row_normalize(Mat: sp.csr_matrix) -> sp.csr_matrix:
    rs = np.asarray(Mat.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ Mat


def ppr_power_iteration(P: sp.csr_matrix, s: np.ndarray, alpha: float,
                        n_iter: int = N_ITER, tol: float = TOL) -> tuple[np.ndarray, int]:
    """r ← α P^T r + (1-α) s  (row-norm => PageRank standard)."""
    PT = P.T.tocsr()
    r = s.copy()
    for t in range(n_iter):
        r_new = alpha * (PT @ r) + (1 - alpha) * s
        if np.abs(r_new - r).sum() < tol:
            return r_new, t + 1
        r = r_new
    return r, n_iter


def build_seed(variant: str, n_jp: int, N_total: int,
               top_art_pks, art_sims, top_jp_ids, jp_sims,
               artid_to_graphcol, jpid_to_graphrow) -> np.ndarray | None:
    """Construit s normalisé (None si somme = 0)."""
    s = np.zeros(N_total)
    if variant in ("art_only", "both"):
        for pk, sim in zip(top_art_pks, art_sims):
            col = artid_to_graphcol.get(pk)
            if col is not None:
                s[n_jp + col] += max(float(sim), 0.0)
    if variant in ("jp_only", "both"):
        for jid, sim in zip(top_jp_ids, jp_sims):
            row = jpid_to_graphrow.get(jid)
            if row is not None:
                s[row] += max(float(sim), 0.0)
    ssum = s.sum()
    if ssum == 0:
        return None
    s /= ssum
    return s


def main(limit_q: int | None = None) -> int:
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

    jpid_to_graphrow = {jid: i for i, jid in enumerate(jp_ids_graph)}
    artid_to_graphcol = {aid: i for i, aid in enumerate(article_ids_graph)}
    pool_articles_set = set(art_order.tolist())
    pool_jp_set = set(jp_order.tolist())

    print("\n══ Block-bipartite + row-normalize ────────────────────────")
    N_total = n_jp + n_art
    G_full = sp.bmat([[None, G], [G.T, None]], format="csr")
    print(f"  G_full {G_full.shape}  nnz {G_full.nnz:,}")
    P_row = row_normalize(G_full)
    print(f"  P_row ready  (t={time.time()-t0:.1f}s)")

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

    print("\n══ Chargement cohorte ─────────────────────────────────────")
    questions = load_cohort_977()
    pourvoi_map = build_pourvoi_map()
    Q_emb = np.load(Q_EMB_CACHE)
    cached_qids = np.load(Q_IDS_CACHE, allow_pickle=True).tolist()
    qid_set = set(cached_qids)
    questions = [q for q in questions if q["id"] in qid_set]
    qid_to_emb = {qid: i for i, qid in enumerate(cached_qids)}
    Q = np.asarray([Q_emb[qid_to_emb[q["id"]]] for q in questions])
    if limit_q is not None:
        questions = questions[:limit_q]
        Q = Q[:limit_q]
    print(f"  questions évaluées : {len(questions)}")

    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T
    sim_jp = Q @ jp_emb.T
    print(f"  sim shapes {sim_art.shape} / {sim_jp.shape}  (t={time.time()-t0:.1f}s)")

    K_MAX = max(K_INS)
    print(f"\n══ Pré-calcul top-{K_MAX} cosine seeds ─────────────────")
    top_art_full = np.argpartition(-sim_art, K_MAX, axis=1)[:, :K_MAX]
    top_jp_full = np.argpartition(-sim_jp, K_MAX, axis=1)[:, :K_MAX]
    # Tri pour que top-K_in (K_in < K_MAX) = prefix correct des K_in meilleurs.
    for qi in range(len(questions)):
        order = np.argsort(-sim_art[qi, top_art_full[qi]])
        top_art_full[qi] = top_art_full[qi][order]
        order_j = np.argsort(-sim_jp[qi, top_jp_full[qi]])
        top_jp_full[qi] = top_jp_full[qi][order_j]
    print(f"  pré-calcul fait (t={time.time()-t0:.1f}s)")

    print("\n══ Boucle PPR sweep ───────────────────────────────────────")
    rows = []
    n_skip = 0
    for qi, q in enumerate(questions):
        if qi % 100 == 0:
            print(f"  q {qi}/{len(questions)}  (t={time.time()-t0:.1f}s, rows={len(rows)})")

        gt_s = q["gt_strict"] & pool_articles_set
        gt_e = q["gt_ext"] & pool_articles_set
        gold_jp = ({jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
                   & pool_jp_set)
        if not gt_s and not gold_jp:
            n_skip += 1
            continue

        for k_in in K_INS:
            top_art_emb_idx = top_art_full[qi, :k_in]
            top_art_pks = art_order[top_art_emb_idx]
            art_sims = sim_art[qi, top_art_emb_idx]
            top_jp_emb_idx = top_jp_full[qi, :k_in]
            top_jp_ids_arr = jp_order[top_jp_emb_idx]
            jp_sims = sim_jp[qi, top_jp_emb_idx]

            for variant in SEED_VARIANTS:
                s = build_seed(variant, n_jp, N_total,
                               top_art_pks, art_sims, top_jp_ids_arr, jp_sims,
                               artid_to_graphcol, jpid_to_graphrow)
                if s is None:
                    continue
                for alpha in ALPHAS:
                    r, n_used = ppr_power_iteration(P_row, s, alpha)

                    r_art_pool = r[art_pool_graph_idx]
                    top_art_idx = np.argsort(-r_art_pool)[:K_OUT]
                    ranked_art = list(art_pool_pks[top_art_idx])

                    r_jp_pool = r[jp_pool_graph_idx]
                    top_jp_idx = np.argsort(-r_jp_pool)[:K_OUT]
                    ranked_jp = list(jp_pool_ids[top_jp_idx])

                    art_metrics = M.panel_strict_ext(ranked_art, gt_s, gt_e, K_OUT)
                    art_metrics = {f"{k}_art": v for k, v in art_metrics.items()}
                    jp_metrics = {f"{k}_jp": v for k, v in
                                  M.all_metrics(ranked_jp, gold_jp, K_OUT).items()}
                    rows.append({
                        "qid": q["id"],
                        "k_in": k_in,
                        "seed_variant": variant,
                        "alpha": alpha,
                        "n_iter_used": n_used,
                        **art_metrics,
                        **jp_metrics,
                    })

    print(f"\n  skipped (GT vide) : {n_skip}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n✓ {OUT_CSV}  ({len(df)} lignes)")

    print("\n══ Agrégats par (k_in, seed, α) ───────────────────────────")
    cols = (
        [f"{m}_strict_art" for m in M.METRIC_NAMES] +
        [f"{m}_ext_art"    for m in M.METRIC_NAMES] +
        [f"{m}_jp"         for m in M.METRIC_NAMES]
    )
    summary = {}
    for (k_in, variant, alpha), sub in df.groupby(["k_in", "seed_variant", "alpha"]):
        means = {c: float(sub[c].mean()) for c in cols}
        iters = float(sub["n_iter_used"].mean())
        summary[f"k{k_in}|{variant}|a{alpha}"] = {
            "k_in": int(k_in), "seed_variant": variant, "alpha": float(alpha),
            "n_iter_avg": iters, "n_rows": int(len(sub)), **means
        }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✓ {OUT_SUMMARY}")

    s_df = pd.DataFrame(summary.values())
    for metric in ["m1_ext_art", "ndcg_ext_art", "mrr_strict_art", "ndcg_jp", "hit_ext_art"]:
        if metric not in s_df.columns:
            continue
        top = s_df.sort_values(metric, ascending=False).head(5)
        print(f"\n  TOP-5 sur {metric}:")
        for _, r in top.iterrows():
            print(f"    k_in={int(r['k_in']):>2} {r['seed_variant']:>9} α={r['alpha']:.2f}  "
                  f"{metric}={r[metric]:.4f}")

    print(f"\n  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    limit = None
    if len(sys.argv) > 1 and sys.argv[1].startswith("--limit="):
        limit = int(sys.argv[1].split("=", 1)[1])
        print(f"[mode sanity check : limit={limit}]")
    sys.exit(main(limit))
