"""Courbes diagnostic PPR :
  (a) convergence ||r^(t+1) - r^t||_1 vs itération, sweep α
  (b) sweep α fin sur M1 strict / étendu / JP

Sortie : fig_ppr_curves.png (2 panneaux côte à côte)
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
from etape1 import config  # noqa: E402

BENCH = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench/bench_global.json"
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
Q_EMB_CACHE = OUT_DIR / "questions_977_emb.npy"
Q_IDS_CACHE = OUT_DIR / "questions_977_ids.npy"
FIG_OUT = OUT_DIR / "fig_ppr_curves.png"
_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

K_OUT = 10
K_IN = 10
N_ITER_MAX = 30
ALPHAS_CONV = [0.50, 0.70, 0.85, 0.95]        # courbe convergence
ALPHAS_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95]  # sweep M1/M2
N_QUERIES_CONV = 100  # sous-échantillon pour la courbe de convergence


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


def m1(ranked, gt, k):
    if not gt: return float("nan")
    return len(set(ranked[:k]) & gt) / len(gt)


def m2(ranked, gt, k):
    if not gt: return float("nan")
    n = len(gt)
    pos = {a: i+1 for i, a in enumerate(ranked[:k])}
    ranks = [pos.get(a, k+1) for a in gt]
    x = sum(ranks) / n
    b_clip = min(n, k)
    denom = (b_clip+1)/2 - (k+1)
    if denom == 0: return float("nan")
    return (x - (k+1)) / denom


def row_normalize(M):
    rs = np.asarray(M.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0/rs) @ M


def ppr_with_log(P_T, s, alpha, n_iter):
    """Power iteration row-norm, retourne (r_final, list_residuals)."""
    r = s.copy()
    resids = []
    for _ in range(n_iter):
        r_new = alpha * (P_T @ r) + (1 - alpha) * s
        resids.append(float(np.abs(r_new - r).sum()))
        r = r_new
    return r, resids


def main():
    t0 = time.time()
    print("══ Chargement ─────────────────────────────────────────────")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    jp_emb = np.load(config.EMB_JP_SYNTHESE)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    n_jp, n_art = G.shape
    jpid_to_graphrow = {jid: i for i, jid in enumerate(jp_ids_graph)}
    artid_to_graphcol = {aid: i for i, aid in enumerate(article_ids_graph)}

    pool_art_set = set(art_order.tolist())
    pool_jp_set = set(jp_order.tolist())

    N_total = n_jp + n_art
    G_full = sp.bmat([[None, G], [G.T, None]], format="csr")
    P = row_normalize(G_full)
    P_T = P.T.tocsr()
    print(f"  P_T ready  (t={time.time()-t0:.1f}s)")

    art_pool_graph_idx = np.array(
        [n_jp + artid_to_graphcol[pk] for pk in art_order if pk in artid_to_graphcol],
        dtype=np.int64)
    art_pool_pks = np.array([pk for pk in art_order if pk in artid_to_graphcol])
    jp_pool_graph_idx = np.array(
        [jpid_to_graphrow[j] for j in jp_order if j in jpid_to_graphrow],
        dtype=np.int64)
    jp_pool_ids = np.array([j for j in jp_order if j in jpid_to_graphrow])

    questions = load_cohort_977()
    Q_emb = np.load(Q_EMB_CACHE)
    cached_qids = np.load(Q_IDS_CACHE, allow_pickle=True).tolist()
    qid_set = set(cached_qids)
    questions = [q for q in questions if q["id"] in qid_set]
    qid_to_emb = {qid: i for i, qid in enumerate(cached_qids)}
    Q = np.asarray([Q_emb[qid_to_emb[q["id"]]] for q in questions])
    print(f"  {len(questions)} questions")
    pourvoi_map = build_pourvoi_map()

    sim_art = Q @ art_emb.T
    sim_jp = Q @ jp_emb.T

    # ─────────────────────────────────────────────────────────────
    # (a) Convergence : 100 questions × 4 α
    # ─────────────────────────────────────────────────────────────
    print("\n══ (a) Convergence ────────────────────────────────────────")
    rng = np.random.default_rng(seed=42)
    sample_idx = rng.choice(len(questions), size=min(N_QUERIES_CONV, len(questions)),
                            replace=False)
    conv_data = {alpha: np.zeros(N_ITER_MAX) for alpha in ALPHAS_CONV}
    for qi in sample_idx:
        top_a = np.argpartition(-sim_art[qi], K_IN)[:K_IN]
        top_j = np.argpartition(-sim_jp[qi], K_IN)[:K_IN]
        s = np.zeros(N_total)
        for pk, sim in zip(art_order[top_a], sim_art[qi, top_a]):
            col = artid_to_graphcol.get(pk)
            if col is not None: s[n_jp+col] = max(float(sim), 0.0)
        for jid, sim in zip(jp_order[top_j], sim_jp[qi, top_j]):
            row = jpid_to_graphrow.get(jid)
            if row is not None: s[row] = max(float(sim), 0.0)
        if s.sum() == 0: continue
        s /= s.sum()
        for alpha in ALPHAS_CONV:
            _, resids = ppr_with_log(P_T, s, alpha, N_ITER_MAX)
            conv_data[alpha] += np.array(resids)
    for alpha in ALPHAS_CONV:
        conv_data[alpha] /= len(sample_idx)
    print(f"  done (t={time.time()-t0:.1f}s)")

    # ─────────────────────────────────────────────────────────────
    # (b) Sweep α fin : 971 q × 6 α
    # ─────────────────────────────────────────────────────────────
    print("\n══ (b) Sweep α ────────────────────────────────────────────")
    sweep_results = {a: {"m1s_art": [], "m1e_art": [], "m1_jp": [],
                         "m2s_art": [], "m2e_art": [], "m2_jp": []}
                     for a in ALPHAS_SWEEP}
    for qi, q in enumerate(questions):
        if qi % 200 == 0:
            print(f"  q {qi}/{len(questions)}  (t={time.time()-t0:.1f}s)")

        gt_s = q["gt_strict"] & pool_art_set
        gt_e = q["gt_ext"] & pool_art_set
        gold_jp = ({jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
                   & pool_jp_set)
        if not gt_s and not gold_jp:
            continue

        top_a = np.argpartition(-sim_art[qi], K_IN)[:K_IN]
        top_j = np.argpartition(-sim_jp[qi], K_IN)[:K_IN]
        s = np.zeros(N_total)
        for pk, sim in zip(art_order[top_a], sim_art[qi, top_a]):
            col = artid_to_graphcol.get(pk)
            if col is not None: s[n_jp+col] = max(float(sim), 0.0)
        for jid, sim in zip(jp_order[top_j], sim_jp[qi, top_j]):
            row = jpid_to_graphrow.get(jid)
            if row is not None: s[row] = max(float(sim), 0.0)
        if s.sum() == 0: continue
        s /= s.sum()

        for alpha in ALPHAS_SWEEP:
            r, _ = ppr_with_log(P_T, s, alpha, N_ITER_MAX)
            r_art = r[art_pool_graph_idx]
            top_a_out = np.argsort(-r_art)[:K_OUT]
            ranked_art = list(art_pool_pks[top_a_out])
            r_jp = r[jp_pool_graph_idx]
            top_j_out = np.argsort(-r_jp)[:K_OUT]
            ranked_jp = list(jp_pool_ids[top_j_out])

            sweep_results[alpha]["m1s_art"].append(m1(ranked_art, gt_s, K_OUT))
            sweep_results[alpha]["m1e_art"].append(m1(ranked_art, gt_e, K_OUT))
            sweep_results[alpha]["m1_jp"].append(m1(ranked_jp, gold_jp, K_OUT))
            sweep_results[alpha]["m2s_art"].append(m2(ranked_art, gt_s, K_OUT))
            sweep_results[alpha]["m2e_art"].append(m2(ranked_art, gt_e, K_OUT))
            sweep_results[alpha]["m2_jp"].append(m2(ranked_jp, gold_jp, K_OUT))

    sweep_means = {}
    for alpha in ALPHAS_SWEEP:
        sweep_means[alpha] = {k: float(np.nanmean(v)) for k, v in sweep_results[alpha].items()}
    print(f"  done (t={time.time()-t0:.1f}s)")
    print("\n  α      M1s_art M1e_art M1_jp  M2s_art M2e_art M2_jp")
    for alpha in ALPHAS_SWEEP:
        m = sweep_means[alpha]
        print(f"  {alpha:.2f}   {m['m1s_art']:.3f}  {m['m1e_art']:.3f}  {m['m1_jp']:.3f}  "
              f"{m['m2s_art']:.3f}  {m['m2e_art']:.3f}  {m['m2_jp']:.3f}")

    # Dump JSON
    (OUT_DIR / "ppr_curves_data.json").write_text(json.dumps({
        "convergence": {str(a): conv_data[a].tolist() for a in ALPHAS_CONV},
        "sweep_alpha": {str(a): sweep_means[a] for a in ALPHAS_SWEEP},
    }, ensure_ascii=False, indent=2))

    # ─────────────────────────────────────────────────────────────
    # Figure 2 panneaux
    # ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    # Panneau gauche : convergence
    ax = axes[0]
    colors = ["#888", "#4c8b9b", "#295c9b", "#8b2e2e"]
    for alpha, color in zip(ALPHAS_CONV, colors):
        ax.semilogy(range(1, N_ITER_MAX+1), conv_data[alpha], "-o",
                    color=color, markersize=3, lw=1.4, label=f"α = {alpha}")
    ax.set_xlabel("itération t")
    ax.set_ylabel(r"$\|r^{(t+1)} - r^{(t)}\|_1$ (échelle log)")
    ax.set_title(f"Convergence PPR (moyenne sur {N_QUERIES_CONV} questions)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.axvline(20, color="gray", ls=":", alpha=0.5)
    ax.text(20, ax.get_ylim()[1]*0.5, " N_iter\n actuel", fontsize=8, color="gray")

    # Panneau droit : sweep α
    ax = axes[1]
    alphas_arr = np.array(ALPHAS_SWEEP)
    m1s = [sweep_means[a]["m1s_art"] for a in ALPHAS_SWEEP]
    m1e = [sweep_means[a]["m1e_art"] for a in ALPHAS_SWEEP]
    m1j = [sweep_means[a]["m1_jp"] for a in ALPHAS_SWEEP]
    ax.plot(alphas_arr, m1s, "-o", color="#295c9b", label="M1 art. strict")
    ax.plot(alphas_arr, m1e, "-s", color="#8b4513", label="M1 art. étendu")
    ax.plot(alphas_arr, m1j, "-^", color="#2f7a3a", label="M1 JP")
    # Lignes de référence
    ax.axhline(0.711, color="#295c9b", ls="--", alpha=0.4, lw=0.9)
    ax.text(0.07, 0.715, "B3-e strict 0.711", fontsize=7, color="#295c9b")
    ax.axhline(0.399, color="#8b4513", ls="--", alpha=0.4, lw=0.9)
    ax.text(0.07, 0.405, "B3-e étendu 0.399", fontsize=7, color="#8b4513")
    ax.axhline(0.416, color="#2f7a3a", ls="--", alpha=0.4, lw=0.9)
    ax.text(0.07, 0.422, "B4-e JP 0.416", fontsize=7, color="#2f7a3a")
    ax.set_xlabel("α (damping)")
    ax.set_ylabel("M1 (recall@10)")
    ax.set_title("Sweep α — M1 par modalité (971 questions)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.8)

    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"\n✓ {FIG_OUT}")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
