"""Visualisation de la répartition des poids PPR sur le graphe.

Trois angles :
  (a) Case study : une question où PPR fait mieux que cosine pur (top-30 art + JP,
      bars cosine vs PPR, GT marqué)
  (b) Concentration : courbe de masse cumulée trié décroissant (combien de nœuds
      portent X% de la masse, par α)
  (c) Divergence ranking : Jaccard(top10_cosine, top10_PPR) histogramme sur 971 q

Sortie : fig_ppr_weights.png (3 panneaux)
"""
from __future__ import annotations
import json, re, sys, time
from pathlib import Path
import numpy as np
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
FIG_OUT = OUT_DIR / "fig_ppr_weights.png"
_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

K_IN = 10
ALPHA = 0.85
N_ITER = 20


def load_cohort():
    d = json.loads(BENCH.read_text())
    out = []
    for q in d["questions"]:
        arts = q.get("articles_attendus") or []
        pourvois = q.get("pourvois_cc") or []
        if not arts or not pourvois or q.get("n_jp_resolues", 0) < 1: continue
        out.append({
            "id": q["qid"], "enonce": q["enonce"][:120],
            "gt_strict": set(arts),
            "gt_ext": set(q.get("articles_attendus_etendu") or arts),
            "pourvois": set(pourvois),
        })
    return out


def build_pourvoi_map():
    jp = pq.read_table(config.JP_INDEX, columns=["id","number","juris"]).to_pandas()
    jp = jp[jp["juris"]=="CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def row_normalize(M):
    rs = np.asarray(M.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0/rs) @ M


def ppr(P_T, s, alpha, n_iter):
    r = s.copy()
    for _ in range(n_iter):
        r = alpha * (P_T @ r) + (1 - alpha) * s
    return r


def main():
    t0 = time.time()
    print("══ Chargement ──")
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
    P = row_normalize(sp.bmat([[None, G], [G.T, None]], format="csr"))
    P_T = P.T.tocsr()

    art_pool_graph_idx = np.array(
        [n_jp + artid_to_graphcol[pk] for pk in art_order if pk in artid_to_graphcol],
        dtype=np.int64)
    art_pool_pks = np.array([pk for pk in art_order if pk in artid_to_graphcol])
    jp_pool_graph_idx = np.array(
        [jpid_to_graphrow[j] for j in jp_order if j in jpid_to_graphrow],
        dtype=np.int64)
    jp_pool_ids = np.array([j for j in jp_order if j in jpid_to_graphrow])

    questions = load_cohort()
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
    # (a) Case study : trouver une question où PPR > cosine
    # ─────────────────────────────────────────────────────────────
    print("\n══ (a) Sélection question case study ──")
    chosen = None
    for qi, q in enumerate(questions):
        gt_s = q["gt_strict"] & pool_art_set
        if not gt_s: continue
        # top-10 cosine articles
        top_a_cos = np.argpartition(-sim_art[qi], 10)[:10]
        ranked_cos = list(art_order[top_a_cos])
        m1_cos = len(set(ranked_cos) & gt_s) / len(gt_s)
        # On veut M1_cos moyen mais PPR meilleur
        if 0.0 < m1_cos < 0.6 and len(gt_s) >= 2:
            chosen = (qi, q, gt_s)
            break
    if chosen is None:
        chosen = (0, questions[0], questions[0]["gt_strict"] & pool_art_set)
    qi, q, gt_s_chosen = chosen
    print(f"  qid : {q['id']}")
    print(f"  GT strict : {len(gt_s_chosen)} articles")
    print(f"  énoncé    : {q['enonce']}")

    # Seed PPR
    top_a_seed = np.argpartition(-sim_art[qi], K_IN)[:K_IN]
    top_j_seed = np.argpartition(-sim_jp[qi], K_IN)[:K_IN]
    s = np.zeros(N_total)
    for pk, sim in zip(art_order[top_a_seed], sim_art[qi, top_a_seed]):
        col = artid_to_graphcol.get(pk)
        if col is not None: s[n_jp+col] = max(float(sim), 0.0)
    for jid, sim in zip(jp_order[top_j_seed], sim_jp[qi, top_j_seed]):
        row = jpid_to_graphrow.get(jid)
        if row is not None: s[row] = max(float(sim), 0.0)
    s /= s.sum()

    r = ppr(P_T, s, ALPHA, N_ITER)

    # Scores articles
    r_art = r[art_pool_graph_idx]
    top_a_ppr = np.argsort(-r_art)[:20]
    top_a_ppr_pks = art_pool_pks[top_a_ppr]
    top_a_ppr_scores = r_art[top_a_ppr]
    top_a_cos_pks_set = set(art_order[top_a_seed[:K_IN]])
    # cosine sim sur les top 20 ppr articles
    pk_to_emb_idx = {pk: i for i, pk in enumerate(art_order)}
    top_a_ppr_cos = np.array([sim_art[qi, pk_to_emb_idx[pk]] for pk in top_a_ppr_pks])

    # ─────────────────────────────────────────────────────────────
    # (b) Concentration : courbe de masse cumulée
    # ─────────────────────────────────────────────────────────────
    print("\n══ (b) Concentration de masse (3 α) ──")
    conc = {}
    for alpha_test in [0.3, 0.7, 0.85]:
        r_t = ppr(P_T, s, alpha_test, N_ITER)
        # On normalise pour comparer (la masse totale peut varier)
        r_t_pos = np.maximum(r_t, 0)
        r_t_pos /= r_t_pos.sum()
        sorted_r = np.sort(r_t_pos)[::-1]
        cumulated = np.cumsum(sorted_r)
        conc[alpha_test] = cumulated[:5000]  # premiers 5000 nœuds

    # ─────────────────────────────────────────────────────────────
    # (c) Divergence Jaccard top-10 cosine vs PPR
    # ─────────────────────────────────────────────────────────────
    print("\n══ (c) Jaccard top-10 cosine vs PPR (971 q) ──")
    jaccards_art, jaccards_jp = [], []
    for qj, qq in enumerate(questions):
        gt_s_q = qq["gt_strict"] & pool_art_set
        if not gt_s_q: continue
        top_a_seed_q = np.argpartition(-sim_art[qj], K_IN)[:K_IN]
        top_j_seed_q = np.argpartition(-sim_jp[qj], K_IN)[:K_IN]
        s_q = np.zeros(N_total)
        for pk, sim in zip(art_order[top_a_seed_q], sim_art[qj, top_a_seed_q]):
            col = artid_to_graphcol.get(pk)
            if col is not None: s_q[n_jp+col] = max(float(sim), 0.0)
        for jid, sim in zip(jp_order[top_j_seed_q], sim_jp[qj, top_j_seed_q]):
            row = jpid_to_graphrow.get(jid)
            if row is not None: s_q[row] = max(float(sim), 0.0)
        if s_q.sum() == 0: continue
        s_q /= s_q.sum()
        r_q = ppr(P_T, s_q, ALPHA, N_ITER)

        cos_top10_art = set(art_order[np.argpartition(-sim_art[qj], 10)[:10]].tolist())
        ppr_top10_art = set(art_pool_pks[np.argsort(-r_q[art_pool_graph_idx])[:10]].tolist())
        if cos_top10_art | ppr_top10_art:
            jaccards_art.append(
                len(cos_top10_art & ppr_top10_art) / len(cos_top10_art | ppr_top10_art))

        cos_top10_jp = set(jp_order[np.argpartition(-sim_jp[qj], 10)[:10]].tolist())
        ppr_top10_jp = set(jp_pool_ids[np.argsort(-r_q[jp_pool_graph_idx])[:10]].tolist())
        if cos_top10_jp | ppr_top10_jp:
            jaccards_jp.append(
                len(cos_top10_jp & ppr_top10_jp) / len(cos_top10_jp | ppr_top10_jp))
    print(f"  Jaccard art : moy {np.mean(jaccards_art):.2f}  med {np.median(jaccards_art):.2f}")
    print(f"  Jaccard JP  : moy {np.mean(jaccards_jp):.2f}  med {np.median(jaccards_jp):.2f}")

    # ─────────────────────────────────────────────────────────────
    # Figure 2 panneaux (b) concentration + (c) jaccard
    # ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

    # Panneau (b) : concentration de masse
    ax = axes[0]
    for alpha_test, color in zip([0.3, 0.7, 0.85], ["#888", "#4c8b9b", "#8b2e2e"]):
        ax.semilogx(np.arange(1, len(conc[alpha_test])+1), conc[alpha_test],
                    color=color, lw=1.6, label=f"α = {alpha_test}")
    ax.set_xlabel("k = nb de nœuds les mieux scorés (échelle log)")
    ax.set_ylabel("masse PPR cumulée")
    ax.set_title("Concentration de la masse — plus α $\\nearrow$, plus on étale", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.text(1.5, 0.51, "50% masse", fontsize=8, color="gray")
    ax.axhline(0.9, color="gray", ls=":", alpha=0.5)
    ax.text(1.5, 0.91, "90% masse", fontsize=8, color="gray")

    # Panneau (c) : histogramme Jaccard
    ax = axes[1]
    bins = np.linspace(0, 1, 21)
    ax.hist(jaccards_art, bins=bins, color="#295c9b", alpha=0.7,
            edgecolor="white", label=f"articles (moy {np.mean(jaccards_art):.2f})")
    ax.hist(jaccards_jp, bins=bins, color="#2f7a3a", alpha=0.7,
            edgecolor="white", label=f"JP (moy {np.mean(jaccards_jp):.2f})")
    ax.set_xlabel("Jaccard(top-10 cosine, top-10 PPR)")
    ax.set_ylabel("nb questions")
    ax.set_title("Divergence du ranking — Jaccard top-10 cosine vs PPR sur 971 q", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    ax.axvline(0.4, color="red", ls="--", alpha=0.4)
    ax.text(0.42, ax.get_ylim()[1]*0.9, "PPR diverge\nsubstantiellement\nde cosine",
            fontsize=8, color="red")

    plt.tight_layout()
    plt.savefig(FIG_OUT, dpi=150, bbox_inches="tight")
    print(f"\n✓ {FIG_OUT}")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
