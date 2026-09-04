"""LightGCN — design A — graphe Art↔JP, questions = users BGE-M3 SANS arête.

MVP (cf. ADR-001 + advisor) : 3 points d'ablation sur la MÊME cohorte / pool /
métriques / schéma que 20_ppr_naive.py, pour une comparaison publiable vs PPR.

  (a) cosine_raw     : non entraîné, K=0  → score = q · article_BGE  (= reproduit B3)
  (b) trained_K0     : LightGCN entraîné, 0 couche de propagation
  (c) trained_K3     : LightGCN entraîné, 3 couches de propagation

Mécanique (design A) :
  - Graphe de propagation = bloc-bipartite Art↔JP (mêmes citations que PPR),
    normalisé symétrique D^{-1/2} A D^{-1/2}. AUCUN nœud question.
  - On apprend E^(0) des items (JP+articles), init BGE-M3 (aléatoire si absent).
  - Questions = vecteurs BGE-M3 FIGÉS (frozen), hors graphe.
  - BPR : positif (q, article_GT), négatif (q, article aléatoire du pool).
  - lr PETIT (init BGE-M3 à préserver — pas le 0.01 du notebook de Johnny).

Réutilise 20_ppr_naive.py (cohorte, graphe, pools, ranking_rows) via importlib.
Sortie : lightgcn_eval.csv (mêmes colonnes panel que ppr_naive_eval.csv).
"""
from __future__ import annotations
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPTS))
from etape1 import config  # noqa: E402
import metrics as M  # noqa: E402

# ── Réutilisation de 20_ppr_naive.py (nom de module = chiffre → importlib) ──
_spec = importlib.util.spec_from_file_location("ppr_naive", SCRIPTS / "20_ppr_naive.py")
ppr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ppr)  # exécute le module (constantes + fonctions, pas main)

OUT_DIR = ROOT / "data/global_bench"
DQ_EMB = ROOT / "data/doctrine_qgen/questions_emb.npy"
DQ_IDS = ROOT / "data/doctrine_qgen/questions_ids.npy"

K_OUT = ppr.K_OUT          # 10, identique à PPR/B*
N_LAYERS = 3
TRAIN_K = int(__import__("os").environ.get("TRAIN_K", 2))  # K du modèle entraîné (best untrained)
LR = 1e-3
EPOCHS = 30
TAU = 0.1                  # température du cosine-BPR (cosine ∈ [-1,1] → logits)
LAMBDA_ANCHOR = 1.0        # ancrage ‖E0 − BGE‖² sur les nœuds embeddés (anti-drift)
DEVICE = "cpu"             # spmm sparse fiable sur CPU (MPS souvent non supporté)
SEED = int(os.environ.get("SEED", 42))   # SEED=n pour la validation multi-seed


def sparse_scipy_to_torch(A: sp.csr_matrix) -> torch.Tensor:
    A = A.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape)).coalesce()


class LightGCNa(nn.Module):
    """Design A : seuls les items (JP+articles) sont des paramètres."""

    def __init__(self, e0_init: torch.Tensor, n_layers: int):
        super().__init__()
        self.emb = nn.Parameter(e0_init.clone())   # [N_total, D]
        self.n_layers = n_layers

    def propagate(self, adj: torch.Tensor) -> torch.Tensor:
        outs = [self.emb]
        x = self.emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            outs.append(x)
        return sum(outs) / (self.n_layers + 1)      # layer combination = moyenne


def _l2(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)


def bpr_loss(q_vec, item_final, pos_idx, neg_idx, tau=TAU):
    """BPR en COSINUS (cohérent avec l'éval) + température.

    Le produit scalaire brut laisse la norme gonflée par la propagation dominer
    (biais hubs). On normalise q et items → cosine, divisé par tau pour un
    gradient exploitable (cosine ∈ [-1,1]).
    """
    qn = _l2(q_vec)
    pos = (qn * _l2(item_final[pos_idx])).sum(1) / tau
    neg = (qn * _l2(item_final[neg_idx])).sum(1) / tau
    return -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()


def main() -> int:
    t0 = time.time()
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    print("══ Chargement graphe + embeddings (mirror script 20) ──────────")

    art_emb = np.load(config.EMB_ARTICLES_ALL).astype(np.float32)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    jp_emb = np.load(config.EMB_JP_SYNTHESE).astype(np.float32)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    D = art_emb.shape[1]

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    n_jp, n_art = G.shape
    N_total = n_jp + n_art
    print(f"  G {G.shape}  nnz {G.nnz:,}  →  N_total {N_total:,}")

    # Adjacence LightGCN = bloc-bipartite symétrique normalisée (= P_sym de 20)
    G_full = sp.bmat([[None, G], [G.T, None]], format="csr")
    A_sym = ppr.sym_normalize(G_full)
    adj = sparse_scipy_to_torch(A_sym).to(DEVICE)
    print(f"  A_sym {A_sym.shape}  nnz {A_sym.nnz:,}  (t={time.time()-t0:.1f}s)")

    # ── E^(0) en ORDRE GRAPHE [JP 0..n_jp-1, art n_jp..], init BGE-M3 ──
    artid_to_graphcol = {aid: i for i, aid in enumerate(article_ids_graph)}
    jpid_to_graphrow = {jid: i for i, jid in enumerate(jp_ids_graph)}
    pk_to_emb = {pk: i for i, pk in enumerate(art_order)}
    jid_to_emb = {jid: i for i, jid in enumerate(jp_order)}

    e0 = np.zeros((N_total, D), dtype=np.float32)
    has_init = np.zeros(N_total, dtype=bool)
    for j, jid in enumerate(jp_ids_graph):
        k = jid_to_emb.get(jid)
        if k is not None:
            e0[j] = jp_emb[k]; has_init[j] = True
    for a, aid in enumerate(article_ids_graph):
        k = pk_to_emb.get(aid)
        if k is not None:
            e0[n_jp + a] = art_emb[k]; has_init[a + n_jp] = True
    # nœuds sans init : aléatoire à la norme médiane des embeddings BGE présents
    med_norm = float(np.median(np.linalg.norm(e0[has_init], axis=1)))
    miss = ~has_init
    rand = rng.standard_normal((miss.sum(), D)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9
    e0[miss] = rand * med_norm
    print(f"  E0 init BGE-M3 : {has_init.sum():,}/{N_total:,} "
          f"(aléatoire pour {miss.sum():,}, norme méd. {med_norm:.2f})")

    # ── Pools de ranking (IDENTIQUES à 20 : embeddés ∩ graphe) ──
    art_pool_graph_idx = np.array(
        [n_jp + artid_to_graphcol[pk] for pk in art_order if pk in artid_to_graphcol],
        dtype=np.int64)
    art_pool_pks = np.array([pk for pk in art_order if pk in artid_to_graphcol])
    jp_pool_graph_idx = np.array(
        [jpid_to_graphrow[j] for j in jp_order if j in jpid_to_graphrow], dtype=np.int64)
    jp_pool_ids = np.array([j for j in jp_order if j in jpid_to_graphrow])
    pool_art_set = set(art_order.tolist())
    pool_jp_set = set(jp_order.tolist())
    # col graphe (bloc article) des articles du pool, pour les positifs
    art_pk_to_itemidx = {pk: n_jp + artid_to_graphcol[pk]
                         for pk in art_order if pk in artid_to_graphcol}
    print(f"  pool art {len(art_pool_graph_idx):,}  |  pool JP {len(jp_pool_graph_idx):,}")

    # ── Cohorte d'éval (held-out) — réutilise la logique exacte de 20 ──
    cohort = ppr.load_cohort_977()
    Q_emb = np.load(ppr.Q_EMB_CACHE).astype(np.float32)
    cohort_ids = np.load(ppr.Q_IDS_CACHE, allow_pickle=True).tolist()
    coh_id_to_emb = {qid: i for i, qid in enumerate(cohort_ids)}
    cohort = [q for q in cohort if q["id"] in coh_id_to_emb]
    cohort_qids = {q["id"] for q in cohort}
    pourvoi_map = ppr.build_pourvoi_map()
    print(f"  cohorte évaluée : {len(cohort)}")

    # ── Questions d'ENTRAÎNEMENT = doctrine_qgen hors cohorte (anti-leak) ──
    dq_emb = np.load(DQ_EMB).astype(np.float32)
    dq_ids = np.load(DQ_IDS, allow_pickle=True)
    dq_id_to_emb = {qid: i for i, qid in enumerate(dq_ids)}
    bench = json.loads((OUT_DIR / "bench_global.json").read_text())["questions"]
    train_pos = []   # (q_emb_idx_train, item_graph_idx)
    train_q_rows = []
    seen_q = {}
    for q in bench:
        qid = q["qid"]
        if qid in cohort_qids:
            continue                       # ANTI-LEAK
        if qid not in dq_id_to_emb:
            continue
        gts = set(q.get("articles_attendus_etendu") or q.get("articles_attendus") or [])
        gts &= pool_art_set
        if not gts:
            continue
        if qid not in seen_q:
            seen_q[qid] = len(train_q_rows)
            train_q_rows.append(dq_emb[dq_id_to_emb[qid]])
        qi = seen_q[qid]
        for pk in gts:
            train_pos.append((qi, art_pk_to_itemidx[pk]))
    assert cohort_qids.isdisjoint(set(seen_q)), "FUITE train/test !"
    Q_train = torch.tensor(np.asarray(train_q_rows), dtype=torch.float32, device=DEVICE)
    pos_q = torch.tensor([p[0] for p in train_pos], dtype=torch.long, device=DEVICE)
    pos_item = torch.tensor([p[1] for p in train_pos], dtype=torch.long, device=DEVICE)
    print(f"  train : {len(seen_q)} questions, {len(train_pos)} paires positives "
          f"(GT étendu) — anti-leak OK")

    e0_t = torch.tensor(e0, dtype=torch.float32, device=DEVICE)
    # Variante init=0 pour les nœuds non-embeddés : pour le diagnostic de
    # propagation NON entraînée, on ne veut propager que le vrai signal BGE-M3
    # (pas le bruit aléatoire des 57k nœuds sans texte).
    e0_zero = e0.copy(); e0_zero[miss] = 0.0
    e0_zero_t = torch.tensor(e0_zero, dtype=torch.float32, device=DEVICE)
    Q_coh = torch.tensor(np.asarray([Q_emb[coh_id_to_emb[q["id"]]] for q in cohort]),
                         dtype=torch.float32, device=DEVICE)
    pool_item_idx_t = torch.tensor(art_pool_graph_idx, device=DEVICE)
    jp_pool_idx_t = torch.tensor(jp_pool_graph_idx, device=DEVICE)

    # ── Éval : produit les lignes panel (mêmes colonnes que 20) ──
    def evaluate(item_final: torch.Tensor, variant: str) -> list[dict]:
        # Scoring en COSINUS (L2-normalisation) : la propagation gonfle la norme
        # des hubs ; sans ça le produit scalaire est dominé par la norme → biais
        # de popularité. BGE est déjà normé (médiane 1,00) → cosine_raw inchangé.
        def _n(x):
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)
        art_final_pool = _n(item_final[pool_item_idx_t])  # [Pa, D]
        jp_final_pool = _n(item_final[jp_pool_idx_t])     # [Pj, D]
        Qn = _n(Q_coh)
        rows = []
        with torch.no_grad():
            sc_art = (Qn @ art_final_pool.T).cpu().numpy()
            sc_jp = (Qn @ jp_final_pool.T).cpu().numpy()
        for i, q in enumerate(cohort):
            gt_s = q["gt_strict"] & pool_art_set
            gt_e = q["gt_ext"] & pool_art_set
            gold_jp = ({jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
                       & pool_jp_set)
            if not gt_s and not gold_jp:
                continue
            top_a = np.argsort(-sc_art[i])[:K_OUT]
            ranked_art = list(art_pool_pks[top_a])
            top_j = np.argsort(-sc_jp[i])[:K_OUT]
            ranked_jp = list(jp_pool_ids[top_j])
            am = {f"{k}_art": v for k, v in
                  M.panel_strict_ext(ranked_art, gt_s, gt_e, K_OUT).items()}
            jm = {f"{k}_jp": v for k, v in
                  M.all_metrics(ranked_jp, gold_jp, K_OUT).items()}
            rows.append({"qid": q["id"], "variant": variant, **am, **jm})
        return rows

    anchor_idx = torch.tensor(np.where(has_init)[0], dtype=torch.long, device=DEVICE)

    def train(n_layers: int) -> torch.Tensor:
        torch.manual_seed(SEED)
        model = LightGCNa(e0_t, n_layers).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
        n_pos = pos_q.shape[0]
        e0_ref = e0_t[anchor_idx].detach().clone()    # init BGE-M3 de référence
        for ep in range(EPOCHS):
            model.train()
            item_final = model.propagate(adj)
            neg = torch.tensor(rng.integers(0, len(art_pool_graph_idx), n_pos),
                               dtype=torch.long, device=DEVICE)
            neg_item = pool_item_idx_t[neg]
            bpr = bpr_loss(Q_train[pos_q], item_final, pos_item, neg_item)
            # ancrage anti-drift : pénalise l'éloignement de l'init BGE-M3
            anchor = ((model.emb[anchor_idx] - e0_ref) ** 2).sum(1).mean()
            loss = bpr + LAMBDA_ANCHOR * anchor
            opt.zero_grad(); loss.backward(); opt.step()
            if ep % 10 == 0 or ep == EPOCHS - 1:
                print(f"    K{n_layers} ep {ep:>3d}/{EPOCHS}  bpr {bpr.item():.4f}"
                      f"  anchor {anchor.item():.4f}  (t={time.time()-t0:.1f}s)")
        model.eval()
        with torch.no_grad():
            return model.propagate(adj)

    all_rows = []
    print("\n══ cosine_raw (non entraîné, K=0) = B2-a ──────────────────")
    all_rows += evaluate(e0_t, "cosine_raw")
    # DIAGNOSTIC CLÉ : propagation de l'init BGE-M3 FIGÉ (sans entraînement).
    # Isole « le graphe de citations aide-t-il le signal sémantique ? » du
    # confound d'entraînement. Init non-embeddés = 0 (propage le vrai signal seul).
    print("══ untrained_K1/2/3 (propagation BGE figé, pas d'entraînement) ─")
    for k in (1, 2, 3):
        m = LightGCNa(e0_zero_t, k).to(DEVICE)
        with torch.no_grad():
            fk = m.propagate(adj)
        all_rows += evaluate(fk, f"untrained_K{k}")
        print(f"    untrained_K{k} évalué  (t={time.time()-t0:.1f}s)")
    if os.environ.get("NOTRAIN") != "1":
        print(f"══ trained_K{TRAIN_K} (BPR cosinus τ={TAU} + ancrage λ={LAMBDA_ANCHOR}) ──")
        all_rows += evaluate(train(TRAIN_K), f"trained_K{TRAIN_K}")

    df = pd.DataFrame(all_rows)
    out_csv = OUT_DIR / "lightgcn_eval.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}  ({len(df)} lignes)")

    # ── Agrégats + comparaison vs PPR champions (depuis ppr_naive_summary) ──
    cols = ([f"{m}_strict_art" for m in M.METRIC_NAMES]
            + [f"{m}_ext_art" for m in M.METRIC_NAMES]
            + [f"{m}_jp" for m in M.METRIC_NAMES])
    print("\n══ LightGCN — moyennes par variante ───────────────────────")
    print(f"  {'variant':>12s} | {'M1s_a':>6s} {'Hits_a':>6s} {'NDCs_a':>6s} | "
          f"{'M1e_a':>6s} {'Hite_a':>6s} {'NDCe_a':>6s} | {'M1_jp':>6s} {'NDC_jp':>6s}")
    print("  " + "─" * 92)
    summ = {}
    for v, sub in df.groupby("variant"):
        m = {c: float(sub[c].mean()) for c in cols}
        summ[v] = m
        print(f"  {v:>12s} | {m['m1_strict_art']:>6.3f} {m['hit_strict_art']:>6.3f} "
              f"{m['ndcg_strict_art']:>6.3f} | {m['m1_ext_art']:>6.3f} "
              f"{m['hit_ext_art']:>6.3f} {m['ndcg_ext_art']:>6.3f} | "
              f"{m['m1_jp']:>6.3f} {m['ndcg_jp']:>6.3f}")

    # Référence PPR (si dispo) + garde anti-drift
    ppr_sum_path = OUT_DIR / "ppr_naive_summary.json"
    if ppr_sum_path.exists():
        ps = json.loads(ppr_sum_path.read_text())
        champ = ps.get("row|alpha=0.95", {})
        print(f"\n  [réf] PPR row α=0.95 : M1e_a={champ.get('m1_ext_art', float('nan')):.3f} "
              f"NDCe_a={champ.get('ndcg_ext_art', float('nan')):.3f} "
              f"M1_jp={champ.get('m1_jp', float('nan')):.3f}")
    # Garde anti-drift : l'entraînement doit au moins égaler la propagation
    # NON entraînée au même K, sur le strict (régime gagnant). Sinon → drift.
    unt = summ.get(f"untrained_K{TRAIN_K}", {})
    tr = summ.get(f"trained_K{TRAIN_K}", {})
    if unt and tr:
        delta = tr["m1_strict_art"] - unt["m1_strict_art"]
        verdict = "AIDE ✅" if delta > 0.003 else ("NEUTRE ≈" if delta > -0.003 else "DRIFT ⚠️")
        print(f"\n  [entraînement] trained_K{TRAIN_K} vs untrained_K{TRAIN_K} "
              f"(M1 strict) : {tr['m1_strict_art']:.3f} vs {unt['m1_strict_art']:.3f} "
              f"(Δ={delta:+.3f}) → {verdict}")

    (OUT_DIR / "lightgcn_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2))
    print(f"\n✓ {OUT_DIR}/lightgcn_summary.json   (t total {time.time()-t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
