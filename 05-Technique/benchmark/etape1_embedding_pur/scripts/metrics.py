"""Panel métriques retrieval pour la cohorte 977 (chantier 1 Week-10).

5 fonctions atomiques (M1, M2, Hit@K, MRR@K, NDCG@K) + helpers de regroupement.
Importé par scripts 18 (B*), 20 (PPR), 23 (M3 LLM-judge, futur), 31 (LightGCN, futur).

Conventions retenues post-Week-9 (cf. présentation Week-10) :
- Hit@K = |GT ∩ R[:K]| / min(|GT|, K) (couverture atteignable)
- MRR cappé à K (rang > K → 0)
- NDCG rel binaire ({0,1}), pas multi-niveau
- Gt vide → NaN (la question est skip côté agrégateur)
"""
from __future__ import annotations
import math


def _unique_ranked(ranked: list, k: int) -> list:
    """Top-k sans doublons, en conservant le premier rang observé."""
    out = []
    seen = set()
    for item in ranked:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= k:
            break
    return out


def m1_recall(ranked: list, gt: set, k: int) -> float:
    """M1 = Recall@K = |GT ∩ R[:K]| / |GT|. NaN si gt vide."""
    if not gt:
        return float("nan")
    ranked_k = _unique_ranked(ranked, k)
    return len(set(ranked_k) & gt) / len(gt)


def m2_rank(ranked: list, gt: set, k: int) -> float:
    """M2 = rang moyen normalisé (custom).

    x = (1/|GT|) Σ rang(a)   avec rang(a) = pos+1 si ≤k sinon k+1
    b' = min(|GT|, k)        (clip pour atteignabilité)
    f(x) = (x - (k+1)) / ((b'+1)/2 - (k+1))    ∈ [0, 1]
    f(k+1) = 0 (pire) ; f((b'+1)/2) = 1 (meilleur : tous au sommet).
    """
    if not gt:
        return float("nan")
    n = len(gt)
    ranked_k = _unique_ranked(ranked, k)
    pos = {a: i + 1 for i, a in enumerate(ranked_k)}
    ranks = [pos.get(a, k + 1) for a in gt]
    x = sum(ranks) / n
    b_clip = min(n, k)
    denom = (b_clip + 1) / 2 - (k + 1)
    if denom == 0:
        return float("nan")
    return (x - (k + 1)) / denom


def hit_at_k(ranked: list, gt: set, k: int) -> float:
    """Hit@K = |GT ∩ R[:K]| / min(|GT|, K). NaN si gt vide."""
    if not gt:
        return float("nan")
    ranked_k = _unique_ranked(ranked, k)
    return len(set(ranked_k) & gt) / min(len(gt), k)


def mrr_at_k(ranked: list, gt: set, k: int) -> float:
    """MRR@K = 1/rang du premier GT (cappé à k, 0 si aucun dans top-K). NaN si gt vide."""
    if not gt:
        return float("nan")
    ranked_k = _unique_ranked(ranked, k)
    for i, item in enumerate(ranked_k):
        if item in gt:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: list, gt: set, k: int) -> float:
    """NDCG@K avec rel binaire (rel_i = 1 si R[i] ∈ GT sinon 0).

    DCG@K  = Σ_{i=1..K}        rel_i / log2(i+1)
    IDCG@K = Σ_{i=1..min(|GT|,K)} 1 / log2(i+1)
    """
    if not gt:
        return float("nan")
    ranked_k = _unique_ranked(ranked, k)
    # i 0-indexé → rang = i+1 → log2(rang+1) = log2(i+2)
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(ranked_k)
        if item in gt
    )
    n_max = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_max))
    if idcg == 0:
        return float("nan")
    return dcg / idcg


# ────────────────────────────────────────────────────────────────────────
# Helpers de regroupement
# ────────────────────────────────────────────────────────────────────────
METRIC_NAMES = ("m1", "m2", "hit", "mrr", "ndcg")


def all_metrics(ranked: list, gt: set, k: int, suffix: str = "") -> dict:
    """Renvoie le panel complet sous forme de dict.

    suffix : chaîne ajoutée à chaque clé (ex: "strict" → m1_strict, m2_strict, ...).
    Sans suffix → clés brutes m1, m2, hit, mrr, ndcg.
    """
    s = f"_{suffix}" if suffix else ""
    return {
        f"m1{s}":   m1_recall(ranked, gt, k),
        f"m2{s}":   m2_rank(ranked, gt, k),
        f"hit{s}":  hit_at_k(ranked, gt, k),
        f"mrr{s}":  mrr_at_k(ranked, gt, k),
        f"ndcg{s}": ndcg_at_k(ranked, gt, k),
    }


def panel_strict_ext(ranked: list, gt_strict: set, gt_ext: set, k: int) -> dict:
    """Panel articles : 10 colonnes (5 strict + 5 étendu)."""
    return {
        **all_metrics(ranked, gt_strict, k, suffix="strict"),
        **all_metrics(ranked, gt_ext,    k, suffix="ext"),
    }
