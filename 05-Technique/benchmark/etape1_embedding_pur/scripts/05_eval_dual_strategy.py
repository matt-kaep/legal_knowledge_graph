"""Éval dual-strategy : open retrieval vs filtered retrieval, multi-branches.

Pour chaque question (8 pénal + 4 affaires = 12) :
  Stratégie 1 (open)      : cosine vs TOUS les articles embeddés (~31 357)
  Stratégie 2 (filtered)  : cosine vs articles de la branche de la question

Métriques (cf. critère adopté Week 7) :
  HARD : recall@10_art (oblig) ≥ 0.5  ET  recall@5_jp_via_graph ≥ 0.5
  EASY : recall@20_art (oblig) ≥ 0.5  ET  recall@10_jp_via_graph ≥ 0.5

Sortie :
  data/recall_dual_strategy.csv          (question × stratégie × side × k × recall)
  data/recall_dual_strategy_summary.json (par question : pass HARD/EASY, K* synthétique)
"""
from __future__ import annotations
import json
import re
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from etape1 import config
from etape1.eval_recall import recall_at_k, kstar, extract_pourvoi_numbers

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")


def _load_questions() -> list[dict]:
    qs = json.loads(config.RUBRICS.read_text())["questions"]
    qs += json.loads(config.RUBRICS_AFFAIRES.read_text())["questions"]
    return qs


def _build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for row in jp.itertuples():
        n = (row.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(row.id)
    return out


def _eval_one(
    question_text: str,
    branche: str,
    oblig: set[str],
    gold_jp_ids: set[str],
    Q: np.ndarray,                # (1, dim)
    art_emb: np.ndarray,
    art_order: np.ndarray,
    art_codes: np.ndarray,
    p2col: np.ndarray,
    G: sp.csr_matrix,
    jp_ids: np.ndarray,
    branche_codes: set[str] | None,   # None → open ; set → filtered
) -> tuple[list[dict], dict]:
    """Renvoie (rows, summary) pour une question × une stratégie."""
    # 1. Restreindre le pool si stratégie filtered
    if branche_codes is None:
        mask = np.ones(len(art_order), dtype=bool)
        strategy = "open"
    else:
        mask = np.array([c in branche_codes for c in art_codes])
        strategy = "filtered"

    sim = (Q @ art_emb[mask].T)[0]  # (n_kept,)
    order_local = np.argsort(-sim)
    ranked_pks = list(art_order[mask][order_local])
    ranked_cols = p2col[mask][order_local]

    # 2. Métriques articles
    rows = []
    for k in config.KS:
        rows.append({"strategy": strategy, "side": "article",
                      "k": k, "recall": recall_at_k(ranked_pks, oblig, k)})
    r10_art = recall_at_k(ranked_pks, oblig, 10)
    r20_art = recall_at_k(ranked_pks, oblig, 20)

    # 3. JP via graphe
    r5_jp = r10_jp = None
    if gold_jp_ids:
        for k in config.KS:
            top = ranked_cols[:k]
            jp_mask = (G[:, top].sum(axis=1) > 0).A1
            hit = len(set(jp_ids[jp_mask].tolist()) & gold_jp_ids)
            r = hit / len(gold_jp_ids)
            rows.append({"strategy": strategy, "side": "jp_via_graph",
                          "k": k, "recall": r})
        # Re-calculer r@5 et r@10 pour le summary
        def _rk(k):
            top = ranked_cols[:k]
            jp_mask = (G[:, top].sum(axis=1) > 0).A1
            return len(set(jp_ids[jp_mask].tolist()) & gold_jp_ids) / len(gold_jp_ids)
        r5_jp = _rk(5); r10_jp = _rk(10)

    # 4. Pass critère dur/facile
    pass_hard = pass_easy = None
    if gold_jp_ids:
        pass_hard = r10_art >= config.KSTAR_THRESHOLD and r5_jp >= config.KSTAR_THRESHOLD
        pass_easy = r20_art >= config.KSTAR_THRESHOLD and r10_jp >= config.KSTAR_THRESHOLD
    else:
        # Branche sans gold JP (affaires) → critère article seul
        pass_hard = r10_art >= config.KSTAR_THRESHOLD
        pass_easy = r20_art >= config.KSTAR_THRESHOLD

    return rows, {
        "strategy":  strategy,
        "branche":   branche,
        "n_pool":    int(mask.sum()),
        "r10_art":   r10_art,
        "r20_art":   r20_art,
        "r5_jp":     r5_jp,
        "r10_jp":    r10_jp,
        "pass_hard": pass_hard,
        "pass_easy": pass_easy,
    }


def main() -> int:
    print("Chargement embeddings…")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    print(f"  pool: {len(art_order)} articles")

    # Mapping code_slug par article (via le graphe)
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    full_codes = z["article_codes"]
    art_codes = full_codes[p2col]  # code_slug aligné sur art_emb

    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids = z["jp_ids"]

    rubrics = _load_questions()
    print(f"  questions: {len(rubrics)}")
    pourvoi_map = _build_pourvoi_to_jpid_map()

    # Encode questions (CPU pour ne pas concurrencer l'embed en cours)
    m = SentenceTransformer(config.MODEL_ID, device="cpu")
    m.max_seq_length = config.BATCH_MAX_LEN
    Q_all = m.encode([q["question"] for q in rubrics],
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=True).astype(np.float32)

    all_rows: list[dict] = []
    summary: list[dict] = []
    for qi, q in enumerate(rubrics):
        qid = q["id"]
        branche = q.get("branche", "?")
        oblig = set(q["articles_attendus"].get("obligatoires", []))
        # JP gold
        jp_gold_pourvois = set()
        for jp in q.get("jp_attendues", []):
            jp_gold_pourvois.update(extract_pourvoi_numbers(jp.get("short_ref") or ""))
        gold_jp_ids = {jpid for p in jp_gold_pourvois for jpid in pourvoi_map.get(p, [])}

        Q = Q_all[qi : qi + 1]

        for strat_codes in (None, set(config.BRANCHES.get(branche, []))):
            rows, sm = _eval_one(q["question"], branche, oblig, gold_jp_ids,
                                  Q, art_emb, art_order, art_codes, p2col,
                                  G, jp_ids, strat_codes if strat_codes else None)
            for r in rows:
                r["question_id"] = qid
                r["branche"] = branche
            all_rows.extend(rows)
            sm["question_id"] = qid
            sm["n_gold_oblig"] = len(oblig)
            sm["n_gold_jp"] = len(gold_jp_ids)
            summary.append(sm)

    pd.DataFrame(all_rows).to_csv(config.DATA / "recall_dual_strategy.csv", index=False)
    (config.DATA / "recall_dual_strategy_summary.json").write_text(
        json.dumps({"per_question": summary, "ks": list(config.KS),
                     "threshold": config.KSTAR_THRESHOLD},
                    ensure_ascii=False, indent=2))
    print(f"\n✓ {config.DATA / 'recall_dual_strategy.csv'}")
    print(f"✓ {config.DATA / 'recall_dual_strategy_summary.json'}")

    # Récap stdout
    print(f"\n{'='*100}")
    print(f"{'qid':22s} {'br':9s} {'strat':9s} {'pool':>6s} {'r10_art':>8s} "
          f"{'r5_jp':>7s} {'hard':>5s} {'easy':>5s}")
    print("-" * 100)
    for s in summary:
        r5 = f"{s['r5_jp']:.2f}" if s['r5_jp'] is not None else "  —  "
        print(f"  {s['question_id']:20s} {s['branche']:9s} {s['strategy']:9s} "
              f"{s['n_pool']:>6d} {s['r10_art']:>8.2f} {r5:>7s} "
              f"{'✓' if s['pass_hard'] else '✗':>5s} {'✓' if s['pass_easy'] else '✗':>5s}")

    # Stat aggrégée
    pen_open = [s for s in summary if s['branche']=='penal' and s['strategy']=='open']
    pen_filt = [s for s in summary if s['branche']=='penal' and s['strategy']=='filtered']
    aff_open = [s for s in summary if s['branche']=='affaires' and s['strategy']=='open']
    aff_filt = [s for s in summary if s['branche']=='affaires' and s['strategy']=='filtered']
    print(f"\n{'='*60}")
    print("Pénal (8 q) : open vs filtered — questions passant le critère DUR")
    print(f"  open    : {sum(s['pass_hard'] for s in pen_open)}/{len(pen_open)}")
    print(f"  filtered: {sum(s['pass_hard'] for s in pen_filt)}/{len(pen_filt)}")
    print("Affaires (4 q) :")
    print(f"  open    : {sum(s['pass_hard'] for s in aff_open)}/{len(aff_open)}")
    print(f"  filtered: {sum(s['pass_hard'] for s in aff_filt)}/{len(aff_filt)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
