"""Éval méthode #4 : JP-direct (cosine question × emb_jp_synthese) vs JP-via-graph.

Pour les 38 questions du benchmark CNB 2025, calcule trois variantes JP-side :

  (A) jp_direct      : cosine(q, emb_jp_synthese) → top-K JP
  (B) jp_via_graph   : top-K articles → JPs citantes via graphe (back-edge)
  (C) jp_hybrid_max  : max(score_direct, score_via_graph) sur chaque JP candidate

Métriques (constantes avec le reste du projet) :
  HARD  recall@5_jp  ≥ 0.5
  EASY  recall@10_jp ≥ 0.5

Le côté article reste tel que `05_eval_dual_strategy.py` le calcule — pas redondant ici.

Notes (TODO ultérieures, cf. demande superviseur) :
  - Variantes expansion 1-hop intersection/union (équivalent strats 5 mai)
    seront couvertes par un script séparé `09_eval_jp_expansion.py`.
  - Gold JP reste CC-only via regex pourvoi sur `short_ref`. Avec emb sur CA/TJ
    désormais à 100% couverture, on pourrait étendre via DB.source_id.

Sortie :
  data/recall_jp_methods.csv          (question × method × side × k × recall)
  data/recall_jp_methods_summary.json (par question × method : r@5, r@10, pass)
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
from etape1.eval_recall import recall_at_k, extract_pourvoi_numbers

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")


def _load_questions() -> list[dict]:
    """Charge TOUS les fichiers cnb-*-2025-consolidated.json + le rubrics_penal."""
    import glob
    files = sorted(glob.glob(str(config.RUBRICS_AFFAIRES.parent / "cnb-*-2025-consolidated.json")))
    all_qs, seen = [], set()
    for f in files:
        d = json.loads(open(f).read())
        qs = d["questions"] if isinstance(d, dict) and "questions" in d else d
        for q in qs:
            if q["id"] not in seen:
                seen.add(q["id"])
                all_qs.append(q)
    return all_qs


def _build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def main() -> int:
    # ─── Chargement ──────────────────────────────────────────────
    print("Chargement embeddings…")
    art_emb   = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col     = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    jp_emb    = np.load(config.EMB_JP_SYNTHESE)
    jp_order  = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    jp_to_row = np.load(config.JP_SUMMARY_TO_GRAPHROW)

    print(f"  articles : {art_emb.shape}, JP synth : {jp_emb.shape}")

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]

    # Coverage : quelles JP CC du corpus pourvoi-mappables sont aussi dans emb_jp ?
    pourvoi_map = _build_pourvoi_to_jpid_map()

    # ─── Questions ───────────────────────────────────────────────
    rubrics = _load_questions()
    print(f"  questions : {len(rubrics)}")

    # Encoder les questions (CPU pour ne pas concurrencer rien)
    m = SentenceTransformer(config.MODEL_ID, device="cpu")
    m.max_seq_length = config.BATCH_MAX_LEN
    Q = m.encode([q["question"] for q in rubrics],
                  normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=True).astype(np.float32)
    print(f"  questions encodées : {Q.shape}")

    # ─── Pré-calculs ─────────────────────────────────────────────
    sim_arts = Q @ art_emb.T   # (n_q, n_art)
    sim_jp   = Q @ jp_emb.T    # (n_q, n_jp_emb)
    print(f"  cos sim : arts {sim_arts.shape}, jp_direct {sim_jp.shape}")

    # Pour back-edge : (n_q, n_art) → on prendra top-K articles puis JP citantes
    # Pour hybrid : il faut mapper le score "via_graph" sur les MÊMES JP que emb_jp.
    # On définit score_via_graph[q, jp_idx_in_emb] = max sim(q, art) sur les articles
    # cités par cette JP. = (G_jp @ sim_arts.T).T avec G_jp = ligne du graphe pour cette JP.

    # ─── Loop par question ──────────────────────────────────────
    KS = config.KS
    rows = []
    summary = []
    for qi, q in enumerate(rubrics):
        qid     = q["id"]
        branche = q.get("branche", "?")
        # Gold JP
        gp = set()
        for jp in q.get("jp_attendues", []):
            gp.update(extract_pourvoi_numbers(jp.get("short_ref") or ""))
        gold_jp_ids = {jid for p in gp for jid in pourvoi_map.get(p, [])}

        sm = {"question_id": qid, "branche": branche,
              "n_gold_jp": len(gold_jp_ids), "n_gold_jp_pourvois": len(gp)}

        if not gold_jp_ids:
            # Question sans gold JP évaluable
            for method in ("jp_direct", "jp_via_graph", "jp_hybrid_max"):
                for k in KS:
                    rows.append({"question_id": qid, "branche": branche,
                                  "method": method, "k": k, "recall": None})
                sm[f"r5_{method}"] = sm[f"r10_{method}"] = None
                sm[f"pass_hard_{method}"] = sm[f"pass_easy_{method}"] = None
            summary.append(sm)
            continue

        # === (A) JP-direct ===
        order_direct = np.argsort(-sim_jp[qi])
        ranked_direct = list(jp_order[order_direct])
        # === (B) JP-via-graph : top-K articles → JPs citantes ===
        order_arts = np.argsort(-sim_arts[qi])
        ranked_cols = p2col[order_arts]
        # === (C) Hybride : score_jp = max(sim_direct, sim_max_via_art_cited) ===
        # Pour chaque JP de emb_jp, son score via_graph = max(sim_arts) parmi les
        # articles que cette JP cite ET qui sont dans le pool art_order.
        # G[jp_to_row[j], p2col] : pour jp d'index j dans emb_jp, dit quels articles cités.
        # On peut le calculer en bloc : (n_jp_emb, n_art_emb) = G[jp_to_row][:, p2col] (sparse)
        # Trop gros pour matériel-isation directe ; on fait au cas par cas.

        # Préparer le mapping jp_order ↔ jp_emb_idx pour le hybride
        jp_id_to_emb_idx = {jpid: i for i, jpid in enumerate(jp_order)}

        for k in KS:
            # JP-direct
            r_dir = recall_at_k(ranked_direct, gold_jp_ids, k)
            rows.append({"question_id": qid, "branche": branche,
                          "method": "jp_direct", "k": k, "recall": r_dir})
            # JP-via-graph (mêmes formules que 05_eval_dual_strategy)
            top_cols = ranked_cols[:k]
            jp_mask = (G[:, top_cols].sum(axis=1) > 0).A1
            recovered = set(jp_ids_graph[jp_mask].tolist())
            r_via = len(recovered & gold_jp_ids) / len(gold_jp_ids)
            rows.append({"question_id": qid, "branche": branche,
                          "method": "jp_via_graph", "k": k, "recall": r_via})
            # Hybride : prendre l'union des top-K direct et top-K via_graph,
            # puis re-ranker par max des deux scores.
            top_direct_ids = list(ranked_direct[:k])
            top_via_ids = [jid for jid in jp_ids_graph[jp_mask].tolist()
                           if jid in jp_id_to_emb_idx]
            cand = list(dict.fromkeys(top_direct_ids + top_via_ids))[:k]
            r_hyb = len(set(cand) & gold_jp_ids) / len(gold_jp_ids)
            rows.append({"question_id": qid, "branche": branche,
                          "method": "jp_hybrid_max", "k": k, "recall": r_hyb})

        # Summary
        for method in ("jp_direct", "jp_via_graph", "jp_hybrid_max"):
            rec_at = {r["k"]: r["recall"] for r in rows
                      if r["question_id"] == qid and r["method"] == method}
            sm[f"r5_{method}"]  = rec_at.get(5)
            sm[f"r10_{method}"] = rec_at.get(10)
            sm[f"pass_hard_{method}"] = bool(rec_at.get(5, 0) >= config.KSTAR_THRESHOLD)
            sm[f"pass_easy_{method}"] = bool(rec_at.get(10, 0) >= config.KSTAR_THRESHOLD)
        summary.append(sm)

    # ─── Écriture ──────────────────────────────────────────────
    out_csv = config.DATA / "recall_jp_methods.csv"
    out_json = config.DATA / "recall_jp_methods_summary.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(
        {"per_question": summary, "ks": list(KS),
         "threshold": config.KSTAR_THRESHOLD,
         "methods": ["jp_direct", "jp_via_graph", "jp_hybrid_max"]},
        ensure_ascii=False, indent=2))
    print(f"\n✓ {out_csv}")
    print(f"✓ {out_json}")

    # ─── Récap ─────────────────────────────────────────────────
    df = pd.DataFrame(summary)
    print(f"\n{'='*92}")
    print(f"{'qid':22s} {'branche':12s} {'gold_jp':>7s} "
          f"{'r5 direct':>10s} {'r5 graph':>10s} {'r5 hybrid':>11s} "
          f"{'pass dur':>10s}")
    print("-" * 92)
    for s in summary:
        gn = s["n_gold_jp"]
        if not gn:
            print(f"  {s['question_id']:20s} {s['branche']:12s} {0:>7d}  (no gold)")
            continue
        d5 = s.get("r5_jp_direct") or 0
        v5 = s.get("r5_jp_via_graph") or 0
        h5 = s.get("r5_jp_hybrid_max") or 0
        pass_d = s.get("pass_hard_jp_direct")
        pass_v = s.get("pass_hard_jp_via_graph")
        pass_h = s.get("pass_hard_jp_hybrid_max")
        mark = lambda b: "✓" if b else "✗"
        print(f"  {s['question_id']:20s} {s['branche']:12s} {gn:>7d}  "
              f"{d5:>10.2f} {v5:>10.2f} {h5:>11.2f}  "
              f"D:{mark(pass_d)} G:{mark(pass_v)} H:{mark(pass_h)}")

    # Agrégat par méthode
    print(f"\n{'='*60}")
    print("Pass critère DUR (r@5_jp ≥ 0.5) par méthode :")
    evaluable = [s for s in summary if s["n_gold_jp"] > 0]
    for method in ("jp_direct", "jp_via_graph", "jp_hybrid_max"):
        n_pass = sum(s.get(f"pass_hard_{method}", False) for s in evaluable)
        print(f"  {method:20s} {n_pass:>3d} / {len(evaluable):<3d}  "
              f"({100*n_pass/max(len(evaluable),1):.0f} %)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
