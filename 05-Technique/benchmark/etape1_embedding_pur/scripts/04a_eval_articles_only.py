"""Éval article-first (Phase D du journal 5 mai) — sans attendre l'embed JP.

Pour chaque question CRFPA :
  (A) question → top-K articles (cos sim)            → recall vs articles_attendus
  (B) top-K articles → JP qui citent via le graphe   → recall vs jp_attendues (pourvoi-CC)

Le (B) utilise uniquement la matrice CSR JP×article du graphe (back-edge) :
  pour chaque article du top-K, on récupère les JP qui le citent, on agrège.
Pas besoin de `emb_jp.npy` — on peut évaluer dès que l'embed articles est là.

Sortie :
  recall_curves_articles_only.csv      (question, side, k, recall)
  recall_kstar_articles_only.json      (K* par question, deux sides)
"""
from __future__ import annotations
import json
import re
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from etape1 import config
from etape1.eval_recall import recall_at_k, kstar, extract_pourvoi_numbers

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")


def _encode_questions(questions: list[str], device: str = "cpu") -> np.ndarray:
    """Encode questions avec BGE-M3. CPU par défaut (MPS occupé par embed JP en parallèle)."""
    model = SentenceTransformer(config.MODEL_ID, device=device)
    model.max_seq_length = config.BATCH_MAX_LEN
    return model.encode(questions, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True).astype(np.float32)


def _build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
    """Pourvoi (XX-XX.XXX) → liste de jp_id, restreint aux JP CC."""
    import pyarrow.parquet as pq
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for row in jp.itertuples():
        n = (row.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(row.id)
    return out


def main() -> int:
    # ─── Chargement ────────────────────────────────────────────────
    emb_arts = np.load(config.EMB_ARTICLES)
    art_order = np.load(config.ARTICLES_ORDER, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL)  # idx d'embedding → colonne dans le graphe biparti
    print(f"emb_articles : {emb_arts.shape}, articles_order : {art_order.shape}, p2col : {p2col.shape}")

    # Graphe biparti JP × article (CSR), shape (n_jp, n_articles)
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids = z["jp_ids"]   # axis 0 ordering
    print(f"graphe biparti : {G.shape} ({G.nnz} arêtes)")

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    q_texts = [q["question"] for q in rubrics]
    print(f"Encodage {len(q_texts)} questions (CPU)…")
    Q = _encode_questions(q_texts, device="cpu")
    sim = Q @ emb_arts.T  # (n_q, n_articles_emb)
    print(f"similarité Q×art : {sim.shape}")

    # Map pourvoi → jp_id (CC seulement)
    print("Map pourvoi → jp_id…")
    pourvoi_to_jpid = _build_pourvoi_to_jpid_map()
    print(f"  {len(pourvoi_to_jpid)} pourvois CC indexables")

    rows = []
    kstar_summary = []

    # ─── Évaluation par question ───────────────────────────────────
    for qi, q in enumerate(rubrics):
        qid = q["id"]

        # Classement articles
        order = np.argsort(-sim[qi])
        ranked_arts = list(art_order[order])  # pair_keys ordonnés
        ranked_cols = p2col[order]            # colonnes graphe ordonnées

        # === (A) ARTICLE-SIDE : recall vs articles_attendus ===
        oblig = set(q["articles_attendus"].get("obligatoires", []))
        optio = set(q["articles_attendus"].get("optionnels", []))
        all_a = oblig | optio
        for k in config.KS:
            rows.append({"question_id": qid, "side": "article", "metric": "obligatoires",
                          "k": k, "recall": recall_at_k(ranked_arts, oblig, k)})
            rows.append({"question_id": qid, "side": "article", "metric": "obligatoires_optionnels",
                          "k": k, "recall": recall_at_k(ranked_arts, all_a, k)})
        kstar_a_oblig = kstar(ranked_arts, oblig, config.KS, config.KSTAR_THRESHOLD)
        kstar_a_all   = kstar(ranked_arts, all_a, config.KS, config.KSTAR_THRESHOLD)

        # === (B) JP-VIA-GRAPH : top-K articles → JP citantes ===
        jp_gold_pourvois = set()
        for jp in q["jp_attendues"]:
            short = jp.get("short_ref") or ""
            jp_gold_pourvois.update(extract_pourvoi_numbers(short))
        jp_gold_ids = {jpid for p in jp_gold_pourvois for jpid in pourvoi_to_jpid.get(p, [])}

        # Pour chaque K, prendre top-K cols, agréger JP qui citent au moins l'une d'elles
        # On utilise G[:, cols].sum(axis=1) > 0 pour trouver les JP citantes
        for k in config.KS:
            top_cols = ranked_cols[:k]
            jp_mask = (G[:, top_cols].sum(axis=1) > 0).A1  # array 1D booléen sur jp_ids
            jp_set = set(jp_ids[jp_mask].tolist())
            rec = (len(jp_set & jp_gold_ids) / max(len(jp_gold_ids), 1)) if jp_gold_ids else 0.0
            rows.append({"question_id": qid, "side": "jp_via_graph", "metric": "pourvoi_resolved",
                          "k": k, "recall": rec})
        # K* = plus petit K tel que JP-via-graph ≥ 0.5
        def _kstar_jp_via_graph() -> int | None:
            for k in sorted(config.KS):
                top_cols = ranked_cols[:k]
                jp_mask = (G[:, top_cols].sum(axis=1) > 0).A1
                jp_set = set(jp_ids[jp_mask].tolist())
                rec = (len(jp_set & jp_gold_ids) / max(len(jp_gold_ids), 1)) if jp_gold_ids else 0.0
                if rec >= config.KSTAR_THRESHOLD:
                    return k
            return None
        kstar_jp = _kstar_jp_via_graph()

        kstar_summary.append({
            "question_id":         qid,
            "n_gold_oblig":        len(oblig),
            "n_gold_oblig_optio":  len(all_a),
            "kstar_art_oblig":     kstar_a_oblig,
            "kstar_art_all":       kstar_a_all,
            "n_gold_jp":           len(jp_gold_ids),
            "n_gold_jp_pourvois":  len(jp_gold_pourvois),
            "kstar_jp_via_graph":  kstar_jp,
        })

    # ─── Écriture ──────────────────────────────────────────────────
    out_csv = config.DATA / "recall_curves_articles_only.csv"
    out_json = config.DATA / "recall_kstar_articles_only.json"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({
        "ks":           list(config.KS),
        "threshold":    config.KSTAR_THRESHOLD,
        "model":        config.MODEL_ID,
        "n_articles":   int(emb_arts.shape[0]),
        "n_questions":  len(rubrics),
        "per_question": kstar_summary,
    }, ensure_ascii=False, indent=2))
    print(f"✓ {out_csv}")
    print(f"✓ {out_json}")

    # ─── Récap stdout ──────────────────────────────────────────────
    print(f"\n{'='*88}")
    print(f"{'Question':22s}  {'gold_o':>6}  {'K*_o':>5}  {'gold_oo':>7}  {'K*_oo':>5}  "
          f"{'gold_jp':>7}  {'K*_jp':>5}")
    print(f"{'-'*88}")
    for s in kstar_summary:
        print(f"  {s['question_id']:22s}  {s['n_gold_oblig']:6d}  "
              f"{str(s['kstar_art_oblig'] or '—'):>5}  {s['n_gold_oblig_optio']:7d}  "
              f"{str(s['kstar_art_all'] or '—'):>5}  {s['n_gold_jp']:7d}  "
              f"{str(s['kstar_jp_via_graph'] or '—'):>5}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
