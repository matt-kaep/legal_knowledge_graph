"""Éval recall sur le full corpus (tous codes).

Variante de 04a_eval_articles_only.py : utilise emb_articles_all.npy +
articles_order_all.npy + pairkey_to_graphcol_all.npy. Pool de candidats étendu
(~tous les codes mappés au lieu des 4 pénaux).

Sortie :
  recall_curves_articles_all.csv     (question, side, k, recall)
  recall_kstar_articles_all.json     (K* par question, deux sides)

Affiche aussi, pour chaque question dont le recall@10 article a chuté
par rapport au pool pénal, les top-3 articles non-pénaux classés au-dessus
des articles attendus.
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
    model = SentenceTransformer(config.MODEL_ID, device=device)
    model.max_seq_length = config.BATCH_MAX_LEN
    return model.encode(questions, normalize_embeddings=True,
                         convert_to_numpy=True, show_progress_bar=True).astype(np.float32)


def _build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
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
    PENAL = set(config.PENAL_CODES.keys())

    # ─── Chargement full corpus ────────────────────────────────────
    emb_arts = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    print(f"emb_articles_all : {emb_arts.shape}, art_order : {art_order.shape}, p2col : {p2col.shape}")

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids = z["jp_ids"]
    article_codes = z["article_codes"]  # parallel à article_ids (axis 1 du graphe)
    print(f"graphe biparti : {G.shape} ({G.nnz} arêtes)")

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    q_texts = [q["question"] for q in rubrics]
    print(f"Encodage {len(q_texts)} questions (CPU)…")
    Q = _encode_questions(q_texts, device="cpu")
    sim = Q @ emb_arts.T

    print("Map pourvoi → jp_id…")
    pourvoi_to_jpid = _build_pourvoi_to_jpid_map()
    print(f"  {len(pourvoi_to_jpid)} pourvois CC indexables")

    # ─── Comparaison avec pénal seulement (si dispo) ──────────────
    penal_kstar = None
    if config.RECALL_KSTAR.exists() or (config.DATA / "recall_kstar_articles_only.json").exists():
        p = config.DATA / "recall_kstar_articles_only.json"
        if p.exists():
            penal_kstar = {s["question_id"]: s for s in json.loads(p.read_text())["per_question"]}
    penal_curves = None
    pc = config.DATA / "recall_curves_articles_only.csv"
    if pc.exists():
        penal_curves = pd.read_csv(pc)

    rows = []
    kstar_summary = []
    diagnostics = []  # questions dont recall@10 chute

    for qi, q in enumerate(rubrics):
        qid = q["id"]
        order = np.argsort(-sim[qi])
        ranked_arts = list(art_order[order])
        ranked_cols = p2col[order]
        # Codes du ranking
        ranked_codes = article_codes[ranked_cols]

        # (A) ARTICLE-SIDE
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
        r10_all = recall_at_k(ranked_arts, all_a, 10)

        # (B) JP-VIA-GRAPH
        jp_gold_pourvois = set()
        for jp in q["jp_attendues"]:
            short = jp.get("short_ref") or ""
            jp_gold_pourvois.update(extract_pourvoi_numbers(short))
        jp_gold_ids = {jpid for p in jp_gold_pourvois for jpid in pourvoi_to_jpid.get(p, [])}

        r5_jp = None
        for k in config.KS:
            top_cols = ranked_cols[:k]
            jp_mask = (G[:, top_cols].sum(axis=1) > 0).A1
            jp_set = set(jp_ids[jp_mask].tolist())
            rec = (len(jp_set & jp_gold_ids) / max(len(jp_gold_ids), 1)) if jp_gold_ids else 0.0
            rows.append({"question_id": qid, "side": "jp_via_graph", "metric": "pourvoi_resolved",
                          "k": k, "recall": rec})
            if k == 5:
                r5_jp = rec

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

        # ─── Diagnostic : top-3 non-pénaux dans le top-10 ─────────
        top10_idx = order[:10]
        non_penal_top = []
        for j, idx in enumerate(top10_idx):
            code_slug = ranked_codes[j] if j < len(ranked_codes) else None
            pk = ranked_arts[j]
            if code_slug not in PENAL and pk not in all_a:
                non_penal_top.append({
                    "rank": j + 1,
                    "pair_key": str(pk),
                    "code_slug": str(code_slug),
                    "score": float(sim[qi, idx]),
                })
            if len(non_penal_top) >= 3:
                break

        # Comparer à recall@10 du pool pénal
        penal_r10 = None
        if penal_curves is not None:
            sel = penal_curves[(penal_curves["question_id"] == qid) &
                                (penal_curves["side"] == "article") &
                                (penal_curves["metric"] == "obligatoires_optionnels") &
                                (penal_curves["k"] == 10)]
            if len(sel):
                penal_r10 = float(sel["recall"].iloc[0])
        if penal_r10 is not None and r10_all < penal_r10 - 1e-9:
            diagnostics.append({
                "question_id": qid,
                "r10_penal_only": penal_r10,
                "r10_all": r10_all,
                "drop": penal_r10 - r10_all,
                "non_penal_top3_in_top10": non_penal_top,
            })

        kstar_summary.append({
            "question_id":         qid,
            "n_gold_oblig":        len(oblig),
            "n_gold_oblig_optio":  len(all_a),
            "kstar_art_oblig":     kstar_a_oblig,
            "kstar_art_all":       kstar_a_all,
            "r10_art_all":         r10_all,
            "n_gold_jp":           len(jp_gold_ids),
            "n_gold_jp_pourvois":  len(jp_gold_pourvois),
            "kstar_jp_via_graph":  kstar_jp,
            "r5_jp_via_graph":     r5_jp,
            "non_penal_top3_in_top10": non_penal_top,
        })

    out_csv = config.RECALL_CURVES_ALL
    out_json = config.RECALL_KSTAR_ALL
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps({
        "ks":           list(config.KS),
        "threshold":    config.KSTAR_THRESHOLD,
        "model":        config.MODEL_ID,
        "n_articles":   int(emb_arts.shape[0]),
        "n_questions":  len(rubrics),
        "per_question": kstar_summary,
        "diagnostics_drops_vs_penal_only": diagnostics,
    }, ensure_ascii=False, indent=2))
    print(f"✓ {out_csv}")
    print(f"✓ {out_json}")

    print(f"\n{'='*100}")
    print(f"{'Question':22s}  {'gold_o':>6}  {'K*_o':>5}  {'gold_oo':>7}  {'K*_oo':>5}  "
          f"{'r10_oo':>6}  {'gold_jp':>7}  {'K*_jp':>5}  {'r5_jp':>5}")
    print(f"{'-'*100}")
    for s in kstar_summary:
        print(f"  {s['question_id']:22s}  {s['n_gold_oblig']:6d}  "
              f"{str(s['kstar_art_oblig'] or '—'):>5}  {s['n_gold_oblig_optio']:7d}  "
              f"{str(s['kstar_art_all'] or '—'):>5}  "
              f"{s['r10_art_all']:6.3f}  {s['n_gold_jp']:7d}  "
              f"{str(s['kstar_jp_via_graph'] or '—'):>5}  "
              f"{(s['r5_jp_via_graph'] if s['r5_jp_via_graph'] is not None else 0):.3f}")

    if diagnostics:
        print(f"\n=== Diagnostics : {len(diagnostics)} question(s) avec drop r@10 vs pool pénal ===")
        for d in diagnostics:
            print(f"\n  {d['question_id']} : {d['r10_penal_only']:.3f} → {d['r10_all']:.3f} (drop {d['drop']:.3f})")
            for t in d["non_penal_top3_in_top10"]:
                print(f"    rank {t['rank']:>2} score={t['score']:.4f}  [{t['code_slug']}]  {t['pair_key']}")
    else:
        print("\n=== Aucune chute de r@10 vs pool pénal seul ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
