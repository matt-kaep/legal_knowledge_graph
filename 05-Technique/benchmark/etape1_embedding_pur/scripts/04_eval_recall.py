"""CLI : recall@K question↔article + question↔JP, courbes + K*."""
from __future__ import annotations
import json
import re
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from etape1 import config
from etape1.eval_recall import recall_at_k, kstar, extract_pourvoi_numbers

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")


def _encode_questions(questions: list[str]) -> np.ndarray:
    model = SentenceTransformer(config.MODEL_ID)
    embs = model.encode(questions, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    return embs


def _build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
    """pourvoi (XX-XX.XXX) → [jp_id, ...] depuis jp_index_penal.parquet, JP CC."""
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for row in jp.itertuples():
        n = (row.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(row.id)
    return out


def main() -> int:
    emb_arts = np.load(config.EMB_ARTICLES)
    emb_jp = np.load(config.EMB_JP)
    art_order = np.load(config.ARTICLES_ORDER, allow_pickle=True)
    jp_order = np.load(config.JP_ORDER, allow_pickle=True)

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    q_texts = [q["question"] for q in rubrics]
    print(f"Encodage de {len(q_texts)} questions…")
    Q = _encode_questions(q_texts)

    sim_arts = Q @ emb_arts.T
    sim_jp = Q @ emb_jp.T

    print("Construction map pourvoi → jp_id…")
    pourvoi_to_jpid = _build_pourvoi_to_jpid_map()
    print(f"  {len(pourvoi_to_jpid)} pourvois CC indexables")

    rows = []
    kstar_summary = []
    for qi, q in enumerate(rubrics):
        qid = q["id"]

        oblig = set(q["articles_attendus"].get("obligatoires", []))
        ranked_arts = list(art_order[np.argsort(-sim_arts[qi])])
        for k in config.KS:
            rows.append({"question_id": qid, "side": "article",
                          "metric": "obligatoires", "k": k,
                          "recall": recall_at_k(ranked_arts, oblig, k)})
        kstar_a = kstar(ranked_arts, oblig, config.KS, config.KSTAR_THRESHOLD)

        jp_gold_pourvois = set()
        for jp in q["jp_attendues"]:
            short = jp.get("short_ref") or ""
            jp_gold_pourvois.update(extract_pourvoi_numbers(short))
        jp_gold_ids = {jpid for p in jp_gold_pourvois for jpid in pourvoi_to_jpid.get(p, [])}

        ranked_jp = list(jp_order[np.argsort(-sim_jp[qi])])
        for k in config.KS:
            rows.append({"question_id": qid, "side": "jp",
                          "metric": "pourvoi_resolved", "k": k,
                          "recall": recall_at_k(ranked_jp, jp_gold_ids, k)})
        kstar_j = kstar(ranked_jp, jp_gold_ids, config.KS, config.KSTAR_THRESHOLD)

        kstar_summary.append({
            "question_id":   qid,
            "n_gold_oblig":  len(oblig),
            "kstar_article": kstar_a,
            "n_gold_jp":     len(jp_gold_ids),
            "n_gold_jp_extractible": len(jp_gold_pourvois),
            "kstar_jp":      kstar_j,
        })

    pd.DataFrame(rows).to_csv(config.RECALL_CURVES, index=False)
    config.RECALL_KSTAR.write_text(json.dumps({
        "ks":            list(config.KS),
        "threshold":     config.KSTAR_THRESHOLD,
        "per_question":  kstar_summary,
    }, ensure_ascii=False, indent=2))
    print(f"✓ {config.RECALL_CURVES}")
    print(f"✓ {config.RECALL_KSTAR}")
    print("\nRécap K* par question :")
    for s in kstar_summary:
        print(f"  {s['question_id']:20s}  K*_art={s['kstar_article']:>4}  "
              f"K*_jp={str(s['kstar_jp']):>5}  (gold_oblig={s['n_gold_oblig']}, "
              f"gold_jp={s['n_gold_jp']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
