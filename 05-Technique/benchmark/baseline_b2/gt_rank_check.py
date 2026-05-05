#!/usr/bin/env python3
"""A.X — À quel rang les pourvois GT remontent-ils dans le retrieval naïf ?

Pour chaque question pénale :
  1. Embed la question
  2. Calcule la similarité cosinus avec TOUTES les 118k JP
  3. Pour chaque pourvoi GT (toutes strates) : trouve son rang dans le tri
     décroissant des similarités

Diagnostic :
  - Si rang << 100 → l'embedding capte le signal, le top-K=10 est juste trop petit
  - Si rang >> 1000 → l'embedding rate la JP, augmenter K ne suffira pas
  - Distribution des rangs → indique le K minimal nécessaire pour récupérer
    une fraction décente des GT

Sortie :
  Results/gt_ranks.csv    rang de chaque pourvoi GT par question/strate
  Results/gt_ranks.txt    histogramme texte par question
"""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE     = Path(__file__).parent.resolve()
PARQUET  = HERE / "jp_index_penal.parquet"
RUBRICS  = HERE / "rubrics_penal.json"
EMB_DIR  = HERE / "embeddings"
RESULTS  = HERE / "Results"
RESULTS.mkdir(exist_ok=True)

EMB_FILE = EMB_DIR / "jp_embeddings_e5-base.npy"
IDS_FILE = EMB_DIR / "jp_order_e5-base.npy"
EMB_DIM  = 768
MODEL    = "intfloat/multilingual-e5-base"

POURVOI_RE = re.compile(r"\b(\d{2})[\s\-]*(\d{2})[.\-]?(\d{3})\b")


def extract_pourvoi(text: str) -> str | None:
    if not text:
        return None
    m = POURVOI_RE.search(text)
    return f"{m.group(1)}-{m.group(2)}.{m.group(3)}" if m else None


def main():
    from sentence_transformers import SentenceTransformer

    print("Lecture corpus + index pourvoi → jp_id…", flush=True)
    df = pq.read_table(PARQUET, columns=["id", "number"]).to_pandas()
    # Mapping pourvoi → liste de positions dans le parquet (si doublon, plusieurs positions)
    pourvoi2positions: dict[str, list[int]] = {}
    for i, num in enumerate(df["number"]):
        if num:
            pourvoi2positions.setdefault(num, []).append(i)

    parquet_ids = df["id"].to_numpy()

    print("Chargement embeddings…", flush=True)
    emb_ids = np.load(IDS_FILE, allow_pickle=True)
    n_total = len(emb_ids)
    emb = np.array(np.memmap(EMB_FILE, dtype=np.float32, mode="r", shape=(n_total, EMB_DIM)))

    # Mapping id → position dans la matrice d'embeddings
    id2embpos = {uid: i for i, uid in enumerate(emb_ids)}

    # Pour chaque pourvoi, on prend les jp_ids correspondants → positions emb
    def pourvoi_to_emb_positions(pourvoi: str) -> list[int]:
        if pourvoi not in pourvoi2positions:
            return []
        positions = []
        for parquet_pos in pourvoi2positions[pourvoi]:
            uid = parquet_ids[parquet_pos]
            if uid in id2embpos:
                positions.append(id2embpos[uid])
        return positions

    print(f"Chargement modèle {MODEL}…", flush=True)
    model = SentenceTransformer(MODEL)

    questions = json.loads(RUBRICS.read_text(encoding="utf-8"))["questions"]
    print(f"Questions : {len(questions)}\n", flush=True)

    rows = []
    txt_lines = ["# Rangs des pourvois GT dans le retrieval naïf",
                 f"# Modèle : {MODEL}",
                 f"# Corpus : {n_total} JP pénales\n"]

    for q in questions:
        qid = q["id"]
        rubric = q.get("rubric") or {}

        # Embed question
        q_vec = model.encode(["query: " + q["question"]],
                              normalize_embeddings=True,
                              convert_to_numpy=True)[0]

        # Score vs tout le corpus
        scores = emb @ q_vec
        # rang[i] = position de la JP i dans le tri décroissant des scores
        # méthode : argsort renvoie indices triés croissants → on inverse
        order_desc = np.argsort(-scores)        # indices triés par score décroissant
        rank_of = np.empty(n_total, dtype=np.int32)
        rank_of[order_desc] = np.arange(n_total)  # rank_of[i] = rang de la JP i (0-indexé)

        txt_lines.append(f"\n{'='*78}")
        txt_lines.append(f"  {qid}  [{q.get('branche')}]")
        txt_lines.append(f"  {q.get('specialisation','')[:90]}")
        txt_lines.append(f"{'='*78}")

        for strate in ("core", "expected", "expert"):
            for item in (rubric.get(strate) or []):
                ref = item.get("linked_jp", "")
                if not ref:
                    continue
                pourvoi = extract_pourvoi(ref)
                if pourvoi is None:
                    txt_lines.append(f"  [{strate:<8s}] ⚠ non parsable : {ref[:60]}")
                    continue
                emb_positions = pourvoi_to_emb_positions(pourvoi)
                if not emb_positions:
                    txt_lines.append(f"  [{strate:<8s}] ✗ pourvoi {pourvoi} absent du corpus indexé")
                    rows.append({"qid": qid, "strate": strate, "pourvoi": pourvoi,
                                 "ref_text": ref[:80], "rank": "absent",
                                 "n_collisions": 0})
                    continue

                # Si plusieurs jp avec le même pourvoi (collision) : on prend le meilleur rang
                ranks = [rank_of[p] for p in emb_positions]
                best_rank = min(ranks) + 1  # 1-indexé pour lisibilité
                txt_lines.append(f"  [{strate:<8s}] {pourvoi}  rang #{best_rank:>6d}  "
                                 f"(score {scores[emb_positions[ranks.index(best_rank-1)]]:.4f})  "
                                 f"← {ref[:60]}")
                rows.append({
                    "qid": qid, "strate": strate, "pourvoi": pourvoi,
                    "ref_text": ref[:80],
                    "rank": int(best_rank),
                    "n_collisions": len(emb_positions),
                })

    # Écriture CSV
    csv_path = RESULTS / "gt_ranks.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["qid", "strate", "pourvoi", "ref_text",
                                           "rank", "n_collisions"])
        w.writeheader()
        w.writerows(rows)

    # Stats agrégées
    txt_lines.append(f"\n\n{'═'*78}")
    txt_lines.append("  STATISTIQUES")
    txt_lines.append(f"{'═'*78}\n")

    valid = [r for r in rows if isinstance(r["rank"], int)]
    if valid:
        ranks = [r["rank"] for r in valid]
        thresholds = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
        txt_lines.append(f"Total pourvois GT évalués : {len(valid)}")
        txt_lines.append(f"Rang médian : {sorted(ranks)[len(ranks)//2]}")
        txt_lines.append(f"Rang moyen  : {sum(ranks)//len(ranks)}")
        txt_lines.append(f"Min / Max   : {min(ranks)} / {max(ranks)}\n")
        txt_lines.append("Recall cumulé par K :")
        for t in thresholds:
            n_in = sum(1 for r in ranks if r <= t)
            txt_lines.append(f"  top-{t:<5d}  : {n_in:>3d}/{len(valid)}  ({100*n_in/len(valid):.0f}%)")

        # Par strate
        txt_lines.append("\nRang médian par strate :")
        for s in ("core", "expected", "expert"):
            rk = [r["rank"] for r in valid if r["strate"] == s]
            if rk:
                txt_lines.append(f"  {s:<10s}  median={sorted(rk)[len(rk)//2]:>6d}  "
                                 f"min={min(rk)}  max={max(rk)}  n={len(rk)}")

    out_txt = RESULTS / "gt_ranks.txt"
    out_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    print(f"\nDétail → {out_txt}")
    print(f"CSV    → {csv_path}\n")
    # Affichage des stats
    for line in txt_lines[-25:]:
        print(line)


if __name__ == "__main__":
    main()
