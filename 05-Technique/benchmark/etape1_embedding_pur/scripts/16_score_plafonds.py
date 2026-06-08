"""Calcule les scores plafond (oracle) par K, strate de gold et modalité.

Pour chaque question :
  - precision_plafond(K) = min(|R|, K) / K           — borne par |R| ou K
  - recall_plafond(K) = min(|R ∩ pool|, K) / |R|     — borne par K et par la couverture du pool

On moyenne sur les 1 707 questions et on rapporte :
  - articles (strict + étendu) à K=10, K=20
  - JP CC résolu à K=5, K=10

Sortie : score_plafonds.json + table console.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import pyarrow.parquet as pq

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
from etape1 import config  # noqa: E402

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")
CORPUS_PATH = REPO / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_strict_gemma4-26B-A4B.json"
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen"


def main() -> int:
    d = json.loads(CORPUS_PATH.read_text())
    qs = d["questions"]

    # Charger pool article (ce qui est indexé via BGE-M3)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    pool_art = set(art_order.tolist())  # pairkeys indexés (31 357 articles)

    # Charger pool JP synthèses (ce qui est indexé)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    pool_jp = set(jp_order.tolist())  # 116 755 JP indexées

    # Charger graphe pour gold étendu
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    jpid_to_row = {j: i for i, j in enumerate(jp_ids_graph)}

    # Charger map pourvoi -> jpid
    jp_idx = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp_idx = jp_idx[jp_idx["juris"] == "CC"]
    pourvoi_map: dict[str, list] = {}
    for r in jp_idx.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            pourvoi_map.setdefault(n, []).append(r.id)

    KS_ARTICLES = [10, 20]
    KS_JP = [5, 10]

    # Calcul par question
    accum = {
        ("art_strict", K): {"prec": [], "recall": [], "n_gold": []}
        for K in KS_ARTICLES
    }
    accum.update({
        ("art_extended", K): {"prec": [], "recall": [], "n_gold": []}
        for K in KS_ARTICLES
    })
    accum.update({
        ("jp", K): {"prec": [], "recall": [], "n_gold": []}
        for K in KS_JP
    })

    for q in qs:
        # Gold articles strict
        oblig = {f'{a["code_slug"]}:{a["article_num"]}' for a in q.get("articles_attendus", [])}
        # Gold JP résolu
        pourvois = [
            p for j in q.get("jp_attendues", [])
            if (p := (j.get("pourvoi") or "").strip()) and _POURVOI_RE.fullmatch(p)
        ]
        gold_jp = {jid for p in pourvois for jid in pourvoi_map.get(p, [])}
        # Gold articles étendu
        oblig_ext = set(oblig)
        for jid in gold_jp:
            if jid in jpid_to_row:
                for col in G[jpid_to_row[jid]].indices:
                    oblig_ext.add(article_ids_graph[int(col)])

        # ─── plafonds articles (strict)
        n_R = len(oblig)
        n_R_in_pool = len(oblig & pool_art)
        for K in KS_ARTICLES:
            if n_R > 0:
                # precision_plafond = min(n_R, K) / K
                p_max = min(n_R, K) / K
                # recall_plafond = min(n_R_in_pool, K) / n_R
                r_max = min(n_R_in_pool, K) / n_R
                accum[("art_strict", K)]["prec"].append(p_max)
                accum[("art_strict", K)]["recall"].append(r_max)
                accum[("art_strict", K)]["n_gold"].append(n_R)

        # ─── plafonds articles (étendu)
        n_R = len(oblig_ext)
        n_R_in_pool = len(oblig_ext & pool_art)
        for K in KS_ARTICLES:
            if n_R > 0:
                p_max = min(n_R, K) / K
                r_max = min(n_R_in_pool, K) / n_R
                accum[("art_extended", K)]["prec"].append(p_max)
                accum[("art_extended", K)]["recall"].append(r_max)
                accum[("art_extended", K)]["n_gold"].append(n_R)

        # ─── plafonds JP (CC résolu)
        n_R = len(gold_jp)
        n_R_in_pool = len(gold_jp & pool_jp)
        for K in KS_JP:
            if n_R > 0:
                p_max = min(n_R, K) / K
                r_max = min(n_R_in_pool, K) / n_R
                accum[("jp", K)]["prec"].append(p_max)
                accum[("jp", K)]["recall"].append(r_max)
                accum[("jp", K)]["n_gold"].append(n_R)

    # Aggrégats
    summary = {}
    print(f"\n{'modalité':<14s} {'K':>3s} {'n_q':>5s} {'|R| moy':>8s} "
          f"{'précision plafond':>18s} {'recall plafond':>16s}")
    print("─" * 75)
    for (modality, K), data in accum.items():
        n_q = len(data["prec"])
        if n_q == 0:
            continue
        mean_p = np.mean(data["prec"])
        mean_r = np.mean(data["recall"])
        mean_n = np.mean(data["n_gold"])
        summary[f"{modality}|K={K}"] = {
            "n_q": n_q,
            "mean_n_gold": float(mean_n),
            "mean_precision_plafond": float(mean_p),
            "mean_recall_plafond": float(mean_r),
            "interpretation": f"À K={K}, on ne peut pas dépasser {100*mean_p:.1f}% de précision moyenne ni {100*mean_r:.1f}% de recall moyen (oracle = retriever idéal qui ramène uniquement le gold).",
        }
        print(f"{modality:<14s} {K:>3d} {n_q:>5d} {mean_n:>8.2f} "
              f"{100*mean_p:>17.1f}% {100*mean_r:>15.1f}%")

    out = OUT_DIR / "score_plafonds.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n✓ {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
