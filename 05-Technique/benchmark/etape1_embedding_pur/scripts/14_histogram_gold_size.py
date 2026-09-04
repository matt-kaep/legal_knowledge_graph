"""Histogramme du nombre d'articles et de JP attendus par question.

Sortie : data/doctrine_qgen/fig_gold_size_articles.png
         data/doctrine_qgen/fig_gold_size_jp.png
         data/doctrine_qgen/gold_size_stats.json
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
import scipy.sparse as sp

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
from etape1 import config  # noqa: E402

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")
CORPUS_PATH = REPO / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_strict_gemma4-26B-A4B.json"
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def main() -> int:
    d = json.loads(CORPUS_PATH.read_text())
    n_art_list = []
    n_art_ext_list = []
    n_jp_pourvois_list = []
    n_jp_resolved_list = []

    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    pourvoi_map: dict[str, list] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            pourvoi_map.setdefault(n, []).append(r.id)

    # Charger le graphe pour calculer le gold étendu
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    jpid_to_row = {j: i for i, j in enumerate(jp_ids_graph)}

    for q in d["questions"]:
        oblig = {
            f'{a["code_slug"]}:{a["article_num"]}'
            for a in q.get("articles_attendus", [])
        }
        n_art_list.append(len(oblig))
        pourvois = {
            p for j in q.get("jp_attendues", [])
            if (p := (j.get("pourvoi") or "").strip()) and _POURVOI_RE.fullmatch(p)
        }
        n_jp_pourvois_list.append(len(pourvois))
        gold_jp_ids = [jid for p in pourvois for jid in pourvoi_map.get(p, [])]
        n_jp_resolved_list.append(len(gold_jp_ids))

        # Extension via graphe
        ext = set(oblig)
        for jid in gold_jp_ids:
            if jid in jpid_to_row:
                row = jpid_to_row[jid]
                for col in G[row].indices:
                    ext.add(article_ids_graph[int(col)])
        n_art_ext_list.append(len(ext))

    n_art = np.array(n_art_list)
    n_art_ext = np.array(n_art_ext_list)
    n_jp_p = np.array(n_jp_pourvois_list)
    n_jp_r = np.array(n_jp_resolved_list)

    stats = {
        "n_questions": len(n_art),
        "articles": {
            "mean": float(n_art.mean()),
            "median": float(np.median(n_art)),
            "min": int(n_art.min()),
            "max": int(n_art.max()),
            "q25": float(np.percentile(n_art, 25)),
            "q75": float(np.percentile(n_art, 75)),
            "p90": float(np.percentile(n_art, 90)),
            "p95": float(np.percentile(n_art, 95)),
            "pct_zero": float((n_art == 0).mean() * 100),
            "pct_single": float((n_art == 1).mean() * 100),
            "histogram": dict(Counter(n_art.tolist())),
        },
        "articles_extended": {
            "mean": float(n_art_ext.mean()),
            "median": float(np.median(n_art_ext)),
            "min": int(n_art_ext.min()),
            "max": int(n_art_ext.max()),
            "q25": float(np.percentile(n_art_ext, 25)),
            "q75": float(np.percentile(n_art_ext, 75)),
            "p90": float(np.percentile(n_art_ext, 90)),
            "p95": float(np.percentile(n_art_ext, 95)),
            "pct_same_as_strict": float((n_art_ext == n_art).mean() * 100),
            "histogram_capped30": dict(Counter(min(int(x), 30) for x in n_art_ext.tolist())),
        },
        "jp_pourvois_total": {
            "mean": float(n_jp_p.mean()),
            "median": float(np.median(n_jp_p)),
            "min": int(n_jp_p.min()),
            "max": int(n_jp_p.max()),
            "q25": float(np.percentile(n_jp_p, 25)),
            "q75": float(np.percentile(n_jp_p, 75)),
            "p90": float(np.percentile(n_jp_p, 90)),
            "p95": float(np.percentile(n_jp_p, 95)),
            "pct_zero": float((n_jp_p == 0).mean() * 100),
            "histogram": dict(Counter(n_jp_p.tolist())),
        },
        "jp_resolved_cc": {
            "mean": float(n_jp_r.mean()),
            "median": float(np.median(n_jp_r)),
            "min": int(n_jp_r.min()),
            "max": int(n_jp_r.max()),
            "q25": float(np.percentile(n_jp_r, 25)),
            "q75": float(np.percentile(n_jp_r, 75)),
            "p90": float(np.percentile(n_jp_r, 90)),
            "p95": float(np.percentile(n_jp_r, 95)),
            "pct_zero": float((n_jp_r == 0).mean() * 100),
            "n_q_at_least_one": int((n_jp_r >= 1).sum()),
            "histogram": dict(Counter(n_jp_r.tolist())),
        },
    }

    (OUT_DIR / "gold_size_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2)
    )
    print(f"✓ {OUT_DIR}/gold_size_stats.json")

    # ─── Histogramme articles : strict + étendu côte à côte
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)

    # Subplot 1 : strict
    max_show = min(int(np.percentile(n_art, 99)) + 1, n_art.max() + 1)
    bins = np.arange(0, max_show + 2) - 0.5
    counts1, _, _ = ax1.hist(n_art, bins=bins, color="#3b6ea5", edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Nombre d'articles gold strict ($|R^{art}_{strict}|$)", fontsize=10)
    ax1.set_ylabel("Nombre de questions", fontsize=10)
    ax1.set_title(
        f"Gold strict — articles_attendus du JSON\n"
        f"N = {len(n_art)}, médiane = {int(np.median(n_art))}, moyenne = {n_art.mean():.2f}, max = {n_art.max()}",
        fontsize=10,
    )
    ax1.axvline(np.median(n_art), color="#c0392b", linestyle="--", linewidth=1.2,
                label=f"médiane = {int(np.median(n_art))}")
    ax1.axvline(n_art.mean(), color="#27ae60", linestyle=":", linewidth=1.2,
                label=f"moyenne = {n_art.mean():.2f}")
    ax1.set_xticks(range(0, max_show + 1))
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for i, c in enumerate(counts1):
        if c > 0:
            ax1.text(i, c + max(counts1) * 0.01, f"{int(c)}", ha="center", fontsize=8)

    # Subplot 2 : étendu (cap visuel à 30 pour éviter l'écrasement par le max=57)
    cap = 30
    n_art_ext_capped = np.minimum(n_art_ext, cap)
    bins2 = np.arange(0, cap + 2) - 0.5
    counts2, _, _ = ax2.hist(n_art_ext_capped, bins=bins2,
                             color="#d97706", edgecolor="white", alpha=0.85)
    ax2.set_xlabel(f"Nombre d'articles gold étendu ($|R^{{art}}_{{ext}}|$, $\\geq{cap}$ regroupés)",
                   fontsize=10)
    ax2.set_ylabel("Nombre de questions", fontsize=10)
    ax2.set_title(
        f"Gold étendu — strict ∪ articles cités par JP gold via graphe\n"
        f"N = {len(n_art_ext)}, médiane = {int(np.median(n_art_ext))}, "
        f"moyenne = {n_art_ext.mean():.2f}, max = {n_art_ext.max()}",
        fontsize=10,
    )
    ax2.axvline(np.median(n_art_ext), color="#c0392b", linestyle="--", linewidth=1.2,
                label=f"médiane = {int(np.median(n_art_ext))}")
    ax2.axvline(min(n_art_ext.mean(), cap), color="#27ae60", linestyle=":", linewidth=1.2,
                label=f"moyenne = {n_art_ext.mean():.2f}")
    ax2.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    pct_same = (n_art_ext == n_art).mean() * 100
    ax2.text(
        0.98, 0.55,
        f"{pct_same:.1f} % des questions :\nstrict = étendu\n(pas de JP résolue\nou JP citant déjà\nles articles gold)",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#d4a017", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_gold_size_articles.png", bbox_inches="tight")
    plt.close()
    print(f"✓ {OUT_DIR}/fig_gold_size_articles.png")

    # ─── Histogramme JP (résolu CC)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    max_show = min(int(np.percentile(n_jp_r, 99)) + 1, n_jp_r.max() + 1)
    bins = np.arange(0, max_show + 2) - 0.5
    counts, _, _ = ax.hist(n_jp_r, bins=bins, color="#a85b3b", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Nombre de JP gold résolues CC ($|R_{Q_a}^{jp}|$)", fontsize=11)
    ax.set_ylabel("Nombre de questions", fontsize=11)
    ax.set_title(
        f"Distribution du nombre de JP attendues (résolues côté Cour de cass.)  "
        f"(N = {len(n_jp_r)}, médiane = {int(np.median(n_jp_r))}, moyenne = {n_jp_r.mean():.1f})",
        fontsize=11,
    )
    ax.axvline(np.median(n_jp_r), color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"médiane = {int(np.median(n_jp_r))}")
    ax.axvline(n_jp_r.mean(), color="#27ae60", linestyle=":", linewidth=1.2,
               label=f"moyenne = {n_jp_r.mean():.1f}")
    ax.set_xticks(range(0, max_show + 1))
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    pct_zero = (n_jp_r == 0).mean() * 100
    ax.text(
        0.98, 0.65,
        f"{pct_zero:.1f} % des questions n'ont aucune JP résolue\n"
        f"(soit {int((n_jp_r == 0).sum())}/{len(n_jp_r)}, hors champ CC)",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#d4a017", alpha=0.9),
    )
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c + max(counts) * 0.01, f"{int(c)}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "fig_gold_size_jp.png", bbox_inches="tight")
    plt.close()
    print(f"✓ {OUT_DIR}/fig_gold_size_jp.png")

    # ─── Résumé console
    print("\n═══ Résumé ═══")
    print(f"  N questions               : {len(n_art)}")
    print(f"  articles strict : mean={n_art.mean():.2f}  median={np.median(n_art):.0f}  "
          f"max={n_art.max()}  p95={np.percentile(n_art, 95):.0f}  "
          f"pct(=0)={(n_art == 0).mean()*100:.1f}%  pct(=1)={(n_art == 1).mean()*100:.1f}%")
    print(f"  articles étendu : mean={n_art_ext.mean():.2f}  median={np.median(n_art_ext):.0f}  "
          f"max={n_art_ext.max()}  p95={np.percentile(n_art_ext, 95):.0f}  "
          f"p99={np.percentile(n_art_ext, 99):.0f}  "
          f"pct(strict=étendu)={(n_art_ext == n_art).mean()*100:.1f}%")
    print(f"  JP pourv.       : mean={n_jp_p.mean():.2f}  median={np.median(n_jp_p):.0f}  "
          f"max={n_jp_p.max()}  p95={np.percentile(n_jp_p, 95):.0f}  "
          f"pct(=0)={(n_jp_p == 0).mean()*100:.1f}%")
    print(f"  JP CC résolu    : mean={n_jp_r.mean():.2f}  median={np.median(n_jp_r):.0f}  "
          f"max={n_jp_r.max()}  p95={np.percentile(n_jp_r, 95):.0f}  "
          f"pct(=0)={(n_jp_r == 0).mean()*100:.1f}%  "
          f"n_q(≥1)={(n_jp_r >= 1).sum()}/{len(n_jp_r)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
