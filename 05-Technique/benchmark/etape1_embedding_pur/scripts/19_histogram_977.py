"""Histogrammes de distribution de la GT sur la cohorte 977 (GT ∩ pool).

Trois sous-figures :
  - GT articles strict (taille de |articles_attendus ∩ pool_articles|)
  - GT articles étendu (taille de |articles_attendus_etendu ∩ pool_articles|)
  - GT JP CC résolu (taille de |gold_jp ∩ pool_jp|)

Sortie : fig_hist_977_gt.png
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
from etape1 import config

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")
BENCH = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench/bench_global.json"
OUT = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench/fig_hist_977_gt.png"

def main():
    pool_art = set(np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True).tolist())
    pool_jp = set(np.load(config.JP_SUMMARY_ORDER, allow_pickle=True).tolist())

    jp_idx = pq.read_table(config.JP_INDEX, columns=["id","number","juris"]).to_pandas()
    jp_idx = jp_idx[jp_idx["juris"]=="CC"]
    pmap = {}
    for r in jp_idx.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            pmap.setdefault(n, []).append(r.id)

    qs = json.loads(BENCH.read_text())["questions"]
    n_art_s, n_art_e, n_jp = [], [], []
    for q in qs:
        arts = q.get("articles_attendus") or []
        pourvois = q.get("pourvois_cc") or []
        n_jp_res = q.get("n_jp_resolues", 0)
        if not arts or not pourvois or n_jp_res < 1:
            continue
        gt_s = set(arts) & pool_art
        gt_e = (set(q.get("articles_attendus_etendu") or arts)) & pool_art
        gold_jp = {jid for p in pourvois for jid in pmap.get(p, [])} & pool_jp
        n_art_s.append(len(gt_s))
        n_art_e.append(len(gt_e))
        n_jp.append(len(gold_jp))
    n_art_s = np.array(n_art_s)
    n_art_e = np.array(n_art_e)
    n_jp = np.array(n_jp)
    print(f"N = {len(n_art_s)}")
    print(f"Articles strict : mean={n_art_s.mean():.2f} med={np.median(n_art_s):.0f} max={n_art_s.max()}")
    print(f"Articles étendu : mean={n_art_e.mean():.2f} med={np.median(n_art_e):.0f} max={n_art_e.max()}")
    print(f"JP CC résolu     : mean={n_jp.mean():.2f} med={np.median(n_jp):.0f} max={n_jp.max()}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    accent = "#295c9b"
    accent2 = "#8b4513"

    ax = axes[0]
    ax.hist(np.clip(n_art_s, 0, 10), bins=np.arange(0, 11)-0.5,
            color=accent, edgecolor="white")
    ax.set_title("GT articles — strict", fontsize=11)
    ax.set_xlabel("|GT strict ∩ pool|")
    ax.set_ylabel("# questions")
    ax.text(0.97, 0.95, f"N = {len(n_art_s)}\nmean = {n_art_s.mean():.2f}\nmed = {int(np.median(n_art_s))}\nmax = {n_art_s.max()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none"))

    ax = axes[1]
    ax.hist(np.clip(n_art_e, 0, 30), bins=np.arange(0, 31)-0.5,
            color=accent2, edgecolor="white")
    ax.set_title("GT articles — étendu (∪ via JP)", fontsize=11)
    ax.set_xlabel("|GT étendu ∩ pool| (cap 30)")
    ax.text(0.97, 0.95, f"N = {len(n_art_e)}\nmean = {n_art_e.mean():.2f}\nmed = {int(np.median(n_art_e))}\nmax = {n_art_e.max()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none"))

    ax = axes[2]
    ax.hist(np.clip(n_jp, 0, 6), bins=np.arange(0, 7)-0.5,
            color="#2f7a3a", edgecolor="white")
    ax.set_title("GT JP — pourvois CC résolus", fontsize=11)
    ax.set_xlabel("|GT JP ∩ pool|")
    ax.text(0.97, 0.95, f"N = {len(n_jp)}\nmean = {n_jp.mean():.2f}\nmed = {int(np.median(n_jp))}\nmax = {n_jp.max()}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="none"))

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"✓ {OUT}")

if __name__ == "__main__":
    main()
