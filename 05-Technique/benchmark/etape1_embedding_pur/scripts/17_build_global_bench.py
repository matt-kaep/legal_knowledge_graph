"""Bench global pénal = doctrine_qgen clean (3 421) + CRFPA pénal (8).

Sorties (dans data/global_bench/) :
  - bench_global.json                 — 3 429 questions au format normalisé
  - bench_article.json                — questions avec ≥1 article gold
  - bench_jp.json                     — questions avec ≥1 JP gold résolue
  - bench_article_extended.json       — bench_article + gold étendu via JP
  - bench_jp.json (pas d'étendu : circulaire)
  - stats.json                        — métriques par split
  - fig_hist_articles_global.png      — histogramme global articles (strict + étendu)
  - fig_hist_jp_global.png            — histogramme global JP
  - fig_hist_articles_per_source.png  — comparaison doctrine_qgen vs CRFPA articles
  - fig_hist_jp_per_source.png        — comparaison doctrine_qgen vs CRFPA JP
"""
from __future__ import annotations
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pyarrow.parquet as pq

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
from etape1 import config  # noqa: E402

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

DOCTRINE_PATH = REPO / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_clean_gemma4-26B-A4B.json"
CRFPA_PENAL_PATH = REPO / "05-Technique/benchmark/data/rubrics/cnb-penal-2025-consolidated.json"
CRFPA_PROC_PENAL_PATH = REPO / "05-Technique/benchmark/data/rubrics/cnb-procedure-penale-2025-consolidated.json"

OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def normalize_doctrine_question(q: dict) -> dict:
    """Aplatit une question doctrine_qgen au format normalisé."""
    articles = [
        f'{a["code_slug"]}:{a["article_num"]}'
        for a in q.get("articles_attendus", [])
    ]
    pourvois = [
        p for j in q.get("jp_attendues", [])
        if (p := (j.get("pourvoi") or "").strip()) and _POURVOI_RE.fullmatch(p)
    ]
    return {
        "qid": q["qid"],
        "source": "doctrine_qgen",
        "branche": q.get("branche", "penal"),
        "doc_id": q.get("doc_id"),
        "section_id": q.get("section_id"),
        "theme": q.get("theme"),
        "enonce": q["enonce"],
        "articles_attendus": articles,
        "pourvois_cc": pourvois,
    }


def normalize_crfpa_question(q: dict) -> dict:
    """Aplatit une question CRFPA au format normalisé. articles = obligatoires + optionnels."""
    aa = q.get("articles_attendus", {})
    articles = list(aa.get("obligatoires", [])) + list(aa.get("optionnels", []))
    # JP : extraire les pourvois depuis short_ref
    pourvois = []
    for jp in q.get("jp_attendues", []):
        ref = jp.get("short_ref") or jp.get("ref") or ""
        m = _POURVOI_RE.search(ref.replace(" ", ""))
        if m:
            pourvois.append(m.group(0))
    return {
        "qid": q["id"],
        "source": "crfpa",
        "branche": q.get("branche", "penal"),
        "doc_id": q.get("specialisation"),
        "section_id": None,
        "theme": q.get("specialisation"),
        "enonce": q.get("question", ""),
        "articles_attendus": articles,
        "pourvois_cc": pourvois,
        "articles_obligatoires": list(aa.get("obligatoires", [])),  # garde la strate dure
        "articles_optionnels": list(aa.get("optionnels", [])),
    }


def build_pourvoi_map() -> dict[str, list[str]]:
    """Pourvoi CC -> liste de jpid Judilibre."""
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def expand_articles_via_jp(questions: list[dict], pourvoi_map, G, jp_ids_graph,
                           article_ids_graph) -> None:
    """Mute chaque question pour ajouter articles_attendus_etendu = strict ∪ via JP."""
    jpid_to_row = {j: i for i, j in enumerate(jp_ids_graph)}
    for q in questions:
        strict = set(q["articles_attendus"])
        gold_jp_ids = [
            jid for p in q["pourvois_cc"] for jid in pourvoi_map.get(p, [])
        ]
        ext = set(strict)
        for jid in gold_jp_ids:
            if jid in jpid_to_row:
                for col in G[jpid_to_row[jid]].indices:
                    ext.add(article_ids_graph[int(col)])
        q["articles_attendus_etendu"] = sorted(ext)
        q["n_articles_strict"] = len(strict)
        q["n_articles_etendu"] = len(ext)
        q["n_pourvois_cc"] = len(q["pourvois_cc"])
        q["n_jp_resolues"] = len(set(gold_jp_ids))


def compute_stats(questions: list[dict], label: str) -> dict:
    n_art_strict = [q["n_articles_strict"] for q in questions]
    n_art_ext = [q["n_articles_etendu"] for q in questions]
    n_jp = [q["n_jp_resolues"] for q in questions]
    n_pourv = [q["n_pourvois_cc"] for q in questions]
    return {
        "label": label,
        "n_questions": len(questions),
        "by_source": dict(Counter(q["source"] for q in questions)),
        "articles_strict": {
            "mean": float(np.mean(n_art_strict)) if n_art_strict else 0,
            "median": float(np.median(n_art_strict)) if n_art_strict else 0,
            "min": int(np.min(n_art_strict)) if n_art_strict else 0,
            "max": int(np.max(n_art_strict)) if n_art_strict else 0,
            "p95": float(np.percentile(n_art_strict, 95)) if n_art_strict else 0,
            "pct_zero": float((np.array(n_art_strict) == 0).mean() * 100) if n_art_strict else 0,
            "histo": dict(Counter(min(int(x), 30) for x in n_art_strict)),
        },
        "articles_etendu": {
            "mean": float(np.mean(n_art_ext)) if n_art_ext else 0,
            "median": float(np.median(n_art_ext)) if n_art_ext else 0,
            "min": int(np.min(n_art_ext)) if n_art_ext else 0,
            "max": int(np.max(n_art_ext)) if n_art_ext else 0,
            "p95": float(np.percentile(n_art_ext, 95)) if n_art_ext else 0,
            "histo": dict(Counter(min(int(x), 60) for x in n_art_ext)),
        },
        "jp_resolues_cc": {
            "mean": float(np.mean(n_jp)) if n_jp else 0,
            "median": float(np.median(n_jp)) if n_jp else 0,
            "max": int(np.max(n_jp)) if n_jp else 0,
            "p95": float(np.percentile(n_jp, 95)) if n_jp else 0,
            "pct_zero": float((np.array(n_jp) == 0).mean() * 100) if n_jp else 0,
            "histo": dict(Counter(min(int(x), 10) for x in n_jp)),
        },
        "pourvois_total": {
            "mean": float(np.mean(n_pourv)) if n_pourv else 0,
            "max": int(np.max(n_pourv)) if n_pourv else 0,
        },
    }


def plot_dual_hist(arr_strict: np.ndarray, arr_ext: np.ndarray, title_prefix: str,
                   out_path: Path):
    """Histogramme strict + étendu côte à côte."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5), dpi=120)
    # Strict
    max_show = min(int(np.percentile(arr_strict, 99)) + 1, int(arr_strict.max()) + 1)
    bins = np.arange(0, max_show + 2) - 0.5
    counts, _, _ = ax1.hist(arr_strict, bins=bins, color="#3b6ea5",
                            edgecolor="white", alpha=0.85)
    ax1.set_xlabel("Nombre d'articles gold strict", fontsize=10)
    ax1.set_ylabel("Nombre de questions", fontsize=10)
    ax1.set_title(
        f"{title_prefix} — strict\n"
        f"N={len(arr_strict)}, médiane={int(np.median(arr_strict))}, "
        f"moyenne={arr_strict.mean():.2f}, max={arr_strict.max()}",
        fontsize=10)
    ax1.axvline(np.median(arr_strict), color="#c0392b", linestyle="--", linewidth=1.2)
    ax1.axvline(arr_strict.mean(), color="#27ae60", linestyle=":", linewidth=1.2)
    ax1.set_xticks(range(0, max_show + 1))
    ax1.grid(axis="y", alpha=0.3)
    for i, c in enumerate(counts):
        if c > 0:
            ax1.text(i, c + max(counts) * 0.01, f"{int(c)}", ha="center", fontsize=8)
    # Étendu
    cap = 30
    arr_ext_capped = np.minimum(arr_ext, cap)
    bins2 = np.arange(0, cap + 2) - 0.5
    counts2, _, _ = ax2.hist(arr_ext_capped, bins=bins2,
                             color="#d97706", edgecolor="white", alpha=0.85)
    ax2.set_xlabel(f"Nombre d'articles gold étendu (≥{cap} regroupés)", fontsize=10)
    ax2.set_ylabel("Nombre de questions", fontsize=10)
    ax2.set_title(
        f"{title_prefix} — étendu (∪ articles via JP)\n"
        f"N={len(arr_ext)}, médiane={int(np.median(arr_ext))}, "
        f"moyenne={arr_ext.mean():.2f}, max={arr_ext.max()}",
        fontsize=10)
    ax2.axvline(np.median(arr_ext), color="#c0392b", linestyle="--", linewidth=1.2)
    ax2.axvline(min(arr_ext.mean(), cap), color="#27ae60", linestyle=":", linewidth=1.2)
    ax2.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax2.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_jp_hist(arr: np.ndarray, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    max_show = min(int(np.percentile(arr, 99)) + 1, int(arr.max()) + 1)
    bins = np.arange(0, max_show + 2) - 0.5
    counts, _, _ = ax.hist(arr, bins=bins, color="#a85b3b", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Nombre de JP gold résolues CC", fontsize=11)
    ax.set_ylabel("Nombre de questions", fontsize=11)
    ax.set_title(
        f"{title}\n"
        f"N={len(arr)}, médiane={int(np.median(arr))}, moyenne={arr.mean():.2f}, max={arr.max()}",
        fontsize=11)
    ax.axvline(np.median(arr), color="#c0392b", linestyle="--", linewidth=1.2,
               label=f"médiane = {int(np.median(arr))}")
    ax.axvline(arr.mean(), color="#27ae60", linestyle=":", linewidth=1.2,
               label=f"moyenne = {arr.mean():.2f}")
    ax.set_xticks(range(0, max_show + 1))
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    for i, c in enumerate(counts):
        if c > 0:
            ax.text(i, c + max(counts) * 0.01, f"{int(c)}", ha="center", fontsize=8)
    pct_zero = (arr == 0).mean() * 100
    if pct_zero > 1:
        ax.text(0.98, 0.65,
                f"{pct_zero:.1f} % des questions\nn'ont aucune JP résolue\n"
                f"(soit {int((arr == 0).sum())}/{len(arr)})",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd",
                          edgecolor="#d4a017", alpha=0.9))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_per_source_hist(questions: list[dict], field: str, max_x: int,
                         title: str, out_path: Path):
    """Compare doctrine_qgen vs CRFPA sur le même histogramme superposé."""
    by_src = {}
    for q in questions:
        by_src.setdefault(q["source"], []).append(q[field])
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=120)
    bins = np.arange(0, max_x + 2) - 0.5
    colors = {"doctrine_qgen": "#3b6ea5", "crfpa": "#c0392b"}
    for src, vals in by_src.items():
        vals_cap = [min(int(x), max_x) for x in vals]
        ax.hist(vals_cap, bins=bins, color=colors[src], alpha=0.6, edgecolor="white",
                label=f"{src} (N={len(vals)})", density=True)
    ax.set_xlabel(field.replace("_", " "), fontsize=11)
    ax.set_ylabel("Densité", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xticks(range(0, max_x + 1, max(1, max_x // 10)))
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main() -> int:
    print("══ Chargement ─────────────────────────────────────────────")
    # 1) doctrine_qgen clean
    d = json.loads(DOCTRINE_PATH.read_text())
    doctrine_qs = [normalize_doctrine_question(q) for q in d["questions"]]
    print(f"  doctrine_qgen clean : {len(doctrine_qs)} questions")
    # 2) CRFPA pénal (penal + procédure pénale)
    crfpa_qs = []
    for fpath in (CRFPA_PENAL_PATH, CRFPA_PROC_PENAL_PATH):
        d2 = json.loads(fpath.read_text())
        for q in d2["questions"]:
            crfpa_qs.append(normalize_crfpa_question(q))
    print(f"  CRFPA pénal : {len(crfpa_qs)} questions ({len(json.loads(CRFPA_PENAL_PATH.read_text())['questions'])} penal + {len(json.loads(CRFPA_PROC_PENAL_PATH.read_text())['questions'])} proc-pénale)")
    # 3) Fusion
    bench_global = doctrine_qs + crfpa_qs
    print(f"  bench global : {len(bench_global)} questions")

    # 4) Charger graphe + pourvoi map pour l'extension
    print("\n══ Extension articles via JP gold ─────────────────────────")
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    pourvoi_map = build_pourvoi_map()
    expand_articles_via_jp(bench_global, pourvoi_map, G, jp_ids_graph,
                           article_ids_graph)

    # 5) Splits
    print("\n══ Splits ─────────────────────────────────────────────────")
    bench_article = [q for q in bench_global if q["n_articles_strict"] >= 1]
    bench_jp = [q for q in bench_global if q["n_jp_resolues"] >= 1]
    bench_article_ext = [q for q in bench_global if q["n_articles_etendu"] >= 1]
    print(f"  bench_global : {len(bench_global)}")
    print(f"  bench_article (≥1 art. strict) : {len(bench_article)}")
    print(f"  bench_article_etendu (≥1 art. étendu) : {len(bench_article_ext)}")
    print(f"  bench_jp (≥1 JP résolue) : {len(bench_jp)}")

    # 6) Sauvegarder
    print("\n══ Sauvegarde JSON ────────────────────────────────────────")
    (OUT_DIR / "bench_global.json").write_text(
        json.dumps({"questions": bench_global}, ensure_ascii=False, indent=2))
    (OUT_DIR / "bench_article.json").write_text(
        json.dumps({"questions": bench_article}, ensure_ascii=False, indent=2))
    (OUT_DIR / "bench_jp.json").write_text(
        json.dumps({"questions": bench_jp}, ensure_ascii=False, indent=2))
    (OUT_DIR / "bench_article_etendu.json").write_text(
        json.dumps({"questions": bench_article_ext}, ensure_ascii=False, indent=2))
    print(f"  ✓ {OUT_DIR}/bench_global.json ({len(bench_global)} q)")
    print(f"  ✓ {OUT_DIR}/bench_article.json ({len(bench_article)} q)")
    print(f"  ✓ {OUT_DIR}/bench_jp.json ({len(bench_jp)} q)")
    print(f"  ✓ {OUT_DIR}/bench_article_etendu.json ({len(bench_article_ext)} q)")

    # 7) Stats par split
    print("\n══ Stats ──────────────────────────────────────────────────")
    stats = {
        "bench_global": compute_stats(bench_global, "Bench global (doctrine_qgen + CRFPA)"),
        "bench_article": compute_stats(bench_article, "Bench article (≥1 article strict)"),
        "bench_article_etendu": compute_stats(bench_article_ext, "Bench article étendu (≥1 article étendu)"),
        "bench_jp": compute_stats(bench_jp, "Bench JP (≥1 JP résolue CC)"),
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"  ✓ {OUT_DIR}/stats.json")

    # 8) Histogrammes
    print("\n══ Histogrammes ───────────────────────────────────────────")
    arr_art_strict = np.array([q["n_articles_strict"] for q in bench_global])
    arr_art_ext = np.array([q["n_articles_etendu"] for q in bench_global])
    arr_jp = np.array([q["n_jp_resolues"] for q in bench_global])
    plot_dual_hist(arr_art_strict, arr_art_ext, "Articles gold — bench global",
                   OUT_DIR / "fig_hist_articles_global.png")
    plot_jp_hist(arr_jp, "JP gold résolues — bench global",
                 OUT_DIR / "fig_hist_jp_global.png")
    plot_per_source_hist(bench_global, "n_articles_strict", 5,
                         "Articles gold strict — doctrine_qgen vs CRFPA (densité normalisée)",
                         OUT_DIR / "fig_hist_articles_per_source.png")
    plot_per_source_hist(bench_global, "n_jp_resolues", 10,
                         "JP gold résolues — doctrine_qgen vs CRFPA (densité normalisée)",
                         OUT_DIR / "fig_hist_jp_per_source.png")
    print(f"  ✓ 4 figures dans {OUT_DIR}")

    # 9) Résumé console
    print("\n═══ Résumé global ═══")
    for key, s in stats.items():
        print(f"\n  [{s['label']}] N = {s['n_questions']}  par_source = {s['by_source']}")
        print(f"    articles strict : médiane {s['articles_strict']['median']:.0f}  "
              f"moy {s['articles_strict']['mean']:.2f}  max {s['articles_strict']['max']}  "
              f"pct=0 {s['articles_strict']['pct_zero']:.1f}%")
        print(f"    articles étendu : médiane {s['articles_etendu']['median']:.0f}  "
              f"moy {s['articles_etendu']['mean']:.2f}  max {s['articles_etendu']['max']}")
        print(f"    JP résolues CC  : médiane {s['jp_resolues_cc']['median']:.0f}  "
              f"moy {s['jp_resolues_cc']['mean']:.2f}  max {s['jp_resolues_cc']['max']}  "
              f"pct=0 {s['jp_resolues_cc']['pct_zero']:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
