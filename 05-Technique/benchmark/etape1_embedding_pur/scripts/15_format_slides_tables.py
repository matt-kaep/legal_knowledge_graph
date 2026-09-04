"""Formate recall_at_k_summary.json en lignes LaTeX prêtes à coller dans la prez.

Usage : python 15_format_slides_tables.py
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
SUMMARY = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen/recall_at_k_summary.json"


def fmt_pct(x: float) -> str:
    return f"{100*x:.1f} \\%".replace(".", ",")


def fmt_recall(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def main() -> int:
    d = json.loads(SUMMARY.read_text())

    # Tri pour lecture humaine
    keys = sorted(d.keys())

    print("══ TABLEAU ARTICLES (slide 15) ──────────────────────────────")
    print(f"{'method':<28s} {'kin':>4s} {'K':>3s} {'n_q':>5s} {'mean_S':>8s} "
          f"{'recall':>7s} {'prec':>7s} {'r≥.5':>11s}")
    for K in (10, 20):
        for kin_filter in (None, 10, 20, 50):
            for method in (
                "B2-a_articles_open", "B2-b_articles_strict",
                "B3-e_art_via_jp",
                "B4-a_cross_art_union", "B4-b_cross_art_inter",
            ):
                kin_str = "nan" if kin_filter is None else f"{kin_filter}.0"
                key = f"{method}|kin={kin_str}|K={K}"
                if key not in d:
                    continue
                row = d[key]
                kin_disp = "-" if kin_filter is None else str(kin_filter)
                pct = row["pct_pass_50"]
                print(f"  {method:<28s} {kin_disp:>4s} {K:>3d} {row['n_q']:>5d} "
                      f"{row['mean_S']:>8.1f} {row['mean_recall']:>7.3f} "
                      f"{row['mean_precision']:>7.3f} {row['n_pass_50']:>4d}({pct:>3.0f}%)")

    print("\n══ TABLEAU JP (slide 16) ────────────────────────────────────")
    print(f"{'method':<28s} {'kin':>4s} {'K':>3s} {'n_q':>5s} {'mean_S':>8s} "
          f"{'recall':>7s} {'prec':>7s} {'r≥.5':>11s}")
    for K in (5, 10):
        for kin_filter in (None, 10, 20, 50):
            for method in (
                "B3-a_jp_direct",
                "B3-b_jp_via_graph",
                "B4-c_cross_jp_union", "B4-d_cross_jp_inter",
            ):
                kin_str = "nan" if kin_filter is None else f"{kin_filter}.0"
                key = f"{method}|kin={kin_str}|K={K}"
                if key not in d:
                    continue
                row = d[key]
                kin_disp = "-" if kin_filter is None else str(kin_filter)
                pct = row["pct_pass_50"]
                print(f"  {method:<28s} {kin_disp:>4s} {K:>3d} {row['n_q']:>5d} "
                      f"{row['mean_S']:>8.1f} {row['mean_recall']:>7.3f} "
                      f"{row['mean_precision']:>7.3f} {row['n_pass_50']:>4d}({pct:>3.0f}%)")

    print("\n══ LIGNES LATEX (slide 15 ARTICLES) ─────────────────────────")
    label_map = {
        "B2-a_articles_open": ("B2-a", "cosine art. pool ouvert"),
        "B2-b_articles_strict": ("B2-b", "cosine art. pool pénal strict"),
        "B3-e_art_via_jp": ("B3-e", "JP $\\to$ Art via graphe"),
        "B4-a_cross_art_union": ("B4-a", "cross-union art."),
        "B4-b_cross_art_inter": ("B4-b", "cross-inter art."),
    }
    for method, (code, label) in label_map.items():
        for kin_filter in (None, 10, 20, 50):
            for K in (10, 20):
                kin_str = "nan" if kin_filter is None else f"{kin_filter}.0"
                key = f"{method}|kin={kin_str}|K={K}"
                if key not in d:
                    continue
                r = d[key]
                kin_disp = "--" if kin_filter is None else str(kin_filter)
                print(
                    f"    {code} & {label:<28s} & {kin_disp:>3s} & {K:>3d} & "
                    f"{r['mean_S']:>6.1f} & {fmt_recall(r['mean_recall'])} & "
                    f"{fmt_pct(r['mean_precision'])} & {fmt_pct(r['pct_pass_50']/100)} \\\\"
                )

    print("\n══ LIGNES LATEX (slide 16 JP) ───────────────────────────────")
    label_map_jp = {
        "B3-a_jp_direct": ("B3-a", "cosine direct synthèses JP"),
        "B3-b_jp_via_graph": ("B3-b", "Art $\\to$ JP via graphe"),
        "B4-c_cross_jp_union": ("B4-c", "cross-union JP"),
        "B4-d_cross_jp_inter": ("B4-d", "cross-inter JP"),
    }
    for method, (code, label) in label_map_jp.items():
        for kin_filter in (None, 10, 20, 50):
            for K in (5, 10):
                kin_str = "nan" if kin_filter is None else f"{kin_filter}.0"
                key = f"{method}|kin={kin_str}|K={K}"
                if key not in d:
                    continue
                r = d[key]
                kin_disp = "--" if kin_filter is None else str(kin_filter)
                print(
                    f"    {code} & {label:<28s} & {kin_disp:>3s} & {K:>3d} & "
                    f"{r['mean_S']:>6.1f} & {fmt_recall(r['mean_recall'])} & "
                    f"{fmt_pct(r['mean_precision'])} & {fmt_pct(r['pct_pass_50']/100)} \\\\"
                )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
