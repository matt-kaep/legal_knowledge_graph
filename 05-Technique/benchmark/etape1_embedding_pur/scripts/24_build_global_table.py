"""Construit le grand tableau global de synthèse (chantier 2 Week-10).

Lit eval_m1_m2.csv (script 18) + ppr_naive_eval.csv (script 20), produit :
  - global_table_articles.csv  : 1 ligne / méthode champion / 10 cols panel × strict/ext
  - global_table_jp.csv        : 1 ligne / méthode champion / 5 cols panel
  - global_table.md            : rendu Markdown lisible pour la présentation
  - global_table_full.csv      : tout, y compris non-champions (pour annexe)

Méthodes incluses (cf. handoff M3) :
  Articles : B2-a, B3-e (k_in=10 champion), PPR-row α=0,85 / 0,95
  JP       : B3-a, B4-d (k_in=50), B4-e (k_in=20 champion), B4-f (k_in=10), PPR-row α=0,95

Méthodes droppées (dégénérescence ou collapse) :
  B4-a (≡ B2-a), B4-c (≡ B3-a), PPR-sym (≡ cosine)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
DATA = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"

# Suffixes du panel
METRICS = ("m1", "hit", "mrr", "ndcg", "m2")
SUFFIXES_ART = [f"{m}_{r}" for r in ("strict", "ext") for m in METRICS]
SUFFIXES_JP = list(METRICS)

# Sélection des champions à afficher dans le tableau public
ART_SELECTION = [
    # (label, source df, filtre)
    ("B2-a (cosine articles)",       "b", lambda d: (d["method"] == "B2-a")),
    ("B3-e — JP→Art via graphe (k_in=10)", "b",
        lambda d: (d["method"] == "B3-e") & (d["k_in"] == 10)),
    ("PPR row-norm α=0,85",          "p", lambda d: (d["norm"] == "row") & (d["alpha"] == 0.85)),
    ("PPR row-norm α=0,95",          "p", lambda d: (d["norm"] == "row") & (d["alpha"] == 0.95)),
]

JP_SELECTION = [
    ("B3-a (cosine JP)",                "b", lambda d: (d["method"] == "B3-a")),
    ("B4-d — intersection (k_in=50)",   "b",
        lambda d: (d["method"] == "B4-d") & (d["k_in"] == 50)),
    ("B4-e — RRF (k_in=20)",            "b",
        lambda d: (d["method"] == "B4-e") & (d["k_in"] == 20)),
    ("B4-f — citation-weighted (k_in=10)", "b",
        lambda d: (d["method"] == "B4-f") & (d["k_in"] == 10)),
    ("PPR row-norm α=0,95 (côté JP)",   "p", lambda d: (d["norm"] == "row") & (d["alpha"] == 0.95)),
]


def aggregate_b(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Agrège df (eval_m1_m2.csv) côté articles (modality=art) ou JP (modality=jp).

    Renvoie un index (method, k_in) avec les colonnes du panel selon le côté.
    """
    sub = df[df["modality"] == ("art" if side == "art" else "jp")].copy()
    cols = (
        [f"{m}_{r}" for m in METRICS for r in ("strict", "ext")]
        if side == "art" else
        [f"{m}_strict" for m in METRICS]  # côté JP : strict == ext (gold_jp_ext = gold_jp)
    )
    agg = sub.groupby(["method", "k_in"], dropna=False)[cols].mean()
    if side == "jp":
        # Rename strict → plain pour JP
        agg = agg.rename(columns={f"{m}_strict": m for m in METRICS})
    return agg


def aggregate_p(df_ppr: pd.DataFrame, side: str) -> pd.DataFrame:
    """Agrège ppr_naive_eval.csv. Renvoie index (norm, alpha) avec cols panel."""
    if side == "art":
        cols = [f"{m}_{r}_art" for m in METRICS for r in ("strict", "ext")]
    else:
        cols = [f"{m}_jp" for m in METRICS]
    agg = df_ppr.groupby(["norm", "alpha"])[cols].mean()
    if side == "art":
        # Strip _art suffix
        agg.columns = [c.replace("_art", "") for c in agg.columns]
    else:
        agg.columns = [c.replace("_jp", "") for c in agg.columns]
    return agg


def collect_rows(selection, df_b, df_p, side: str) -> pd.DataFrame:
    """Pour chaque entrée de selection, extrait la ligne agrégée correspondante."""
    agg_b = aggregate_b(df_b, side)
    agg_p = aggregate_p(df_p, side)
    rows = []
    for label, source, filt in selection:
        if source == "b":
            sub = df_b[filt(df_b) & (df_b["modality"] == ("art" if side == "art" else "jp"))]
            if sub.empty:
                continue
            row = aggregate_b(sub, side).iloc[0].to_dict()
        else:
            sub = df_p[filt(df_p)]
            if sub.empty:
                continue
            row = aggregate_p(sub, side).iloc[0].to_dict()
        rows.append({"méthode": label, **row})
    return pd.DataFrame(rows)


def fmt_table_articles(df: pd.DataFrame) -> str:
    """Rendu Markdown 11 cols (label + 5 strict + 5 ext) avec format 3 décimales."""
    cols_strict = [f"{m}_strict" for m in METRICS]
    cols_ext    = [f"{m}_ext"    for m in METRICS]
    headers = (
        ["méthode"] +
        [f"M1_s", "Hit_s", "MRR_s", "NDCG_s", "M2_s"] +
        [f"M1_e", "Hit_e", "MRR_e", "NDCG_e", "M2_e"]
    )
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = [str(r["méthode"])]
        for c in cols_strict + cols_ext:
            v = r.get(c, float("nan"))
            cells.append(f"{v:.3f}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_table_jp(df: pd.DataFrame) -> str:
    """Rendu Markdown 6 cols (label + 5 panel)."""
    headers = ["méthode", "M1", "Hit", "MRR", "NDCG", "M2"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = [str(r["méthode"])]
        for c in METRICS:
            v = r.get(c, float("nan"))
            cells.append(f"{v:.3f}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    df_b = pd.read_csv(DATA / "eval_m1_m2.csv")
    df_p = pd.read_csv(DATA / "ppr_naive_eval.csv")
    print(f"  eval_m1_m2.csv : {len(df_b)} lignes")
    print(f"  ppr_naive_eval.csv : {len(df_p)} lignes")

    art = collect_rows(ART_SELECTION, df_b, df_p, "art")
    jp  = collect_rows(JP_SELECTION,  df_b, df_p, "jp")

    out_art = DATA / "global_table_articles.csv"
    out_jp  = DATA / "global_table_jp.csv"
    art.to_csv(out_art, index=False)
    jp.to_csv(out_jp, index=False)
    print(f"\n✓ {out_art}  ({len(art)} méthodes)")
    print(f"✓ {out_jp}  ({len(jp)} méthodes)")

    # Markdown global
    md_lines = [
        "# Grand tableau global — cohorte 971, K=10",
        "",
        "Panel : M1=Recall@K, Hit=Hit@K, MRR=MRR@K, NDCG=NDCG@K, M2=rang moyen normalisé.",
        "Côté articles : strict (|GT| moy 1,23) + étendu (|GT| moy 7,39).",
        "Côté JP : modalité unique (|GT| moy 1,13).",
        "",
        "## Articles",
        "",
        fmt_table_articles(art),
        "",
        "## JP",
        "",
        fmt_table_jp(jp),
        "",
        "## Méthodes droppées (collapse vers parent)",
        "",
        "- **B4-a** ≡ B2-a (re-rank cosine sur union dégénère vers top cosine)",
        "- **B4-c** ≡ B3-a (idem côté JP)",
        "- **PPR sym-norm** ≡ B2-a (matrice non-stochastique → mass decay)",
    ]
    out_md = DATA / "global_table.md"
    out_md.write_text("\n".join(md_lines))
    print(f"✓ {out_md}")

    print("\n══ Articles ────────────────────────────────────────────────")
    print(fmt_table_articles(art))
    print("\n══ JP ──────────────────────────────────────────────────────")
    print(fmt_table_jp(jp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
