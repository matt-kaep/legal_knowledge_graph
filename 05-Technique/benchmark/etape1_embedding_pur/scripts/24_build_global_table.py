"""Construit le grand tableau global de synthèse (chantier 2 Week-10).

Lit eval_m1_m2.csv (18) + ppr_naive_eval.csv (20) + ppr_kin_sweep_eval.csv
+ lightgcn_eval.csv (31), produit :
  - global_table_articles.csv  : 1 ligne / méthode champion / 10 cols panel × strict/ext
  - global_table_jp.csv        : 1 ligne / méthode champion / 5 cols panel
  - global_table.md            : rendu Markdown lisible pour la présentation
  - global_table_full.csv      : tout, y compris non-champions (pour annexe)

Méthodes incluses (cf. handoff LLM Judge) :
  Articles : B2-a, B3-e (k_in=10 champion), PPR-row α=0,85 / 0,95,
             PPR sweep s=5 seed JP-only α=0,70
  JP       : B3-a, B4-d (k_in=50), B4-e (k_in=20 champion), B4-f (k_in=10), PPR-row α=0,95

Méthodes droppées (dégénérescence ou collapse) :
  B4-a (≡ B2-a), B4-c (≡ B3-a), PPR-sym (≡ cosine)
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import pandas as pd

REPO = Path(os.environ.get(
    "LKG_REPO",
    str(Path(__file__).resolve().parents[4]),
))
DEFAULT_DATA = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
M3_K = 10

# Suffixes du panel
METRICS = ("m1", "hit", "mrr", "ndcg", "m2")
SUFFIXES_ART = [f"{m}_{r}" for r in ("strict", "ext") for m in METRICS]
SUFFIXES_JP = list(METRICS)

# Sélection des champions à afficher dans le tableau public
ART_SELECTION = [
    # (label, source df, filtre)
    ("LLM seul", "b", lambda d: (d["method"] == "LLM-seul")),
    ("LLM + RAG", "b", lambda d: (d["method"] == "LLM-RAG") & (d["k_in"] == 10)),
    ("B2-a (cosine articles)",       "b", lambda d: (d["method"] == "B2-a")),
    ("B3-e — JP→Art via graphe (k_in=10)", "b",
        lambda d: (d["method"] == "B3-e") & (d["k_in"] == 10)),
    ("PPR row-norm α=0,85",          "p", lambda d: (d["norm"] == "row") & (d["alpha"] == 0.85)),
    ("PPR row-norm α=0,95",          "p", lambda d: (d["norm"] == "row") & (d["alpha"] == 0.95)),
    ("PPR sweep s=50 both α=0,50 (champion strict)", "ps",
        lambda d: (d["k_in"] == 50) & (d["seed_variant"] == "both") & (d["alpha"] == 0.50)),
    ("PPR sweep s=50 both α=0,70 (champion étendu)", "ps",
        lambda d: (d["k_in"] == 50) & (d["seed_variant"] == "both") & (d["alpha"] == 0.70)),
    ("LightGCN K=2 (BGE propagé, cosinus)", "lg", lambda d: d["variant"] == "untrained_K2"),
    ("LightGCN K=2 entraîné (BPR cosinus)", "lg", lambda d: d["variant"] == "trained_K2"),
]

JP_SELECTION = [
    ("LLM seul JP",                     "b", lambda d: (d["method"] == "LLM-seul-JP")),
    ("LLM + RAG JP",                    "b",
        lambda d: (d["method"] == "LLM-RAG-JP") & (d["k_in"] == 10)),
    ("B3-a (cosine JP)",                "b", lambda d: (d["method"] == "B3-a")),
    ("B4-d — intersection (k_in=50)",   "b",
        lambda d: (d["method"] == "B4-d") & (d["k_in"] == 50)),
    ("B4-e — RRF (k_in=20)",            "b",
        lambda d: (d["method"] == "B4-e") & (d["k_in"] == 20)),
    ("B4-f — citation-weighted (k_in=10)", "b",
        lambda d: (d["method"] == "B4-f") & (d["k_in"] == 10)),
    ("PPR row-norm α=0,95 (côté JP)",   "p", lambda d: (d["norm"] == "row") & (d["alpha"] == 0.95)),
    ("PPR sweep s=20 both α=0,50 (côté JP)", "ps",
        lambda d: (d["k_in"] == 20) & (d["seed_variant"] == "both") & (d["alpha"] == 0.50)),
    ("LightGCN K=2 (BGE propagé, cosinus)", "lg", lambda d: d["variant"] == "untrained_K2"),
    ("LightGCN K=2 entraîné (BPR cosinus)", "lg", lambda d: d["variant"] == "trained_K2"),
]

M3_KEYS = {
    "LLM seul": "LLM-seul|art|kin=-",
    "LLM + RAG": "LLM-RAG|art|kin=10",
    "LLM + RAG JP": "LLM-RAG-JP|jp|kin=10",
    "B2-a (cosine articles)": "B2-a|art|kin=-",
    "B3-e — JP→Art via graphe (k_in=10)": "B3-e|art|kin=10",
    "PPR row-norm α=0,85": "PPR-row-a0.85|art|kin=10",
    "PPR row-norm α=0,95": "PPR-row-a0.95|art|kin=10",
    "PPR sweep s=50 both α=0,50 (champion strict)": "PPR-sweep-k50-both-a0.5|art|kin=50",
    "PPR sweep s=50 both α=0,70 (champion étendu)": "PPR-sweep-k50-both-a0.7|art|kin=50",
    "B3-a (cosine JP)": "B3-a|jp|kin=-",
    "B4-d — intersection (k_in=50)": "B4-d|jp|kin=50",
    "B4-e — RRF (k_in=20)": "B4-e|jp|kin=20",
    "B4-f — citation-weighted (k_in=10)": "B4-f|jp|kin=10",
    "PPR row-norm α=0,95 (côté JP)": "PPR-row-a0.95|jp|kin=10",
    "PPR sweep s=20 both α=0,50 (côté JP)": "PPR-sweep-k20-both-a0.5|jp|kin=20",
    ("art", "LightGCN K=2 (BGE propagé, cosinus)"): "LightGCN-untrained_K2|art|kin=2",
    ("jp", "LightGCN K=2 (BGE propagé, cosinus)"): "LightGCN-untrained_K2|jp|kin=2",
    ("art", "LightGCN K=2 entraîné (BPR cosinus)"): "LightGCN-trained_K2|art|kin=2",
    ("jp", "LightGCN K=2 entraîné (BPR cosinus)"): "LightGCN-trained_K2|jp|kin=2",
}


def _m3_group_key(method: str, modality: str, k_in) -> str:
    kin = "-" if pd.isna(k_in) else str(int(k_in))
    return f"{method}|{modality}|kin={kin}"


def aggregate_m3(df_m3: pd.DataFrame) -> pd.DataFrame:
    """Agrège LLM Judge par méthode et prépare l'affichage score (n2/n1/n0) sur K=10."""
    df = df_m3.copy()
    df["m3_key"] = [
        _m3_group_key(method, modality, k_in)
        for method, modality, k_in in zip(df["method"], df["modality"], df["k_in"])
    ]
    df["n0_display"] = df["n0"] + (M3_K - df["n_judged"]).clip(lower=0)
    agg = df.groupby("m3_key", dropna=False).agg(
        m3=("m3", "mean"),
        m3_n2_avg=("n2", "mean"),
        m3_n1_avg=("n1", "mean"),
        m3_n0_avg=("n0_display", "mean"),
    )
    agg["m3_display"] = agg.apply(
        lambda r: (
            f"{r['m3']:.3f} "
            f"({r['m3_n2_avg']:.1f}/{r['m3_n1_avg']:.1f}/{r['m3_n0_avg']:.1f})"
        ),
        axis=1,
    )
    return agg


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


def aggregate_ppr_sweep(df_ppr: pd.DataFrame, side: str) -> pd.DataFrame:
    """Agrège ppr_kin_sweep_eval.csv. Index = (k_in, seed_variant, alpha)."""
    if side == "art":
        cols = [f"{m}_{r}_art" for m in METRICS for r in ("strict", "ext")]
    else:
        cols = [f"{m}_jp" for m in METRICS]
    agg = df_ppr.groupby(["k_in", "seed_variant", "alpha"])[cols].mean()
    if side == "art":
        agg.columns = [c.replace("_art", "") for c in agg.columns]
    else:
        agg.columns = [c.replace("_jp", "") for c in agg.columns]
    return agg


def aggregate_lg(df_lg: pd.DataFrame, side: str) -> pd.DataFrame:
    """Agrège lightgcn_eval.csv (script 31). Index = variant, cols panel."""
    if side == "art":
        cols = [f"{m}_{r}_art" for m in METRICS for r in ("strict", "ext")]
    else:
        cols = [f"{m}_jp" for m in METRICS]
    agg = df_lg.groupby("variant")[cols].mean()
    agg.columns = [c.replace("_art", "").replace("_jp", "") for c in agg.columns]
    return agg


def collect_rows(selection, df_b, df_p, df_ps, df_lg, df_m3, side: str) -> pd.DataFrame:
    """Pour chaque entrée de selection, extrait la ligne agrégée correspondante."""
    rows = []
    for label, source, filt in selection:
        if source == "b":
            sub = df_b[filt(df_b) & (df_b["modality"] == ("art" if side == "art" else "jp"))]
            if sub.empty:
                continue
            row = aggregate_b(sub, side).iloc[0].to_dict()
        elif source == "p":
            if df_p is None:
                continue
            sub = df_p[filt(df_p)]
            if sub.empty:
                continue
            row = aggregate_p(sub, side).iloc[0].to_dict()
        elif source == "ps":
            if df_ps is None:
                continue
            sub = df_ps[filt(df_ps)]
            if sub.empty:
                continue
            row = aggregate_ppr_sweep(sub, side).iloc[0].to_dict()
        else:  # source == "lg"
            if df_lg is None:
                continue
            sub = df_lg[filt(df_lg)]
            if sub.empty:
                continue
            row = aggregate_lg(sub, side).iloc[0].to_dict()
        m3_key = M3_KEYS.get((side, label), M3_KEYS.get(label))
        if m3_key == "soon":
            row.update(
                {
                    "m3": float("nan"),
                    "m3_n2_avg": float("nan"),
                    "m3_n1_avg": float("nan"),
                    "m3_n0_avg": float("nan"),
                    "m3_display": "soon",
                }
            )
        elif m3_key and df_m3 is None:
            row.update(
                {
                    "m3": float("nan"),
                    "m3_n2_avg": float("nan"),
                    "m3_n1_avg": float("nan"),
                    "m3_n0_avg": float("nan"),
                    "m3_display": "soon",
                }
            )
        elif m3_key and df_m3 is not None and m3_key in df_m3.index:
            row.update(df_m3.loc[m3_key].to_dict())
        else:
            row.update(
                {
                    "m3": float("nan"),
                    "m3_n2_avg": float("nan"),
                    "m3_n1_avg": float("nan"),
                    "m3_n0_avg": float("nan"),
                    "m3_display": "—",
                }
            )
        rows.append({"méthode": label, **row})
    return pd.DataFrame(rows)


def fmt_table_articles(df: pd.DataFrame) -> str:
    """Rendu Markdown 11 cols (label + 5 strict + 5 ext) avec format 3 décimales."""
    cols_strict = [f"{m}_strict" for m in METRICS]
    cols_ext    = [f"{m}_ext"    for m in METRICS]
    headers = (
        ["méthode", "LLM Judge (n2/n1/n0)"] +
        ["Recall_s", "Hit_s", "MRR_s", "NDCG_s", "NormRank_s"] +
        ["Recall_e", "Hit_e", "MRR_e", "NDCG_e", "NormRank_e"]
    )
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = [str(r["méthode"]), str(r.get("m3_display", "—"))]
        for c in cols_strict + cols_ext:
            v = r.get(c, float("nan"))
            cells.append(f"{v:.3f}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def fmt_table_jp(df: pd.DataFrame) -> str:
    """Rendu Markdown 6 cols (label + 5 panel)."""
    headers = ["méthode", "LLM Judge (n2/n1/n0)", "Recall", "Hit", "MRR", "NDCG", "NormRank"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in df.iterrows():
        cells = [str(r["méthode"]), str(r.get("m3_display", "—"))]
        for c in METRICS:
            v = r.get(c, float("nan"))
            cells.append(f"{v:.3f}" if pd.notna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def normalize_target(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"art", "article", "articles", "articles_strict", "article_strict"}:
        return "articles_strict"
    if raw in {"jp", "jps"}:
        return "jp"
    if raw in {"articles_extended", "article_extended", "art_ext"}:
        return "articles_extended"
    return raw or "unknown"


def build_final_target_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Construit un tableau final cible depuis un résumé normalisé des champions."""
    wanted_target = normalize_target(target)
    frame = df.copy()
    if "target" not in frame.columns and "modality" in frame.columns:
        frame["target"] = frame["modality"].map(normalize_target)
    elif "target" in frame.columns:
        frame["target"] = frame["target"].map(normalize_target)
    else:
        frame["target"] = wanted_target
    frame = frame[frame["target"] == wanted_target].copy()
    if frame.empty:
        return frame

    if "method_label" not in frame.columns:
        if "method" in frame.columns:
            frame["method_label"] = frame["method"].astype(str)
        else:
            frame["method_label"] = ""
    frame = frame.rename(columns={"method_label": "méthode"})

    cols = [
        "méthode",
        "family",
        "graph_version",
        "m3_display",
        "m1",
        "hit",
        "mrr",
        "ndcg",
        "m2",
        "coverage_articles",
        "coverage_jp",
        "question_coverage",
        "n_questions_covered",
        "n_questions_benchmark",
    ]
    if wanted_target == "articles_strict":
        cols.extend(
            [
                "m1_ext",
                "hit_ext",
                "mrr_ext",
                "ndcg_ext",
                "m2_ext",
            ]
        )
    existing = [column for column in cols if column in frame.columns]
    order_cols = [column for column in ["graph_version", "family", "méthode"] if column in frame.columns]
    if order_cols:
        frame = frame.sort_values(order_cols).reset_index(drop=True)
    return frame[existing]


def build_graph_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise la comparaison inter-graphes en conservant les colonnes de couverture."""
    frame = df.copy()
    if "target" not in frame.columns and "modality" in frame.columns:
        frame["target"] = frame["modality"].map(normalize_target)
    elif "target" in frame.columns:
        frame["target"] = frame["target"].map(normalize_target)
    if "method_label" not in frame.columns:
        if "method" in frame.columns:
            frame["method_label"] = frame["method"].astype(str)
        else:
            frame["method_label"] = ""
    cols = [
        "graph_version",
        "target",
        "family",
        "method_label",
        "hit",
        "ndcg",
        "mrr",
        "m1",
        "m2",
        "hit_ext",
        "ndcg_ext",
        "mrr_ext",
        "m1_ext",
        "m2_ext",
        "coverage_questions",
        "coverage_articles",
        "coverage_jp",
        "coverage_articles_occ_pct",
        "coverage_articles_occ_present",
        "coverage_articles_occ_total",
        "coverage_articles_unique_pct",
        "coverage_articles_unique_present",
        "coverage_articles_unique_total",
        "coverage_articles_q_all_pct",
        "coverage_articles_q_any_pct",
        "coverage_articles_extended_occ_pct",
        "coverage_articles_extended_occ_present",
        "coverage_articles_extended_occ_total",
        "coverage_articles_extended_unique_pct",
        "coverage_articles_extended_unique_present",
        "coverage_articles_extended_unique_total",
        "coverage_jp_occ_pct",
        "coverage_jp_occ_present",
        "coverage_jp_occ_total",
        "coverage_jp_unique_pct",
        "coverage_jp_unique_present",
        "coverage_jp_unique_total",
        "coverage_jp_q_all_pct",
        "coverage_jp_q_any_pct",
        "question_coverage",
        "n_questions_covered",
        "n_questions_benchmark",
    ]
    existing = [c for c in cols if c in frame.columns]
    sort_cols = [c for c in ["target", "method_label", "graph_version"] if c in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols).reset_index(drop=True)
    return frame[existing]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--label", default="benchmark strict")
    args = parser.parse_args()
    data = args.bench_dir

    df_b = pd.read_csv(data / "eval_m1_m2.csv")
    p_path = data / "ppr_naive_eval.csv"
    df_p = pd.read_csv(p_path) if p_path.exists() else None
    ps_path = data / "ppr_kin_sweep_eval.csv"
    df_ps = pd.read_csv(ps_path) if ps_path.exists() else None
    lg_path = data / "lightgcn_eval.csv"
    df_lg = pd.read_csv(lg_path) if lg_path.exists() else None
    m3_path = data / "eval_m3.csv"
    df_m3 = aggregate_m3(pd.read_csv(m3_path)) if m3_path.exists() else None
    print(f"  eval_m1_m2.csv : {len(df_b)} lignes")
    print(f"  ppr_naive_eval.csv : {len(df_p) if df_p is not None else 'absent'} lignes")
    print(f"  ppr_kin_sweep_eval.csv : {len(df_ps) if df_ps is not None else 'absent'} lignes")
    print(f"  lightgcn_eval.csv : {len(df_lg) if df_lg is not None else 'absent'} lignes")
    print(f"  eval_m3.csv : {len(df_m3) if df_m3 is not None else 'absent'} groupes LLM Judge")

    art = collect_rows(ART_SELECTION, df_b, df_p, df_ps, df_lg, df_m3, "art")
    jp  = collect_rows(JP_SELECTION,  df_b, df_p, df_ps, df_lg, df_m3, "jp")

    out_art = data / "global_table_articles.csv"
    out_jp  = data / "global_table_jp.csv"
    art.to_csv(out_art, index=False)
    jp.to_csv(out_jp, index=False)
    print(f"\n✓ {out_art}  ({len(art)} méthodes)")
    print(f"✓ {out_jp}  ({len(jp)} méthodes)")

    # Markdown global
    md_lines = [
        f"# Grand tableau global — {args.label}, K=10",
        "",
        "Panel : Recall@K, Hit=|R∩GT|/min(|GT|,K), MRR=MRR@K, NDCG=NDCG@K, Normalized Rank.",
        "LLM Judge est affiché sous la forme score (n2/n1/n0 moyens sur K=10). Les non-jugés éventuels sont comptés dans n0 pour l'affichage.",
        "Côté articles : strict + étendu via les JP gold.",
        "Côté JP : modalité unique.",
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
    out_md = data / "global_table.md"
    out_md.write_text("\n".join(md_lines))
    print(f"✓ {out_md}")

    print("\n══ Articles ────────────────────────────────────────────────")
    print(fmt_table_articles(art))
    print("\n══ JP ──────────────────────────────────────────────────────")
    print(fmt_table_jp(jp))

    return 0


if __name__ == "__main__":
    sys.exit(main())
