import importlib.util
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "24_build_global_table.py"
)
spec = importlib.util.spec_from_file_location("build_global_table", SCRIPT)
build_global_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_global_table)


def test_aggregate_m3_counts_missing_judgements_as_n0_for_display():
    df = pd.DataFrame(
        [
            {
                "method": "B3-e",
                "modality": "art",
                "k_in": 10,
                "n_judged": 10,
                "n2": 2,
                "n1": 3,
                "n0": 5,
                "m3": 0.35,
            },
            {
                "method": "B3-e",
                "modality": "art",
                "k_in": 10,
                "n_judged": 8,
                "n2": 1,
                "n1": 2,
                "n0": 5,
                "m3": 0.2,
            },
        ]
    )

    summary = build_global_table.aggregate_m3(df)
    row = summary.loc["B3-e|art|kin=10"]

    assert row["m3"] == 0.275
    assert row["m3_n2_avg"] == 1.5
    assert row["m3_n1_avg"] == 2.5
    assert row["m3_n0_avg"] == 6.0
    assert row["m3_display"] == "0.275 (1.5/2.5/6.0)"


def test_aggregate_ppr_sweep_renames_article_and_jp_metrics():
    df = pd.DataFrame(
        [
            {
                "k_in": 5,
                "seed_variant": "jp_only",
                "alpha": 0.70,
                "m1_strict_art": 0.62,
                "hit_strict_art": 0.65,
                "mrr_strict_art": 0.33,
                "ndcg_strict_art": 0.39,
                "m2_strict_art": 0.47,
                "m1_ext_art": 0.52,
                "hit_ext_art": 0.91,
                "mrr_ext_art": 0.68,
                "ndcg_ext_art": 0.51,
                "m2_ext_art": 0.47,
                "m1_jp": 0.38,
                "hit_jp": 0.39,
                "mrr_jp": 0.26,
                "ndcg_jp": 0.28,
                "m2_jp": 0.32,
            }
        ]
    )

    art = build_global_table.aggregate_ppr_sweep(df, "art")
    jp = build_global_table.aggregate_ppr_sweep(df, "jp")

    art_row = art.loc[(5, "jp_only", 0.70)]
    jp_row = jp.loc[(5, "jp_only", 0.70)]

    assert art_row["m1_strict"] == 0.62
    assert art_row["ndcg_ext"] == 0.51
    assert jp_row["m1"] == 0.38
    assert jp_row["ndcg"] == 0.28


def test_graph_comparison_table_keeps_graph_version_and_coverage():
    df = pd.DataFrame(
        [
            {
                "graph_version": "G0",
                "target": "articles_strict",
                "method_label": "B2-a",
                "hit": 0.42,
                "coverage_articles": 1.0,
                "coverage_jp": 1.0,
            },
            {
                "graph_version": "G1",
                "target": "articles_strict",
                "method_label": "B2-a",
                "hit": 0.45,
                "coverage_articles": 1.0,
                "coverage_jp": 1.0,
            },
        ]
    )

    out = build_global_table.build_graph_comparison_table(df)

    assert list(out["graph_version"]) == ["G0", "G1"]
    assert list(out["coverage_articles"]) == [1.0, 1.0]


def test_build_final_target_table_keeps_extended_article_metrics_as_diagnostic():
    df = pd.DataFrame(
        [
            {
                "graph_version": "G0",
                "target": "articles_strict",
                "family": "ppr",
                "method_label": "PPR-sweep-k50-both-a0.5",
                "hit": 0.52,
                "ndcg": 0.41,
                "mrr": 0.33,
                "m1": 0.58,
                "m2": 0.36,
                "hit_ext": 0.71,
                "ndcg_ext": 0.55,
                "mrr_ext": 0.48,
                "m1_ext": 0.77,
                "m2_ext": 0.49,
            }
        ]
    )

    out = build_global_table.build_final_target_table(df, "articles_strict")

    assert out.loc[0, "méthode"] == "PPR-sweep-k50-both-a0.5"
    assert out.loc[0, "hit"] == 0.52
    assert out.loc[0, "hit_ext"] == 0.71
