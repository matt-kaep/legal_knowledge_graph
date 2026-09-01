from pathlib import Path
import importlib.util

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "42_run_cv_b3_b4.py"
)
spec = importlib.util.spec_from_file_location("cv_b3b4", SCRIPT)
cv_b3b4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_b3b4)

PPR_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "43_run_cv_ppr.py"
)
ppr_spec = importlib.util.spec_from_file_location("cv_ppr", PPR_SCRIPT)
cv_ppr = importlib.util.module_from_spec(ppr_spec)
ppr_spec.loader.exec_module(cv_ppr)

BASELINE_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "26_eval_doctrine_v3plus_m1_m2.py"
)
baseline_spec = importlib.util.spec_from_file_location("baseline_eval", BASELINE_SCRIPT)
baseline_eval = importlib.util.module_from_spec(baseline_spec)
baseline_spec.loader.exec_module(baseline_eval)

GRAPH_PROTOCOL_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "graph_protocol.py"
)
graph_protocol_spec = importlib.util.spec_from_file_location(
    "graph_protocol_cv_selection", GRAPH_PROTOCOL_SCRIPT
)
graph_protocol = importlib.util.module_from_spec(graph_protocol_spec)
assert graph_protocol_spec.loader is not None
graph_protocol_spec.loader.exec_module(graph_protocol)


def test_select_champion_uses_hit_then_ndcg_then_mrr():
    df = pd.DataFrame(
        [
            {
                "method": "B3-e",
                "k_in": 10,
                "modality": "art",
                "hit_strict": 0.50,
                "ndcg_strict": 0.40,
                "mrr_strict": 0.35,
                "m1_strict": 0.51,
                "m2_strict": 0.42,
            },
            {
                "method": "B3-e",
                "k_in": 20,
                "modality": "art",
                "hit_strict": 0.50,
                "ndcg_strict": 0.41,
                "mrr_strict": 0.34,
                "m1_strict": 0.50,
                "m2_strict": 0.41,
            },
        ]
    )
    best = cv_b3b4.select_champion(df, "art")
    assert best["k_in"] == 20


def test_summarize_cv_results_adds_explicit_coverage_columns():
    df = pd.DataFrame(
        [
            {
                "fold": 0,
                "qid": "q1",
                "method": "B3-e",
                "k_in": 10,
                "modality": "art",
                "hit_strict": 0.6,
                "ndcg_strict": 0.5,
                "mrr_strict": 0.4,
                "m1_strict": 0.6,
                "m2_strict": 0.5,
            },
            {
                "fold": 1,
                "qid": "q2",
                "method": "B3-e",
                "k_in": 10,
                "modality": "art",
                "hit_strict": 0.4,
                "ndcg_strict": 0.3,
                "mrr_strict": 0.2,
                "m1_strict": 0.4,
                "m2_strict": 0.3,
            },
        ]
    )

    summary = cv_b3b4.summarize_cv_results(df, "art")

    assert summary.loc[0, "n_questions_covered"] == 2
    assert summary.loc[0, "n_questions_benchmark"] == 2
    assert summary.loc[0, "n_folds_covered"] == 2
    assert summary.loc[0, "question_coverage"] == 1.0
    assert summary.loc[0, "fold_coverage"] == 2 / cv_b3b4.graph_protocol.OFFICIAL_N_FOLDS


def test_grouped_jp_control_aggregates_the_real_strict_metric_columns():
    rows = [
        {
            "fold": fold,
            "qid": f"q{fold}",
            "method": "B3-a",
            "k_in": None,
            "modality": "jp",
            "hit_strict": 0.5 + fold / 100,
            "ndcg_strict": 0.4,
            "mrr_strict": 0.3,
            "m1_strict": 0.2,
        }
        for fold in range(5)
    ]

    fold_metrics, summary = cv_b3b4.summarize_grouped_cv_outputs(
        pd.DataFrame(rows),
        "jp",
        n_questions_benchmark=5,
        expected_qids_by_fold={fold: {f"q{fold}"} for fold in range(5)},
    )

    assert fold_metrics["hit_strict"].tolist() == pytest.approx([0.5, 0.51, 0.52, 0.53, 0.54])
    assert summary.loc[0, "jp_hit_at_10_mean"] == pytest.approx(0.52)
    assert summary.loc[0, "eligible_champion"]


def test_ppr_summary_adds_explicit_coverage_denominator_columns():
    df = pd.DataFrame(
        [
            {
                "fold": 0,
                "qid": "q1",
                "k_in": 10,
                "seed_variant": "both",
                "alpha": 0.5,
                "hit_strict_art": 0.6,
                "ndcg_strict_art": 0.5,
                "mrr_strict_art": 0.4,
                "m1_strict_art": 0.6,
                "m2_strict_art": 0.5,
            },
            {
                "fold": 1,
                "qid": "q2",
                "k_in": 10,
                "seed_variant": "both",
                "alpha": 0.5,
                "hit_strict_art": 0.5,
                "ndcg_strict_art": 0.4,
                "mrr_strict_art": 0.3,
                "m1_strict_art": 0.5,
                "m2_strict_art": 0.4,
            },
        ]
    )

    summary = cv_ppr.summarize_cv_results(df, "art")

    assert summary.loc[0, "n_questions_covered"] == 2
    assert summary.loc[0, "n_questions_benchmark"] == 2
    assert summary.loc[0, "question_coverage"] == 1.0
    assert summary.loc[0, "n_folds_covered"] == 2
    assert summary.loc[0, "fold_coverage"] == 2 / cv_ppr.graph_protocol.OFFICIAL_N_FOLDS


@pytest.mark.parametrize("module", [cv_b3b4, cv_ppr])
def test_validate_fold_assignments_rejects_duplicate_qids(module):
    folds = pd.DataFrame(
        [
            {"qid": "q1", "fold": 0},
            {"qid": "q1", "fold": 1},
            {"qid": "q2", "fold": 2},
        ]
    )

    with pytest.raises(ValueError, match="duplicate qids"):
        module.validate_fold_assignments(folds, {"q1", "q2"})


@pytest.mark.parametrize("module", [cv_b3b4, cv_ppr])
def test_validate_fold_assignments_rejects_missing_and_extra_qids(module):
    folds = pd.DataFrame(
        [
            {"qid": "q1", "fold": 0},
            {"qid": "q3", "fold": 1},
        ]
    )

    with pytest.raises(ValueError, match="missing qids=.*q2.*extra qids=.*q3"):
        module.validate_fold_assignments(folds, {"q1", "q2"})


def test_b3b4_main_rejects_non_official_split():
    with pytest.raises(ValueError, match="train_augmented_retrievable_strict"):
        cv_b3b4.main(["--split", "eval_rich_retrievable_strict"])


def test_ppr_main_rejects_non_official_split():
    with pytest.raises(ValueError, match="train_augmented_retrievable_strict"):
        cv_ppr.main(["--split", "eval_rich_retrievable_strict"])


def test_subset_cached_questions_reuses_superset_embeddings(tmp_path):
    cache_dir = tmp_path / "bench"
    cache_dir.mkdir()
    ids = pd.Series(["q1", "q2", "q3"], dtype=object).to_numpy()
    emb = pd.DataFrame([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype="float32").to_numpy()
    np.save(cache_dir / "questions_ids.npy", ids)
    np.save(cache_dir / "questions_emb.npy", emb)

    subset = baseline_eval._subset_cached_questions(["q3", "q1"], cache_dir)

    assert subset is not None
    assert subset.shape == (2, 2)
    assert subset.tolist() == [[3.0, 0.0], [1.0, 0.0]]


def test_cv_summary_averages_fold_means_not_question_rows():
    raw = pd.DataFrame(
        [
            {"fold": 0, "config": "candidate", "hit_strict_art": 1.0},
            {"fold": 1, "config": "candidate", "hit_strict_art": 0.0},
            {"fold": 1, "config": "candidate", "hit_strict_art": 0.0},
            {"fold": 1, "config": "candidate", "hit_strict_art": 0.0},
        ]
    )

    fold_metrics, summary = graph_protocol.summarize_fold_metrics(
        raw,
        config_columns=["config"],
        metric_columns={"article_hit_at_10": "hit_strict_art"},
        expected_folds=2,
    )

    assert fold_metrics["hit_strict_art"].tolist() == [1.0, 0.0]
    assert summary.loc[0, "article_hit_at_10_mean"] == 0.5
    assert summary.loc[0, "eligible_champion"]


def test_cv_summary_contains_std_and_95_percent_ci():
    raw = pd.DataFrame(
        [
            {"fold": fold, "config": "candidate", "hit_strict_art": value}
            for fold, value in enumerate([0.0, 1.0, 0.0, 1.0, 0.0])
        ]
    )

    _, summary = graph_protocol.summarize_fold_metrics(
        raw,
        config_columns=["config"],
        metric_columns={"article_hit_at_10": "hit_strict_art"},
    )

    assert summary.loc[0, "article_hit_at_10_std"] == pytest.approx(0.5477225575)
    assert summary.loc[0, "article_hit_at_10_ci95_low"] == pytest.approx(-0.2799783526)
    assert summary.loc[0, "article_hit_at_10_ci95_high"] == pytest.approx(1.0799783526)


def test_grouped_cv_rejects_duplicate_question_rows_within_a_config_fold():
    raw = pd.DataFrame([
        {"fold": 0, "qid": "q1", "config": "candidate", "hit_strict_art": 1.0},
        {"fold": 0, "qid": "q1", "config": "candidate", "hit_strict_art": 0.0},
    ])

    with pytest.raises(ValueError, match="duplicate question rows"):
        graph_protocol.summarize_fold_metrics(
            raw,
            config_columns=["config"],
            metric_columns={"article_hit_at_10": "hit_strict_art"},
            expected_qids_by_fold={0: {"q1"}},
        )


def test_paired_delta_uses_matching_fold_ids():
    candidate = pd.DataFrame(
        [
            {"fold": 2, "hit_strict_art": 0.9},
            {"fold": 0, "hit_strict_art": 0.6},
            {"fold": 1, "hit_strict_art": 0.8},
        ]
    )
    control = pd.DataFrame(
        [
            {"fold": 1, "hit_strict_art": 0.5},
            {"fold": 2, "hit_strict_art": 0.7},
            {"fold": 0, "hit_strict_art": 0.5},
        ]
    )

    delta = graph_protocol.summarize_paired_fold_delta(
        candidate, control, "hit_strict_art", expected_folds=3
    )

    assert delta["eligible_comparison"]
    assert delta["hit_strict_art_delta_mean"] == pytest.approx(0.2)


def test_metric_order_is_recall_then_ndcg_then_mrr_for_articles():
    assert graph_protocol.champion_sort_columns("article") == [
        ("article_recall_at_10_mean", False),
        ("article_ndcg_at_10_mean", False),
        ("article_mrr_at_10_mean", False),
    ]


def test_metric_order_is_hit_then_ndcg_then_mrr_for_jp():
    assert graph_protocol.champion_sort_columns("jp") == [
        ("jp_hit_at_10_mean", False),
        ("jp_ndcg_at_10_mean", False),
        ("jp_mrr_at_10_mean", False),
    ]


def test_article_champion_prefers_recall_even_when_hit_is_lower():
    summary = pd.DataFrame(
        [
            {
                "config": "recall-winner",
                "eligible_champion": True,
                "article_recall_at_10_mean": 0.70,
                "article_hit_at_10_mean": 0.60,
                "article_ndcg_at_10_mean": 0.40,
                "article_mrr_at_10_mean": 0.30,
            },
            {
                "config": "hit-winner",
                "eligible_champion": True,
                "article_recall_at_10_mean": 0.69,
                "article_hit_at_10_mean": 0.90,
                "article_ndcg_at_10_mean": 0.80,
                "article_mrr_at_10_mean": 0.70,
            },
        ]
    )

    assert cv_ppr.select_champion(summary, "art")["config"] == "recall-winner"


def test_grouped_selection_rejects_missing_official_metric_columns():
    summary = pd.DataFrame(
        [
            {
                "config": "incomplete-schema",
                "eligible_champion": True,
                "article_hit_at_10_mean": 0.90,
            }
        ]
    )

    with pytest.raises(KeyError, match="article_recall_at_10_mean"):
        cv_ppr.select_champion(summary, "art")


def test_missing_fold_rejects_champion_selection():
    raw = pd.DataFrame(
        [
            {"fold": 0, "config": "complete", "hit_strict_art": 0.4, "m1_strict_art": 0.4, "ndcg_strict_art": 0.4, "mrr_strict_art": 0.4},
            {"fold": 1, "config": "complete", "hit_strict_art": 0.4, "m1_strict_art": 0.4, "ndcg_strict_art": 0.4, "mrr_strict_art": 0.4},
            {"fold": 0, "config": "missing", "hit_strict_art": 0.9, "m1_strict_art": 0.9, "ndcg_strict_art": 0.9, "mrr_strict_art": 0.9},
        ]
    )
    _, summary = graph_protocol.summarize_fold_metrics(
        raw,
        config_columns=["config"],
            metric_columns={
                "article_hit_at_10": "hit_strict_art",
                "article_recall_at_10": "m1_strict_art",
                "article_ndcg_at_10": "ndcg_strict_art",
            "article_mrr_at_10": "mrr_strict_art",
        },
        expected_folds=2,
    )

    assert not summary.loc[summary["config"] == "missing", "eligible_champion"].item()
    assert cv_ppr.select_champion(summary, "art")["config"] == "complete"


def test_partial_qid_coverage_rejects_champion_with_all_five_folds():
    expected_qids_by_fold = {fold: {f"q{fold}a", f"q{fold}b"} for fold in range(5)}
    raw = pd.DataFrame(
        [
            {
                "fold": fold,
                "qid": qid,
                "config": "partial",
                "hit_strict_art": 0.9,
            }
            for fold, qids in expected_qids_by_fold.items()
            for qid in sorted(qids - ({f"q{fold}b"} if fold == 3 else set()))
        ]
    )

    _, summary = graph_protocol.summarize_fold_metrics(
        raw,
        config_columns=["config"],
        metric_columns={"article_hit_at_10": "hit_strict_art"},
        expected_qids_by_fold=expected_qids_by_fold,
    )

    assert summary.loc[0, "n_folds_covered"] == 5
    assert summary.loc[0, "question_coverage"] == pytest.approx(0.9)
    assert not summary.loc[0, "eligible_champion"]
