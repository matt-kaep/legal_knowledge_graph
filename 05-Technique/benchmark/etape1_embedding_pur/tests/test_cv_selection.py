from pathlib import Path
import importlib.util

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
