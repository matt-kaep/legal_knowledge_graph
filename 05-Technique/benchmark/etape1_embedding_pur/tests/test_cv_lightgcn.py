import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "44_run_cv_lightgcn.py"
)
spec = importlib.util.spec_from_file_location("cv_lightgcn", SCRIPT)
cv_lightgcn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_lightgcn)


def test_validate_fold_assignments_rejects_duplicate_qids():
    folds = pd.DataFrame(
        [
            {"qid": "q1", "fold": 0},
            {"qid": "q1", "fold": 1},
            {"qid": "q2", "fold": 2},
        ]
    )

    with pytest.raises(ValueError, match="duplicate qids"):
        cv_lightgcn.validate_fold_assignments(folds, {"q1", "q2"})


def test_validate_fold_assignments_rejects_missing_and_extra_qids():
    folds = pd.DataFrame(
        [
            {"qid": "q1", "fold": 0},
            {"qid": "q3", "fold": 1},
        ]
    )

    with pytest.raises(ValueError, match="missing qids=.*q2.*extra qids=.*q3"):
        cv_lightgcn.validate_fold_assignments(folds, {"q1", "q2"})


def test_build_subset_bench_keeps_requested_qids_and_aligned_arrays(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    payload = {
        "questions": [
            {"qid": "q1", "text": "first"},
            {"qid": "q2", "text": "second"},
            {"qid": "q3", "text": "third"},
        ]
    }
    (src_dir / "bench_global.json").write_text(json.dumps(payload))
    np.save(src_dir / "questions_ids.npy", np.array(["q3", "q1", "q2"], dtype=object))
    np.save(
        src_dir / "questions_emb.npy",
        np.array(
            [
                [3.0, 30.0],
                [1.0, 10.0],
                [2.0, 20.0],
            ],
            dtype=np.float32,
        ),
    )

    dst_dir = tmp_path / "dst"
    cv_lightgcn.build_subset_bench(src_dir, {"q1", "q3"}, dst_dir)

    out_payload = json.loads((dst_dir / "bench_global.json").read_text())
    out_ids = np.load(dst_dir / "questions_ids.npy", allow_pickle=True).tolist()
    out_emb = np.load(dst_dir / "questions_emb.npy")

    assert [row["qid"] for row in out_payload["questions"]] == ["q1", "q3"]
    assert out_ids == ["q3", "q1"]
    assert out_emb.tolist() == [[3.0, 30.0], [1.0, 10.0]]


def test_select_champion_uses_official_metric_priority():
    df = pd.DataFrame(
        [
            {
                "method": "LightGCN-trained_K2-s11-lr0.001-e30-la1",
                "variant": "trained_K2",
                "train_k": 2,
                "seed": 11,
                "lr": 1e-3,
                "epochs": 30,
                "lambda_anchor": 1.0,
                "graph_version": "canonical",
                "modality": "art",
                "hit_strict": 0.50,
                "ndcg_strict": 0.40,
                "mrr_strict": 0.32,
                "m1_strict": 0.50,
                "m2_strict": 0.40,
            },
            {
                "method": "LightGCN-trained_K3-s17-lr0.001-e30-la1",
                "variant": "trained_K3",
                "train_k": 3,
                "seed": 17,
                "lr": 1e-3,
                "epochs": 30,
                "lambda_anchor": 1.0,
                "graph_version": "canonical",
                "modality": "art",
                "hit_strict": 0.50,
                "ndcg_strict": 0.41,
                "mrr_strict": 0.31,
                "m1_strict": 0.49,
                "m2_strict": 0.39,
            },
        ]
    )

    best = cv_lightgcn.select_champion(df, "art")

    assert best["variant"] == "trained_K3"
    assert best["train_k"] == 3


def test_summarize_cv_results_preserves_negative_sampling_strategy():
    df = pd.DataFrame(
        [
            {
                "qid": "q1",
                "fold": 0,
                "variant": "trained_K2",
                "train_k": 2,
                "seed": 42,
                "lr": 1e-3,
                "epochs": 30,
                "lambda_anchor": 1.0,
                "graph_version": "G0",
                "negative_sampling_strategy": "hard_negative_cosine_top20",
                "hit_strict_art": 0.5,
                "ndcg_strict_art": 0.4,
                "mrr_strict_art": 0.3,
                "m1_strict_art": 0.5,
                "m2_strict_art": 0.4,
            }
        ]
    )

    out = cv_lightgcn.summarize_cv_results(df, "art", n_questions_benchmark=1)

    assert out.loc[0, "negative_sampling_strategy"] == "hard_negative_cosine_top20"
    assert "neg-hard_negative_cosine_top20" in out.loc[0, "method"]


def test_main_rejects_non_official_split():
    with pytest.raises(ValueError, match="train_augmented_retrievable_strict"):
        cv_lightgcn.main(["--split", "eval_rich_retrievable_strict"])


def test_cv_summary_uses_only_checkpoint_selected_for_requested_target():
    rows = []
    for target in ("art", "jp"):
        for fold in range(5):
            rows.append(
                {
                    "qid": f"q{fold}",
                    "fold": fold,
                    "variant": "trained_K2",
                    "train_k": 2,
                    "seed": 42,
                    "lr": 0.001,
                    "epochs": 3,
                    "lambda_anchor": 1.0,
                    "graph_version": "G1",
                    "negative_sampling_strategy": "random",
                    "selection_target": target,
                    "hit_strict_art": 0.2 if target == "art" else 0.9,
                    "m1_strict_art": 0.7 if target == "art" else 0.1,
                    "ndcg_strict_art": 0.4,
                    "mrr_strict_art": 0.3,
                    "hit_jp": 0.8 if target == "jp" else 0.1,
                    "m1_jp": 0.5,
                    "ndcg_jp": 0.4,
                    "mrr_jp": 0.3,
                }
            )

    _, article_summary = cv_lightgcn.summarize_cv_outputs(pd.DataFrame(rows), "art")
    _, jp_summary = cv_lightgcn.summarize_cv_outputs(pd.DataFrame(rows), "jp")

    assert article_summary["selection_target"].tolist() == ["art"]
    assert article_summary.loc[0, "article_recall_at_10_mean"] == pytest.approx(0.7)
    assert jp_summary["selection_target"].tolist() == ["jp"]
    assert jp_summary.loc[0, "jp_hit_at_10_mean"] == pytest.approx(0.8)
def test_paired_deltas_match_same_lightgcn_config_across_graph_versions():
    base = {
        "variant": "trained_K2", "train_k": 2, "seed": 42, "lr": 1e-3,
        "epochs": 30, "lambda_anchor": 1.0, "negative_sampling_strategy": "random",
        "selection_target": "art",
    }
    candidate = pd.DataFrame([
        {**base, "fold": fold, "graph_version": "G7", "m1_strict_art": 0.6 + fold / 100}
        for fold in range(5)
    ])
    control = pd.DataFrame([
        {**base, "fold": fold, "graph_version": "G1", "m1_strict_art": 0.5 + fold / 100}
        for fold in range(5)
    ])

    deltas = cv_lightgcn.build_paired_deltas(candidate, control, "art")

    assert deltas.loc[0, "eligible_comparison"]
    assert deltas.loc[0, "n_folds_paired"] == 5
    assert deltas.loc[0, "m1_strict_art_delta_mean"] == pytest.approx(0.1)
    assert deltas.loc[0, "candidate_graph_version"] == "G7"
    assert deltas.loc[0, "control_graph_version"] == "G1"
