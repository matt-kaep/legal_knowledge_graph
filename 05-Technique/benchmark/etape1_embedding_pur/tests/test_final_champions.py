import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "45_run_final_champions.py"
)
spec = importlib.util.spec_from_file_location("final_champions", SCRIPT)
final_champions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final_champions)


def _write_bench(bench_dir: Path, questions: list[dict]) -> None:
    bench_dir.mkdir(parents=True, exist_ok=True)
    (bench_dir / "bench_global.json").write_text(
        json.dumps({"questions": questions}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    qids = np.array([q["qid"] for q in questions], dtype=object)
    emb = np.arange(len(questions) * 3, dtype=np.float32).reshape(len(questions), 3)
    np.save(bench_dir / "questions_ids.npy", qids)
    np.save(bench_dir / "questions_emb.npy", emb)


def _write_champions(cv_root: Path) -> None:
    payloads = {
        "b3_b4": {
            "articles": {"method": "B2-a", "modality": "art", "k_in": None},
            "jp": {"method": "B3-a", "modality": "jp", "k_in": None},
        },
        "ppr": {
            "articles": {
                "method": "PPR-sweep-k20-both-a0.5",
                "modality": "art",
                "k_in": 20,
                "seed_variant": "both",
                "alpha": 0.5,
            },
            "jp": {
                "method": "PPR-sweep-k20-both-a0.5",
                "modality": "jp",
                "k_in": 20,
                "seed_variant": "both",
                "alpha": 0.5,
            },
        },
        "lightgcn": {
            "articles": {
                "method": "LightGCN-trained_K2",
                "modality": "art",
                "k_in": 2,
                "variant": "trained_K2",
                "train_k": 2,
                "seed": 42,
                "lr": 0.001,
                "epochs": 30,
                "lambda_anchor": 1.0,
            },
            "jp": {
                "method": "LightGCN-trained_K2",
                "modality": "jp",
                "k_in": 2,
                "variant": "trained_K2",
                "train_k": 2,
                "seed": 42,
                "lr": 0.001,
                "epochs": 30,
                "lambda_anchor": 1.0,
            },
        },
    }
    for family, payload in payloads.items():
        family_dir = cv_root / family
        family_dir.mkdir(parents=True, exist_ok=True)
        (family_dir / "champions.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _write_replay_outputs(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_rows = [
        {
            "qid": "q1",
            "method": "B2-a",
            "modality": "art",
            "k_in": np.nan,
            "m1_strict": 0.5,
            "hit_strict": 0.5,
            "mrr_strict": 0.4,
            "ndcg_strict": 0.45,
            "m2_strict": 0.4,
            "m1_ext": 0.8,
            "hit_ext": 0.9,
            "mrr_ext": 0.6,
            "ndcg_ext": 0.7,
            "m2_ext": 0.3,
        },
        {
            "qid": "q2",
            "method": "B2-a",
            "modality": "art",
            "k_in": np.nan,
            "m1_strict": 0.6,
            "hit_strict": 0.6,
            "mrr_strict": 0.5,
            "ndcg_strict": 0.55,
            "m2_strict": 0.35,
            "m1_ext": 0.85,
            "hit_ext": 0.95,
            "mrr_ext": 0.65,
            "ndcg_ext": 0.75,
            "m2_ext": 0.25,
        },
        {
            "qid": "q1",
            "method": "B3-a",
            "modality": "jp",
            "k_in": np.nan,
            "m1": 0.4,
            "hit": 0.5,
            "mrr": 0.3,
            "ndcg": 0.35,
            "m2": 0.45,
        },
        {
            "qid": "q2",
            "method": "B3-a",
            "modality": "jp",
            "k_in": np.nan,
            "m1": 0.5,
            "hit": 0.6,
            "mrr": 0.35,
            "ndcg": 0.4,
            "m2": 0.4,
        },
    ]
    pd.DataFrame(eval_rows).to_csv(out_dir / "eval_m1_m2.csv", index=False)

    ppr_rows = [
        {
            "qid": "q1",
            "k_in": 20,
            "seed_variant": "both",
            "alpha": 0.5,
            "m1_strict_art": 0.55,
            "hit_strict_art": 0.6,
            "mrr_strict_art": 0.42,
            "ndcg_strict_art": 0.5,
            "m2_strict_art": 0.33,
            "m1_ext_art": 0.82,
            "hit_ext_art": 0.92,
            "mrr_ext_art": 0.63,
            "ndcg_ext_art": 0.74,
            "m2_ext_art": 0.22,
            "m1_jp": 0.46,
            "hit_jp": 0.56,
            "mrr_jp": 0.34,
            "ndcg_jp": 0.39,
            "m2_jp": 0.37,
        },
        {
            "qid": "q2",
            "k_in": 20,
            "seed_variant": "both",
            "alpha": 0.5,
            "m1_strict_art": 0.65,
            "hit_strict_art": 0.7,
            "mrr_strict_art": 0.48,
            "ndcg_strict_art": 0.58,
            "m2_strict_art": 0.28,
            "m1_ext_art": 0.86,
            "hit_ext_art": 0.96,
            "mrr_ext_art": 0.68,
            "ndcg_ext_art": 0.78,
            "m2_ext_art": 0.18,
            "m1_jp": 0.5,
            "hit_jp": 0.6,
            "mrr_jp": 0.38,
            "ndcg_jp": 0.43,
            "m2_jp": 0.32,
        },
    ]
    pd.DataFrame(ppr_rows).to_csv(out_dir / "ppr_kin_sweep_eval.csv", index=False)

    lightgcn_rows = [
        {
            "qid": "q1",
            "variant": "trained_K2",
            "train_k": 2,
            "seed": 42,
            "lr": 0.001,
            "epochs": 30,
            "lambda_anchor": 1.0,
            "hit_strict_art": 0.58,
            "ndcg_strict_art": 0.52,
            "mrr_strict_art": 0.46,
            "m1_strict_art": 0.6,
            "m2_strict_art": 0.31,
            "hit_ext_art": 0.88,
            "ndcg_ext_art": 0.8,
            "mrr_ext_art": 0.7,
            "m1_ext_art": 0.9,
            "m2_ext_art": 0.16,
            "hit_jp": 0.54,
            "ndcg_jp": 0.44,
            "mrr_jp": 0.36,
            "m1_jp": 0.48,
            "m2_jp": 0.34,
        },
        {
            "qid": "q2",
            "variant": "trained_K2",
            "train_k": 2,
            "seed": 42,
            "lr": 0.001,
            "epochs": 30,
            "lambda_anchor": 1.0,
            "hit_strict_art": 0.62,
            "ndcg_strict_art": 0.56,
            "mrr_strict_art": 0.5,
            "m1_strict_art": 0.64,
            "m2_strict_art": 0.27,
            "hit_ext_art": 0.9,
            "ndcg_ext_art": 0.83,
            "mrr_ext_art": 0.74,
            "m1_ext_art": 0.92,
            "m2_ext_art": 0.14,
            "hit_jp": 0.58,
            "ndcg_jp": 0.48,
            "mrr_jp": 0.4,
            "m1_jp": 0.52,
            "m2_jp": 0.3,
        },
    ]
    pd.DataFrame(lightgcn_rows).to_csv(out_dir / "lightgcn_eval.csv", index=False)


def test_replay_b3_b4_accepts_baseline_only_champions(tmp_path, monkeypatch):
    eval_dir = tmp_path / "eval"
    _write_bench(
        eval_dir,
        [
            {
                "qid": "q1",
                "articles_attendus": ["art1"],
                "articles_attendus_etendu": ["art1", "art2"],
                "gold_jp_ids": ["jp1"],
            }
        ],
    )

    class FakeBaselineEval:
        @staticmethod
        def eval_m1_m2(questions, out_dir, limit=None, qid_filter=None, ks_in=None):
            assert ks_in is None
            pd.DataFrame(
                [
                    {
                        "qid": "q1",
                        "method": "B2-a",
                        "modality": "art",
                        "k_in": np.nan,
                        "m1_strict": 0.5,
                        "hit_strict": 0.5,
                        "mrr_strict": 0.4,
                        "ndcg_strict": 0.45,
                        "m2_strict": 0.4,
                        "m1_ext": 0.8,
                        "hit_ext": 0.9,
                        "mrr_ext": 0.6,
                        "ndcg_ext": 0.7,
                        "m2_ext": 0.3,
                    },
                    {
                        "qid": "q1",
                        "method": "B3-a",
                        "modality": "jp",
                        "k_in": np.nan,
                        "m1": 0.4,
                        "hit": 0.5,
                        "mrr": 0.3,
                        "ndcg": 0.35,
                        "m2": 0.45,
                    },
                ]
            ).to_csv(Path(out_dir) / "eval_m1_m2.csv", index=False)

    monkeypatch.setattr(
        final_champions,
        "_load_script_module",
        lambda script_name, module_name: FakeBaselineEval,
    )

    out = final_champions.replay_b3_b4(
        eval_dir,
        {
            "articles": {"method": "B2-a", "modality": "art", "k_in": None},
            "jp": {"method": "B3-a", "modality": "jp", "k_in": None},
        },
    )

    assert set(out["method"]) == {"B2-a", "B3-a"}


def test_main_skip_replay_builds_final_outputs(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    questions = [
        {
            "qid": "q1",
            "articles_attendus": ["art1"],
            "articles_attendus_etendu": ["art1", "art2"],
            "gold_jp_ids": ["jp1"],
        },
        {
            "qid": "q2",
            "articles_attendus": ["art2"],
            "articles_attendus_etendu": ["art2", "art3"],
            "gold_jp_ids": ["jp2"],
        },
    ]
    _write_bench(train_dir, questions)
    _write_bench(eval_dir, questions)

    cv_root = tmp_path / "cv"
    _write_champions(cv_root)

    out_dir = tmp_path / "final"
    _write_replay_outputs(out_dir)

    coverage_path = tmp_path / "coverage.json"
    coverage_path.write_text(
        json.dumps(
            {
                "datasets": {
                    "eval_rich_retrievable_strict": {
                        "g0": {
                            "questions": 2,
                            "strict_occ_total": 4,
                            "strict_occ_present": 4,
                            "strict_occ_pct": 91.0,
                            "strict_unique_total": 3,
                            "strict_unique_present": 3,
                            "strict_unique_pct": 89.0,
                            "strict_q_all_pct": 100.0,
                            "strict_q_any_pct": 95.0,
                            "ext_occ_total": 8,
                            "ext_occ_present": 8,
                            "ext_occ_pct": 92.0,
                            "ext_unique_total": 5,
                            "ext_unique_present": 5,
                            "ext_unique_pct": 90.0,
                            "jp_occ_total": 2,
                            "jp_occ_present": 2,
                            "jp_occ_pct": 90.0,
                            "jp_unique_total": 2,
                            "jp_unique_present": 2,
                            "jp_unique_pct": 88.0,
                            "jp_q_all_pct": 100.0,
                            "jp_q_any_pct": 93.0,
                        }
                    }
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    def fake_resolve_graph_bench_dir(graph_version: str, split: str) -> Path:
        if split == final_champions.graph_protocol.OFFICIAL_TRAIN_SPLIT:
            return train_dir
        if split == "eval_rich_retrievable_strict":
            return eval_dir
        raise AssertionError(f"unexpected split {split}")

    monkeypatch.setattr(
        final_champions.graph_protocol,
        "resolve_graph_bench_dir",
        fake_resolve_graph_bench_dir,
    )

    rc = final_champions.main(
        [
            "--graph-version",
            "G0",
            "--cv-root",
            str(cv_root),
            "--out-dir",
            str(out_dir),
            "--coverage-summary",
            str(coverage_path),
            "--skip-replay",
        ]
    )

    assert rc == 0

    summary = pd.read_csv(out_dir / "final_champions_summary.csv")
    articles = pd.read_csv(out_dir / "global_table_articles.csv")
    jp = pd.read_csv(out_dir / "global_table_jp.csv")
    comparison = pd.read_csv(out_dir / "global_table_graph_comparison.csv")

    assert len(summary) == 6
    assert set(summary["family"]) == {"b3_b4", "ppr", "lightgcn"}
    assert set(summary["target"]) == {"articles_strict", "jp"}
    assert summary["n_questions_benchmark"].eq(2).all()
    assert summary["question_coverage"].eq(1.0).all()
    assert summary["coverage_articles"].eq(95.0).all()
    assert summary["coverage_jp"].eq(93.0).all()
    assert summary["coverage_articles_occ_total"].eq(4).all()
    assert summary["coverage_articles_extended_unique_total"].eq(5).all()
    assert summary["coverage_jp_occ_total"].eq(2).all()

    assert not articles.empty
    assert not jp.empty
    assert not comparison.empty
    assert "coverage_articles" in comparison.columns
    assert "n_questions_benchmark" in comparison.columns
    assert "coverage_articles_occ_total" in comparison.columns
