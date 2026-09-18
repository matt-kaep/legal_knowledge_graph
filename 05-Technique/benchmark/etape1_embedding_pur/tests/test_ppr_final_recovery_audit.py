from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "71_audit_ppr_final_recovery.py"
SPEC = importlib.util.spec_from_file_location("ppr_final_recovery_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


RESULT_MANIFEST_SHA = "a" * 64
EVAL_SHA = "b" * 64
FOLDS_SHA = "c" * 64
MATRIX_SHA = "d" * 64


def _write_fixture(root: Path, *, result_manifest_sha: str = RESULT_MANIFEST_SHA) -> None:
    graph_dir = root / "G1"
    graph_dir.mkdir(parents=True)
    champions = {
        "ppr": {
            "art": {
                "method": "PPR-article",
                "modality": "art",
                "k_in": 50,
                "seed_variant": "both",
                "alpha": 0.5,
                "n_folds_covered": 5,
                "expected_folds": 5,
                "fold_coverage": 1.0,
                "n_questions_expected": 2,
                "n_questions_covered": 2,
                "question_coverage": 1.0,
                "eligible_champion": True,
                "protocol_version": "grouped_v2",
                "dataset_sha256": "train-sha",
                "fold_assignment_sha256": FOLDS_SHA,
            },
            "jp": {
                "method": "PPR-jp",
                "modality": "jp",
                "k_in": 20,
                "seed_variant": "jp_only",
                "alpha": 0.7,
                "n_folds_covered": 5,
                "expected_folds": 5,
                "fold_coverage": 1.0,
                "n_questions_expected": 2,
                "n_questions_covered": 2,
                "question_coverage": 1.0,
                "eligible_champion": True,
                "protocol_version": "grouped_v2",
                "dataset_sha256": "train-sha",
                "fold_assignment_sha256": FOLDS_SHA,
            },
        }
    }
    (graph_dir / "selected_champions.json").write_text(
        json.dumps(champions), encoding="utf-8"
    )
    rows = []
    for target, modality, method, k_in in [
        ("articles_strict", "art", "PPR-article", 50),
        ("jp", "jp", "PPR-jp", 20),
    ]:
        rows.append(
            {
                "graph_version": "G1",
                "family": "ppr",
                "target": target,
                "modality": modality,
                "method": method,
                "k_in": k_in,
                "question_coverage": 1.0,
                "n_questions_covered": 2,
                "n_questions_benchmark": 2,
                "m1": 1.0,
                "hit": 1.0,
                "mrr": 1.0,
                "ndcg": 1.0,
                "seed_variant": "both" if modality == "art" else "jp_only",
                "alpha": 0.5 if modality == "art" else 0.7,
                "protocol_version": "grouped_v2",
                "dataset_sha256": "train-sha",
                "fold_assignment_sha256": FOLDS_SHA,
                "eligible_champion": True,
                "n_folds_covered": 5,
                "manifest_sha256": result_manifest_sha,
                "internal_eval_sha256": EVAL_SHA,
                "graph_matrix_sha256": MATRIX_SHA,
            }
        )
    with (graph_dir / "final_champions_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rankings = []
    for qid, article_id, jp_id in [("q1", "a1", "j1"), ("q2", "a2", "j2")]:
        for rank in range(1, 11):
            rankings.append(
                {
                    "qid": qid,
                    "method": "PPR-article",
                    "k_in": 50,
                    "modality": "art",
                    "rank": rank,
                    "item_id": article_id if rank == 1 else f"article-{qid}-{rank}",
                }
            )
            rankings.append(
                {
                    "qid": qid,
                    "method": "PPR-jp",
                    "k_in": 20,
                    "modality": "jp",
                    "rank": rank,
                    "item_id": jp_id if rank == 1 else f"jp-{qid}-{rank}",
                }
            )
    pd.DataFrame(rankings).to_parquet(graph_dir / "rankings.parquet", index=False)


def _questions():
    return {
        "q1": {"articles_attendus": ["a1"], "gold_jp_ids": ["j1"]},
        "q2": {"articles_attendus": ["a2"], "gold_jp_ids": ["j2"]},
    }


def test_audit_accepts_complete_recovered_ppr_with_a_declared_historical_manifest(tmp_path: Path):
    _write_fixture(tmp_path)

    result = MODULE.audit_ppr_final_outputs(
        final_root=tmp_path,
        questions_by_qid=_questions(),
        graph_matrix_sha256s={"G1": MATRIX_SHA},
        expected_eval_sha256=EVAL_SHA,
        expected_fold_sha256=FOLDS_SHA,
        allowed_result_manifest_sha256s={RESULT_MANIFEST_SHA},
        expected_question_count=2,
    )

    assert result["status"] == "complete"
    assert result["graphs"]["G1"]["status"] == "complete"
    assert result["graphs"]["G1"]["ranking_depth"] == 10
    assert result["graphs"]["G1"]["metrics"]["articles_strict"]["recall_at_10"] == 1.0
    assert result["graphs"]["G1"]["metrics"]["jp"]["official_hit_at_10"] == 1.0


def test_audit_rejects_an_undeclared_historical_manifest(tmp_path: Path):
    _write_fixture(tmp_path, result_manifest_sha="e" * 64)

    result = MODULE.audit_ppr_final_outputs(
        final_root=tmp_path,
        questions_by_qid=_questions(),
        graph_matrix_sha256s={"G1": MATRIX_SHA},
        expected_eval_sha256=EVAL_SHA,
        expected_fold_sha256=FOLDS_SHA,
        allowed_result_manifest_sha256s={RESULT_MANIFEST_SHA},
        expected_question_count=2,
    )

    assert result["status"] == "incomplete_or_invalid"
    assert "undeclared_result_manifest_sha256" in result["graphs"]["G1"]["errors"]


def test_audit_recomputes_top10_from_a_historical_top20_summary(tmp_path: Path):
    _write_fixture(tmp_path)
    graph_dir = tmp_path / "G1"
    champions = json.loads((graph_dir / "selected_champions.json").read_text(encoding="utf-8"))
    for champion in champions["ppr"].values():
        champion["n_questions_expected"] = 3
        champion["n_questions_covered"] = 3
    (graph_dir / "selected_champions.json").write_text(json.dumps(champions), encoding="utf-8")

    rankings = []
    for qid, article_id, jp_id in [("q1", "a1", "j1"), ("q2", "a2", "j2")]:
        for method, k_in, modality, gold_id in [
            ("PPR-article", 50, "art", article_id),
            ("PPR-jp", 20, "jp", jp_id),
        ]:
            for rank in range(1, 21):
                rankings.append(
                    {
                        "qid": qid,
                        "method": method,
                        "k_in": k_in,
                        "modality": modality,
                        "rank": rank,
                        "item_id": gold_id if rank == 20 else f"{modality}-{qid}-{rank}",
                    }
                )
    pd.DataFrame(rankings).to_parquet(graph_dir / "rankings.parquet", index=False)

    summary_rows = list(csv.DictReader((graph_dir / "final_champions_summary.csv").open(encoding="utf-8")))
    for row in summary_rows:
        row.update({"m1": "1.0", "hit": "1.0", "mrr": "0.05", "ndcg": str(1 / math.log2(21))})
    with (graph_dir / "final_champions_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    result = MODULE.audit_ppr_final_outputs(
        final_root=tmp_path,
        questions_by_qid=_questions(),
        graph_matrix_sha256s={"G1": MATRIX_SHA},
        expected_eval_sha256=EVAL_SHA,
        expected_fold_sha256=FOLDS_SHA,
        allowed_result_manifest_sha256s={RESULT_MANIFEST_SHA},
        expected_question_count=2,
        expected_selection_question_count=3,
        historical_summary_top_k=20,
    )

    assert result["status"] == "complete"
    assert result["graphs"]["G1"]["metrics"]["articles_strict"]["recall_at_10"] == 0.0
    assert result["graphs"]["G1"]["metrics"]["jp"]["official_hit_at_10"] == 0.0
