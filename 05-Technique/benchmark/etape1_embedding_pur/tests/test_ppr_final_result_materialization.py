from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "73_materialize_e022_ppr_results.py"


def _load_module():
    assert SCRIPT.exists(), "PPR result materializer is missing"
    spec = importlib.util.spec_from_file_location("ppr_final_result_materialization", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_materializer_exports_exact_top10_means_dispersion_and_separate_tables(tmp_path: Path):
    graph_dir = tmp_path / "G1"
    graph_dir.mkdir()
    champions = {
        "ppr": {
            "art": {"method": "PPR-art", "modality": "art", "k_in": 50, "seed_variant": "both", "alpha": 0.5},
            "jp": {"method": "PPR-jp", "modality": "jp", "k_in": 20, "seed_variant": "jp_only", "alpha": 0.7},
        }
    }
    champions_path = graph_dir / "selected_champions.json"
    champions_path.write_text(json.dumps(champions), encoding="utf-8")
    summary_path = graph_dir / "final_champions_summary.csv"
    summary_path.write_text("placeholder\n", encoding="utf-8")
    rankings = []
    for qid, article_id, jp_id in [("q1", "a1", "j1"), ("q2", "a2", "j2")]:
        for method, k_in, modality, gold_id, gold_rank in [
            ("PPR-art", 50, "art", article_id, 1 if qid == "q1" else 11),
            ("PPR-jp", 20, "jp", jp_id, 2 if qid == "q1" else 1),
        ]:
            for rank in range(1, 11):
                rankings.append(
                    {
                        "qid": qid,
                        "method": method,
                        "k_in": k_in,
                        "modality": modality,
                        "rank": rank,
                        "item_id": gold_id if rank == gold_rank else f"{modality}-{qid}-{rank}",
                    }
                )
    rankings_path = graph_dir / "rankings.parquet"
    pd.DataFrame(rankings).to_parquet(rankings_path, index=False)
    audit = {
        "status": "complete",
        "top_k_metrics": 10,
        "historical_summary_top_k": 20,
        "graphs": {
            "G1": {
                "status": "complete",
                "errors": [],
                "files": {
                    "selected_champions.json": _sha(champions_path),
                    "final_champions_summary.csv": _sha(summary_path),
                    "rankings.parquet": _sha(rankings_path),
                },
                "metrics": {
                    "articles_strict": {
                        "recall_at_10": 0.5,
                        "official_hit_at_10": 0.5,
                        "mrr_at_10": 0.5,
                        "ndcg_at_10": 0.5,
                    },
                    "jp": {
                        "recall_at_10": 1.0,
                        "official_hit_at_10": 1.0,
                        "mrr_at_10": 0.75,
                        "ndcg_at_10": 0.8154648767857288,
                    },
                },
                "duplicate_item_questions_in_raw_top_10": {"articles_strict": 0, "jp": 0},
            }
        },
    }
    questions = {
        "q1": {"articles_attendus": ["a1"], "gold_jp_ids": ["j1"]},
        "q2": {"articles_attendus": ["a2"], "gold_jp_ids": ["j2"]},
    }

    module = _load_module()
    rows = module.build_result_rows(audit=audit, final_root=tmp_path, questions_by_qid=questions)

    assert len(rows) == 6
    article_recall = next(row for row in rows if row["task"] == "articles" and row["metric"] == "Recall@10")
    assert article_recall["mean"] == 0.5
    assert article_recall["sample_std"] == pytest.approx(0.7071067811865476)
    assert article_recall["configuration"] == "PPR-art; k_in=50; seed_variant=both; alpha=0.5"
    jp_mrr = next(row for row in rows if row["task"] == "jp" and row["metric"] == "MRR@10")
    assert jp_mrr["mean"] == 0.75

    output_dir = tmp_path / "out"
    module.write_outputs(rows, output_dir=output_dir, audit=audit)
    assert (output_dir / "internal_eval_ppr_exact.csv").is_file()
    assert (output_dir / "table_articles_ppr_exact.csv").is_file()
    assert (output_dir / "table_jp_ppr_exact.csv").is_file()
    assert json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))["rows"] == 6
