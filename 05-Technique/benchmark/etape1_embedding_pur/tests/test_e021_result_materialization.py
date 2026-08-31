from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "74_materialize_e021_reranking_results.py"


def _load_module():
    assert SCRIPT.exists(), "E021 result materializer is missing"
    spec = importlib.util.spec_from_file_location("e021_result_materialization", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _family(hit: float) -> dict:
    return {
        "expected_questions": 2,
        "valid_responses": 2,
        "missing_questions": 0,
        "coverage": 1.0,
        "status": "complete",
        "metrics": {"official_hit_at_10": hit, "ndcg_at_10": hit / 2, "mrr_at_10": hit / 3},
        "dispersion": {"official_hit_at_10": 0.1, "ndcg_at_10": 0.2, "mrr_at_10": 0.3},
    }


def test_materializer_requires_complete_receipt_and_exports_long_and_table_rows(tmp_path: Path):
    metrics_path = tmp_path / "metrics.json"
    manifest_path = tmp_path / "manifest.json"
    receipt_path = tmp_path / "receipt.json"
    metrics = {"families": {"cosine_bge_m3": _family(0.2), "ppr": _family(0.3), "lightgcn": _family(0.4)}}
    manifest = {
        "experiment_id": "E021-resume-v3",
        "k_in": 20,
        "k_out": 10,
        "reranker": {
            "model_id": "model",
            "model_revision": "revision",
            "temperature": 0,
            "local_parser": "strict_json_subset_then_stable_dedup_and_pool_order_completion",
        },
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {
        "status": "complete",
        "metrics_sha256": _sha(metrics_path),
        "resume_manifest_sha256": _sha(manifest_path),
        "families": {
            name: {"checks": {"metrics_coverage_complete": True, "response_history_has_every_key": True}}
            for name in metrics["families"]
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    module = _load_module()
    rows = module.build_rows(
        metrics=metrics,
        receipt=receipt,
        manifest=manifest,
        metrics_sha256=_sha(metrics_path),
        receipt_sha256=_sha(receipt_path),
        manifest_sha256=_sha(manifest_path),
    )

    assert len(rows) == 9
    hit = next(row for row in rows if row["family"] == "ppr" and row["metric"] == "Hit@10")
    assert hit["mean"] == 0.3
    assert hit["questions"] == 2
    assert "K_in=20" in hit["configuration"]

    out = tmp_path / "out"
    module.write_outputs(rows, output_dir=out, receipt=receipt, manifest=manifest)
    assert (out / "internal_eval_jp_reranking_exact.csv").is_file()
    assert (out / "table_jp_reranking_exact.csv").is_file()
    assert b"\r\n" not in (out / "internal_eval_jp_reranking_exact.csv").read_bytes()
    export_manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert export_manifest["rows"] == 9
    assert export_manifest["completion_status"] == "complete"
