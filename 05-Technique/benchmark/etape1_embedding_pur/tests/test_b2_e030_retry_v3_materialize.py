"""Regression contract for the fail-closed E030 r6 materializer."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "122_materialize_b2_e030_retry_v3.py"


def _job(position: int, *, zero_slot: bool = False) -> dict:
    return {
        "job_id": f"job-{position}",
        "family": "reranked_lightgcn_seed43",
        "modality": "jp",
        "qid": "q1",
        "position": position,
        "question": "Question test",
        "candidate_id_internal": None if zero_slot else f"JP-{position}",
        "document": None if zero_slot else {"synthese": "Synthèse test."},
        "zero_slot": zero_slot,
        "model_id": "model",
        "model_revision": "revision",
        "temperature": 0,
    }


class RetryV3MaterializationTests(unittest.TestCase):
    def test_explicit_zero_slot_is_a_fixed_k_zero_not_a_transport_error(self) -> None:
        spec = importlib.util.spec_from_file_location("retry_v3_materialize", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        jobs = [_job(position, zero_slot=position == 1) for position in range(1, 11)]
        responses = [
            {"job_id": "job-1", "status": "zero_slot", "judgment": None, "validation_reason": "unresolved_zero_slot"},
            *[
                {"job_id": f"job-{position}", "status": "ok", "judgment": {"classe": "A", "justification": "Pertinent."}}
                for position in range(2, 11)
            ],
        ]
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory) / "materialized"
            summary = module.materialize_retry_v3(jobs, responses, out_dir)

            self.assertEqual(summary[0]["judge_score_at_10"], 0.9)
            rows = [json.loads(line) for line in (out_dir / "per_position_judgments.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["response_status"], "zero_slot")
            self.assertEqual(rows[0]["label"], "non_jugeable")


if __name__ == "__main__":
    unittest.main()
