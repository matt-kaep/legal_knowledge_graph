"""Contracts for the E030 r6 post-execution audit."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "123_aggregate_b2_e030_retry_v3.py"


class RetryV3AuditTests(unittest.TestCase):
    def test_explicit_zero_slot_is_allowed_but_invalid_model_response_is_not(self) -> None:
        spec = importlib.util.spec_from_file_location("retry_v3_audit", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        jobs = [{"job_id": "zero", "zero_slot": True}, {"job_id": "ok", "zero_slot": False}]
        responses = [{"job_id": "zero", "status": "zero_slot"}, {"job_id": "ok", "status": "ok"}]

        self.assertEqual(module.validate_response_statuses(jobs, responses), {"ok": 1, "zero_slot": 1})
        with self.assertRaisesRegex(ValueError, "invalid"):
            module.validate_response_statuses(jobs, [{"job_id": "zero", "status": "invalid"}, {"job_id": "ok", "status": "ok"}])


if __name__ == "__main__":
    unittest.main()
