#!/usr/bin/env python3
"""Materialize E030 r6 responses while preserving explicit frozen zero slots.

The r6 runner can emit only ``ok`` and ``zero_slot`` responses.  A zero slot
is a known, frozen non-candidate position, not an invalid model completion or
transport error.  This adapter rejects all other statuses before delegating to
the immutable v1 materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any


BASE_PATH = Path(__file__).with_name("110_run_b2_e030_llm_judge.py")
spec = importlib.util.spec_from_file_location("e030_materializer_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(base)

ALLOWED_STATUSES = {"ok", "zero_slot"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def materialize_retry_v3(jobs: list[dict[str, Any]], responses: list[dict[str, Any]], out_dir: Path) -> list[dict[str, Any]]:
    """Write fixed-K scores, accepting only frozen zero slots beyond valid calls."""
    expected_zero_ids = {str(job["job_id"]) for job in jobs if job.get("zero_slot") is True}
    response_by_id = {str(response.get("job_id")): response for response in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate retry-v3 response id")
    statuses = {str(response.get("status")) for response in responses}
    if not statuses <= ALLOWED_STATUSES:
        raise ValueError(f"retry-v3 rejects non-fail-closed response statuses: {sorted(statuses - ALLOWED_STATUSES)}")
    actual_zero_ids = {job_id for job_id, response in response_by_id.items() if response.get("status") == "zero_slot"}
    if actual_zero_ids != expected_zero_ids:
        raise ValueError("zero_slot responses must match exactly the frozen zero_slot jobs")

    original_terminal_statuses = base.TERMINAL_STATUSES
    base.TERMINAL_STATUSES = ALLOWED_STATUSES
    try:
        summary = base.materialize_judgments(jobs, responses, out_dir)
    finally:
        base.TERMINAL_STATUSES = original_terminal_statuses

    wrapper_receipt = {
        "schema_version": "b2-e030-retry-v3-materialization.v1",
        "adapter_sha256": sha256(Path(__file__).resolve()),
        "base_materializer_sha256": sha256(BASE_PATH),
        "responses": len(responses),
        "ok_count": sum(response.get("status") == "ok" for response in responses),
        "zero_slot_count": len(actual_zero_ids),
        "base_receipt_sha256": sha256(out_dir / "materialization_receipt.json"),
    }
    (out_dir / "retry_v3_materialization_receipt.json").write_text(
        json.dumps(wrapper_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_retry_v3(load_jsonl(args.jobs), load_jsonl(args.responses), args.out_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
