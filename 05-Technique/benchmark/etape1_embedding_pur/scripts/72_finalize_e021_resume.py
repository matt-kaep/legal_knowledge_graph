"""Write a hash-bound receipt for an E021 reranking resume attempt.

The E021 response JSONL is append-only.  This receipt records its final byte
hash and checks completeness by the family/question key, so a large response
history cannot conceal missing units or an incomplete metric aggregation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _latest_valid_keys(
    records: Iterable[dict[str, Any]],
    expected_input_sha256s: dict[tuple[str, str], str],
) -> set[tuple[str, str]]:
    """Keep only successful responses for the exact immutable job contract."""
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        key = (str(record["family"]), str(record["qid"]))
        if record.get("input_sha256") != expected_input_sha256s.get(key):
            continue
        latest[key] = record
    return set(latest)


def build_completion_record(
    *,
    jobs_path: Path,
    responses_path: Path,
    metrics_path: Path,
    expected_families: tuple[str, ...],
    expected_questions_per_family: int,
) -> dict[str, Any]:
    """Build a completion verdict from the immutable jobs and aggregated metrics."""
    jobs = _load_jsonl(jobs_path)
    responses = _load_jsonl(responses_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_input_sha256s = {
        (str(row["family"]), str(row["qid"])): str(row["input_sha256"])
        for row in jobs
    }
    jobs_by_family = {
        family: {str(row["qid"]) for row in jobs if str(row["family"]) == family}
        for family in expected_families
    }
    latest_valid = _latest_valid_keys(responses, expected_input_sha256s)
    families: dict[str, Any] = {}
    complete = True
    for family in expected_families:
        family_metrics = metrics.get("families", {}).get(family, {})
        valid_for_family = sum(1 for key in latest_valid if key[0] == family)
        expected_job_qids = jobs_by_family[family]
        checks = {
            "job_question_count_matches": len(expected_job_qids) == expected_questions_per_family,
            "metrics_expected_questions_matches": family_metrics.get("expected_questions") == expected_questions_per_family,
            "metrics_valid_responses_matches": family_metrics.get("valid_responses") == expected_questions_per_family,
            "metrics_missing_questions_zero": family_metrics.get("missing_questions") == 0,
            "metrics_coverage_complete": family_metrics.get("coverage") == 1.0,
            "metrics_status_complete": family_metrics.get("status") == "complete",
            "response_history_has_every_key": valid_for_family == expected_questions_per_family,
        }
        family_complete = all(checks.values())
        complete = complete and family_complete
        families[family] = {
            "status": "complete" if family_complete else "incomplete",
            "checks": checks,
            "valid_response_keys": valid_for_family,
            "missing_questions": family_metrics.get("missing_questions"),
            "metrics": family_metrics,
        }
    return {
        "receipt_kind": "e021_reranking_resume_completion",
        "status": "complete" if complete else "incomplete",
        "jobs_sha256": sha256_file(jobs_path),
        "responses_sha256": sha256_file(responses_path),
        "metrics_sha256": sha256_file(metrics_path),
        "jobs": len(jobs),
        "response_history_records": len(responses),
        "latest_valid_response_keys": len(latest_valid),
        "expected_families": list(expected_families),
        "expected_questions_per_family": expected_questions_per_family,
        "families": families,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-family", action="append", required=True)
    parser.add_argument("--expected-questions-per-family", type=int, required=True)
    parser.add_argument("--resume-manifest", type=Path)
    args = parser.parse_args(argv)

    receipt = build_completion_record(
        jobs_path=args.jobs,
        responses_path=args.responses,
        metrics_path=args.metrics,
        expected_families=tuple(args.expected_family),
        expected_questions_per_family=args.expected_questions_per_family,
    )
    if args.resume_manifest is not None:
        receipt["resume_manifest_sha256"] = sha256_file(args.resume_manifest)
        receipt["resume_manifest"] = str(args.resume_manifest)
    receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["finalizer_script_sha256"] = sha256_file(Path(__file__))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
