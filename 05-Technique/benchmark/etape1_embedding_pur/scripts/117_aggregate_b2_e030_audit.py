#!/usr/bin/env python3
"""Audit and aggregate frozen E030 shards without treating technical zeros as judgments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


TERMINAL_STATUSES = {"ok", "invalid", "error"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def audit_shard(*, shard: dict[str, Any], data_root: Path, output_root: Path, manifest_sha: str) -> dict[str, Any]:
    shard_id = str(shard["id"])
    jobs_path = data_root / shard["jobs"]["path"]
    run_root = output_root / "shards" / shard_id
    responses_path = run_root / "responses.jsonl"
    materialized = run_root / "materialized"
    receipt_path = materialized / "materialization_receipt.json"
    run_receipt_path = run_root / "run_receipt.json"
    required = (jobs_path, responses_path, receipt_path, run_receipt_path)
    if not all(path.is_file() for path in required):
        missing = ";".join(str(path) for path in required if not path.is_file())
        return {"shard": shard_id, "technical_verdict": "missing_artifact", "missing": missing}

    jobs = load_jsonl(jobs_path)
    responses = load_jsonl(responses_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    run_receipt = json.loads(run_receipt_path.read_text(encoding="utf-8"))
    job_ids = [str(job.get("job_id")) for job in jobs]
    response_ids = [str(response.get("job_id")) for response in responses if response.get("status") in TERMINAL_STATUSES]
    statuses = Counter(str(response.get("status")) for response in responses)
    material_files_ok = True
    for filename, expected_hash in receipt.get("files", {}).items():
        path = materialized / filename
        material_files_ok = material_files_ok and path.is_file() and sha256(path) == expected_hash

    per_question_path = materialized / "per_question_judge_scores.csv"
    score_path = materialized / "judge_scores_at_10.csv"
    positions_path = materialized / "per_position_judgments.jsonl"
    if not all(path.is_file() for path in (per_question_path, score_path, positions_path)):
        return {"shard": shard_id, "technical_verdict": "missing_materialized_file"}
    with per_question_path.open(encoding="utf-8") as handle:
        per_question = list(csv.DictReader(handle))
    with score_path.open(encoding="utf-8") as handle:
        score_rows = list(csv.DictReader(handle))
    positions = load_jsonl(positions_path)
    positions_by_question = Counter(str(row.get("qid")) for row in positions)
    expected_jobs = int(shard["jobs"]["count"])
    job_hash_ok = sha256(jobs_path) == str(shard["jobs"]["sha256"]) == str(run_receipt.get("jobs_sha256"))
    coverage_ok = (
        len(jobs) == expected_jobs
        and len(responses) == expected_jobs
        and len(set(job_ids)) == expected_jobs
        and set(job_ids) == set(response_ids)
        and len(response_ids) == len(set(response_ids))
        and len(positions) == expected_jobs
        and len(per_question) == 754
        and len(positions_by_question) == 754
        and set(positions_by_question.values()) == {10}
    )
    manifest_ok = str(run_receipt.get("execution_manifest_sha256")) == manifest_sha
    clean = coverage_ok and job_hash_ok and material_files_ok and manifest_ok and statuses == {"ok": expected_jobs}
    return {
        "shard": shard_id,
        "family": str(score_rows[0].get("family", "")) if score_rows else "",
        "modality": str(shard["modality"]),
        "slurm_job_id": str(run_receipt.get("slurm_job_id", "")),
        "jobs": expected_jobs,
        "questions": len(per_question),
        "judge_score_at_10": str(score_rows[0].get("judge_score_at_10", "")) if score_rows else "",
        "ok_count": statuses.get("ok", 0),
        "invalid_count": statuses.get("invalid", 0),
        "error_count": statuses.get("error", 0),
        "jobs_hash_ok": job_hash_ok,
        "manifest_hash_ok": manifest_ok,
        "materialization_files_ok": material_files_ok,
        "fixed_k_coverage_ok": coverage_ok,
        "gpu": str(run_receipt.get("gpu", "")),
        "technical_verdict": "complete_clean" if clean else "partial_technical",
        "responses_sha256": sha256(responses_path),
        "materialization_receipt_sha256": sha256(receipt_path),
        "run_receipt_sha256": sha256(run_receipt_path),
    }


def aggregate(*, manifest_path: Path, data_root: Path, out_dir: Path) -> dict[str, Path]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable aggregation: {out_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != "E030":
        raise ValueError("expected an E030 execution manifest")
    manifest_sha = sha256(manifest_path)
    output_root = data_root / manifest["outputs"]["root"]
    rows = [audit_shard(shard=shard, data_root=data_root, output_root=output_root, manifest_sha=manifest_sha) for shard in manifest["shards"]]
    if len(rows) != len(manifest["shards"]):
        raise ValueError("shard count mismatch")
    out_dir.mkdir(parents=True)
    columns = [
        "shard", "family", "modality", "slurm_job_id", "jobs", "questions", "judge_score_at_10",
        "ok_count", "invalid_count", "error_count", "jobs_hash_ok", "manifest_hash_ok",
        "materialization_files_ok", "fixed_k_coverage_ok", "gpu", "technical_verdict",
        "responses_sha256", "materialization_receipt_sha256", "run_receipt_sha256", "missing",
    ]
    audit_path = out_dir / "per_shard_audit.csv"
    write_csv(audit_path, rows, columns)
    clean_rows = [row for row in rows if row.get("technical_verdict") == "complete_clean"]
    clean_path = out_dir / "complete_clean_scores.csv"
    write_csv(clean_path, clean_rows, columns)
    lightgcn_rows = [row for row in rows if str(row.get("family", "")).startswith("lightgcn_seed")]
    lightgcn_mean_allowed = len(lightgcn_rows) == 6 and all(row.get("technical_verdict") == "complete_clean" for row in lightgcn_rows)
    receipt = {
        "schema_version": "b2-e030-aggregation-audit.v1",
        "experiment_id": "E030",
        "execution_manifest": str(manifest_path),
        "execution_manifest_sha256": manifest_sha,
        "aggregation_script_sha256": sha256(Path(__file__).resolve()),
        "shards_expected": len(manifest["shards"]),
        "shards_complete_clean": len(clean_rows),
        "shards_partial_or_missing": len(rows) - len(clean_rows),
        "lightgcn_seed_mean_allowed": lightgcn_mean_allowed,
        "campaign_verdict": "complete_clean" if len(clean_rows) == len(rows) else "incomplete_technical_failures",
        "scientific_status": "exploratory_until_lawyer_audit",
        "files": {},
    }
    receipt["files"] = {audit_path.name: sha256(audit_path), clean_path.name: sha256(clean_path)}
    receipt_path = out_dir / "aggregation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"audit": audit_path, "complete_clean": clean_path, "receipt": receipt_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({name: str(path) for name, path in aggregate(
        manifest_path=args.execution_manifest, data_root=args.data_root, out_dir=args.out_dir
    ).items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
