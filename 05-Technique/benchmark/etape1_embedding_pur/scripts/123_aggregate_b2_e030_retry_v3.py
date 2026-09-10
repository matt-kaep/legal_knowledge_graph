#!/usr/bin/env python3
"""Audit complete E030 r6 retries without reusing E030 v2 scores.

Only ``ok`` model judgments and explicitly frozen ``zero_slot`` positions are
terminal for this campaign.  Transport/format failures are rejected rather
than converted into a score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_QUESTIONS = 754
FIXED_K = 10
ALLOWED_STATUSES = {"ok", "zero_slot"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_response_statuses(jobs: list[dict[str, Any]], responses: list[dict[str, Any]]) -> dict[str, int]:
    """Validate r6 terminal statuses and the one-to-one frozen zero-slot map."""
    response_by_id = {str(row.get("job_id")): row for row in responses}
    if len(response_by_id) != len(responses):
        raise ValueError("duplicate response ids")
    statuses = Counter(str(row.get("status")) for row in responses)
    unexpected = set(statuses) - ALLOWED_STATUSES
    if unexpected:
        raise ValueError(f"invalid retry-v3 response statuses: {sorted(unexpected)}")
    frozen_zero_ids = {str(job.get("job_id")) for job in jobs if job.get("zero_slot") is True}
    actual_zero_ids = {job_id for job_id, row in response_by_id.items() if row.get("status") == "zero_slot"}
    if actual_zero_ids != frozen_zero_ids:
        raise ValueError("zero_slot responses do not match frozen zero_slot jobs")
    return dict(statuses)


def audit_shard(*, shard: dict[str, Any], data_root: Path, output_root: Path, manifest_sha: str, adapter_sha: str) -> dict[str, Any]:
    shard_id = str(shard["id"])
    jobs_path = data_root / str(shard["jobs"]["path"])
    run_root = output_root / "shards" / shard_id
    responses_path = run_root / "responses.jsonl"
    run_receipt_path = run_root / "run_receipt.json"
    materialized = run_root / "materialized"
    base_receipt_path = materialized / "materialization_receipt.json"
    retry_receipt_path = materialized / "retry_v3_materialization_receipt.json"
    required = (jobs_path, responses_path, run_receipt_path, base_receipt_path, retry_receipt_path)
    if not all(path.is_file() for path in required):
        return {"shard": shard_id, "technical_verdict": "missing_artifact", "missing": ";".join(str(path) for path in required if not path.is_file())}

    jobs = load_jsonl(jobs_path)
    responses = load_jsonl(responses_path)
    run_receipt = load_json(run_receipt_path)
    base_receipt = load_json(base_receipt_path)
    retry_receipt = load_json(retry_receipt_path)
    expected_jobs = int(shard["jobs"]["count"])
    job_ids = [str(job.get("job_id")) for job in jobs]
    response_ids = [str(row.get("job_id")) for row in responses]

    try:
        statuses = validate_response_statuses(jobs, responses)
        status_ok = True
    except ValueError as exc:
        statuses = Counter(str(row.get("status")) for row in responses)
        status_ok = False
        status_error = str(exc)
    else:
        status_error = ""

    material_files_ok = True
    for filename, expected_hash in base_receipt.get("files", {}).items():
        path = materialized / filename
        material_files_ok = material_files_ok and path.is_file() and sha256(path) == expected_hash
    positions_path = materialized / "per_position_judgments.jsonl"
    questions_path = materialized / "per_question_judge_scores.csv"
    scores_path = materialized / "judge_scores_at_10.csv"
    material_files_ok = material_files_ok and all(path.is_file() for path in (positions_path, questions_path, scores_path))
    positions = load_jsonl(positions_path) if positions_path.is_file() else []
    with questions_path.open(encoding="utf-8") as handle:
        questions = list(csv.DictReader(handle))
    with scores_path.open(encoding="utf-8") as handle:
        scores = list(csv.DictReader(handle))
    positions_by_question = Counter(str(row.get("qid")) for row in positions)
    coverage_ok = (
        len(jobs) == expected_jobs == EXPECTED_QUESTIONS * FIXED_K
        and len(responses) == expected_jobs
        and len(set(job_ids)) == expected_jobs
        and set(job_ids) == set(response_ids)
        and len(set(response_ids)) == expected_jobs
        and len(positions) == expected_jobs
        and len(questions) == EXPECTED_QUESTIONS
        and len(positions_by_question) == EXPECTED_QUESTIONS
        and set(positions_by_question.values()) == {FIXED_K}
    )
    job_hash_ok = sha256(jobs_path) == str(shard["jobs"]["sha256"]) == str(run_receipt.get("jobs_sha256"))
    manifest_ok = str(run_receipt.get("execution_manifest_sha256")) == manifest_sha
    adapter_ok = (
        str(retry_receipt.get("adapter_sha256")) == adapter_sha
        and str(retry_receipt.get("base_receipt_sha256")) == sha256(base_receipt_path)
        and int(retry_receipt.get("responses", -1)) == expected_jobs
    )
    clean = coverage_ok and status_ok and material_files_ok and job_hash_ok and manifest_ok and adapter_ok and len(scores) == 1
    score = scores[0] if len(scores) == 1 else {}
    return {
        "shard": shard_id,
        "family": str(score.get("family", "")),
        "modality": str(shard.get("modality", "")),
        "slurm_job_id": str(run_receipt.get("slurm_job_id", "")),
        "judge_score_at_10": str(score.get("judge_score_at_10", "")),
        "jobs": expected_jobs,
        "questions": len(questions),
        "ok_count": statuses.get("ok", 0),
        "zero_slot_count": statuses.get("zero_slot", 0),
        "jobs_hash_ok": job_hash_ok,
        "manifest_hash_ok": manifest_ok,
        "adapter_hash_ok": adapter_ok,
        "materialization_files_ok": material_files_ok,
        "fixed_k_coverage_ok": coverage_ok,
        "status_contract_ok": status_ok,
        "status_error": status_error,
        "gpu": str(run_receipt.get("gpu", "")),
        "technical_verdict": "complete_clean" if clean else "partial_technical",
        "responses_sha256": sha256(responses_path),
        "base_materialization_receipt_sha256": sha256(base_receipt_path),
        "retry_materialization_receipt_sha256": sha256(retry_receipt_path),
        "run_receipt_sha256": sha256(run_receipt_path),
    }


def lightgcn_means(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        match = re.fullmatch(r"(reranked-lightgcn)-seed(42|43|44)-(article|jp)", str(row["shard"]))
        if match and row["technical_verdict"] == "complete_clean":
            groups[(match.group(1), match.group(3))].append(row)
    means = []
    for (family, modality), items in sorted(groups.items()):
        seeds = {re.search(r"seed(42|43|44)", str(item["shard"])).group(1) for item in items}
        if seeds == {"42", "43", "44"}:
            means.append({"family": family, "modality": modality, "seeds": "42;43;44", "judge_score_at_10_mean": sum(float(item["judge_score_at_10"]) for item in items) / 3, "shards": 3})
    return means


def aggregate(*, manifest_path: Path, data_root: Path, out_dir: Path, adapter_path: Path) -> dict[str, Path]:
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable aggregation: {out_dir}")
    manifest = load_json(manifest_path)
    if manifest.get("experiment_id") != "E030" or len(manifest.get("shards", [])) != 17:
        raise ValueError("expected the 17-shard E030 retry-v3 manifest")
    manifest_sha = sha256(manifest_path)
    adapter_sha = sha256(adapter_path)
    output_root = data_root / str(manifest["outputs"]["root"])
    rows = [audit_shard(shard=shard, data_root=data_root, output_root=output_root, manifest_sha=manifest_sha, adapter_sha=adapter_sha) for shard in manifest["shards"]]
    out_dir.mkdir(parents=True)
    columns = ["shard", "family", "modality", "slurm_job_id", "judge_score_at_10", "jobs", "questions", "ok_count", "zero_slot_count", "jobs_hash_ok", "manifest_hash_ok", "adapter_hash_ok", "materialization_files_ok", "fixed_k_coverage_ok", "status_contract_ok", "status_error", "gpu", "technical_verdict", "responses_sha256", "base_materialization_receipt_sha256", "retry_materialization_receipt_sha256", "run_receipt_sha256", "missing"]
    audit_path = out_dir / "per_shard_audit.csv"
    write_csv(audit_path, rows, columns)
    clean_rows = [row for row in rows if row["technical_verdict"] == "complete_clean"]
    scores_path = out_dir / "complete_clean_scores.csv"
    write_csv(scores_path, clean_rows, columns)
    means = lightgcn_means(clean_rows)
    means_path = out_dir / "lightgcn_seed_mean_scores.csv"
    write_csv(means_path, means, ["family", "modality", "seeds", "judge_score_at_10_mean", "shards"])
    receipt = {
        "schema_version": "b2-e030-retry-v3-aggregation-audit.v1",
        "experiment_id": "E030",
        "execution_manifest": str(manifest_path),
        "execution_manifest_sha256": manifest_sha,
        "aggregation_script_sha256": sha256(Path(__file__).resolve()),
        "materialization_adapter_sha256": adapter_sha,
        "retry_shards_expected": 17,
        "retry_shards_complete_clean": len(clean_rows),
        "retry_shards_partial_or_missing": len(rows) - len(clean_rows),
        "archived_v2_complete_clean_diagnostic_only": 5,
        "global_conditions_audited": 22,
        "v2_scores_reused": False,
        "lightgcn_seed_mean_allowed": len(means) == 2,
        "campaign_verdict": "retry_complete_clean" if len(clean_rows) == len(rows) else "incomplete_technical_failures",
        "scientific_status": "exploratory_until_lawyer_audit",
        "files": {},
    }
    receipt["files"] = {path.name: sha256(path) for path in (audit_path, scores_path, means_path)}
    receipt_path = out_dir / "aggregation_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"audit": audit_path, "scores": scores_path, "lightgcn_means": means_path, "receipt": receipt_path}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--materialization-adapter", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps({name: str(path) for name, path in aggregate(manifest_path=args.execution_manifest, data_root=args.data_root, out_dir=args.out_dir, adapter_path=args.materialization_adapter).items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
