#!/usr/bin/env python3
"""Create the immutable E030 v3 retry manifest from the audited v2 campaign.

This is deliberately a manifest transformation, not a result transformation:
it carries forward only frozen inputs and selects only v2 shards with a
technical verdict other than ``complete_clean``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_PARTIAL_SHARDS = 17
FIRST_RESERVED_PORT = 18401


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _partial_shard_ids(v2: dict[str, Any], audit_csv_path: Path) -> list[str]:
    with audit_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {str(row.get("shard")): str(row.get("technical_verdict")) for row in rows}
    v2_ids = [str(shard.get("id")) for shard in v2.get("shards", [])]
    if set(by_id) != set(v2_ids):
        raise ValueError("v2 audit shard identifiers differ from the frozen v2 manifest")
    partial = [shard_id for shard_id in v2_ids if by_id[shard_id] != "complete_clean"]
    if len(partial) != EXPECTED_PARTIAL_SHARDS:
        raise ValueError(f"v2 audit must select exactly {EXPECTED_PARTIAL_SHARDS} partial shards, got {len(partial)}")
    return partial


def create_manifest(
    *,
    v2_manifest_path: Path,
    audit_csv_path: Path,
    output_path: Path,
    launcher_path: Path,
    smoke_launcher_path: Path,
) -> dict[str, Any]:
    """Write an immutable E030 v3 preflight manifest and return its content."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable E030 v3 manifest: {output_path}")
    for path in (v2_manifest_path, audit_csv_path, launcher_path, smoke_launcher_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    v2 = _load_json(v2_manifest_path)
    if v2.get("experiment_id") != "E030":
        raise ValueError("v2 input is not an E030 manifest")
    partial_ids = _partial_shard_ids(v2, audit_csv_path)
    v2_by_id = {str(shard["id"]): shard for shard in v2["shards"]}
    retry_shards = []
    for index, shard_id in enumerate(partial_ids):
        shard = copy.deepcopy(v2_by_id[shard_id])
        shard["vllm_port"] = FIRST_RESERVED_PORT + index
        retry_shards.append(shard)

    payload: dict[str, Any] = {
        "manifest_id": "b2-e030-llm-as-a-judge-a3-retry-v3-preflight-2026-09-09",
        "campaign_id": "b2-llm-as-a-judge-a3-retry-v3-preflight-2026-09-09",
        "schema_version": "b2-llm-as-a-judge-retry.v3",
        "experiment_id": "E030",
        "status": "frozen_preflight_only",
        "scientific_status": "exploratory_until_lawyer_audit",
        "source_v2": {
            "manifest_path": str(v2_manifest_path),
            "sha256": sha256(v2_manifest_path),
            "audit_csv_path": str(audit_csv_path),
            "audit_csv_sha256": sha256(audit_csv_path),
            "complete_clean_archived_only": 5,
            "partial_technical_retried": EXPECTED_PARTIAL_SHARDS,
        },
        "a3": copy.deepcopy(v2["a3"]),
        "e029": copy.deepcopy(v2["e029"]),
        "model": copy.deepcopy(v2["model"]),
        "prompts": copy.deepcopy(v2["prompts"]),
        "grading": copy.deepcopy(v2["grading"]),
        "visibility_contract": copy.deepcopy(v2["visibility_contract"]),
        "code_bundle": {
            "v3_launcher": {"path": str(launcher_path), "sha256": sha256(launcher_path)},
            "v3_smoke_launcher": {"path": str(smoke_launcher_path), "sha256": sha256(smoke_launcher_path)},
            "v2_judge_runner": copy.deepcopy(v2["code_bundle"]["judge_runner"]),
            "context_auditor": copy.deepcopy(v2["code_bundle"]["context_auditor"]),
        },
        "outputs": {
            "root": "doctrine_v3plus_bench/_campaign_b2_e030_a3_retry_v3_20260909",
            "immutable": True,
        },
        "retry_policy": {
            "reuse_v2_scores": False,
            "reuse_v2_responses": False,
            "retry_only_v2_partial_technical_shards": True,
            "complete_clean_v2_shards": "archived_technical_diagnostics_only_not_aggregated",
            "http_failure": "fail_closed_no_label_no_zero_no_judge_response",
            "zero_slot": "explicit_non_candidate_fixed_k_position_not_a_judge_response",
        },
        "service_identity_contract": {
            "bind_host": "127.0.0.1",
            "ports": {shard["id"]: shard["vllm_port"] for shard in retry_shards},
            "pre_start": ["port_must_be_unbound", "no_listener_on_reserved_port"],
            "post_start": ["health_endpoint", "v1_models_endpoint", "served_model_id_includes_frozen_revision"],
            "receipt_fields": ["slurm_job_id", "node", "port", "listener", "gpu", "model_id", "model_revision", "v1_models"],
        },
        "execution_gate": {
            "smoke_test": "required_before_any_full_shard_submission",
            "full_retry": "requires_successful_smoke_receipt_and_explicit_submission",
            "smoke_anomaly": "stop_and_wait_for_explicit_decision",
        },
        "post_execution_audit": {
            "all_conditions": 22,
            "each_retry_shard": {"questions": 754, "positions_per_question": 10, "expected_rows": 7540},
            "lightgcn_seed_mean": "forbidden_unless_all_three_seeds_clean_per_modality",
            "paper_transfer": "complete_hashed_aggregates_only_exploratory_until_lawyer_agreement",
            "lawyer_agreement_required_for_comparative_claim": "lawyer_agreement.json",
        },
        "shards": retry_shards,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-manifest", type=Path, required=True)
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--smoke-launcher", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = create_manifest(
        v2_manifest_path=args.v2_manifest,
        audit_csv_path=args.audit_csv,
        output_path=args.output,
        launcher_path=args.launcher,
        smoke_launcher_path=args.smoke_launcher,
    )
    print(json.dumps({"manifest": str(args.output), "sha256": sha256(args.output), "shards": len(payload["shards"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
