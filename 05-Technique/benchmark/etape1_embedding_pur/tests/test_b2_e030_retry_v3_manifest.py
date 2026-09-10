from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "120_prepare_b2_e030_retry_v3.py"
V2_MANIFEST = ROOT / "configs" / "b2_llm_as_judge_a3_execution_v2_gpu70.json"
AUDIT = ROOT.parents[2] / "results" / "benchmark-a3-b1" / "e030-aggregation-audit-v1" / "per_shard_audit.csv"


def _load_module():
    spec = importlib.util.spec_from_file_location("b2_e030_retry_v3", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v3_manifest_retries_only_the_17_partial_shards_with_unique_ports(tmp_path):
    retry = _load_module()
    output = tmp_path / "e030-v3.json"

    retry.create_manifest(
        v2_manifest_path=V2_MANIFEST,
        audit_csv_path=AUDIT,
        output_path=output,
        launcher_path=ROOT / "scripts" / "sbatch_b2_e030_judge_v3.sh",
        smoke_launcher_path=ROOT / "scripts" / "sbatch_b2_e030_smoke_v3.sh",
        fail_closed_runner_path=ROOT / "scripts" / "121_run_b2_e030_judge_fail_closed_v3.py",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    shards = payload["shards"]
    assert len(shards) == 17
    assert {shard["id"] for shard in shards}.isdisjoint({
        "cosine-jp", "ppr-article", "ppr-jp", "lightgcn-seed42-article", "lightgcn-seed42-jp",
    })
    assert len({shard["vllm_port"] for shard in shards}) == 17
    assert all(18401 <= shard["vllm_port"] <= 18417 for shard in shards)
    assert payload["retry_policy"]["reuse_v2_scores"] is False
    assert payload["execution_gate"]["full_retry"] == "requires_successful_smoke_receipt_and_explicit_submission"
    assert payload["model"]["revision"] == "4033b16200f4152e55e100ea12dc388c537df622"


def test_v3_manifest_refuses_an_audit_that_does_not_select_exactly_17_shards(tmp_path):
    retry = _load_module()
    damaged_audit = tmp_path / "audit.csv"
    damaged_audit.write_text("shard,technical_verdict\ncosine-jp,complete_clean\n", encoding="utf-8")

    try:
        retry.create_manifest(
            v2_manifest_path=V2_MANIFEST,
            audit_csv_path=damaged_audit,
            output_path=tmp_path / "should-not-exist.json",
            launcher_path=ROOT / "scripts" / "sbatch_b2_e030_judge_v3.sh",
            smoke_launcher_path=ROOT / "scripts" / "sbatch_b2_e030_smoke_v3.sh",
            fail_closed_runner_path=ROOT / "scripts" / "121_run_b2_e030_judge_fail_closed_v3.py",
        )
    except ValueError as exc:
        assert "shard identifiers" in str(exc)
    else:
        raise AssertionError("a damaged v2 audit must block v3 manifest creation")
