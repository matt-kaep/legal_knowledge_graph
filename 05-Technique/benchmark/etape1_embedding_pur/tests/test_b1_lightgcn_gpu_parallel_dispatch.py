"""Contract checks for the resumable GPU-parallel B1-r2 LightGCN dispatch."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "97_run_b1_a3_r2_lightgcn_tasks.py"
MANIFEST = ROOT / "configs" / "confirmatory_campaign_b1_a3_r2.json"
PARENT_MANIFEST = ROOT / "configs" / "confirmatory_campaign_b1_a3_r1.json"
PROBE = ROOT / "scripts" / "sbatch_b1_a3_r2_lightgcn_gpu_probe.sh"
WORKER = ROOT / "scripts" / "sbatch_b1_a3_r2_lightgcn_cv.sh"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b1_r2_lightgcn_tasks", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_b1_r2_lightgcn_runner_is_an_atomic_gpu_dispatcher():
    assert RUNNER.is_file(), "B1-r2 requires an atomic LightGCN task runner"

    source = RUNNER.read_text(encoding="utf-8")
    assert "build_atomic_task_specs" in source
    assert "validate_atomic_task_receipts" in source
    assert 'device="cuda"' in source
    assert "preflight" in source


def test_load_campaign_requires_the_hashed_parent_manifest(tmp_path):
    runner = _load_runner()
    parent = tmp_path / "parent.json"
    parent.write_text(json.dumps({"campaign_id": "b1-r1", "outputs": {"root": "old"}}))
    parent_hash = hashlib.sha256(parent.read_bytes()).hexdigest()
    child = tmp_path / "child.json"
    child.write_text(
        json.dumps(
            {
                "campaign_id": "b1-r2",
                "parent_campaign": {"manifest_path": str(parent), "sha256": parent_hash},
                "outputs": {"root": "new"},
            }
        )
    )

    merged = runner.load_campaign(child)

    assert merged["campaign_id"] == "b1-r2"
    assert merged["outputs"] == {"root": "new"}

    child.write_text(
        json.dumps(
            {
                "campaign_id": "b1-r2",
                "parent_campaign": {"manifest_path": str(parent), "sha256": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="parent manifest hash mismatch"):
        runner.load_campaign(child)


def test_b1_r2_manifest_and_slurm_wrappers_seal_the_cuda_task_grid():
    assert MANIFEST.is_file(), "B1-r2 needs its own immutable manifest"
    assert PROBE.is_file(), "B1-r2 needs a full-epoch CUDA probe wrapper"
    assert WORKER.is_file(), "B1-r2 needs its resumable CUDA array wrapper"

    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    parent_hash = hashlib.sha256(PARENT_MANIFEST.read_bytes()).hexdigest()
    assert payload["parent_campaign"] == {
        "manifest_path": "05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r1.json",
        "sha256": parent_hash,
        "immutable": True,
    }
    assert payload["execution"]["lightgcn"]["device"] == "cuda"
    assert payload["execution"]["lightgcn"]["task_count"] == 1320
    assert "r2_cuda_atomic" in payload["outputs"]["lightgcn_cv"]

    for wrapper in (PROBE.read_text(encoding="utf-8"), WORKER.read_text(encoding="utf-8")):
        assert "confirmatory_campaign_b1_a3_r2.json" in wrapper
        assert "97_run_b1_a3_r2_lightgcn_tasks.py" in wrapper
        assert "LKG_PYTHON" in wrapper
    assert "#SBATCH --array=0-1319%32" in WORKER.read_text(encoding="utf-8")
