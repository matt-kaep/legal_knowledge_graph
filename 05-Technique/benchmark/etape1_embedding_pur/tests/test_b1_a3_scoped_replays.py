import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIGHTGCN_SCRIPT = ROOT / "scripts" / "101_aggregate_b1_a3_scoped_lightgcn.py"
PPR_SCRIPT = ROOT / "scripts" / "102_replay_b1_a3_scoped_ppr.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _lightgcn_manifest():
    return {
        "scope": {"graph_version": "G1", "atomic_cv": {"task_count": 120, "fold_count": 5, "selection_targets": ["art", "jp"]}},
        "selection": {"scope": "train_cv_only", "freeze_before_evaluation": True},
        "replay": {"final_seeds": [42, 43, 44], "top_k_out": 100},
    }


def test_scoped_lightgcn_requires_one_complete_graph_scope():
    runner = _load(LIGHTGCN_SCRIPT, "scoped_lightgcn")
    manifest = _lightgcn_manifest()
    runner.validate_manifest(manifest)
    tasks = [{"task_id": f"g1-{index}", "graph_version": "G1"} for index in range(120)]
    assert len(runner.select_scoped_tasks(manifest, {"tasks": tasks})) == 120
    with pytest.raises(ValueError, match="expected 120"):
        runner.select_scoped_tasks(manifest, {"tasks": tasks[:-1]})


def test_scoped_ppr_rejects_unfrozen_or_non_a3_contracts():
    runner = _load(PPR_SCRIPT, "scoped_ppr")
    manifest = {
        "a3": {"sha256": "c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3"},
        "selection": {"scope": "train_cv_only", "freeze_before_evaluation": True},
        "replay": {"top_k_out": 100},
        "conditions": [{"id": "g1-art"}],
    }
    runner.validate_manifest(manifest)
    manifest["selection"]["freeze_before_evaluation"] = False
    with pytest.raises(ValueError, match="frozen"):
        runner.validate_manifest(manifest)
