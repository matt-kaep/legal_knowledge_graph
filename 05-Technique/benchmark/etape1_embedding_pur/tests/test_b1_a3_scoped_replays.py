import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
LIGHTGCN_SCRIPT = ROOT / "scripts" / "101_aggregate_b1_a3_scoped_lightgcn.py"
PPR_SCRIPT = ROOT / "scripts" / "102_replay_b1_a3_scoped_ppr.py"
PPR_V2_SCRIPT = ROOT / "scripts" / "103_replay_b1_a3_scoped_ppr_v2.py"


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


def test_scoped_ppr_accepts_a_graph_scoped_champion_without_a_redundant_graph_field(tmp_path):
    runner = _load(PPR_V2_SCRIPT, "scoped_ppr_graph_scope")
    source = tmp_path / "G1" / "champions.json"
    source.parent.mkdir()
    champion = {
        "modality": "art", "eligible_champion": True, "n_folds_covered": 5,
        "question_coverage": 1.0, "dataset_sha256": "train", "fold_assignment_sha256": "folds",
        "k_in": 50, "seed_variant": "both", "alpha": 0.5, "method": "PPR-sweep-k50-both-a0.5",
    }
    condition = {
        "graph_version": "G1", "target": "art",
        "selected_configuration": {"k_in": 50, "seed_variant": "both", "alpha": 0.5, "method": "PPR-sweep-k50-both-a0.5"},
    }
    payload = {"datasets": {"train": {"sha256": "train"}}, "folds": {"sha256": "folds"}}

    runner._validate_champion(champion, condition, payload, source)
