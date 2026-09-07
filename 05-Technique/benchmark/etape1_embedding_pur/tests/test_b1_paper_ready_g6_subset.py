import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "99_aggregate_b1_a3_paper_ready_subset.py"
MANIFEST = ROOT / "configs" / "confirmatory_campaign_b1_a3_paper_ready_g6.json"


def _load_runner():
    spec = importlib.util.spec_from_file_location("b1_paper_ready_subset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manifest():
    return {
        "campaign_id": "b1-paper-ready-g6-a3",
        "scope": {
            "graph_version": "G6-citation-AA-knn5",
            "atomic_cv": {"task_count": 120, "fold_count": 5, "selection_targets": ["art", "jp"]},
        },
        "selection": {"scope": "train_cv_only", "freeze_before_evaluation": True},
        "replay": {"final_seeds": [42, 43, 44], "top_k_out": 100},
    }


def test_paper_ready_subset_requires_exactly_one_complete_g6_scope():
    runner = _load_runner()

    runner.validate_paper_ready_manifest(_manifest())

    wrong_graph = _manifest()
    wrong_graph["scope"]["graph_version"] = "G7-citation-AA-cit1-sem025-knn5"
    with pytest.raises(ValueError, match="G6-citation-AA-knn5"):
        runner.validate_paper_ready_manifest(wrong_graph)

    incomplete = _manifest()
    incomplete["scope"]["atomic_cv"]["task_count"] = 119
    with pytest.raises(ValueError, match="task_count"):
        runner.validate_paper_ready_manifest(incomplete)


def test_paper_ready_manifest_seals_the_existing_b1_r2_evidence_and_new_subset_runner():
    assert MANIFEST.is_file()
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert payload["parent_b1_r2"]["manifest_sha256"] == (
        "9427430f436ca5fc1e2d2bdc9858880c7ed45daaaf48d6b88ca170df614e440c"
    )
    assert payload["parent_b1_r2"]["task_plan_sha256"] == (
        "37fce90ff05643b54502ae5b40b22154792d94bb691880401de6ec0e89de5c4d"
    )
    assert payload["parent_b1_r2"]["preflight_sha256"] == (
        "9d816a7ec6c9af83d7e66a2c7a403257e990e388aecd692181b57a72240d3f0a"
    )
    assert payload["scope"]["graph_version"] == "G6-citation-AA-knn5"
    assert payload["replay"]["final_seeds"] == [42, 43, 44]
    assert payload["code_bundle"]["subset_aggregator"]["sha256"] == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()


def test_subset_selection_refuses_missing_or_foreign_g6_tasks():
    runner = _load_runner()
    manifest = _manifest()
    tasks = [
        {"task_id": "g1", "graph_version": "G1"},
        *[
            {"task_id": f"g6-{index}", "graph_version": "G6-citation-AA-knn5"}
            for index in range(120)
        ],
    ]

    selected = runner.select_scoped_tasks(manifest, {"tasks": tasks})
    assert len(selected) == 120
    assert {task["graph_version"] for task in selected} == {"G6-citation-AA-knn5"}

    with pytest.raises(ValueError, match="expected 120"):
        runner.select_scoped_tasks(manifest, {"tasks": tasks[:-1]})


def test_aggregation_writes_its_receipt_after_the_isolated_output_is_created(tmp_path, monkeypatch):
    runner = _load_runner()
    parent_manifest = tmp_path / "b1-r2.json"
    parent_manifest.write_text("{}", encoding="utf-8")
    parent_sha = hashlib.sha256(parent_manifest.read_bytes()).hexdigest()
    tasks = [
        {"task_id": f"g6-{index}", "graph_version": "G6-citation-AA-knn5"}
        for index in range(120)
    ]
    task_plan = tmp_path / "task_plan.json"
    task_plan.write_text(json.dumps({"campaign_manifest_sha256": parent_sha, "tasks": tasks}), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"ok": True, "manifest_sha256": parent_sha}), encoding="utf-8")
    manifest = {
        **_manifest(),
        "parent_b1_r2": {
            "manifest_path": str(parent_manifest),
            "manifest_sha256": parent_sha,
            "task_plan_path": str(task_plan),
            "task_plan_sha256": hashlib.sha256(task_plan.read_bytes()).hexdigest(),
            "preflight_path": str(preflight),
            "preflight_sha256": hashlib.sha256(preflight.read_bytes()).hexdigest(),
            "task_root": str(tmp_path / "tasks"),
        },
        "outputs": {"root": str(tmp_path / "paper-ready")},
    }
    manifest_path = tmp_path / "paper-ready.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    class FakeParentRunner:
        @staticmethod
        def load_campaign(_path):
            return {"campaign_id": "b1-r2"}

        @staticmethod
        def _aggregate_graph(_payload, **kwargs):
            kwargs["out_dir"].mkdir(parents=True)
            return {"task_receipt_sha256": {task["task_id"]: "receipt" for task in kwargs["tasks"]}}

    monkeypatch.setattr(runner, "_load_r2_runner", lambda: FakeParentRunner)

    receipt_path = runner.aggregate_g6(manifest_path)

    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["scope"]["graph_version"] == "G6-citation-AA-knn5"
    assert len(receipt["task_receipt_sha256"]) == 120
