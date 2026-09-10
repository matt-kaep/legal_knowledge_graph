#!/usr/bin/env python3
"""Aggregate one complete, sealed A3 LightGCN CV graph without inter-graph selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _code_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else CODE_REPO / path


def _data_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_REPO / path


def _load_r2_runner():
    path = SCRIPTS / "97_run_b1_a3_r2_lightgcn_tasks.py"
    spec = importlib.util.spec_from_file_location("b1_r2_lightgcn_tasks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_manifest(payload: dict[str, Any]) -> None:
    """Require a complete one-graph CV scope before any aggregation is visible."""
    scope = payload.get("scope", {})
    graph = str(scope.get("graph_version", ""))
    if not graph:
        raise ValueError("scoped LightGCN manifest needs scope.graph_version")
    atomic = scope.get("atomic_cv", {})
    if int(atomic.get("task_count", -1)) != 120:
        raise ValueError("scoped LightGCN task_count must be 120")
    if int(atomic.get("fold_count", -1)) != 5:
        raise ValueError("scoped LightGCN fold_count must be 5")
    if set(atomic.get("selection_targets", [])) != {"art", "jp"}:
        raise ValueError("scoped LightGCN targets must be art and jp")
    selection = payload.get("selection", {})
    if selection.get("scope") != "train_cv_only" or selection.get("freeze_before_evaluation") is not True:
        raise ValueError("scoped LightGCN selection must be frozen train/CV only")
    replay = payload.get("replay", {})
    if replay.get("final_seeds") != [42, 43, 44] or int(replay.get("top_k_out", -1)) != 100:
        raise ValueError("scoped LightGCN replay must use three seeds and top-100")


def select_scoped_tasks(payload: dict[str, Any], task_plan: dict[str, Any]) -> list[dict[str, Any]]:
    validate_manifest(payload)
    tasks = task_plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("parent task plan is missing tasks")
    graph = payload["scope"]["graph_version"]
    selected = [task for task in tasks if task.get("graph_version") == graph]
    expected = int(payload["scope"]["atomic_cv"]["task_count"])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} tasks for graph={graph}, found {len(selected)}")
    ids = [str(task.get("task_id", "")) for task in selected]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("scoped task plan has missing or duplicate task IDs")
    if {task.get("graph_version") for task in selected} != {graph}:
        raise ValueError("scoped task selection contains a foreign graph")
    return selected


def aggregate(manifest_path: Path) -> Path:
    """Verify the parent receipts and materialize a new, append-only scoped receipt."""
    payload = _load_json(manifest_path)
    validate_manifest(payload)
    parent = payload["parent_b1_r2"]
    parent_manifest = _code_path(parent["manifest_path"])
    if _sha256(parent_manifest) != parent["manifest_sha256"]:
        raise ValueError("parent B1-r2 manifest hash mismatch")
    plan_path = _data_path(parent["task_plan_path"])
    if _sha256(plan_path) != parent["task_plan_sha256"]:
        raise ValueError("parent B1-r2 task-plan hash mismatch")
    preflight_path = _data_path(parent["preflight_path"])
    if _sha256(preflight_path) != parent["preflight_sha256"]:
        raise ValueError("parent B1-r2 preflight hash mismatch")
    preflight = _load_json(preflight_path)
    if preflight.get("ok") is not True or preflight.get("manifest_sha256") != parent["manifest_sha256"]:
        raise ValueError("parent B1-r2 preflight is not a successful matching receipt")
    plan = _load_json(plan_path)
    if plan.get("campaign_manifest_sha256") != parent["manifest_sha256"]:
        raise ValueError("parent task plan belongs to another campaign")
    tasks = select_scoped_tasks(payload, plan)

    out_root = _data_path(payload["outputs"]["root"])
    out_dir = _data_path(payload["outputs"]["lightgcn_cv"]) / payload["scope"]["graph_version"]
    receipt_path = out_root / "scoped_cv_aggregation.json"
    if out_root.exists() or out_dir.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite scoped LightGCN evidence")
    out_root.mkdir(parents=True, exist_ok=False)

    runner = _load_r2_runner()
    parent_payload = runner.load_campaign(parent_manifest)
    aggregation = runner._aggregate_graph(
        parent_payload,
        graph_version=payload["scope"]["graph_version"],
        tasks=tasks,
        task_root=_data_path(parent["task_root"]),
        task_plan_path=plan_path,
        preflight_path=preflight_path,
        out_dir=out_dir,
    )
    receipt = {
        "schema_version": "b1-a3-scoped-lightgcn-aggregation.v1",
        "status": "complete",
        "campaign_id": payload["campaign_id"],
        "campaign_manifest_sha256": _sha256(manifest_path),
        "scope": payload["scope"],
        "parent_b1_r2": {
            "manifest_sha256": parent["manifest_sha256"],
            "task_plan_sha256": parent["task_plan_sha256"],
            "preflight_sha256": parent["preflight_sha256"],
        },
        "aggregation": aggregation,
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(aggregate(_code_path(str(args.manifest))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
