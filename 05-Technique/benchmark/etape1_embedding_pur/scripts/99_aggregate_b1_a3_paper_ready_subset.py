#!/usr/bin/env python3
"""Aggregate the complete G6 CV subset from immutable B1-r2 atomic receipts."""

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
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3_paper_ready_g6.json"
G6_GRAPH = "G6-citation-AA-knn5"
G6_TASK_COUNT = 120


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


def validate_paper_ready_manifest(payload: dict[str, Any]) -> None:
    """Reject a paper-ready manifest that is not the complete G6-only protocol."""
    scope = payload.get("scope", {})
    if scope.get("graph_version") != G6_GRAPH:
        raise ValueError(f"paper-ready graph_version must be {G6_GRAPH}")
    atomic_cv = scope.get("atomic_cv", {})
    if atomic_cv.get("task_count") != G6_TASK_COUNT:
        raise ValueError(f"paper-ready task_count must be {G6_TASK_COUNT}")
    if atomic_cv.get("fold_count") != 5:
        raise ValueError("paper-ready fold_count must be 5")
    if set(atomic_cv.get("selection_targets", [])) != {"art", "jp"}:
        raise ValueError("paper-ready selection_targets must contain art and jp")
    selection = payload.get("selection", {})
    if selection.get("scope") != "train_cv_only":
        raise ValueError("paper-ready selection must be train_cv_only")
    if selection.get("freeze_before_evaluation") is not True:
        raise ValueError("paper-ready selection must freeze before evaluation")
    replay = payload.get("replay", {})
    if replay.get("final_seeds") != [42, 43, 44]:
        raise ValueError("paper-ready final_seeds must be [42, 43, 44]")
    if replay.get("top_k_out") != 100:
        raise ValueError("paper-ready top_k_out must be 100")


def select_scoped_tasks(payload: dict[str, Any], task_plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Return exactly the sealed G6 task grid, never a partial or foreign subset."""
    validate_paper_ready_manifest(payload)
    tasks = task_plan.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("parent task plan is missing tasks")
    selected = [task for task in tasks if task.get("graph_version") == G6_GRAPH]
    expected = int(payload["scope"]["atomic_cv"]["task_count"])
    if len(selected) != expected:
        raise ValueError(f"expected {expected} G6 tasks, found {len(selected)}")
    task_ids = [str(task.get("task_id", "")) for task in selected]
    if not all(task_ids) or len(set(task_ids)) != len(task_ids):
        raise ValueError("G6 task plan contains missing or duplicate task_id values")
    if {str(task.get("graph_version")) for task in selected} != {G6_GRAPH}:
        raise ValueError("paper-ready task selection contains a foreign graph")
    return selected


def aggregate_g6(manifest_path: Path) -> Path:
    """Verify parent evidence and write a new, non-overlapping G6 aggregation receipt."""
    payload = _load_json(manifest_path)
    validate_paper_ready_manifest(payload)
    parent = payload.get("parent_b1_r2", {})
    parent_manifest = _code_path(str(parent["manifest_path"]))
    if _sha256(parent_manifest) != parent.get("manifest_sha256"):
        raise ValueError("parent B1-r2 manifest hash mismatch")
    task_plan_path = _data_path(str(parent["task_plan_path"]))
    if _sha256(task_plan_path) != parent.get("task_plan_sha256"):
        raise ValueError("parent B1-r2 task-plan hash mismatch")
    preflight_path = _data_path(str(parent["preflight_path"]))
    if _sha256(preflight_path) != parent.get("preflight_sha256"):
        raise ValueError("parent B1-r2 preflight hash mismatch")
    preflight = _load_json(preflight_path)
    if preflight.get("ok") is not True:
        raise ValueError("parent B1-r2 preflight is not successful")
    if preflight.get("manifest_sha256") != parent.get("manifest_sha256"):
        raise ValueError("parent B1-r2 preflight belongs to another manifest")

    task_plan = _load_json(task_plan_path)
    if task_plan.get("campaign_manifest_sha256") != parent.get("manifest_sha256"):
        raise ValueError("parent task plan belongs to another manifest")
    tasks = select_scoped_tasks(payload, task_plan)
    task_root = _data_path(str(parent["task_root"]))
    out_root = _data_path(str(payload["outputs"]["root"]))
    out_dir = out_root / "lightgcn_cv" / G6_GRAPH
    receipt_path = out_root / "g6_subset_aggregation.json"
    if out_root.exists() or out_dir.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite paper-ready G6 aggregation evidence")
    out_root.mkdir(parents=True, exist_ok=False)

    runner = _load_r2_runner()
    parent_payload = runner.load_campaign(parent_manifest)
    aggregation = runner._aggregate_graph(
        parent_payload,
        graph_version=G6_GRAPH,
        tasks=tasks,
        task_root=task_root,
        task_plan_path=task_plan_path,
        preflight_path=preflight_path,
        out_dir=out_dir,
    )
    receipt = {
        "schema_version": "b1-paper-ready-g6-subset-aggregation.v1",
        "status": "complete",
        "campaign_id": payload["campaign_id"],
        "campaign_manifest_sha256": _sha256(manifest_path),
        "parent_b1_r2": {
            "manifest_sha256": parent["manifest_sha256"],
            "task_plan_sha256": parent["task_plan_sha256"],
            "preflight_sha256": parent["preflight_sha256"],
        },
        "scope": payload["scope"],
        "task_receipt_sha256": aggregation["task_receipt_sha256"],
        "aggregation": aggregation,
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    print(aggregate_g6(_code_path(str(args.manifest))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
