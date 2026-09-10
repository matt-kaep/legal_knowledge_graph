"""Run the sealed B1-r2 LightGCN CV grid as resumable CUDA tasks.

Each Slurm array element owns exactly one (graph, fold, configuration,
selection target) combination.  A task becomes visible only after its CSVs,
input receipt and SHA-256 receipt have been atomically published.  Aggregation
refuses incomplete task sets, so it cannot select a champion from partial CV.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3_r2.json"


def _load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cv_lightgcn = _load_module("44_run_cv_lightgcn.py", "b1_r2_cv_lightgcn")
b1_runner = _load_module("94_run_b1_a3_campaign.py", "b1_r2_campaign_runner")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _data_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_REPO / path


def _code_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else CODE_REPO / path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, child_value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(child_value, dict):
            merged[key] = _deep_merge(parent_value, child_value)
        else:
            merged[key] = child_value
    return merged


def load_campaign(manifest_path: Path) -> dict[str, Any]:
    """Load an immutable B1-r2 manifest layered over its hash-checked parent."""
    payload = _load_json(manifest_path)
    parent_ref = payload.get("parent_campaign")
    if parent_ref is None:
        return payload
    parent_path = _code_path(parent_ref["manifest_path"])
    if _sha256(parent_path) != parent_ref.get("sha256"):
        raise ValueError(f"parent manifest hash mismatch: {parent_path}")
    return _deep_merge(load_campaign(parent_path), payload)


def build_tasks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the complete, stable B1-r2 CUDA CV task grid."""
    grid = payload["parameters"]["lightgcn"]
    return cv_lightgcn.build_atomic_task_specs(
        graph_versions=[str(graph) for graph in payload["graphs"]],
        folds=list(range(int(payload["folds"]["count"]))),
        train_ks=[int(value) for value in grid["train_k"]],
        seeds=[int(value) for value in grid["cv_seeds"]],
        lrs=[float(value) for value in grid["learning_rate"]],
        epochs_list=[int(grid["epochs"])],
        lambda_anchors=[float(value) for value in grid["lambda_anchor"]],
        negative_sampling_strategies=[str(value) for value in grid["negative_sampling_strategy"]],
        selection_targets=["art", "jp"],
    )


def _task_plan_payload(manifest_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    tasks = build_tasks(payload)
    return {
        "schema_version": "b1-a3-r2-lightgcn-atomic-task-plan.v1",
        "campaign_id": payload["campaign_id"],
        "campaign_manifest_path": str(manifest_path),
        "campaign_manifest_sha256": _sha256(manifest_path),
        "execution_device": "cuda",
        "task_count": len(tasks),
        "tasks": tasks,
    }


def write_task_plan(manifest_path: Path, payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Write the immutable task plan once, or verify an existing identical plan."""
    expected = _task_plan_payload(manifest_path, payload)
    if path.exists():
        current = _load_json(path)
        if current != expected:
            raise ValueError(f"existing task plan differs from B1-r2 manifest: {path}")
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return expected


def write_preflight(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    task_plan_path: Path,
    out_path: Path,
) -> dict[str, Any]:
    """Run B1 input/code verification once before dispatching array tasks."""
    task_plan = write_task_plan(manifest_path, payload, task_plan_path)
    report = b1_runner.preflight(payload, manifest_path=manifest_path)
    report.update(
        {
            "lightgcn_execution": "atomic_cuda",
            "task_plan_path": str(task_plan_path),
            "task_plan_sha256": _sha256(task_plan_path),
            "task_count": int(task_plan["task_count"]),
            "created_at": _utc_now(),
        }
    )
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite B1-r2 preflight evidence: {out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _require_preflight(
    manifest_path: Path,
    task_plan_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    report = _load_json(preflight_path)
    if report.get("ok") is not True:
        raise ValueError("B1-r2 CUDA task requires a successful preflight report")
    if report.get("manifest_sha256") != _sha256(manifest_path):
        raise ValueError("B1-r2 preflight report belongs to another campaign manifest")
    if report.get("task_plan_sha256") != _sha256(task_plan_path):
        raise ValueError("B1-r2 preflight report belongs to another atomic task plan")
    if report.get("lightgcn_execution") != "atomic_cuda":
        raise ValueError("B1-r2 preflight report does not authorize atomic CUDA LightGCN")
    return report


def _task_dir(task_root: Path, task: dict[str, Any]) -> Path:
    return task_root / str(task["task_id"])


def _receipt_path(task_root: Path, task: dict[str, Any]) -> Path:
    return _task_dir(task_root, task) / "receipt.json"


def _validate_task_receipt(task_root: Path, task: dict[str, Any]) -> dict[str, Any]:
    receipt_path = _receipt_path(task_root, task)
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing atomic LightGCN receipt: {receipt_path}")
    receipt = _load_json(receipt_path)
    if receipt.get("status") != "complete" or receipt.get("task") != task:
        raise ValueError(f"invalid atomic LightGCN receipt: {receipt_path}")
    task_dir = receipt_path.parent
    for relative, expected_hash in receipt.get("files", {}).items():
        path = task_dir / relative
        if not path.is_file() or _sha256(path) != expected_hash:
            raise ValueError(f"atomic LightGCN task file hash mismatch: {path}")
    return receipt


def _fold_inputs(payload: dict[str, Any]) -> tuple[Path, pd.DataFrame, set[str]]:
    bench_dir = _data_path(payload["datasets"]["train"]["directory"])
    questions = cv_lightgcn.graph_protocol.load_bench_questions(bench_dir)
    qids = {str(question["qid"]) for question in questions}
    folds, _ = cv_lightgcn.load_fold_assignments(
        bench_dir,
        qids,
        split=str(payload["datasets"]["train"]["split"]),
        protocol_version=str(payload["protocol_version"]),
    )
    return bench_dir, folds, qids


def _decorate_task_outputs(
    raw: pd.DataFrame,
    history: pd.DataFrame,
    task: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    decorated_raw = raw.copy()
    decorated_raw.insert(0, "fold", int(task["fold"]))
    for key in (
        "train_k",
        "seed",
        "lr",
        "epochs",
        "lambda_anchor",
        "negative_sampling_strategy",
        "graph_version",
        "selection_target",
    ):
        decorated_raw[key] = task[key]
    decorated_history = history.copy()
    if not decorated_history.empty:
        decorated_history.insert(0, "fold", int(task["fold"]))
        for key in (
            "train_k",
            "seed",
            "lr",
            "epochs",
            "lambda_anchor",
            "negative_sampling_strategy",
            "selection_target",
        ):
            decorated_history[key] = task[key]
    return decorated_raw, decorated_history


def run_task(
    payload: dict[str, Any],
    *,
    task: dict[str, Any],
    task_root: Path,
    task_plan_path: Path,
    preflight_path: Path,
) -> dict[str, Any]:
    """Execute one sealed CV task and atomically publish its checked receipt."""
    manifest_path = _code_path(payload["_manifest_path"])
    _require_preflight(manifest_path, task_plan_path, preflight_path)
    final_dir = _task_dir(task_root, task)
    if final_dir.exists():
        return _validate_task_receipt(task_root, task)

    bench_dir, folds, bench_qids = _fold_inputs(payload)
    val_qids = set(folds.loc[folds["fold"].astype(int).eq(int(task["fold"])), "qid"].astype(str))
    train_qids = bench_qids - val_qids
    if not val_qids or not train_qids:
        raise ValueError(f"invalid empty fold split for task={task['task_id']}")

    task_root.mkdir(parents=True, exist_ok=True)
    pending_dir = Path(tempfile.mkdtemp(prefix=f".{task['task_id']}.pending-", dir=task_root))
    try:
        train_dir = pending_dir / "train"
        val_dir = pending_dir / "val"
        cv_lightgcn.build_subset_bench(bench_dir, train_qids, train_dir)
        cv_lightgcn.build_subset_bench(bench_dir, val_qids, val_dir)
        raw, history = cv_lightgcn.run_lightgcn_config(
            train_dir,
            val_dir,
            graph_version=str(task["graph_version"]),
            fold=int(task["fold"]),
            train_k=int(task["train_k"]),
            seed=int(task["seed"]),
            lr=float(task["lr"]),
            epochs=int(task["epochs"]),
            lambda_anchor=float(task["lambda_anchor"]),
            negative_sampling_strategy=str(task["negative_sampling_strategy"]),
            selection_target=str(task["selection_target"]),
            include_baselines=False,
            train_variant=True,
            device="cuda",
        )
        if history is None or history.empty:
            raise ValueError(f"missing epoch history for task={task['task_id']}")
        raw, history = _decorate_task_outputs(raw, history, task)
        raw_path = pending_dir / "raw.csv"
        history_path = pending_dir / "history.csv"
        raw.to_csv(raw_path, index=False)
        history.to_csv(history_path, index=False)
        source_inputs = next(val_dir.glob("lightgcn_inputs_*.json"), None)
        if source_inputs is None:
            raise FileNotFoundError(f"missing LightGCN runtime inputs for task={task['task_id']}")
        inputs_path = pending_dir / "lightgcn_inputs.json"
        shutil.copy2(source_inputs, inputs_path)
        receipt = {
            "schema_version": "b1-a3-r2-lightgcn-atomic-task-receipt.v1",
            "status": "complete",
            "task_id": task["task_id"],
            "task": task,
            "task_plan_sha256": _sha256(task_plan_path),
            "preflight_sha256": _sha256(preflight_path),
            "files": {
                "raw.csv": _sha256(raw_path),
                "history.csv": _sha256(history_path),
                "lightgcn_inputs.json": _sha256(inputs_path),
            },
            "rows": int(len(raw)),
            "history_rows": int(len(history)),
            "completed_at": _utc_now(),
        }
        (pending_dir / "receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        pending_dir.replace(final_dir)
        return receipt
    except Exception:
        shutil.rmtree(pending_dir, ignore_errors=True)
        raise


def _aggregate_graph(
    payload: dict[str, Any],
    *,
    graph_version: str,
    tasks: list[dict[str, Any]],
    task_root: Path,
    task_plan_path: Path,
    preflight_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    receipts = [_validate_task_receipt(task_root, task) for task in tasks]
    cv_lightgcn.validate_atomic_task_receipts(tasks, receipts)
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite aggregated LightGCN CV output: {out_dir}")
    bench_dir, folds, bench_qids = _fold_inputs(payload)
    expected_qids_by_fold = cv_lightgcn.graph_protocol.expected_qids_by_fold(folds)
    raw_df = pd.concat(
        [pd.read_csv(_task_dir(task_root, task) / "raw.csv") for task in tasks],
        ignore_index=True,
    )
    history_df = pd.concat(
        [pd.read_csv(_task_dir(task_root, task) / "history.csv") for task in tasks],
        ignore_index=True,
    )
    outputs = [
        cv_lightgcn.summarize_cv_outputs(
            raw_df,
            "art",
            n_questions_benchmark=len(bench_qids),
            expected_qids_by_fold=expected_qids_by_fold,
        ),
        cv_lightgcn.summarize_cv_outputs(
            raw_df,
            "jp",
            n_questions_benchmark=len(bench_qids),
            expected_qids_by_fold=expected_qids_by_fold,
        ),
    ]
    fold_metrics_df = pd.concat(
        [fold_metrics for fold_metrics, _ in outputs if not fold_metrics.empty], ignore_index=True
    )
    summary_df = pd.concat(
        [summary for _, summary in outputs if not summary.empty], ignore_index=True
    )
    champions = {
        modality: cv_lightgcn.select_champion(
            summary_df.loc[summary_df["modality"].eq(modality)], modality
        )
        for modality in ("art", "jp")
    }
    _, fold_metadata = cv_lightgcn.load_fold_assignments(
        bench_dir,
        bench_qids,
        split=str(payload["datasets"]["train"]["split"]),
        protocol_version=str(payload["protocol_version"]),
    )
    run_metadata = {
        key: fold_metadata[key]
        for key in ("protocol_version", "dataset_sha256", "fold_assignment_sha256")
    }
    run_metadata.update(
        {
            "campaign_id": payload["campaign_id"],
            "graph_version": graph_version,
            "execution_device": "cuda",
            "atomic_task_plan_sha256": _sha256(task_plan_path),
            "atomic_preflight_sha256": _sha256(preflight_path),
            "atomic_tasks_expected": len(tasks),
            "atomic_tasks_complete": len(receipts),
        }
    )
    for key, value in run_metadata.items():
        summary_df[key] = value
    champions = cv_lightgcn.attach_replay_epochs(champions, history_df)
    for champion in champions.values():
        champion.update(run_metadata)

    out_dir.mkdir(parents=True, exist_ok=False)
    raw_df.to_csv(out_dir / "raw.csv", index=False)
    fold_metrics_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    history_df.to_csv(out_dir / "lightgcn_history_all.csv", index=False)
    (out_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "champions.json").write_text(
        json.dumps(champions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    aggregation = {
        "schema_version": "b1-a3-r2-lightgcn-atomic-aggregation.v1",
        "status": "complete",
        "graph_version": graph_version,
        "task_count": len(tasks),
        "task_receipt_sha256": {
            str(task["task_id"]): _sha256(_receipt_path(task_root, task)) for task in tasks
        },
        "files": {
            name: _sha256(out_dir / name)
            for name in (
                "raw.csv",
                "fold_metrics.csv",
                "summary.csv",
                "lightgcn_history_all.csv",
                "run_metadata.json",
                "champions.json",
            )
        },
    }
    (out_dir / "atomic_aggregation.json").write_text(
        json.dumps(aggregation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return aggregation


def aggregate_all(
    manifest_path: Path,
    payload: dict[str, Any],
    *,
    task_plan_path: Path,
    preflight_path: Path,
    task_root: Path,
    out_root: Path,
) -> list[dict[str, Any]]:
    _require_preflight(manifest_path, task_plan_path, preflight_path)
    plan = _load_json(task_plan_path)
    expected_plan = _task_plan_payload(manifest_path, payload)
    if plan != expected_plan:
        raise ValueError("B1-r2 task plan no longer matches its campaign manifest")
    tasks = plan["tasks"]
    all_receipts = [_validate_task_receipt(task_root, task) for task in tasks]
    cv_lightgcn.validate_atomic_task_receipts(tasks, all_receipts)
    results = []
    for graph_version in payload["graphs"]:
        graph_tasks = [task for task in tasks if task["graph_version"] == graph_version]
        results.append(
            _aggregate_graph(
                payload,
                graph_version=str(graph_version),
                tasks=graph_tasks,
                task_root=task_root,
                task_plan_path=task_plan_path,
                preflight_path=preflight_path,
                out_dir=out_root / str(graph_version),
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--write-preflight", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args(argv)
    manifest_path = _code_path(args.manifest)
    payload = load_campaign(manifest_path)
    payload["_manifest_path"] = str(manifest_path)
    out_root = _data_path(payload["outputs"]["lightgcn_cv"])

    if args.write_preflight:
        write_preflight(
            manifest_path,
            payload,
            task_plan_path=args.task_plan,
            out_path=args.preflight,
        )
        return 0
    if args.aggregate:
        aggregate_all(
            manifest_path,
            payload,
            task_plan_path=args.task_plan,
            preflight_path=args.preflight,
            task_root=args.task_root,
            out_root=out_root,
        )
        return 0
    if args.task_index is None:
        raise ValueError("provide --task-index, --write-preflight, or --aggregate")
    task_plan = _load_json(args.task_plan)
    tasks = task_plan.get("tasks")
    if not isinstance(tasks, list) or task_plan != _task_plan_payload(manifest_path, payload):
        raise ValueError("B1-r2 atomic task plan is missing or does not match the manifest")
    if args.task_index < 0 or args.task_index >= len(tasks):
        raise ValueError(f"task index outside B1-r2 task grid: {args.task_index}")
    run_task(
        payload,
        task=tasks[args.task_index],
        task_root=args.task_root,
        task_plan_path=args.task_plan,
        preflight_path=args.preflight,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
