#!/usr/bin/env python3
"""Run one isolated E017 LightGCN CV or replay task."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).expanduser().resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DATA_ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"
DEFAULT_CONFIG = ROOT / "configs/e017_intergraph_graded_jp_cluster.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_task(path: Path, index: int) -> dict[str, Any]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if index < 0 or index >= len(rows):
        raise IndexError(f"task index {index} outside 0..{len(rows) - 1}")
    task = rows[index]
    if int(task["task_id"]) != index:
        raise ValueError(f"task identity mismatch at index {index}")
    return task


def task_roots(task: dict[str, Any], config: dict[str, Any], repo: Path) -> tuple[Path, Path]:
    suffix = Path(str(task["graph_id"])) / f"seed_{int(task['seed'])}"
    return (
        repo / config["outputs"]["cv_root"] / suffix,
        repo / config["outputs"]["final_root"] / suffix,
    )


def build_cv_command(
    task: dict[str, Any],
    config: dict[str, Any],
    *,
    repo: Path = REPO,
    python: str = sys.executable,
) -> list[str]:
    cv_root, _final_root = task_roots(task, config, repo)
    train_bench = (
        repo
        / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
        / "train_augmented_retrievable_strict"
    )
    command = [
        python,
        str(repo / "05-Technique/benchmark/etape1_embedding_pur/scripts/44_run_cv_lightgcn.py"),
        "--graph-version",
        str(task["graph_id"]),
        "--bench-dir",
        str(train_bench),
        "--out-dir",
        str(cv_root / "lightgcn"),
        "--train-k",
        str(int(task["train_k"])),
        "--seed",
        str(int(task["seed"])),
        "--lr",
        str(float(task["learning_rate"])),
        "--epochs",
        str(int(task["epochs"])),
        "--lambda-anchor",
        str(float(task["lambda_anchor"])),
        "--negative-sampling-strategy",
        str(task["negative_sampling_strategy"]),
    ]
    for target in task["selection_targets"]:
        command.extend(["--selection-target", str(target)])
    return command


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_cv(task: dict[str, Any], config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    cv_root, _final_root = task_roots(task, config, REPO)
    status_path = cv_root / "status.json"
    command = build_cv_command(task, config)
    if dry_run:
        return {"stage": "cv", "task": task, "command": command, "dry_run": True}
    started_at = utc_now()
    _write_status(
        status_path,
        {"stage": "cv", "status": "running", "task": task, "started_at": started_at},
    )
    try:
        subprocess.run(command, check=True, cwd=REPO)
        champions_path = cv_root / "lightgcn/champions.json"
        champions = json.loads(champions_path.read_text(encoding="utf-8"))
        if set(champions) != {"art", "jp"}:
            raise ValueError(f"incomplete CV champions: {sorted(champions)}")
        if any("replay_epochs" not in row for row in champions.values()):
            raise ValueError("CV champion is missing its validation-selected replay epoch")
        result = {
            "stage": "cv",
            "status": "complete",
            "task": task,
            "started_at": started_at,
            "finished_at": utc_now(),
            "champions_path": str(champions_path),
        }
    except Exception as exc:
        _write_status(
            status_path,
            {
                "stage": "cv",
                "status": "failed",
                "task": task,
                "started_at": started_at,
                "finished_at": utc_now(),
                "error": repr(exc),
            },
        )
        raise
    _write_status(status_path, result)
    return result


def run_replay(task: dict[str, Any], config: dict[str, Any], *, dry_run: bool) -> dict[str, Any]:
    cv_root, final_root = task_roots(task, config, REPO)
    champions_path = cv_root / "lightgcn/champions.json"
    if dry_run:
        return {
            "stage": "replay",
            "task": task,
            "champions_path": str(champions_path),
            "out_dir": str(final_root),
            "dry_run": True,
        }
    champions = json.loads(champions_path.read_text(encoding="utf-8"))
    final = _load_module(SCRIPTS / "45_run_final_champions.py", "e017_final_replay")
    bench_root = DATA_ROOT / "data/doctrine_v3plus_bench"
    train_bench = bench_root / "train_augmented_retrievable_strict"
    eval_bench = bench_root / "eval_rich_retrievable_strict"
    final_root.mkdir(parents=True, exist_ok=True)
    status_path = final_root / "status.json"
    started_at = utc_now()
    _write_status(
        status_path,
        {"stage": "replay", "status": "running", "task": task, "started_at": started_at},
    )
    try:
        eval_df, history_df, rankings_df = final.replay_lightgcn(
            train_bench,
            eval_bench,
            str(task["graph_id"]),
            champions,
            top_k_out=10,
        )
        jp_rankings = rankings_df.loc[
            (rankings_df["modality"].astype(str) == "jp")
            & (rankings_df["selection_target"].astype(str) == "jp")
            & (rankings_df["rank"].astype(int) <= 10)
        ]
        if jp_rankings["qid"].nunique() != 754 or len(jp_rankings) != 7540:
            raise ValueError(
                "replay JP coverage mismatch: "
                f"questions={jp_rankings['qid'].nunique()} positions={len(jp_rankings)}"
            )
        eval_df.to_csv(final_root / "lightgcn_eval.csv", index=False)
        if not history_df.empty:
            history_df.to_csv(final_root / "lightgcn_history.csv", index=False)
        rankings_df.to_parquet(final_root / "rankings.parquet", index=False)
        (final_root / "champions.json").write_text(
            json.dumps(champions, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result = {
            "stage": "replay",
            "status": "complete",
            "task": task,
            "started_at": started_at,
            "finished_at": utc_now(),
            "ranking_questions": int(jp_rankings["qid"].nunique()),
            "ranking_positions": int(len(jp_rankings)),
        }
    except Exception as exc:
        _write_status(
            status_path,
            {
                "stage": "replay",
                "status": "failed",
                "task": task,
                "started_at": started_at,
                "finished_at": utc_now(),
                "error": repr(exc),
            },
        )
        raise
    _write_status(status_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cv", "replay"), required=True)
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    tasks_path = args.tasks or (
        REPO / config["outputs"]["campaign_root"] / "tasks.jsonl"
    )
    task = load_task(tasks_path, args.task_index)
    result = (
        run_cv(task, config, dry_run=args.dry_run)
        if args.stage == "cv"
        else run_replay(task, config, dry_run=args.dry_run)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
