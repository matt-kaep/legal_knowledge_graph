#!/usr/bin/env python3
"""Replay explicitly frozen per-graph A3 PPR champions without reselection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else CODE_REPO / path


def _data_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_REPO / path


def _load_replay_helper():
    path = SCRIPTS / "45_run_final_champions.py"
    spec = importlib.util.spec_from_file_location("scoped_ppr_replay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _validate_champion(champion: dict[str, Any], condition: dict[str, Any], payload: dict[str, Any]) -> None:
    target = condition["target"]
    if champion.get("graph_version") != condition["graph_version"]:
        raise ValueError("PPR champion graph does not match its sealed condition")
    if champion.get("modality") != target or champion.get("eligible_champion") is not True:
        raise ValueError("PPR champion is not eligible for its sealed target")
    if int(champion.get("n_folds_covered", -1)) != 5 or float(champion.get("question_coverage", -1)) != 1.0:
        raise ValueError("PPR champion lacks complete train/CV coverage")
    if champion.get("dataset_sha256") != payload["datasets"]["train"]["sha256"]:
        raise ValueError("PPR champion belongs to another training snapshot")
    if champion.get("fold_assignment_sha256") != payload["folds"]["sha256"]:
        raise ValueError("PPR champion belongs to another fold assignment")
    selected = condition["selected_configuration"]
    for field in ("k_in", "seed_variant", "alpha", "method"):
        if champion.get(field) != selected.get(field):
            raise ValueError(f"PPR champion differs from sealed {field}")


def validate_manifest(payload: dict[str, Any]) -> None:
    if payload.get("a3", {}).get("sha256") != "c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3":
        raise ValueError("scoped PPR replay requires A3")
    selection = payload.get("selection", {})
    if selection.get("scope") != "train_cv_only" or selection.get("freeze_before_evaluation") is not True:
        raise ValueError("scoped PPR selections must be frozen from train/CV")
    if int(payload.get("replay", {}).get("top_k_out", -1)) != 100:
        raise ValueError("scoped PPR replay requires top-100 rankings")
    conditions = payload.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ValueError("scoped PPR replay requires at least one condition")
    ids = [str(item.get("id", "")) for item in conditions]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("scoped PPR condition IDs must be unique")


def replay(manifest_path: Path) -> Path:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(payload)
    out_dir = _data_path(payload["outputs"]["root"])
    if out_dir.exists():
        raise FileExistsError("refusing to overwrite scoped PPR replay evidence")
    out_dir.mkdir(parents=True, exist_ok=False)
    helper = _load_replay_helper()
    eval_dir = _data_path(payload["datasets"]["evaluation"]["directory"])
    metrics_frames: list[pd.DataFrame] = []
    rankings_frames: list[pd.DataFrame] = []
    timings: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for condition in payload["conditions"]:
        source = _data_path(condition["champions_path"])
        if _sha256(source) != condition["champions_sha256"]:
            raise ValueError(f"PPR champions hash mismatch for {condition['id']}")
        champions = json.loads(source.read_text(encoding="utf-8"))
        champion = champions.get(condition["target"])
        if not isinstance(champion, dict):
            raise ValueError(f"PPR champion missing target={condition['target']}")
        _validate_champion(champion, condition, payload)
        started = time.perf_counter()
        metrics, rankings = helper.replay_ppr(
            eval_dir,
            {condition["target"]: champion},
            condition["graph_version"],
            top_k_out=100,
        )
        elapsed = time.perf_counter() - started
        for frame in (metrics, rankings):
            frame["condition_id"] = condition["id"]
            frame["selected_graph_version"] = condition["graph_version"]
            frame["selected_target"] = condition["target"]
            frame["champions_sha256"] = condition["champions_sha256"]
        metrics_frames.append(metrics)
        rankings_frames.append(rankings)
        timings.append({
            "condition_id": condition["id"],
            "target": condition["target"],
            "graph_version": condition["graph_version"],
            "offline_and_retrieval_seconds": elapsed,
            "n_questions": int(payload["datasets"]["evaluation"]["questions"]),
            "mean_seconds_per_question": elapsed / int(payload["datasets"]["evaluation"]["questions"]),
            "median_seconds_per_question": elapsed / int(payload["datasets"]["evaluation"]["questions"]),
        })
    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    rankings_df = pd.concat(rankings_frames, ignore_index=True)
    timings_df = pd.DataFrame(timings)
    metrics_path = out_dir / "ppr_eval_per_question.csv"
    rankings_path = out_dir / "ppr_rankings_top100.parquet"
    timings_path = out_dir / "ppr_timings.csv"
    metrics_df.to_csv(metrics_path, index=False)
    rankings_df.to_parquet(rankings_path, index=False)
    timings_df.to_csv(timings_path, index=False)
    receipt = {
        "schema_version": "b1-a3-scoped-ppr-replay.v1",
        "status": "complete",
        "campaign_id": payload["campaign_id"],
        "campaign_manifest_sha256": _sha256(manifest_path),
        "conditions": [{key: condition[key] for key in ("id", "graph_version", "target", "champions_sha256")} for condition in payload["conditions"]],
        "files": {path.name: _sha256(path) for path in (metrics_path, rankings_path, timings_path)},
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    receipt_path = out_dir / "replay_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    print(replay(_code_path(str(args.manifest))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
