#!/usr/bin/env python3
"""Successor replay for graph-scoped A3 PPR champions without reselection."""

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
    spec = importlib.util.spec_from_file_location("scoped_ppr_replay_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    merged = dict(parent)
    for key, child_value in child.items():
        parent_value = merged.get(key)
        if isinstance(parent_value, dict) and isinstance(child_value, dict):
            merged[key] = _deep_merge(parent_value, child_value)
        else:
            merged[key] = child_value
    return merged


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    parent = payload.get("parent_manifest")
    if not parent:
        return payload
    parent_path = _code_path(parent["path"])
    if _sha256(parent_path) != parent["sha256"]:
        raise ValueError("parent scoped PPR manifest hash mismatch")
    return _deep_merge(_load_manifest(parent_path), payload)


def _validate_champion(
    champion: dict[str, Any], condition: dict[str, Any], payload: dict[str, Any], source: Path
) -> None:
    """Accept sealed per-graph sources that do not redundantly store graph_version."""
    target = condition["target"]
    if source.name != "champions.json" or source.parent.name != condition["graph_version"]:
        raise ValueError("PPR champions source path does not match its sealed graph")
    recorded_graph = champion.get("graph_version")
    if recorded_graph is not None and recorded_graph != condition["graph_version"]:
        raise ValueError("PPR champion graph does not match its sealed condition")
    if champion.get("modality") != target or champion.get("eligible_champion") is not True:
        raise ValueError("PPR champion is not eligible for its sealed target")
    if int(champion.get("n_folds_covered", -1)) != 5 or float(champion.get("question_coverage", -1)) != 1.0:
        raise ValueError("PPR champion lacks complete train/CV coverage")
    if champion.get("dataset_sha256") != payload["datasets"]["train"]["sha256"]:
        raise ValueError("PPR champion belongs to another training snapshot")
    if champion.get("fold_assignment_sha256") != payload["folds"]["sha256"]:
        raise ValueError("PPR champion belongs to another fold assignment")
    for field in ("k_in", "seed_variant", "alpha", "method"):
        if champion.get(field) != condition["selected_configuration"].get(field):
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
    payload = _load_manifest(manifest_path)
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
    for condition in payload["conditions"]:
        source = _data_path(condition["champions_path"])
        if _sha256(source) != condition["champions_sha256"]:
            raise ValueError(f"PPR champions hash mismatch for {condition['id']}")
        champions = json.loads(source.read_text(encoding="utf-8"))
        champion = champions.get(condition["target"])
        if not isinstance(champion, dict):
            raise ValueError(f"PPR champion missing target={condition['target']}")
        _validate_champion(champion, condition, payload, source)
        started = time.perf_counter()
        metrics, rankings = helper.replay_ppr(
            eval_dir, {condition["target"]: champion}, condition["graph_version"], top_k_out=100
        )
        elapsed = time.perf_counter() - started
        for frame in (metrics, rankings):
            frame["condition_id"] = condition["id"]
            frame["selected_graph_version"] = condition["graph_version"]
            frame["selected_target"] = condition["target"]
            frame["champions_sha256"] = condition["champions_sha256"]
        metrics_frames.append(metrics)
        rankings_frames.append(rankings)
        timings.append({"condition_id": condition["id"], "target": condition["target"], "graph_version": condition["graph_version"], "offline_and_retrieval_seconds": elapsed, "n_questions": int(payload["datasets"]["evaluation"]["questions"]), "mean_seconds_per_question": elapsed / int(payload["datasets"]["evaluation"]["questions"]), "median_seconds_per_question": elapsed / int(payload["datasets"]["evaluation"]["questions"])})
    metrics_path = out_dir / "ppr_eval_per_question.csv"
    rankings_path = out_dir / "ppr_rankings_top100.parquet"
    timings_path = out_dir / "ppr_timings.csv"
    pd.concat(metrics_frames, ignore_index=True).to_csv(metrics_path, index=False)
    pd.concat(rankings_frames, ignore_index=True).to_parquet(rankings_path, index=False)
    pd.DataFrame(timings).to_csv(timings_path, index=False)
    receipt = {"schema_version": "b1-a3-scoped-ppr-replay.v2", "status": "complete", "campaign_id": payload["campaign_id"], "campaign_manifest_sha256": _sha256(manifest_path), "conditions": [{key: condition[key] for key in ("id", "graph_version", "target", "champions_sha256")} for condition in payload["conditions"]], "files": {path.name: _sha256(path) for path in (metrics_path, rankings_path, timings_path)}, "script_sha256": _sha256(Path(__file__).resolve())}
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
