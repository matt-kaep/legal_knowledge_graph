"""Replay B1 frozen champions on the internal evaluation, at top-100 only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import pandas as pd


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3.json"


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _manifest_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _data_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_REPO / path


def _load_frozen(payload: dict, family: str, *, manifest_sha256: str) -> dict:
    path = _data_path(payload["outputs"]["root"]) / "frozen" / f"{family}_champions.json"
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("campaign_id") != payload["campaign_id"]:
        raise ValueError("Frozen champion campaign_id mismatch")
    if frozen.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Frozen champion manifest hash mismatch")
    if frozen.get("a3_sha256") != payload["a3"]["sha256"]:
        raise ValueError("Frozen champion A3 hash mismatch")
    if frozen.get("selection_data") != "train_cv_only":
        raise ValueError("Frozen champion was not selected from train/CV only")
    return frozen


def _write_outputs(out_dir: Path, *, family: str, metrics: pd.DataFrame, rankings: pd.DataFrame, timings: list[dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / f"{family}_eval_per_question.csv", index=False)
    rankings.to_parquet(out_dir / f"{family}_rankings_top100.parquet", index=False)
    pd.DataFrame(timings).to_csv(out_dir / f"{family}_timings.csv", index=False)


def replay_ppr(payload: dict, *, manifest_sha256: str) -> Path:
    frozen = _load_frozen(payload, "ppr", manifest_sha256=manifest_sha256)
    replay = _load_script("45_run_final_champions.py", "b1_final_replay")
    eval_dir = _data_path(payload["datasets"]["evaluation"]["directory"])
    frames, ranking_frames, timings = [], [], []
    for target, champion in frozen["champions"].items():
        started = time.perf_counter()
        metrics, rankings = replay.replay_ppr(
            eval_dir,
            {target: champion},
            str(champion["graph_version"]),
            top_k_out=int(payload["parameters"]["ppr"]["top_k_out"]),
        )
        elapsed = time.perf_counter() - started
        metrics["selected_graph_version"] = champion["graph_version"]
        metrics["selected_target"] = target
        rankings["selected_graph_version"] = champion["graph_version"]
        rankings["selected_target"] = target
        frames.append(metrics)
        ranking_frames.append(rankings)
        timings.append({"target": target, "graph_version": champion["graph_version"], "offline_and_retrieval_seconds": elapsed, "n_questions": 754, "mean_seconds_per_question": elapsed / 754.0, "median_seconds_per_question": elapsed / 754.0})
    out_dir = _data_path(payload["outputs"]["ppr_final"])
    _write_outputs(out_dir, family="ppr", metrics=pd.concat(frames, ignore_index=True), rankings=pd.concat(ranking_frames, ignore_index=True), timings=timings)
    return out_dir


def replay_lightgcn(payload: dict, *, manifest_sha256: str) -> Path:
    frozen = _load_frozen(payload, "lightgcn", manifest_sha256=manifest_sha256)
    replay = _load_script("45_run_final_champions.py", "b1_final_lightgcn")
    train_dir = _data_path(payload["datasets"]["train"]["directory"])
    eval_dir = _data_path(payload["datasets"]["evaluation"]["directory"])
    frames, ranking_frames, timings = [], [], []
    for target, frozen_champion in frozen["champions"].items():
        for seed in payload["parameters"]["lightgcn"]["final_seeds"]:
            champion = dict(frozen_champion)
            champion["seed"] = int(seed)
            champion["method"] = (
                f"LightGCN-{champion['variant']}-K{int(champion['train_k'])}-s{int(seed)}"
                f"-lr{float(champion['lr']):g}-e{int(champion['epochs'])}-la{float(champion['lambda_anchor']):g}"
                f"-neg{champion.get('negative_sampling_strategy', 'random')}"
            )
            started = time.perf_counter()
            metrics, _history, rankings = replay.replay_lightgcn(
                train_dir,
                eval_dir,
                str(champion["graph_version"]),
                {target: champion},
                top_k_out=int(payload["parameters"]["lightgcn"]["top_k_out"]),
            )
            elapsed = time.perf_counter() - started
            metrics["selected_graph_version"] = champion["graph_version"]
            metrics["selected_target"] = target
            metrics["replay_seed"] = seed
            rankings["selected_graph_version"] = champion["graph_version"]
            rankings["selected_target"] = target
            rankings["replay_seed"] = seed
            frames.append(metrics)
            ranking_frames.append(rankings)
            timings.append({"target": target, "graph_version": champion["graph_version"], "seed": seed, "offline_and_retrieval_seconds": elapsed, "n_questions": 754, "mean_seconds_per_question": elapsed / 754.0, "median_seconds_per_question": elapsed / 754.0})
    out_dir = _data_path(payload["outputs"]["lightgcn_final"])
    _write_outputs(out_dir, family="lightgcn", metrics=pd.concat(frames, ignore_index=True), rankings=pd.concat(ranking_frames, ignore_index=True), timings=timings)
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--family", choices=("ppr", "lightgcn"), required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    print(
        replay_ppr(payload, manifest_sha256=manifest_sha256)
        if args.family == "ppr"
        else replay_lightgcn(payload, manifest_sha256=manifest_sha256)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
