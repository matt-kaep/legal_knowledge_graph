#!/usr/bin/env python3
"""Materialize the sealed task and transfer contract for E017."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


CODE_REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).expanduser().resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DEFAULT_CONFIG = ROOT / "configs/e017_intergraph_graded_jp_cluster.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_paths(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str):
            yield path
        for nested in value.values():
            yield from _walk_paths(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_paths(nested)


def build_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one isolated CV/replay task per graph and robustness seed."""
    screen = manifest["lightgcn"]["screen"]
    seeds = manifest["lightgcn"]["robustness"]["seed"]
    train_ks = screen["train_k"]
    learning_rates = screen["learning_rate"]
    lambda_anchors = screen["lambda_anchor"]
    strategies = screen["negative_sampling_strategy"]
    if not (
        train_ks == [2]
        and learning_rates == [0.001]
        and lambda_anchors == [1.0]
        and strategies == ["random"]
    ):
        raise ValueError("E017 requires the frozen LightGCN screen configuration")
    tasks: list[dict[str, Any]] = []
    for graph in manifest["graphs"]:
        for seed in seeds:
            tasks.append(
                {
                    "task_id": len(tasks),
                    "graph_id": graph["graph_id"],
                    "graph_matrix_path": graph["matrix_path"],
                    "graph_matrix_sha256": graph["matrix_sha256"],
                    "seed": int(seed),
                    "train_k": 2,
                    "learning_rate": 0.001,
                    "lambda_anchor": 1.0,
                    "epochs": int(screen["epochs"]),
                    "negative_sampling_strategy": "random",
                    "selection_targets": ["art", "jp"],
                }
            )
    return tasks


def build_transfer_paths(manifest: dict[str, Any]) -> list[str]:
    """Resolve the minimal source-manifest input set, never credentials."""
    paths = set(_walk_paths(manifest.get("datasets", {})))
    paths.update(_walk_paths(manifest.get("folds", {})))
    metadata_path = manifest.get("folds", {}).get("metadata_path")
    if isinstance(metadata_path, str):
        paths.add(metadata_path)
    paths.update(_walk_paths(manifest.get("shared_hybrid_ids", {})))
    paths.update(_walk_paths(manifest.get("immutable_inputs", {})))
    paths.update(_walk_paths(manifest.get("code_bundle", {})))
    paths.update(str(graph["matrix_path"]) for graph in manifest["graphs"])
    hybrid_filenames = (
        "jp_ids.npy",
        "article_ids.npy",
        "node_ids.npy",
        "article_codes.npy",
    )
    for graph in manifest["graphs"]:
        if graph["graph_id"] == "G1":
            continue
        graph_root = (
            Path("05-Technique/benchmark/etape1_embedding_pur/data/hybrid_graphs")
            / str(graph["graph_id"])
        )
        paths.update((graph_root / filename).as_posix() for filename in hybrid_filenames)
    paths.add("05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2.json")
    forbidden = (".env", "password", "credential", "judilibre_corpus")
    normalized: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Transfer path escapes repository: {raw_path}")
        if any(marker in raw_path.lower() for marker in forbidden):
            raise ValueError(f"Forbidden transfer path: {raw_path}")
        normalized.append(path.as_posix())
    return sorted(set(normalized))


def verify_inputs(campaign: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    """Return every missing or changed sealed input beneath ``root``."""
    problems: list[dict[str, Any]] = []
    for relative_path, expected in campaign.get("inputs", {}).items():
        path = root / relative_path
        if not path.is_file():
            problems.append({"path": relative_path, "problem": "missing"})
            continue
        actual_size = path.stat().st_size
        if actual_size != int(expected["size_bytes"]):
            problems.append(
                {
                    "path": relative_path,
                    "problem": "size_mismatch",
                    "expected": int(expected["size_bytes"]),
                    "actual": actual_size,
                }
            )
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            problems.append(
                {
                    "path": relative_path,
                    "problem": "sha256_mismatch",
                    "expected": expected["sha256"],
                    "actual": actual_hash,
                }
            )
    return problems


def materialize(
    source_manifest: dict[str, Any],
    out_dir: Path,
    *,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    tasks = build_tasks(source_manifest)
    transfer_paths = sorted(
        set(build_transfer_paths(source_manifest))
        | {str(path) for path in config.get("extra_transfer_paths", [])}
    )
    missing = [path for path in transfer_paths if not (REPO / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing E017 transfer inputs: {missing}")
    inputs = {
        path: {
            "size_bytes": (REPO / path).stat().st_size,
            "sha256": sha256_file(REPO / path),
        }
        for path in transfer_paths
    }
    campaign = {
        "schema_version": 1,
        "experiment_id": "E017",
        "campaign_id": config["campaign_id"],
        "source_campaign_id": source_manifest["campaign_id"],
        "scientific_status": "exploratory_internal_evaluation",
        "internal_eval_authorized": True,
        "n_graphs": len(source_manifest["graphs"]),
        "n_seeds": len({task["seed"] for task in tasks}),
        "n_tasks": len(tasks),
        "lightgcn": config["lightgcn"],
        "cluster": config["cluster"],
        "outputs": config["outputs"],
        "inputs": inputs,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "campaign_manifest.json").write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out_dir / "tasks.jsonl").open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    (out_dir / "transfer_paths.txt").write_text(
        "\n".join(transfer_paths) + "\n",
        encoding="utf-8",
    )
    return {"out_dir": str(out_dir), "n_tasks": len(tasks), "n_inputs": len(inputs)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--verify-campaign", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.verify_campaign is not None:
        campaign = json.loads(args.verify_campaign.read_text(encoding="utf-8"))
        problems = verify_inputs(campaign, REPO)
        print(json.dumps({"ok": not problems, "problems": problems}, ensure_ascii=False, indent=2))
        return 0 if not problems else 2
    config = json.loads(args.config.read_text(encoding="utf-8"))
    source_path = args.source_manifest or (REPO / config["source_manifest"])
    out_dir = args.out_dir or (REPO / config["outputs"]["campaign_root"])
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    result = materialize(source_manifest, out_dir, config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
