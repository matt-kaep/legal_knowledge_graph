"""Run the B1 campaign without mutating legacy grouped-v2 namespaces."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DATA_ROOT = DATA_REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3.json"
DEFAULT_A3 = ROOT / "configs/benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json"
STAGES = ("preflight", "cosine", "ppr-cv", "lightgcn-cv")


def _load_module(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _data_path(raw: str, data_root: Path = DATA_REPO) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else data_root / path


def _code_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else CODE_REPO / path


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_stage_commands(
    payload: dict[str, Any],
    stage: str,
    *,
    data_root: Path = DATA_REPO,
    python: Path = Path(sys.executable),
    graph_id: str | None = None,
) -> list[list[str]]:
    """Render commands without executing them, for tests and Slurm wrappers."""
    if stage not in STAGES:
        raise ValueError(f"Unsupported B1 stage={stage}")
    if stage == "preflight":
        return []
    if graph_id is not None and graph_id not in payload["graphs"]:
        raise ValueError(f"Unknown B1 graph={graph_id}")
    protocol = str(payload["protocol_version"])
    outputs = payload["outputs"]
    if stage == "cosine":
        return [[
            str(python), str(SCRIPTS / "26_eval_doctrine_v3plus_m1_m2.py"),
            "--split", payload["datasets"]["evaluation"]["split"],
            "--graph-version", "G1",
            "--top-k-out", str(payload["parameters"]["cosine"]["top_k_out"]),
            "--out-dir", str(_data_path(outputs["cosine"], data_root)),
            "--direct-cosine-only",
        ]]
    if graph_id is None:
        raise ValueError(f"B1 stage={stage} requires --graph-id")
    if stage == "ppr-cv":
        command = [
            str(python), str(SCRIPTS / "43_run_cv_ppr.py"),
            "--graph-version", graph_id,
            "--split", payload["datasets"]["train"]["split"],
            "--protocol-version", protocol,
            "--out-dir", str(_data_path(f"{outputs['ppr_cv']}/{graph_id}", data_root)),
        ]
        for k_in in payload["parameters"]["ppr"]["k_in"]:
            for variant in payload["parameters"]["ppr"]["seed_variant"]:
                for alpha in payload["parameters"]["ppr"]["alpha"]:
                    command.extend(["--config", f"{k_in}:{variant}:{alpha}"])
        return [command]
    grid = payload["parameters"]["lightgcn"]
    command = [
        str(python), str(SCRIPTS / "44_run_cv_lightgcn.py"),
        "--graph-version", graph_id,
        "--split", payload["datasets"]["train"]["split"],
        "--protocol-version", protocol,
        "--out-dir", str(_data_path(f"{outputs['lightgcn_cv']}/{graph_id}", data_root)),
    ]
    for value in grid["train_k"]:
        command.extend(["--train-k", str(value)])
    for value in grid["cv_seeds"]:
        command.extend(["--seed", str(value)])
    for value in grid["learning_rate"]:
        command.extend(["--lr", str(value)])
    for value in grid["lambda_anchor"]:
        command.extend(["--lambda-anchor", str(value)])
    for value in grid["negative_sampling_strategy"]:
        command.extend(["--negative-sampling-strategy", str(value)])
    command.extend(["--epochs", str(grid["epochs"]), "--selection-target", "art", "--selection-target", "jp"])
    return [command]


def preflight(payload: dict[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    """Verify A3 provenance, immutable inputs and code before running B1."""
    contract = _load_module("b1_campaign_contract.py", "b1_contract")
    a3_path = _code_path(payload["a3"]["manifest_path"])
    a3 = load_manifest(a3_path)
    contract.validate_b1_against_a3(payload, a3, a3_sha256=_sha256(a3_path))
    verified: list[str] = [str(a3_path)]
    for section in ("train", "evaluation"):
        item = payload["datasets"][section]
        path = _data_path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"dataset input mismatch: {section}")
        verified.append(str(path))
    for name in ("assignments", "metadata"):
        item = payload["fold_inputs"][name]
        path = _data_path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"fold input mismatch: {name}")
        verified.append(str(path))
    for name, item in payload["candidate_inputs"].items():
        path = _data_path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"candidate input mismatch: {name}")
        verified.append(str(path))
    for name, item in payload["graph_inputs"].items():
        path = _data_path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"graph input mismatch: {name}")
        verified.append(str(path))
    for name, item in payload["code_bundle"].items():
        path = _code_path(item["path"])
        if not path.is_file() or _sha256(path) != item["sha256"]:
            raise ValueError(f"code bundle mismatch: {name}")
        verified.append(str(path))
    evaluation_questions = json.loads(
        _data_path(payload["datasets"]["evaluation"]["path"]).read_text(encoding="utf-8")
    )["questions"]
    max_strict_articles = max(
        (len(set(map(str, question.get("articles_attendus") or []))) for question in evaluation_questions),
        default=0,
    )
    if max_strict_articles > 10:
        raise ValueError("Hit@10 cannot equal Recall@10: an evaluation question has more than 10 strict articles")
    return {
        "ok": True,
        "campaign_id": payload["campaign_id"],
        "manifest_path": str(manifest_path),
        # The campaign reference is the byte-level SHA-256, exactly like A3.
        # Keep a canonical digest as an additional diagnostic only.
        "manifest_sha256": _sha256(manifest_path),
        "manifest_canonical_sha256": _canonical_sha256(payload),
        "verified_files": len(verified),
        "a3_sha256": _sha256(a3_path),
        "train_questions": payload["datasets"]["train"]["questions"],
        "evaluation_questions": payload["datasets"]["evaluation"]["questions"],
        "max_strict_articles_per_evaluation_question": max_strict_articles,
        "hit_equals_recall_at_10_verified": True,
    }


def _write_status(payload: dict[str, Any], *, stage: str, report: dict[str, Any], output_paths: list[Path]) -> Path:
    status_root = _data_path(payload["outputs"]["root"]) / "status"
    status_root.mkdir(parents=True, exist_ok=True)
    artifacts = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in output_paths
        if path.is_file()
    ]
    record = {
        "stage": stage,
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": report["manifest_sha256"],
        "a3_sha256": report["a3_sha256"],
        "artifacts": artifacts,
    }
    path = status_root / f"{stage}.json"
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_cosine(payload: dict[str, Any], report: dict[str, Any]) -> Path:
    baseline = _load_module("26_eval_doctrine_v3plus_m1_m2.py", "b1_cosine")
    eval_dir = _data_path(payload["datasets"]["evaluation"]["directory"])
    questions = json.loads((eval_dir / "bench_global.json").read_text(encoding="utf-8"))["questions"]
    out_dir = _data_path(payload["outputs"]["cosine"])
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"B1 cosine output already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline.eval_m1_m2(
        questions,
        out_dir,
        ks_in=[10],
        question_cache_dir=eval_dir,
        graph_version="G1",
        top_k_out=int(payload["parameters"]["cosine"]["top_k_out"]),
        direct_cosine_only=True,
    )
    return _write_status(
        payload,
        stage="cosine",
        report=report,
        output_paths=[out_dir / "eval_m1_m2.csv", out_dir / "rankings.parquet", out_dir / "eval_m1_m2_summary.json"],
    )


def run_subprocess_stage(payload: dict[str, Any], stage: str, graph_id: str, report: dict[str, Any]) -> Path:
    commands = build_stage_commands(payload, stage, graph_id=graph_id)
    for command in commands:
        subprocess.run(command, check=True)
    out_key = "ppr_cv" if stage == "ppr-cv" else "lightgcn_cv"
    out_dir = _data_path(f"{payload['outputs'][out_key]}/{graph_id}")
    expected = [out_dir / name for name in ("raw.csv", "fold_metrics.csv", "summary.csv", "champions.json")]
    return _write_status(payload, stage=f"{stage}--{graph_id}", report=report, output_paths=expected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--graph-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    payload = load_manifest(args.manifest)
    report = preflight(payload, manifest_path=args.manifest)
    if args.stage == "preflight":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.dry_run:
        print(json.dumps({"preflight": report, "commands": build_stage_commands(payload, args.stage, graph_id=args.graph_id)}, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "cosine":
        print(run_cosine(payload, report))
        return 0
    if args.graph_id is None:
        parser.error(f"--graph-id is required for stage={args.stage}")
    print(run_subprocess_stage(payload, args.stage, args.graph_id, report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
