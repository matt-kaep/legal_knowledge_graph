"""Manifest-driven orchestration for the grouped_v2 confirmatory campaign."""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import platform
import socket
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import psutil


CODE_REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO)))
REPO = DATA_REPO
BENCH = DATA_REPO / "05-Technique/benchmark/etape1_embedding_pur"
CODE_BENCH = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
PYTHON = Path(os.environ.get("LKG_PYTHON", sys.executable))
SCRIPTS = CODE_BENCH / "scripts"
DEFAULT_MANIFEST = CODE_BENCH / "configs/confirmatory_campaign_grouped_v2_repro_v1.json"
SHORTLIST_PATH = (
    BENCH
    / "data/doctrine_v3plus_bench/_protocol/grouped_v2/lightgcn_shortlist.json"
)
STAGES = (
    "preflight",
    "cosine-control-cv",
    "ppr-cv",
    "lightgcn-screen",
    "lightgcn-shortlist",
    "lightgcn-tune",
    "lightgcn-seeds",
    "freeze-epochs",
    "internal-replay",
    "diagnostics",
    "paper-exports",
)
CANONICAL_GRAPH_IDS = (
    "G1", "G6-citation-AA-knn5", "G6-citation-JJ-knn5",
    "G7-citation-AA-cit1-sem025-knn5", "G7-citation-AA-cit1-sem050-knn5",
    "G7-citation-AA-cit1-sem100-knn5", "G7-citation-AA-cit025-sem1-knn5",
    "G7-citation-JJ-cit1-sem025-knn5", "G7-citation-JJ-cit1-sem050-knn5",
    "G7-citation-JJ-cit1-sem100-knn5", "G7-citation-JJ-cit025-sem1-knn5",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_manifest_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("protocol_version") != "grouped_v2":
        raise ValueError("Confirmatory campaign requires protocol_version=grouped_v2")
    graph_ids = [graph["graph_id"] for graph in payload.get("graphs", [])]
    if tuple(graph_ids) != CANONICAL_GRAPH_IDS:
        raise ValueError("Confirmatory manifest graph matrix differs from the approved canonical matrix")
    expected_selection = {
        "article": ["recall_at_10", "ndcg_at_10", "mrr_at_10"],
        "jp": ["hit_at_10", "ndcg_at_10", "mrr_at_10"],
    }
    for target, metrics in expected_selection.items():
        if payload.get("selection", {}).get(target) != metrics:
            raise ValueError(f"selection.{target} differs from grouped_v2")
    if payload.get("folds", {}).get("count") != 5 or payload.get("folds", {}).get("seed") != 42:
        raise ValueError("fold contract differs from grouped_v2")
    if payload.get("ppr") != {
        "k_in": [5, 10, 20, 50],
        "seed_variant": ["art_only", "jp_only", "both"],
        "alpha": [0.5, 0.7, 0.85, 0.95],
    }:
        raise ValueError("ppr grid differs from the approved campaign")
    lightgcn = payload.get("lightgcn", {})
    required_lightgcn = {
        "screen": {"train_k": [2], "learning_rate": [0.001], "lambda_anchor": [1.0], "seed": [42], "epochs": 30, "negative_sampling_strategy": ["random"], "selection_targets": ["art", "jp"]},
        "tune": {"train_k": [1, 2, 3], "learning_rate": [0.0005, 0.001], "lambda_anchor": [0.5, 1.0], "seed": [42], "epochs": 30, "negative_sampling_strategy": ["random"], "selection_targets": ["art", "jp"]},
        "robustness": {"seed": [42, 43, 44], "negative_sampling_strategy": ["random"], "selection_targets": ["art", "jp"]},
    }
    for section, expected in required_lightgcn.items():
        if lightgcn.get(section) != expected:
            raise ValueError(f"lightgcn.{section} grid differs from the approved campaign")
    expected_outputs = {
        "cv_root": "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_cv_grouped_v2",
        "final_root": "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_final_grouped_v2",
        "status_root": "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_protocol/grouped_v2/status",
        "log_root": "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_protocol/grouped_v2/logs",
        "export_root": "06-Analyses/comparatifs/benchmark-confirmatoire-grouped-v2",
    }
    for key, raw in payload.get("outputs", {}).items():
        candidate = Path(str(raw))
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"outputs.{key} must be a repository-relative path without '..'")
        resolved = (REPO / candidate).resolve(strict=False)
        if not resolved.is_relative_to(REPO.resolve()):
            raise ValueError(f"outputs.{key} escapes the repository")
    for key, expected in expected_outputs.items():
        if payload.get("outputs", {}).get(key) != expected:
            raise ValueError(f"outputs.{key} differs from the approved campaign namespace")
    resources = payload.get("resources", {})
    if set(resources) != {
        "threads_per_process", "cpu_per_graph_job", "cpu_per_ppr_job",
        "ram_minimum_gb_per_graph_job",
        "ram_minimum_gb_per_ppr_job",
        "ram_observed_upper_bound_gb_per_graph_job", "max_parallel_graph_jobs",
        "gpu_required",
    }:
        raise ValueError("resource contract fields differ from the approved schema")
    if (
        resources["threads_per_process"] != 2
        or resources["cpu_per_graph_job"] != 5
        or resources["cpu_per_ppr_job"] != 4
    ):
        raise ValueError("CPU/thread resource contract differs from the approved campaign")
    if resources["ram_observed_upper_bound_gb_per_graph_job"] != 45:
        raise ValueError("observed RAM upper bound differs from the approved evidence")
    measured_minimum = resources["ram_minimum_gb_per_graph_job"]
    if measured_minimum is not None and (
        not isinstance(measured_minimum, (int, float)) or measured_minimum <= 0
    ):
        raise ValueError("measured RAM minimum must be null or a positive number")
    measured_ppr_minimum = resources["ram_minimum_gb_per_ppr_job"]
    if measured_ppr_minimum is not None and (
        not isinstance(measured_ppr_minimum, (int, float)) or measured_ppr_minimum <= 0
    ):
        raise ValueError("measured PPR RAM minimum must be null or a positive number")
    if resources["max_parallel_graph_jobs"] != 2 or resources["gpu_required"] is not False:
        raise ValueError("parallelism/GPU resource contract differs from the approved campaign")
    return payload


def load_manifest(path: Path) -> dict[str, Any]:
    return validate_manifest_payload(json.loads(path.read_text(encoding="utf-8")))


def _resolve(relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else REPO / path


def _resolve_code(relative_path: str) -> Path:
    path = Path(relative_path)
    return path if path.is_absolute() else CODE_REPO / path


def validate_internal_eval_authorization(payload: dict[str, Any], authorization: str | None) -> None:
    if authorization != payload["campaign_id"]:
        raise ValueError("internal evaluation authorization must exactly match campaign_id")


@contextmanager
def stage_lock(lock_path: Path, *, max_parallel_jobs: int):
    """Own both a stage/graph lock and one campaign-wide concurrency slot."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        stage_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"stage already running: {lock_path}") from exc
    slot_path = None
    slot_fd = None
    try:
        owner = {"pid": os.getpid(), "hostname": socket.gethostname(), "started_at": utc_now()}
        os.write(stage_fd, json.dumps(owner).encode())
        slots_dir = lock_path.parent / "slots"
        slots_dir.mkdir(exist_ok=True)
        for index in range(int(max_parallel_jobs)):
            candidate = slots_dir / f"slot-{index}.lock"
            try:
                slot_fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                slot_path = candidate
                os.write(slot_fd, json.dumps(owner).encode())
                break
            except FileExistsError:
                continue
        if slot_fd is None:
            raise RuntimeError(f"max_parallel_graph_jobs={max_parallel_jobs} already reached")
        yield
    finally:
        os.close(stage_fd)
        lock_path.unlink(missing_ok=True)
        if slot_fd is not None:
            os.close(slot_fd)
        if slot_path is not None:
            slot_path.unlink(missing_ok=True)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def recover_stale_locks(locks_root: Path) -> dict[str, Any]:
    """Archive only locks whose local PID is demonstrably dead."""
    recovered = []
    retained = []
    candidates = sorted(path for path in locks_root.rglob("*.lock") if "quarantine" not in path.parts)
    archive_dir = locks_root / "quarantine" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    for path in candidates:
        try:
            owner = json.loads(path.read_text(encoding="utf-8"))
            pid = int(owner["pid"])
            hostname = str(owner["hostname"])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            retained.append({"path": str(path), "reason": "malformed_owner"})
            continue
        if hostname != socket.gethostname():
            retained.append({"path": str(path), "reason": "remote_owner"})
            continue
        if _pid_alive(pid):
            retained.append({"path": str(path), "reason": "owner_alive"})
            continue
        archive_dir.mkdir(parents=True, exist_ok=True)
        destination = archive_dir / f"{len(recovered) + 1:03d}--{path.name}"
        path.replace(destination)
        recovered.append({
            "path": str(path), "archived_path": str(destination),
            "sha256": sha256_file(destination), "owner": owner,
        })
    proof = {"recovered_at": utc_now(), "recovered": recovered, "retained": retained}
    if recovered:
        (archive_dir / "recovery_manifest.json").write_text(
            json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return proof


def refuse_unverified_overwrite(artifacts: list[Path], *, status_path: Path) -> None:
    existing = [path for path in artifacts if path.exists()]
    if existing:
        raise FileExistsError(
            f"unverified campaign artifact would be overwritten: {existing[0]} "
            f"(verified resume status required at {status_path})"
        )


def stage_owned_artifacts(stage: str, artifacts: list[Path]) -> list[Path]:
    if stage == "lightgcn-seeds":
        return [path for path in artifacts if path.name != "champions.json"]
    return artifacts


def stage_owned_paths(
    payload: dict[str, Any], stage: str, graph_id: str | None, artifacts: list[Path]
) -> list[Path]:
    cv_root = _resolve(payload["outputs"]["cv_root"])
    final_root = _resolve(payload["outputs"]["final_root"])
    export_root = _resolve(payload["outputs"]["export_root"])
    owned_directories = {
        "cosine-control-cv": cv_root / "G1" / "b3_b4",
        "ppr-cv": cv_root / str(graph_id) / "ppr",
        "lightgcn-screen": cv_root / str(graph_id) / "lightgcn_screen",
        "lightgcn-tune": cv_root / str(graph_id) / "lightgcn",
        "lightgcn-seeds": cv_root / str(graph_id) / "lightgcn_robustness",
        "internal-replay": final_root / str(graph_id),
        "diagnostics": export_root / "diagnostics",
        "paper-exports": export_root / "paper",
    }
    if stage in owned_directories:
        return [owned_directories[stage]]
    return stage_owned_artifacts(stage, artifacts)


PER_GRAPH_STAGES = {
    "ppr-cv", "lightgcn-screen", "lightgcn-tune", "lightgcn-seeds",
    "freeze-epochs", "internal-replay",
}
GLOBAL_ONLY_STAGES = {"cosine-control-cv", "lightgcn-shortlist", "diagnostics", "paper-exports"}


def validate_stage_invocation(stage: str, graph_id: str | None) -> None:
    if stage in PER_GRAPH_STAGES and graph_id is None:
        raise ValueError(f"stage {stage} requires --graph-id to prevent overlapping output ownership")
    if stage in GLOBAL_ONLY_STAGES and graph_id is not None:
        raise ValueError(f"stage {stage} is global and must not receive --graph-id")


def quarantine_unverified_artifacts(
    artifacts: list[Path], *, quarantine_root: Path, stage: str, graph_id: str | None
) -> dict[str, Any]:
    existing = [path for path in artifacts if path.exists()]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    quarantine_dir = quarantine_root / stage / (graph_id or "all") / run_id
    quarantine_dir.mkdir(parents=True, exist_ok=False)
    moves = []
    for index, source in enumerate(existing, start=1):
        destination = quarantine_dir / f"{index:03d}--{source.name}"
        source.replace(destination)
        move = {
            "original_path": str(source),
            "quarantined_path": str(destination),
        }
        if destination.is_dir():
            files = [
                {"relative_path": str(path.relative_to(destination)), "sha256": sha256_file(path)}
                for path in sorted(destination.rglob("*")) if path.is_file()
            ]
            move["files"] = files
        else:
            move["sha256"] = sha256_file(destination)
        moves.append(move)
    proof = {
        "stage": stage,
        "graph_id": graph_id,
        "quarantined_at": utc_now(),
        "quarantine_dir": str(quarantine_dir),
        "moves": moves,
    }
    (quarantine_dir / "quarantine_manifest.json").write_text(
        json.dumps(proof, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return proof


def _verify_file(path: Path, expected_sha256: str, verify_hashes: bool) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if verify_hashes and sha256_file(path) != expected_sha256:
        raise ValueError(f"sha256 mismatch: {path}")


def assess_resources(
    resources: dict[str, Any], *, stage: str | None = None,
    cpu_available: int | None, ram_available_bytes: int | None
) -> dict[str, Any]:
    ppr_profile_stages = {"cosine-control-cv", "ppr-cv"}
    resource_profile = "ppr" if stage in ppr_profile_stages else "maximum_graph_job"
    required_cpu = int(
        resources.get("cpu_per_ppr_job", resources["cpu_per_graph_job"])
        if resource_profile == "ppr"
        else resources["cpu_per_graph_job"]
    )
    insufficient = []
    minimum_ram_gb = (
        resources.get("ram_minimum_gb_per_ppr_job")
        if resource_profile == "ppr"
        else resources.get("ram_minimum_gb_per_graph_job")
    )
    if minimum_ram_gb is None:
        insufficient.append("ram_minimum_unmeasured")
        required_ram = None
    else:
        required_ram = int(float(minimum_ram_gb) * 1024**3)
    if cpu_available is not None and cpu_available < required_cpu:
        insufficient.append("cpu")
    if required_ram is not None and ram_available_bytes is not None and ram_available_bytes < required_ram:
        insufficient.append("ram")
    cpu_jobs = (cpu_available // required_cpu) if cpu_available is not None else 0
    ram_jobs = (ram_available_bytes // required_ram) if required_ram and ram_available_bytes is not None else 0
    max_safe = min(int(resources["max_parallel_graph_jobs"]), cpu_jobs, ram_jobs)
    return {
        "compatible": not insufficient and max_safe >= 1,
        "insufficient": insufficient,
        "cpu_required_per_job": required_cpu,
        "ram_required_bytes_per_job": required_ram,
        "resource_profile": resource_profile,
        "max_safe_parallel_jobs": int(max_safe),
    }


def preflight(
    payload: dict[str, Any], *, verify_hashes: bool = True,
    resource_stage: str | None = None,
) -> dict[str, Any]:
    """Validate immutable inputs without creating campaign outputs."""
    datasets = payload["datasets"]
    for dataset in datasets.values():
        _verify_file(_resolve(dataset["path"]), dataset["sha256"], verify_hashes)

    folds = payload["folds"]
    fold_path = _resolve(folds["path"])
    _verify_file(fold_path, folds["sha256"], verify_hashes)
    metadata_path = _resolve(folds["metadata_path"])
    _verify_file(metadata_path, folds["metadata_sha256"], verify_hashes)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    required_metadata = {
        "protocol_version": payload["protocol_version"],
        "dataset_sha256": datasets["train"]["sha256"],
        "fold_assignment_sha256": folds["sha256"],
        "n_questions": datasets["train"]["questions"],
        "n_folds": folds["count"],
        "provenance_groups_crossing_folds": 0,
        "normalized_text_groups_crossing_folds": 0,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(f"fold metadata mismatch for {key}")
    fold_rows = pd.read_csv(fold_path)
    for column in ("provenance_fingerprint", "text_fingerprint"):
        crossings = fold_rows.dropna(subset=[column]).groupby(column)["fold"].nunique()
        if int((crossings > 1).sum()) != 0:
            raise ValueError(f"fold leakage recomputation failed for {column}")

    for shared_file in payload["shared_hybrid_ids"].values():
        _verify_file(_resolve(shared_file["path"]), shared_file["sha256"], verify_hashes)
    immutable_inputs = payload.get("immutable_inputs", {})
    if not immutable_inputs:
        raise ValueError("confirmatory manifest must seal immutable_inputs")
    for name, input_file in immutable_inputs.items():
        try:
            _verify_file(_resolve(input_file["path"]), input_file["sha256"], verify_hashes)
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(f"immutable input {name}: {exc}") from exc
    code_bundle = payload.get("code_bundle", {})
    if not code_bundle:
        raise ValueError("confirmatory manifest must seal code_bundle")
    for name, code_file in code_bundle.items():
        try:
            _verify_file(_resolve_code(code_file["path"]), code_file["sha256"], verify_hashes)
        except (FileNotFoundError, ValueError) as exc:
            raise type(exc)(f"code bundle {name}: {exc}") from exc
    graph_input_copies_verified = 0
    for graph in payload["graphs"]:
        _verify_file(_resolve(graph["matrix_path"]), graph["matrix_sha256"], verify_hashes)
        for split_name, dataset in datasets.items():
            graph_split = dataset["split"]
            graph_bench = (
                BENCH / "data/doctrine_v3plus_bench" / graph["graph_id"] / graph_split
                / "bench_global.json"
            )
            if not graph_bench.is_file():
                raise FileNotFoundError(
                    f"Missing {split_name} bench for {graph['graph_id']}: {graph_bench}"
                )
            if verify_hashes and sha256_file(graph_bench) != dataset["sha256"]:
                raise ValueError(f"dataset hash mismatch for graph={graph['graph_id']} split={graph_split}")
            cache_prefix = "train" if split_name == "train" else "eval"
            for filename, suffix in (("questions_emb.npy", "question_embeddings"), ("questions_ids.npy", "question_ids")):
                immutable = immutable_inputs[f"{cache_prefix}_{suffix}"]
                _verify_file(graph_bench.parent / filename, immutable["sha256"], verify_hashes)
                graph_input_copies_verified += 1
        if graph["graph_id"] != "G1":
            graph_dir = _resolve(graph["matrix_path"]).parent
            for filename, shared_key in (
                ("jp_ids.npy", "jp_ids"),
                ("article_ids.npy", "article_ids"),
                ("node_ids.npy", "node_ids"),
                ("article_codes.npy", "article_codes"),
            ):
                _verify_file(
                    graph_dir / filename,
                    payload["shared_hybrid_ids"][shared_key]["sha256"],
                    verify_hashes,
                )
                graph_input_copies_verified += 1

    if not PYTHON.is_file():
        raise FileNotFoundError(f"Missing benchmark Python runtime: {PYTHON}")
    runtime = payload.get("runtime", {})
    if platform.python_version() != runtime.get("python"):
        raise ValueError("runtime python version mismatch")
    import importlib.metadata
    for package, expected in runtime.get("packages", {}).items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"runtime package mismatch: {package}")
    disk = shutil.disk_usage(REPO)
    resources = payload["resources"]
    cpu_available = os.cpu_count()
    memory = psutil.virtual_memory()
    ram_available_bytes = int(memory.available)
    resource_assessment = assess_resources(
        resources,
        stage=resource_stage,
        cpu_available=cpu_available,
        ram_available_bytes=ram_available_bytes,
    )
    return {
        "ok": resource_assessment["compatible"],
        "scientific_inputs_ok": True,
        "campaign_id": payload["campaign_id"],
        "manifest_sha256": manifest_sha256(payload),
        "n_graphs": len(payload["graphs"]),
        "n_train_questions": datasets["train"]["questions"],
        "n_eval_questions": datasets["internal_eval"]["questions"],
        "n_folds": folds["count"],
        "disk_free_bytes": disk.free,
        "cpu_available": cpu_available,
        "ram_available_bytes": ram_available_bytes,
        "ram_total_bytes": int(memory.total),
        "declared_resources": resources,
        "resource_assessment": resource_assessment,
        "immutable_inputs_verified": len(immutable_inputs),
        "code_files_verified": len(code_bundle),
        "graph_input_copies_verified": graph_input_copies_verified,
        "runtime_verified": True,
        "verified_hashes": verify_hashes,
    }


def _graph_ids(payload: dict[str, Any], graph_id: str | None) -> list[str]:
    known = [graph["graph_id"] for graph in payload["graphs"]]
    if graph_id is None:
        return known
    if graph_id not in known:
        raise ValueError(f"Unknown graph_id={graph_id}")
    return [graph_id]


def campaign_shortlist_path(payload: dict[str, Any]) -> Path:
    return _resolve(payload["outputs"]["status_root"]).parent / "lightgcn_shortlist.json"


def _shortlisted_graph_ids(
    payload: dict[str, Any], shortlist_file: Path = SHORTLIST_PATH
) -> list[str]:
    shortlist = json.loads(shortlist_file.read_text(encoding="utf-8"))
    if shortlist.get("manifest_sha256") != manifest_sha256(payload):
        raise ValueError("Frozen LightGCN shortlist does not match the campaign manifest")
    graph_ids = [str(value) for value in shortlist.get("graph_ids", [])]
    known = {graph["graph_id"] for graph in payload["graphs"]}
    if not graph_ids or len(graph_ids) != len(set(graph_ids)):
        raise ValueError("Frozen LightGCN shortlist is empty or contains duplicates")
    unknown = sorted(set(graph_ids) - known)
    if unknown:
        raise ValueError(f"Frozen LightGCN shortlist contains unknown graphs: {unknown}")
    for source in shortlist.get("sources", []):
        source_path = Path(source["path"])
        if not source_path.is_file() or sha256_file(source_path) != source.get("sha256"):
            raise ValueError(f"Frozen LightGCN shortlist source changed: {source_path}")
    return graph_ids


def _lightgcn_robustness_commands(
    payload: dict[str, Any], graph_ids: list[str]
) -> list[list[str]]:
    cv_root = _resolve(payload["outputs"]["cv_root"])
    seeds = payload["lightgcn"]["robustness"]["seed"]
    commands: list[list[str]] = []
    for graph_id in graph_ids:
        champions_path = cv_root / graph_id / "lightgcn" / "champions.json"
        champions = json.loads(champions_path.read_text(encoding="utf-8"))
        for target in ("art", "jp"):
            champion = champions[target]
            if not str(champion.get("variant", "")).startswith("trained_"):
                continue
            command = [
                str(PYTHON), str(SCRIPTS / "44_run_cv_lightgcn.py"),
                "--graph-version", graph_id,
                "--split", payload["datasets"]["train"]["split"],
                "--out-dir", str(cv_root / graph_id / "lightgcn_robustness" / target),
                "--selection-target", target,
                "--train-k", str(int(champion["train_k"])),
                "--lr", str(float(champion["lr"])),
                "--lambda-anchor", str(float(champion["lambda_anchor"])),
                "--epochs", str(int(champion["epochs"])),
                "--negative-sampling-strategy",
                str(champion.get("negative_sampling_strategy", "random")),
            ]
            for seed in seeds:
                command.extend(["--seed", str(seed)])
            commands.append(command)
    return commands


def aggregate_lightgcn_robustness(payload: dict[str, Any], graph_ids: list[str]) -> list[Path]:
    cv_root = _resolve(payload["outputs"]["cv_root"])
    expected_seeds = set(payload["lightgcn"]["robustness"]["seed"])
    written: list[Path] = []
    for graph_id in graph_ids:
        champions_path = cv_root / graph_id / "lightgcn" / "champions.json"
        champions = json.loads(champions_path.read_text(encoding="utf-8"))
        champions_sha256 = sha256_file(champions_path)
        rows = []
        for target in ("art", "jp"):
            champion = champions[target]
            path = cv_root / graph_id / "lightgcn_robustness" / target / "summary.csv"
            if not path.is_file():
                continue
            frame = pd.read_csv(path)
            trained = frame[
                frame["modality"].eq(target)
                & frame["selection_target"].eq(target)
                & frame["variant"].astype(str).str.startswith("trained_")
            ].copy()
            if set(trained["seed"].dropna().astype(int)) != expected_seeds:
                raise ValueError(f"robustness seeds incomplete for graph={graph_id} target={target}")
            if not trained["eligible_champion"].eq(True).all():
                raise ValueError(f"ineligible robustness row for graph={graph_id} target={target}")
            identity = {
                "variant": champion["variant"],
                "train_k": champion["train_k"],
                "lr": champion["lr"],
                "epochs": champion["epochs"],
                "lambda_anchor": champion["lambda_anchor"],
                "negative_sampling_strategy": champion.get("negative_sampling_strategy", "random"),
            }
            for column, expected in identity.items():
                if not trained[column].eq(expected).all():
                    raise ValueError(f"robustness champion mismatch for graph={graph_id} target={target} field={column}")
            metric = "article_recall_at_10_mean" if target == "art" else "jp_hit_at_10_mean"
            rows.append({
                "graph_id": graph_id,
                "target": target,
                "metric": metric,
                "n_seeds": len(expected_seeds),
                "seeds": ";".join(map(str, sorted(expected_seeds))),
                "mean_across_seeds": float(trained[metric].mean()),
                "std_across_seeds": float(trained[metric].std(ddof=1)),
                "protocol_version": payload["protocol_version"],
                "dataset_sha256": payload["datasets"]["train"]["sha256"],
                "fold_assignment_sha256": payload["folds"]["sha256"],
                "champions_path": str(champions_path),
                "champions_sha256": champions_sha256,
                **identity,
                "replay_epochs": champion["replay_epochs"],
            })
        out_path = cv_root / graph_id / "lightgcn_robustness" / "summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            rows,
            columns=[
                "graph_id", "target", "metric", "n_seeds", "seeds",
                "mean_across_seeds", "std_across_seeds", "protocol_version",
                "dataset_sha256", "fold_assignment_sha256",
                "champions_path", "champions_sha256", "variant", "train_k", "lr",
                "epochs", "lambda_anchor", "negative_sampling_strategy", "replay_epochs",
            ],
        ).to_csv(out_path, index=False)
        written.append(out_path)
    return written


def freeze_lightgcn_champions(payload: dict[str, Any], graph_ids: list[str]) -> list[Path]:
    cv_root = _resolve(payload["outputs"]["cv_root"])
    written: list[Path] = []
    for graph_id in graph_ids:
        champions_path = cv_root / graph_id / "lightgcn" / "champions.json"
        champions = json.loads(champions_path.read_text(encoding="utf-8"))
        robustness_path = cv_root / graph_id / "lightgcn_robustness" / "summary.csv"
        robustness = pd.read_csv(robustness_path) if robustness_path.is_file() else pd.DataFrame()
        for target, champion in champions.items():
            if str(champion.get("variant", "")).startswith("trained_"):
                if not {"selected_epoch_index", "replay_epochs"}.issubset(champion):
                    raise ValueError(f"missing frozen epoch for graph={graph_id} target={target}")
                target_rows = robustness[robustness["target"].eq(target)] if not robustness.empty else robustness
                if target_rows.empty:
                    raise ValueError(f"missing robustness aggregate for graph={graph_id} target={target}")
                proof = target_rows.iloc[0]
                expected_identity = {
                    "champions_sha256": sha256_file(champions_path),
                    "variant": champion["variant"],
                    "train_k": champion["train_k"],
                    "lr": champion["lr"],
                    "epochs": champion["epochs"],
                    "lambda_anchor": champion["lambda_anchor"],
                    "negative_sampling_strategy": champion.get("negative_sampling_strategy", "random"),
                    "replay_epochs": champion["replay_epochs"],
                }
                mismatched = [
                    field for field, expected in expected_identity.items()
                    if str(proof[field]) != str(expected)
                ]
                if mismatched:
                    raise ValueError(f"robustness proof mismatch for graph={graph_id} target={target} field={mismatched[0]}")
                champion["robustness_summary_path"] = str(robustness_path)
                champion["robustness_summary_sha256"] = sha256_file(robustness_path)
        out_path = cv_root / graph_id / "lightgcn" / "frozen_champions.json"
        temporary = out_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(champions, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(out_path)
        written.append(out_path)
    return written


def build_stage_commands(
    payload: dict[str, Any],
    stage: str,
    *,
    graph_id: str | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    shortlist_file: Path = SHORTLIST_PATH,
) -> list[list[str]]:
    if stage not in STAGES:
        raise ValueError(f"Unsupported stage: {stage}")
    if stage in {"preflight", "freeze-epochs"}:
        return []

    if stage == "cosine-control-cv":
        if graph_id not in {None, "G1"}:
            raise ValueError("the shared cosine control is computed once on G1")
        return [[
            str(PYTHON), str(SCRIPTS / "42_run_cv_b3_b4.py"),
            "--graph-version", "G1",
            "--split", payload["datasets"]["train"]["split"],
            "--out-dir", str(_resolve(payload["outputs"]["cv_root"]) / "G1" / "b3_b4"),
            "--direct-cosine-only",
        ]]

    outputs = payload["outputs"]
    graphs = (
        _shortlisted_graph_ids(payload, shortlist_file)
        if graph_id is None and stage in {"lightgcn-tune", "lightgcn-seeds"}
        else _graph_ids(payload, graph_id)
    )
    commands: list[list[str]] = []
    for current_graph in graphs:
        if stage == "ppr-cv":
            command = [
                    str(PYTHON), str(SCRIPTS / "43_run_cv_ppr.py"),
                    "--graph-version", current_graph,
                    "--split", payload["datasets"]["train"]["split"],
                    "--out-dir", str(_resolve(outputs["cv_root"]) / current_graph / "ppr"),
            ]
            for k_in in payload["ppr"]["k_in"]:
                for seed_variant in payload["ppr"]["seed_variant"]:
                    for alpha in payload["ppr"]["alpha"]:
                        command.extend(["--config", f"{k_in}:{seed_variant}:{alpha}"])
            command.extend([
                "--control-fold-metrics",
                str(_resolve(outputs["cv_root"]) / "G1" / "ppr" / "fold_metrics.csv"),
            ])
            commands.append(command)
        elif stage in {"lightgcn-screen", "lightgcn-tune"}:
            grid = payload["lightgcn"]["screen" if stage == "lightgcn-screen" else "tune"]
            command = [
                str(PYTHON), str(SCRIPTS / "44_run_cv_lightgcn.py"),
                "--graph-version", current_graph,
                "--split", payload["datasets"]["train"]["split"],
                "--out-dir", str(
                    _resolve(outputs["cv_root"])
                    / current_graph
                    / ("lightgcn_screen" if stage == "lightgcn-screen" else "lightgcn")
                ),
            ]
            for value in grid["train_k"]:
                command.extend(["--train-k", str(value)])
            for value in grid["learning_rate"]:
                command.extend(["--lr", str(value)])
            for value in grid["lambda_anchor"]:
                command.extend(["--lambda-anchor", str(value)])
            for value in grid["seed"]:
                command.extend(["--seed", str(value)])
            command.extend(["--epochs", str(grid["epochs"])])
            for value in grid["negative_sampling_strategy"]:
                command.extend(["--negative-sampling-strategy", value])
            command.extend([
                "--control-fold-metrics",
                str(
                    _resolve(outputs["cv_root"])
                    / "G1"
                    / ("lightgcn_screen" if stage == "lightgcn-screen" else "lightgcn")
                    / "fold_metrics.csv"
                ),
            ])
            commands.append(command)
        elif stage == "lightgcn-shortlist":
            commands = [[
                str(PYTHON), str(SCRIPTS / "65_select_lightgcn_shortlist.py"),
                "--manifest", str(manifest_path),
            ]]
            break
        elif stage == "internal-replay":
            shortlist = set(_shortlisted_graph_ids(payload, shortlist_file))
            commands.append(
                [
                    str(PYTHON), str(SCRIPTS / "45_run_final_champions.py"),
                    "--graph-version", current_graph,
                    "--protocol-version", payload["protocol_version"],
                    "--families", "b3_b4,ppr,lightgcn" if current_graph == "G1" and current_graph in shortlist else ("ppr,lightgcn" if current_graph in shortlist else "ppr"),
                    "--top-k-out", "1000",
                    "--cv-root", str(_resolve(outputs["cv_root"]) / current_graph),
                    "--out-dir", str(_resolve(outputs["final_root"]) / current_graph),
                    "--campaign-manifest", str(manifest_path),
                    "--authorize-internal-eval", payload["campaign_id"],
                ]
            )
        elif stage in {"diagnostics", "paper-exports"}:
            commands = [[
                str(PYTHON), str(SCRIPTS / "66_export_confirmatory_results.py"),
                "--manifest", str(manifest_path),
                "--mode", "diagnostics" if stage == "diagnostics" else "paper-exports",
            ]]
            break
        elif stage == "lightgcn-seeds":
            return _lightgcn_robustness_commands(payload, graphs)
    return commands


def expected_artifacts(
    payload: dict[str, Any],
    stage: str,
    *,
    graph_id: str | None = None,
    shortlist_file: Path = SHORTLIST_PATH,
) -> list[Path]:
    outputs = payload["outputs"]
    graphs = (
        _shortlisted_graph_ids(payload, shortlist_file)
        if graph_id is None and stage in {"lightgcn-tune", "lightgcn-seeds", "freeze-epochs"}
        else _graph_ids(payload, graph_id)
    )
    cv_root = _resolve(outputs["cv_root"])
    final_root = _resolve(outputs["final_root"])
    export_root = _resolve(outputs["export_root"])
    if stage == "cosine-control-cv":
        return [cv_root / "G1" / "b3_b4" / name for name in ("summary.csv", "champions.json", "fold_metrics.csv")]
    if stage == "ppr-cv":
        return [cv_root / graph / "ppr" / name for graph in graphs for name in ("fold_metrics.csv", "summary.csv", "champions.json", "paired_deltas.csv")]
    if stage == "lightgcn-screen":
        return [cv_root / graph / "lightgcn_screen" / name for graph in graphs for name in ("fold_metrics.csv", "summary.csv", "paired_deltas.csv")]
    if stage == "lightgcn-shortlist":
        return [shortlist_file]
    if stage == "lightgcn-tune":
        return [cv_root / graph / "lightgcn" / name for graph in graphs for name in ("fold_metrics.csv", "summary.csv", "champions.json", "paired_deltas.csv")]
    if stage == "lightgcn-seeds":
        paths: list[Path] = []
        for graph in graphs:
            champions = json.loads((cv_root / graph / "lightgcn" / "champions.json").read_text())
            for target in ("art", "jp"):
                if str(champions[target].get("variant", "")).startswith("trained_"):
                    paths.append(cv_root / graph / "lightgcn_robustness" / target / "summary.csv")
        return [
            *paths,
            *[cv_root / graph / "lightgcn_robustness" / "summary.csv" for graph in graphs],
            *[cv_root / graph / "lightgcn" / "champions.json" for graph in graphs],
        ]
    if stage == "freeze-epochs":
        return [cv_root / graph / "lightgcn" / "frozen_champions.json" for graph in graphs]
    if stage == "internal-replay":
        return [
            final_root / graph / name
            for graph in graphs
            for name in ("final_champions_summary.csv", "rankings.parquet", "selected_champions.json")
        ]
    if stage == "diagnostics":
        return [export_root / "diagnostics" / name for name in ("expected_coverage_by_question.csv", "expected_coverage_by_k.csv")]
    if stage == "paper-exports":
        return [export_root / "paper" / name for name in ("internal_eval_results.csv", "internal_eval_articles.csv", "internal_eval_jp.csv", "internal_eval_primary_metrics.png")]
    return []


def write_stage_status(
    path: Path,
    *,
    stage: str,
    manifest_hash: str,
    results: list[dict[str, Any]],
    artifacts: list[Path],
    dependencies: list[str] | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict[str, Any]:
    complete = bool(results) and all(result.get("status") == "complete" for result in results)
    artifact_rows = [
        {"path": str(artifact), "sha256": sha256_file(artifact)}
        for artifact in artifacts
        if artifact.is_file()
    ]
    status = {
        "stage": stage,
        "status": "complete" if complete and len(artifact_rows) == len(artifacts) else "failed",
        "manifest_sha256": manifest_hash,
        "started_at": started_at or utc_now(),
        "finished_at": finished_at or utc_now(),
        "updated_at": utc_now(),
        "dependencies": dependencies or [],
        "results": results,
        "artifacts": artifact_rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return status


def execute_commands(
    commands: list[list[str]], *, dry_run: bool, log_dir: Path | None = None
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    env = os.environ.copy()
    env["LKG_REPO"] = str(DATA_REPO)
    env["LKG_CODE_REPO"] = str(CODE_REPO)
    env.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"})
    for index, command in enumerate(commands, start=1):
        started_at = utc_now()
        if dry_run:
            results.append({"command": command, "status": "dry_run"})
            continue
        log_path = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"command-{index:02d}.log"
        if log_path is None:
            completed = subprocess.run(command, cwd=REPO, env=env, check=False)
        else:
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=REPO,
                    env=env,
                    check=False,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
        results.append(
            {
                "command": command,
                "status": "complete" if completed.returncode == 0 else "failed",
                "returncode": completed.returncode,
                **({"log_path": str(log_path)} if log_path else {}),
                "started_at": started_at,
                "finished_at": utc_now(),
            }
        )
        if completed.returncode != 0:
            break
    return results


def can_resume(
    status: dict[str, Any],
    expected_manifest_sha256: str,
    expected_artifact_paths: set[str] | None = None,
    *,
    allow_artifact_superset: bool = False,
) -> bool:
    if status.get("status") != "complete":
        return False
    if status.get("manifest_sha256") != expected_manifest_sha256:
        return False
    artifacts = status.get("artifacts", [])
    if not artifacts:
        return False
    if expected_artifact_paths is not None:
        recorded_paths = {str(artifact.get("path")) for artifact in artifacts}
        valid_paths = expected_artifact_paths.issubset(recorded_paths) if allow_artifact_superset else recorded_paths == expected_artifact_paths
        if not valid_paths:
            return False
    return all(
        Path(artifact["path"]).is_file()
        and sha256_file(Path(artifact["path"])) == artifact.get("sha256")
        for artifact in artifacts
    )


def require_stage_status(
    payload: dict[str, Any], stage: str, graph_id: str | None
) -> None:
    status_root = _resolve(payload["outputs"]["status_root"])
    candidates = [status_root / f"{stage}--{graph_id or 'all'}.json"]
    if graph_id is not None:
        candidates.append(status_root / f"{stage}--all.json")
    expected_paths = {
        str(item)
        for item in expected_artifacts(
            payload,
            stage,
            graph_id=graph_id,
            shortlist_file=campaign_shortlist_path(payload),
        )
    }
    for path in candidates:
        if path.is_file() and can_resume(
            json.loads(path.read_text(encoding="utf-8")),
            manifest_sha256(payload),
            expected_paths,
            allow_artifact_superset=path.name.endswith("--all.json") and graph_id is not None,
        ):
            return
    raise ValueError(f"required verified stage is incomplete: {stage} graph={graph_id or 'all'}")


def dependency_requirements(
    payload: dict[str, Any], stage: str, graph_id: str | None, shortlist_file: Path
) -> list[tuple[str, str | None]]:
    requirements: list[tuple[str, str | None]] = []
    if stage == "ppr-cv" and graph_id not in {None, "G1"}:
        requirements.append(("ppr-cv", "G1"))
    elif stage == "lightgcn-screen" and graph_id not in {None, "G1"}:
        requirements.append(("lightgcn-screen", "G1"))
    elif stage == "lightgcn-shortlist":
        requirements.extend(("lightgcn-screen", graph) for graph in _graph_ids(payload, None))
    elif stage == "lightgcn-tune":
        _shortlisted_graph_ids(payload, shortlist_file)
        if graph_id not in {None, "G1"}:
            requirements.append(("lightgcn-tune", "G1"))
    elif stage == "lightgcn-seeds":
        for graph in _graph_ids(payload, graph_id) if graph_id else _shortlisted_graph_ids(payload, shortlist_file):
            requirements.append(("lightgcn-tune", graph))
    elif stage == "freeze-epochs":
        for graph in _graph_ids(payload, graph_id) if graph_id else _shortlisted_graph_ids(payload, shortlist_file):
            requirements.append(("lightgcn-seeds", graph))
    elif stage == "internal-replay":
        shortlist = set(_shortlisted_graph_ids(payload, shortlist_file))
        graphs = _graph_ids(payload, graph_id)
        for graph in graphs:
            requirements.append(("ppr-cv", graph))
            if graph == "G1":
                requirements.append(("cosine-control-cv", "G1"))
            if graph in shortlist:
                requirements.append(("freeze-epochs", graph))
    elif stage == "diagnostics":
        for graph in _graph_ids(payload, graph_id):
            requirements.append(("internal-replay", graph))
    elif stage == "paper-exports":
        requirements.append(("diagnostics", graph_id))
    return requirements


def enforce_stage_dependencies(
    payload: dict[str, Any], stage: str, graph_id: str | None, shortlist_file: Path
) -> None:
    for dependency_stage, dependency_graph in dependency_requirements(
        payload, stage, graph_id, shortlist_file
    ):
        require_stage_status(payload, dependency_stage, dependency_graph)


def verified_dependency_labels(
    payload: dict[str, Any], stage: str, graph_id: str | None, shortlist_file: Path
) -> list[str]:
    """Describe the dependencies already enforced before a stage was executed."""
    return [
        f"{dependency_stage}--{dependency_graph or 'all'}"
        for dependency_stage, dependency_graph in dependency_requirements(
            payload, stage, graph_id, shortlist_file
        )
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--graph-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--authorize-internal-eval")
    parser.add_argument("--recover-stale-locks", action="store_true")
    args = parser.parse_args(argv)
    if args.stage != "preflight":
        validate_stage_invocation(args.stage, args.graph_id)

    payload = load_manifest(args.manifest)
    report = preflight(
        payload,
        verify_hashes=True,
        resource_stage=None if args.stage == "preflight" else args.stage,
    )
    if args.recover_stale_locks:
        report["stale_lock_recovery"] = recover_stale_locks(
            _resolve(payload["outputs"]["status_root"]) / "locks"
        )
    if args.stage == "preflight":
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 2
    resolved_manifest = args.manifest.resolve()
    shortlist_file = campaign_shortlist_path(payload)
    artifacts = expected_artifacts(
        payload,
        args.stage,
        graph_id=args.graph_id,
        shortlist_file=shortlist_file,
    )
    status_root = _resolve(payload["outputs"]["status_root"])
    status_name = f"{args.stage}--{args.graph_id or 'all'}.json"
    status_path = status_root / status_name
    if args.check_only:
        expected_paths = {str(path) for path in artifacts}
        if not status_path.is_file():
            check = {"ok": False, "reason": "missing_status", "status_path": str(status_path)}
        else:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            actual_paths = {str(row.get("path")) for row in status.get("artifacts", [])}
            check = {
                "ok": can_resume(status, manifest_sha256(payload), expected_paths),
                "reason": "verified" if can_resume(status, manifest_sha256(payload), expected_paths) else "status_manifest_or_artifact_mismatch",
                "status_path": str(status_path),
            }
        print(json.dumps({**report, "stage": args.stage, "check": check, "artifacts": sorted(expected_paths)}, ensure_ascii=False, indent=2))
        return 0 if check["ok"] else 1
    expected_manifest_hash = manifest_sha256(payload)
    if args.resume and status_path.is_file():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        if can_resume(previous, expected_manifest_hash, {str(path) for path in artifacts}):
            print(json.dumps({"stage": args.stage, "status": "skipped_verified_complete", "status_path": str(status_path)}, ensure_ascii=False, indent=2))
            return 0
    lock_stack = ExitStack()
    if not args.dry_run:
        if not report["resource_assessment"]["compatible"]:
            raise RuntimeError(
                "local resources do not satisfy one graph job: "
                + ",".join(report["resource_assessment"]["insufficient"])
            )
        atexit.register(lock_stack.close)
        lock_path = status_root / "locks" / f"{args.stage}--{args.graph_id or 'all'}.lock"
        lock_stack.enter_context(
            stage_lock(
                lock_path,
                max_parallel_jobs=int(report["resource_assessment"]["max_safe_parallel_jobs"]),
            )
        )
        if args.stage in {"internal-replay", "diagnostics", "paper-exports"}:
            validate_internal_eval_authorization(payload, args.authorize_internal_eval)
        enforce_stage_dependencies(payload, args.stage, args.graph_id, shortlist_file)
        owned_paths = stage_owned_paths(payload, args.stage, args.graph_id, artifacts)
        if args.resume and any(path.exists() for path in owned_paths):
            quarantine_unverified_artifacts(
                owned_paths,
                quarantine_root=status_root / "quarantine",
                stage=args.stage,
                graph_id=args.graph_id,
            )
        else:
            refuse_unverified_overwrite(owned_paths, status_path=status_path)
    commands = build_stage_commands(
        payload,
        args.stage,
        graph_id=args.graph_id,
        manifest_path=resolved_manifest,
        shortlist_file=shortlist_file,
    )
    if args.dry_run and not commands:
        results = [{"command": [f"postprocess:{args.stage}"], "status": "dry_run"}]
    elif args.stage == "freeze-epochs":
        graphs = _graph_ids(payload, args.graph_id) if args.graph_id else _shortlisted_graph_ids(payload, shortlist_file)
        freeze_lightgcn_champions(payload, graphs)
        results = [{"command": ["freeze-lightgcn-champions"], "status": "complete"}]
    elif args.stage == "lightgcn-seeds" and not commands:
        graphs = _graph_ids(payload, args.graph_id) if args.graph_id else _shortlisted_graph_ids(payload, shortlist_file)
        aggregate_lightgcn_robustness(payload, graphs)
        results = [{"command": ["aggregate-baseline-robustness"], "status": "complete"}]
    elif not commands and artifacts:
        results = [{"command": [], "status": "complete" if all(path.is_file() for path in artifacts) else "failed"}]
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        log_dir = _resolve(payload["outputs"]["log_root"]) / args.stage / (args.graph_id or "all") / run_id
        results = execute_commands(commands, dry_run=args.dry_run, log_dir=None if args.dry_run else log_dir)
        if args.stage == "lightgcn-seeds" and all(result["status"] == "complete" for result in results):
            graphs = _graph_ids(payload, args.graph_id) if args.graph_id else _shortlisted_graph_ids(payload, shortlist_file)
            aggregate_lightgcn_robustness(payload, graphs)
    stage_started_at = results[0].get("started_at", utc_now()) if results else utc_now()
    stage_finished_at = results[-1].get("finished_at", utc_now()) if results else utc_now()
    response = {"preflight": report, "stage": args.stage, "results": results}
    final_status = None
    if not args.dry_run:
        response["status_record"] = write_stage_status(
            status_path,
            stage=args.stage,
            manifest_hash=expected_manifest_hash,
            results=results,
            artifacts=artifacts,
            dependencies=verified_dependency_labels(payload, args.stage, args.graph_id, shortlist_file),
            started_at=stage_started_at,
            finished_at=stage_finished_at,
        )
        final_status = response["status_record"]["status"]
    print(json.dumps(response, ensure_ascii=False, indent=2))
    lock_stack.close()
    if not args.dry_run:
        atexit.unregister(lock_stack.close)
    commands_ok = bool(results) and all(result["status"] in {"complete", "dry_run"} for result in results)
    return 0 if commands_ok and (args.dry_run or final_status == "complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
