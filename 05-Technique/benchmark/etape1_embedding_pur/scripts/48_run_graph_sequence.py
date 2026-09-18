from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PACKAGE_ROOT = SCRIPT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from etape1.paths import resolve_data_root, resolve_repo_root  # noqa: E402

import graph_protocol  # noqa: E402


CODE_REPO = resolve_repo_root(Path(__file__))
DATA_REPO = resolve_data_root(Path(__file__))
REPO = CODE_REPO
LOG_DIR = (
    DATA_REPO
    / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_protocol/run_logs"
)
STATUS_PATH = (
    DATA_REPO
    / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_protocol/graph_sequence_status.json"
)


@dataclass(frozen=True)
class Step:
    graph_version: str
    name: str
    cmd: list[str]
    log_path: Path
    expected_path: Path


def build_steps(graph_version: str, protocol_version: str | None = None) -> list[Step]:
    if protocol_version is None:
        cv_root = graph_protocol.resolve_graph_bench_dir(
            graph_version, graph_protocol.OFFICIAL_TRAIN_SPLIT
        ) / "_cv"
        final_path = (
            graph_protocol.resolve_graph_bench_dir(
                graph_version,
                "eval_rich_retrievable_strict",
            )
            / "_final_champions"
            / "final_champions_summary.csv"
        )
    else:
        cv_root = graph_protocol.cv_root(graph_protocol.BENCH_ROOT, protocol_version) / graph_version
        final_path = (
            graph_protocol.final_root(graph_protocol.BENCH_ROOT, protocol_version)
            / graph_version
            / "final_champions_summary.csv"
        )
    return [
        Step(
            graph_version=graph_version,
            name="b3_b4_cv",
            cmd=[
                sys.executable,
                str(SCRIPT_DIR / "42_run_cv_b3_b4.py"),
                "--graph-version",
                graph_version,
            ],
            log_path=LOG_DIR / f"{graph_version.lower()}_b3_b4_cv.log",
            expected_path=cv_root / "b3_b4" / "champions.json",
        ),
        Step(
            graph_version=graph_version,
            name="ppr_cv",
            cmd=[
                sys.executable,
                str(SCRIPT_DIR / "43_run_cv_ppr.py"),
                "--graph-version",
                graph_version,
            ],
            log_path=LOG_DIR / f"{graph_version.lower()}_ppr_cv.log",
            expected_path=cv_root / "ppr" / "champions.json",
        ),
        Step(
            graph_version=graph_version,
            name="lightgcn_cv",
            cmd=[
                sys.executable,
                str(SCRIPT_DIR / "44_run_cv_lightgcn.py"),
                "--graph-version",
                graph_version,
            ],
            log_path=LOG_DIR / f"{graph_version.lower()}_lightgcn_cv.log",
            expected_path=cv_root / "lightgcn" / "champions.json",
        ),
        Step(
            graph_version=graph_version,
            name="final_champions",
            cmd=[
                sys.executable,
                str(SCRIPT_DIR / "45_run_final_champions.py"),
                "--graph-version",
                graph_version,
                "--cv-root",
                str(cv_root),
            ],
            log_path=LOG_DIR / f"{graph_version.lower()}_final_champions.log",
            expected_path=final_path,
        ),
    ]


def write_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(status, ensure_ascii=False, indent=2))


def serialize_step(step: Step) -> dict:
    payload = asdict(step)
    payload["log_path"] = str(step.log_path)
    payload["expected_path"] = str(step.expected_path)
    return payload


def wait_for_path(target: Path, poll_seconds: float) -> None:
    while not target.exists():
        time.sleep(poll_seconds)


def run_step(step: Step) -> dict:
    if step.expected_path.exists():
        return {
            "graph_version": step.graph_version,
            "name": step.name,
            "cmd": step.cmd,
            "log_path": str(step.log_path),
            "expected_path": str(step.expected_path),
            "returncode": 0,
            "started_at": time.time(),
            "finished_at": time.time(),
            "duration_seconds": 0.0,
            "status": "skipped_existing",
        }
    step.log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "2")
    env.setdefault("MKL_NUM_THREADS", "2")
    env.setdefault("OPENBLAS_NUM_THREADS", "2")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "2")
    env.setdefault("NUMEXPR_NUM_THREADS", "2")
    with step.log_path.open("w") as log_file:
        log_file.write(f"$ {' '.join(step.cmd)}\n\n")
        log_file.write(
            "thread_limits: "
            f"OMP={env['OMP_NUM_THREADS']} "
            f"MKL={env['MKL_NUM_THREADS']} "
            f"OPENBLAS={env['OPENBLAS_NUM_THREADS']} "
            f"VECLIB={env['VECLIB_MAXIMUM_THREADS']} "
            f"NUMEXPR={env['NUMEXPR_NUM_THREADS']}\n\n"
        )
        log_file.flush()
        proc = subprocess.Popen(
            step.cmd,
            cwd=str(REPO),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log_file.write(line)
        returncode = proc.wait()
    finished_at = time.time()
    return {
        "graph_version": step.graph_version,
        "name": step.name,
        "cmd": step.cmd,
        "log_path": str(step.log_path),
        "expected_path": str(step.expected_path),
        "returncode": returncode,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": finished_at - started_at,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", action="append", required=True)
    parser.add_argument("--wait-for", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--protocol-version")
    args = parser.parse_args(argv)

    graphs = args.graph
    status = {
        "graphs": graphs,
        "wait_for": str(args.wait_for) if args.wait_for else None,
        "status": "waiting" if args.wait_for else "ready",
        "protocol_version": args.protocol_version,
        "steps": [],
        "results": [],
    }
    write_status(status)
    if args.wait_for is not None:
        print(f"Waiting for {args.wait_for}")
        wait_for_path(args.wait_for, args.poll_seconds)

    all_steps = []
    for graph_version in graphs:
        all_steps.extend(build_steps(graph_version, args.protocol_version))

    status.update(
        {
            "status": "running",
            "steps": [serialize_step(step) for step in all_steps],
        }
    )
    write_status(status)

    for step in all_steps:
        print(f"\n=== {step.graph_version} :: {step.name} ===")
        result = run_step(step)
        status["results"].append(result)
        write_status(status)
        if result["returncode"] != 0:
            print(f"Step failed: {step.graph_version} / {step.name}", file=sys.stderr)
            return int(result["returncode"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
