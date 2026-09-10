from __future__ import annotations

import argparse
import json
import shutil
import sys
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


OFFICIAL_SPLITS = [
    graph_protocol.OFFICIAL_TRAIN_SPLIT,
    "eval_rich_retrievable_strict",
]

DEFAULT_COVERAGE_SUMMARY = (
    CODE_REPO / "01-Projet/specs/graph-g0-g1-g2-g3-dataset-coverage-2026-06-22/summary.json"
)
DEFAULT_GRAPH_VERSIONS = [
    "G0",
    "G1",
    "G2",
    "G3",
    "G4-knn5",
    "G4-knn10",
    "G4-knn20",
    "G4-knn30",
    "G4-knn50",
    "G5-citation-knn5",
    "G5-citation-knn10",
    "G6-citation-AA-knn5",
    "G6-citation-JJ-knn5",
    "G6-citation-AJ-knn5",
    "G6-citation-AA-JJ-knn5",
    "G6-citation-AA-JJ-AJ-knn5",
    "G6U-citation-AA-knn5",
    "G6U-citation-JJ-knn5",
    "G6U-citation-AJ-knn5",
    "G6U-citation-AA-JJ-knn5",
    "G6U-citation-AA-JJ-AJ-knn5",
    "G7-citation-AA-cit1-sem025-knn5",
    "G7-citation-AA-cit1-sem050-knn5",
    "G7-citation-AA-cit1-sem100-knn5",
    "G7-citation-AA-cit025-sem1-knn5",
    "G7-citation-JJ-cit1-sem025-knn5",
    "G7-citation-JJ-cit1-sem050-knn5",
    "G7-citation-JJ-cit1-sem100-knn5",
    "G7-citation-JJ-cit025-sem1-knn5",
]


def _canonical_source_split_dir(split: str) -> Path:
    return graph_protocol.BENCH_ROOT / split


def _copy_if_needed(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            return
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.suffix == ".json":
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        shutil.copy2(src, dst)


def materialize_graph_split(
    graph_version: str,
    split: str,
    coverage_summary: dict | None,
    *,
    force: bool = False,
) -> Path:
    src_dir = _canonical_source_split_dir(split)
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing canonical split dir: {src_dir}")
    dst_dir = graph_protocol.BENCH_ROOT / graph_version / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["bench_global.json", "stats.json", "questions_ids.npy", "questions_emb.npy"]:
        src = src_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing source artifact: {src}")
        _copy_if_needed(src, dst_dir / filename, force)
    if coverage_summary is not None:
        graph_key = (
            "g1"
            if graph_version.lower().startswith(
                ("g4-knn", "g5-citation-knn", "g6-citation-", "g6u-citation-", "g7-citation-")
            )
            else graph_version.lower()
        )
        split_payload = coverage_summary.get("datasets", {}).get(split, {})
        coverage_source_graph = (
            "G1"
            if graph_version.lower().startswith(
                ("g4-knn", "g5-citation-knn", "g6-citation-", "g6u-citation-", "g7-citation-")
            )
            else graph_version
        )
        local_payload = {
            "graph_version": graph_version,
            "split": split,
            "coverage_source_graph": coverage_source_graph,
            "coverage": split_payload.get(graph_key, {}),
        }
        (dst_dir / "coverage_summary.json").write_text(
            json.dumps(local_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return dst_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-version",
        action="append",
        dest="graph_versions",
        help="Graph version(s) to materialize. Repeatable. Defaults to G0,G1,G2,G3.",
    )
    parser.add_argument(
        "--coverage-summary",
        type=Path,
        default=DEFAULT_COVERAGE_SUMMARY,
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graph_versions = args.graph_versions or DEFAULT_GRAPH_VERSIONS
    coverage_summary = (
        json.loads(args.coverage_summary.read_text(encoding="utf-8"))
        if args.coverage_summary.exists()
        else None
    )
    for graph_version in graph_versions:
        for split in OFFICIAL_SPLITS:
            out_dir = materialize_graph_split(
                graph_version,
                split,
                coverage_summary,
                force=args.force,
            )
            print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
