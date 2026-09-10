"""Create the immutable A3 snapshot for the representation-backed retrieval universe."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np


REPO = Path(os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4])))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import benchmark_labels  # noqa: E402


SNAPSHOT_SCHEMA = "effective-retrieval-universe-freeze.v1"
LOCAL_MANIFEST_FILENAME = "effective_retrieval_universe_freeze_manifest.json"
COPY_FILENAMES = ("bench_global.json", "questions_ids.npy", "questions_emb.npy", "stats.json")


def ensure_output_does_not_exist(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite immutable A3 snapshot artifact: {path}")


def _load_identifiers(path: Path) -> list[str]:
    return [str(value) for value in np.load(path, allow_pickle=True).tolist()]


def _relative_or_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return path.name


def _identifier_metadata(path: Path, values: Iterable[object]) -> dict[str, object]:
    values_list = [str(value) for value in values]
    unique = benchmark_labels.stable_unique_strings(values_list)
    return {
        "source_path": _relative_or_name(path),
        "source_sha256": benchmark_labels.sha256_file(path),
        "raw_rows": len(values_list),
        "unique_ids": len(unique),
        "duplicate_rows": len(values_list) - len(unique),
        "stable_unique_sequence_sha256": benchmark_labels.stable_sequence_sha256(values_list),
    }


def _auxiliary_metadata(graph_values: Iterable[object], retrieval_values: Iterable[object]) -> dict[str, object]:
    graph_unique = benchmark_labels.stable_unique_strings(graph_values)
    retrieval_set = set(benchmark_labels.stable_unique_strings(retrieval_values))
    auxiliary = [identifier for identifier in graph_unique if identifier not in retrieval_set]
    return {
        "unique_ids": len(auxiliary),
        "stable_unique_sequence_sha256": benchmark_labels.stable_sequence_sha256(auxiliary),
        "returnability": "auxiliary_non_returnable",
    }


def _effective_retrieval_metadata(
    path: Path,
    representation_values: list[str],
    effective_values: list[str],
) -> dict[str, object]:
    representation_unique = benchmark_labels.stable_unique_strings(representation_values)
    effective_unique = benchmark_labels.stable_unique_strings(effective_values)
    return {
        "source_path": _relative_or_name(path),
        "source_sha256": benchmark_labels.sha256_file(path),
        "representation_rows": len(representation_values),
        "representation_unique_ids": len(representation_unique),
        "unique_ids": len(effective_unique),
        "excluded_representation_only_ids": len(representation_unique) - len(effective_unique),
        "stable_unique_sequence_sha256": benchmark_labels.stable_sequence_sha256(effective_unique),
    }


def _strict_coverage_counts(
    bench_dir: Path,
    *,
    article_ids: list[str],
    jp_ids: list[str],
) -> dict[str, int]:
    questions = json.loads((bench_dir / "bench_global.json").read_text(encoding="utf-8"))["questions"]
    issues = benchmark_labels.strict_candidate_coverage_issues(
        questions,
        article_candidate_ids=article_ids,
        jp_candidate_ids=jp_ids,
    )
    return {
        "articles_absent": sum(bool(issue["missing_articles_attendus"]) for issue in issues),
        "jurisprudence_absent": sum(bool(issue["missing_gold_jp_ids"]) for issue in issues),
    }


def _copy_bench_inputs(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for filename in COPY_FILENAMES:
        origin = source / filename
        if filename == "stats.json" and not origin.exists():
            continue
        if not origin.is_file():
            raise FileNotFoundError(f"Missing frozen benchmark input: {origin}")
        shutil.copy2(origin, destination / filename)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_snapshot_manifest(
    *,
    source_train_bench_dir: Path,
    evaluation_bench_dir: Path,
    output_train_bench_dir: Path,
    article_order_path: Path,
    jp_order_path: Path,
    graph_article_ids_path: Path,
    graph_jp_ids_path: Path,
    article_order_values: list[str],
    jp_order_values: list[str],
    article_ids: list[str],
    jp_ids: list[str],
    graph_article_ids: list[str],
    graph_jp_ids: list[str],
) -> dict[str, object]:
    train_coverage = _strict_coverage_counts(
        output_train_bench_dir, article_ids=article_ids, jp_ids=jp_ids
    )
    evaluation_coverage = _strict_coverage_counts(
        evaluation_bench_dir, article_ids=article_ids, jp_ids=jp_ids
    )
    if any(train_coverage.values()) or any(evaluation_coverage.values()):
        raise ValueError(
            "A3 strict labels are not covered by the effective retrieval candidate universe: "
            f"train={train_coverage}, evaluation={evaluation_coverage}"
        )
    projection_path = output_train_bench_dir / benchmark_labels.LIGHTGCN_PROJECTION_FILENAME
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "source_snapshot": {
            "train_bench": {
                "path": _relative_or_name(source_train_bench_dir / "bench_global.json"),
                "sha256": benchmark_labels.sha256_file(source_train_bench_dir / "bench_global.json"),
            },
            "evaluation_bench": {
                "path": _relative_or_name(evaluation_bench_dir / "bench_global.json"),
                "sha256": benchmark_labels.sha256_file(evaluation_bench_dir / "bench_global.json"),
            },
        },
        "artifacts": {
            filename: benchmark_labels.sha256_file(output_train_bench_dir / filename)
            for filename in COPY_FILENAMES
            if (output_train_bench_dir / filename).is_file()
        }
        | {
            benchmark_labels.LIGHTGCN_PROJECTION_FILENAME: benchmark_labels.sha256_file(projection_path)
        },
        "graph_node_universe": {
            "articles": _identifier_metadata(graph_article_ids_path, graph_article_ids),
            "jurisprudence": _identifier_metadata(graph_jp_ids_path, graph_jp_ids),
        },
        "retrieval_candidate_universe": {
            "definition": "graph_and_representation_backed_stable_unique_order",
            "ranking_rule": "Only these identifiers may appear in cosine, PPR, or LightGCN rankings.",
            "articles": _effective_retrieval_metadata(
                article_order_path, article_order_values, article_ids
            ),
            "jurisprudence": _effective_retrieval_metadata(
                jp_order_path, jp_order_values, jp_ids
            ),
        },
        "auxiliary_non_returnable_nodes": {
            "definition": "graph nodes without a representation-backed slot in the retrieval candidate universe",
            "articles": _auxiliary_metadata(graph_article_ids, article_ids),
            "jurisprudence": _auxiliary_metadata(graph_jp_ids, jp_ids),
        },
        "strict_label_coverage": {
            "train": train_coverage,
            "evaluation": evaluation_coverage,
        },
        "lightgcn_extended_article_positive_projection": {
            "path": benchmark_labels.LIGHTGCN_PROJECTION_FILENAME,
            "sha256": benchmark_labels.sha256_file(projection_path),
            "counts": json.loads(projection_path.read_text(encoding="utf-8"))["counts"],
        },
        "generator": {
            "path": _relative_or_name(Path(__file__)),
            "sha256": benchmark_labels.sha256_file(Path(__file__)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-train-bench-dir", type=Path, required=True)
    parser.add_argument("--evaluation-bench-dir", type=Path, required=True)
    parser.add_argument("--output-train-bench-dir", type=Path, required=True)
    parser.add_argument("--article-order-path", type=Path, required=True)
    parser.add_argument("--jp-order-path", type=Path, required=True)
    parser.add_argument("--graph-article-ids-path", type=Path, required=True)
    parser.add_argument("--graph-jp-ids-path", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args(argv)

    output = args.output_train_bench_dir.resolve()
    manifest_output = (args.manifest_output or output / LOCAL_MANIFEST_FILENAME).resolve()
    ensure_output_does_not_exist(output)
    ensure_output_does_not_exist(manifest_output)
    if not args.source_train_bench_dir.is_dir():
        raise FileNotFoundError(args.source_train_bench_dir)
    if not args.evaluation_bench_dir.is_dir():
        raise FileNotFoundError(args.evaluation_bench_dir)

    article_order_values = _load_identifiers(args.article_order_path)
    jp_order_values = _load_identifiers(args.jp_order_path)
    graph_article_ids = _load_identifiers(args.graph_article_ids_path)
    graph_jp_ids = _load_identifiers(args.graph_jp_ids_path)
    graph_article_set = set(benchmark_labels.stable_unique_strings(graph_article_ids))
    graph_jp_set = set(benchmark_labels.stable_unique_strings(graph_jp_ids))
    article_ids = [identifier for identifier in article_order_values if identifier in graph_article_set]
    jp_ids = [identifier for identifier in jp_order_values if identifier in graph_jp_set]

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.a3-", dir=output.parent) as temp_root:
        temporary_output = Path(temp_root) / output.name
        _copy_bench_inputs(args.source_train_bench_dir, temporary_output)
        projection = benchmark_labels.write_lightgcn_article_positive_projection(
            temporary_output,
            json.loads((temporary_output / "bench_global.json").read_text(encoding="utf-8"))["questions"],
            article_candidate_ids=article_ids,
        )
        if projection["counts"]["questions_without_retrievable_positive"] != 0:
            raise ValueError("A3 LightGCN projection left training questions without positives")
        snapshot = build_snapshot_manifest(
            source_train_bench_dir=args.source_train_bench_dir,
            evaluation_bench_dir=args.evaluation_bench_dir,
            output_train_bench_dir=temporary_output,
            article_order_path=args.article_order_path,
            jp_order_path=args.jp_order_path,
            graph_article_ids_path=args.graph_article_ids_path,
            graph_jp_ids_path=args.graph_jp_ids_path,
            article_order_values=article_order_values,
            jp_order_values=jp_order_values,
            article_ids=article_ids,
            jp_ids=jp_ids,
            graph_article_ids=graph_article_ids,
            graph_jp_ids=graph_jp_ids,
        )
        _write_json(temporary_output / LOCAL_MANIFEST_FILENAME, snapshot)
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary_output, output)

    if manifest_output != output / LOCAL_MANIFEST_FILENAME:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output / LOCAL_MANIFEST_FILENAME, manifest_output)
    print(json.dumps({
        "output_train_bench_dir": str(output),
        "manifest_output": str(manifest_output),
        "articles": len(benchmark_labels.stable_unique_strings(article_ids)),
        "jurisprudence": len(benchmark_labels.stable_unique_strings(jp_ids)),
        "projection_counts": projection["counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
