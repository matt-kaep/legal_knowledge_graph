#!/usr/bin/env python3
"""Export one canonical structural table for G1, G6-AA and G7-AA under A3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etape1 import config, graph_versions  # noqa: E402


RETURNABLE_ARTICLES = 13236
RETURNABLE_JP = 114851
GRAPHS = ("G1", "G6-citation-AA-knn5", "G7-citation-AA-cit1-sem025-knn5")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _degree_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "p99": float(np.percentile(values, 99)) if values.size else 0.0,
        "max": int(values.max()) if values.size else 0,
    }


def _square_adjacency(graph: sp.csr_matrix, n_jp: int, n_articles: int) -> sp.csr_matrix:
    if graph.shape == (n_jp, n_articles):
        return sp.bmat([[None, graph], [graph.T, None]], format="csr")
    if graph.shape[0] != graph.shape[1]:
        raise ValueError("structural graph must be square or JP-by-Article bipartite")
    return graph.tocsr()


def structural_record(
    *,
    graph_version: str,
    graph: sp.csr_matrix,
    jp_ids: np.ndarray,
    article_ids: np.ndarray,
    edges_by_type: dict[str, int],
    normalization: str,
    active_weights: dict[str, float],
) -> dict[str, Any]:
    """Calculate graph-wide quantities while retaining raw JP rows and unique IDs."""
    adjacency = _square_adjacency(graph, len(jp_ids), len(article_ids))
    adjacency = (adjacency != 0).astype(np.int8).tocsr()
    n_nodes = adjacency.shape[0]
    components, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    degrees = np.asarray(adjacency.sum(axis=1)).ravel()
    edge_total = int(sum(edges_by_type.values()))
    if adjacency.nnz != edge_total * 2:
        raise ValueError("typed undirected edge count does not match adjacency")
    return {
        "graph": graph_version,
        "jp_rows_raw": int(len(jp_ids)),
        "jp_decisions_unique": int(len(set(jp_ids.astype(str).tolist()))),
        "article_nodes": int(len(article_ids)),
        "returnable_article_candidates": RETURNABLE_ARTICLES,
        "returnable_jp_candidates": RETURNABLE_JP,
        "edges_by_type": {key: int(value) for key, value in edges_by_type.items()},
        "total_undirected_edges": edge_total,
        "total_nodes": int(n_nodes),
        "density_undirected": float((2 * edge_total) / (n_nodes * (n_nodes - 1))) if n_nodes > 1 else 0.0,
        "connected_components": int(components),
        "largest_component_nodes": int(component_sizes.max()) if component_sizes.size else 0,
        "largest_component_fraction": float(component_sizes.max() / n_nodes) if n_nodes else 0.0,
        "degree": _degree_stats(degrees),
        "isolated_nodes": int((degrees == 0).sum()),
        "normalization": normalization,
        "active_weights": active_weights,
    }


def _hybrid_metadata(graph_version: str) -> tuple[dict[str, Any], Path]:
    path = config.DATA / "hybrid_graphs" / graph_version / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8")), path


def _graph_definition(graph_version: str, graph: sp.csr_matrix) -> tuple[dict[str, int], str, dict[str, float], dict[str, str]]:
    if graph_version == "G1":
        return (
            {"citation_article_jp": int(graph.nnz)},
            "none_binary_citation",
            {"citation": 1.0},
            {"graph_penal.npz": _sha256(config.GRAPH_NPZ), "legi.sqlite": _sha256(config.LEGI_SQLITE)},
        )
    metadata, metadata_path = _hybrid_metadata(graph_version)
    typed = metadata["edge_type_counts_hybrid_undirected"]
    edges = {
        "citation_article_jp": int(typed["article-jp"]),
        "semantic_article_article_AA": int(typed["article-article"]),
    }
    source_path = config.DATA / "hybrid_graphs" / graph_version / "graph_hybrid_mixed.npz"
    return edges, str(metadata["normalization"]), {key: float(value) for key, value in metadata["active_block_weights"].items()} | {"citation": float(metadata["lambda_cit"])}, {"graph_hybrid_mixed.npz": _sha256(source_path), "metadata.json": _sha256(metadata_path)}


def characterize(graph_version: str) -> dict[str, Any]:
    variant = graph_versions.load_graph_variant(graph_version)
    edges, normalization, weights, sources = _graph_definition(graph_version, variant.graph)
    record = structural_record(
        graph_version=graph_version,
        graph=variant.graph,
        jp_ids=variant.jp_ids,
        article_ids=variant.article_ids,
        edges_by_type=edges,
        normalization=normalization,
        active_weights=weights,
    )
    record["source_sha256"] = sources
    return record


def write_outputs(out_dir: Path) -> dict[str, Path]:
    if out_dir.exists():
        raise FileExistsError("refusing to overwrite structural characterization")
    out_dir.mkdir(parents=True, exist_ok=False)
    records = [characterize(graph) for graph in GRAPHS]
    json_path = out_dir / "a3_structural_graphs.json"
    csv_path = out_dir / "a3_structural_graphs.csv"
    json_path.write_text(json.dumps({"schema_version": "a3-structural-characterization.v1", "graphs": records}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    flat = []
    for record in records:
        flat.append({
            "graph": record["graph"],
            "jp_rows_raw": record["jp_rows_raw"],
            "jp_decisions_unique": record["jp_decisions_unique"],
            "article_nodes": record["article_nodes"],
            "returnable_article_candidates": record["returnable_article_candidates"],
            "returnable_jp_candidates": record["returnable_jp_candidates"],
            "edges_by_type_json": json.dumps(record["edges_by_type"], sort_keys=True),
            "total_undirected_edges": record["total_undirected_edges"],
            "connected_components": record["connected_components"],
            "largest_component_fraction": record["largest_component_fraction"],
            "degree_mean": record["degree"]["mean"],
            "degree_median": record["degree"]["median"],
            "degree_p90": record["degree"]["p90"],
            "degree_p99": record["degree"]["p99"],
            "degree_max": record["degree"]["max"],
            "normalization": record["normalization"],
            "active_weights_json": json.dumps(record["active_weights"], sort_keys=True),
            "source_sha256_json": json.dumps(record["source_sha256"], sort_keys=True),
        })
    pd.DataFrame(flat).to_csv(csv_path, index=False)
    manifest_path = out_dir / "a3_structural_graphs_manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": "a3-structural-characterization-receipt.v1",
        "a3_manifest_sha256": "c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3",
        "script_sha256": _sha256(Path(__file__).resolve()),
        "files": {"a3_structural_graphs.json": _sha256(json_path), "a3_structural_graphs.csv": _sha256(csv_path)},
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "manifest": manifest_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    outputs = write_outputs(args.out_dir if args.out_dir.is_absolute() else DATA_REPO / args.out_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
