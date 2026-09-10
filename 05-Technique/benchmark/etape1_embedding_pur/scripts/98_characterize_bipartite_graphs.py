"""Characterize explicitly named bipartite graph inputs with one shared definition."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csr(path: Path) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    required = {"data", "indices", "indptr", "shape", "jp_ids", "article_ids", "article_codes"}
    missing = sorted(required - set(raw.files))
    if missing:
        raise ValueError(f"{path}: unsupported bipartite NPZ; missing {missing}")
    shape = tuple(int(value) for value in raw["shape"].tolist())
    matrix = sp.csr_matrix((raw["data"], raw["indices"], raw["indptr"]), shape=shape)
    return matrix, raw["jp_ids"].astype(str), raw["article_ids"].astype(str), raw["article_codes"].astype(str)


def _degree_stats(values: np.ndarray) -> dict[str, float | int]:
    return {
        "mean": float(values.mean()) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "p90": float(np.percentile(values, 90)) if values.size else 0.0,
        "p99": float(np.percentile(values, 99)) if values.size else 0.0,
        "max": int(values.max()) if values.size else 0,
    }


def characterize(name: str, definition: str, path: Path) -> tuple[dict, pd.DataFrame]:
    matrix, jp_ids, article_ids, article_codes = load_csr(path)
    n_jp, n_articles = matrix.shape
    if len(jp_ids) != n_jp or len(article_ids) != n_articles or len(article_codes) != n_articles:
        raise ValueError(f"{path}: matrix and identifier arrays have inconsistent dimensions")
    degree_jp = np.asarray(matrix.sum(axis=1)).ravel()
    degree_articles = np.asarray(matrix.sum(axis=0)).ravel()
    adjacency = sp.bmat([[None, matrix], [matrix.T, None]], format="csr")
    n_components, labels = connected_components(adjacency, directed=False)
    component_sizes = np.bincount(labels)
    code_counts = Counter(article_codes.tolist())
    stats = {
        "graph": name,
        "definition": definition,
        "source_path": str(path),
        "source_sha256": sha256(path),
        "matrix_shape": [int(n_jp), int(n_articles)],
        "nodes": {"jurisprudence": int(n_jp), "articles": int(n_articles), "total": int(n_jp + n_articles)},
        "edges_by_type": {"jp_article_citation": int(matrix.nnz)},
        "density_bipartite": float(matrix.nnz / (n_jp * n_articles)) if n_jp and n_articles else 0.0,
        "degrees": {"jurisprudence": _degree_stats(degree_jp), "articles": _degree_stats(degree_articles)},
        "isolated_nodes": {"jurisprudence": int((degree_jp == 0).sum()), "articles": int((degree_articles == 0).sum())},
        "connected_components": {"count": int(n_components), "largest_size": int(component_sizes.max()) if component_sizes.size else 0},
    }
    code_frame = pd.DataFrame({"graph": name, "code": list(code_counts), "article_nodes": list(code_counts.values())})
    return stats, code_frame.sort_values(["graph", "article_nodes", "code"], ascending=[True, False, True])


def write_outputs(manifest: dict, items: list[tuple[str, str, Path]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=False)
    records, codes = [], []
    for name, definition, path in items:
        stats, code_frame = characterize(name, definition, path)
        records.append(stats)
        codes.append(code_frame)
        (out_dir / f"{name}_statistics.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    flat: list[dict] = []
    for stats in records:
        base = {"graph": stats["graph"], "definition": stats["definition"], "source_sha256": stats["source_sha256"]}
        flat.extend([
            {**base, "measure": "jurisprudence_nodes", "value": stats["nodes"]["jurisprudence"]},
            {**base, "measure": "article_nodes", "value": stats["nodes"]["articles"]},
            {**base, "measure": "jp_article_citation_edges", "value": stats["edges_by_type"]["jp_article_citation"]},
            {**base, "measure": "density_bipartite", "value": stats["density_bipartite"]},
            {**base, "measure": "connected_components", "value": stats["connected_components"]["count"]},
            {**base, "measure": "largest_component_size", "value": stats["connected_components"]["largest_size"]},
        ])
        for node_type in ("jurisprudence", "articles"):
            for measure, value in stats["degrees"][node_type].items():
                flat.append({**base, "measure": f"degree_{node_type}_{measure}", "value": value})
            flat.append({**base, "measure": f"isolated_{node_type}", "value": stats["isolated_nodes"][node_type]})
    comparison = pd.DataFrame(flat)
    comparison_path = out_dir / "graph_comparison.csv"
    codes_path = out_dir / "article_code_distribution.csv"
    comparison.to_csv(comparison_path, index=False)
    pd.concat(codes, ignore_index=True).to_csv(codes_path, index=False)
    (out_dir / "graph_characterization_manifest.json").write_text(json.dumps({
        "campaign_id": manifest["campaign_id"],
        "shared_definition": "CSR JP x Article; one non-zero is one JP-to-Article citation; components are computed on the undirected bipartite projection.",
        "graphs": records,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot(comparison, out_dir)
    return {"comparison": comparison_path, "code_distribution": codes_path}


def _plot(comparison: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    wanted = comparison.loc[comparison["measure"].isin(["jurisprudence_nodes", "article_nodes", "jp_article_citation_edges"])]
    pivot = wanted.pivot(index="measure", columns="graph", values="value")
    ax = pivot.plot(kind="bar", figsize=(8, 4.2), logy=True)
    ax.set_ylabel("count (log scale)")
    ax.set_xlabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "graph_composition.png", dpi=200)
    ax.figure.savefig(out_dir / "graph_composition.pdf")
    plt.close(ax.figure)


def _parse_pair(raw: str) -> tuple[str, str, Path]:
    name, sep, rest = raw.partition("=")
    definition, sep2, path = rest.partition("::")
    if not sep or not sep2 or not name or not definition or not path:
        raise ValueError("--graph must be NAME=EXPLICIT_DEFINITION::PATH")
    return name, definition, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--graph", action="append", required=True, metavar="NAME=DEFINITION::NPZ")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    items = [_parse_pair(raw) for raw in args.graph]
    if len({name for name, _, _ in items}) != len(items):
        parser.error("duplicate graph name")
    outputs = write_outputs(manifest, items, args.out_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
