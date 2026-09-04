from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import config
from .normalize import legi_num_candidates, parse_pair_key


G3_REMOVE_PAIR_KEYS = {
    "code_de_procedure_penale:592",
    "code_de_procedure_penale:584",
    "code_de_procedure_penale:4",
    "code_de_procedure_penale:préliminaire",
    "code_de_procedure_civile:699",
    "code_de_procedure_penale:6",
    "code_de_procedure_civile:1082",
    "code_de_procedure_civile:805",
    "code_de_procedure_penale:513",
    "code_de_procedure_penale:590",
    "code_de_procedure_penale:606",
    "code_de_procedure_penale:385",
    "code_de_procedure_penale:388",
    "code_de_procedure_penale:802",
    "code_de_procedure_penale:197",
}

EMBEDDING_GRAPHS_ROOT = config.DATA / "embedding_graphs"
HYBRID_GRAPHS_ROOT = config.DATA / "hybrid_graphs"
G4_KNNS = {5, 10, 20, 30, 50}
G5_KNNS = {5, 10}
G6_KNNS = {5, 10}
G6U_KNNS = {5, 10}
G7_KNNS = {5}
G6_BLOCKS = {"AA", "JJ", "AJ"}
G6_BLOCK_ORDER = ("AA", "JJ", "AJ")
G7_WEIGHT_LABELS = {"cit1-sem025", "cit1-sem050", "cit1-sem100", "cit025-sem1"}


@dataclass(frozen=True)
class GraphVariant:
    graph_version: str
    graph: sp.csr_matrix
    jp_ids: np.ndarray
    article_ids: np.ndarray
    article_codes: np.ndarray

    @property
    def jp_index_by_id(self) -> dict[str, int]:
        return {str(jid): idx for idx, jid in enumerate(self.jp_ids.tolist())}

    @property
    def article_index_by_key(self) -> dict[str, int]:
        return {str(pk): idx for idx, pk in enumerate(self.article_ids.tolist())}


@dataclass(frozen=True)
class RetrievalView:
    graph_version: str
    graph: sp.csr_matrix
    jp_ids_graph: np.ndarray
    article_ids_graph: np.ndarray
    article_codes_graph: np.ndarray
    art_emb: np.ndarray
    art_order: np.ndarray
    p2col: np.ndarray
    jp_emb: np.ndarray
    jp_order: np.ndarray
    jp_to_row: np.ndarray


def canonical_graph_version(graph_version: str) -> str:
    raw = str(graph_version or "G0").strip()
    lowered = raw.lower()
    if lowered in {"g0", "canonical"}:
        return "G0"
    if lowered in {"g1", "g2", "g3"}:
        return lowered.upper()
    if lowered.startswith("g4-knn"):
        try:
            knn = int(lowered.split("g4-knn", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Unsupported graph_version={graph_version!r}") from exc
        if knn in G4_KNNS:
            return f"G4-knn{knn}"
    if lowered.startswith("g5-citation-knn"):
        try:
            knn = int(lowered.split("g5-citation-knn", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Unsupported graph_version={graph_version!r}") from exc
        if knn in G5_KNNS:
            return f"G5-citation-knn{knn}"
    if lowered.startswith(("g6-citation-", "g6u-citation-")):
        is_uniform = lowered.startswith("g6u-citation-")
        prefix = "g6u-citation-" if is_uniform else "g6-citation-"
        body = lowered[len(prefix) :]
        if "-knn" not in body:
            raise ValueError(f"Unsupported graph_version={graph_version!r}")
        block_raw, knn_raw = body.rsplit("-knn", 1)
        try:
            knn = int(knn_raw)
        except ValueError as exc:
            raise ValueError(f"Unsupported graph_version={graph_version!r}") from exc
        blocks = tuple(part.upper() for part in block_raw.split("-") if part)
        allowed_knns = G6U_KNNS if is_uniform else G6_KNNS
        if knn in allowed_knns and blocks and set(blocks).issubset(G6_BLOCKS):
            ordered_blocks = tuple(block for block in G6_BLOCK_ORDER if block in blocks)
            family = "G6U" if is_uniform else "G6"
            return f"{family}-citation-{'-'.join(ordered_blocks)}-knn{knn}"
    if lowered.startswith("g7-citation-"):
        body = lowered.removeprefix("g7-citation-")
        if "-knn" not in body or "-cit" not in body:
            raise ValueError(f"Unsupported graph_version={graph_version!r}")
        before_knn, knn_raw = body.rsplit("-knn", 1)
        block_raw, weight_suffix = before_knn.rsplit("-cit", 1)
        try:
            knn = int(knn_raw)
        except ValueError as exc:
            raise ValueError(f"Unsupported graph_version={graph_version!r}") from exc
        blocks = tuple(part.upper() for part in block_raw.split("-") if part)
        weight_label = f"cit{weight_suffix}"
        if (
            knn in G7_KNNS
            and blocks
            and set(blocks).issubset(G6_BLOCKS)
            and weight_label in G7_WEIGHT_LABELS
        ):
            ordered_blocks = tuple(block for block in G6_BLOCK_ORDER if block in blocks)
            return f"G7-citation-{'-'.join(ordered_blocks)}-{weight_label}-knn{knn}"
    raise ValueError(f"Unsupported graph_version={graph_version!r}")


def is_g4_graph_version(graph_version: str) -> bool:
    return canonical_graph_version(graph_version).startswith("G4-knn")


def is_g5_graph_version(graph_version: str) -> bool:
    return canonical_graph_version(graph_version).startswith("G5-citation-knn")


def is_g6_graph_version(graph_version: str) -> bool:
    return canonical_graph_version(graph_version).startswith("G6-citation-")


def is_g6u_graph_version(graph_version: str) -> bool:
    return canonical_graph_version(graph_version).startswith("G6U-citation-")


def load_sparse_npz_manual(path: Path) -> sp.csr_matrix:
    z = np.load(path, allow_pickle=True)
    return sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]),
        shape=tuple(z["shape"].tolist()),
    )


def load_g4_embedding_graph(canonical: str) -> GraphVariant:
    graph_dir = EMBEDDING_GRAPHS_ROOT / canonical
    graph_path = graph_dir / "graph_embedding_mixed.npz"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Missing G4 embedding graph artifact for {canonical}: {graph_path}. "
            "Run scripts/54_build_g4_embedding_graphs.py first."
        )
    graph = load_sparse_npz_manual(graph_path)
    return GraphVariant(
        canonical,
        graph,
        np.load(graph_dir / "jp_ids.npy", allow_pickle=True).astype(str),
        np.load(graph_dir / "article_ids.npy", allow_pickle=True).astype(str),
        np.load(graph_dir / "article_codes.npy", allow_pickle=True).astype(str),
    )


def load_g5_hybrid_graph(canonical: str) -> GraphVariant:
    graph_dir = HYBRID_GRAPHS_ROOT / canonical
    graph_path = graph_dir / "graph_hybrid_mixed.npz"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Missing G5 hybrid graph artifact for {canonical}: {graph_path}. "
            "Run scripts/56_build_g5_hybrid_graphs.py first."
        )
    graph = load_sparse_npz_manual(graph_path)
    return GraphVariant(
        canonical,
        graph,
        np.load(graph_dir / "jp_ids.npy", allow_pickle=True).astype(str),
        np.load(graph_dir / "article_ids.npy", allow_pickle=True).astype(str),
        np.load(graph_dir / "article_codes.npy", allow_pickle=True).astype(str),
    )


def load_g6_typed_hybrid_graph(canonical: str) -> GraphVariant:
    graph_dir = HYBRID_GRAPHS_ROOT / canonical
    graph_path = graph_dir / "graph_hybrid_mixed.npz"
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Missing G6 typed hybrid graph artifact for {canonical}: {graph_path}. "
            "Run scripts/58_build_g6_typed_hybrid_graphs.py first."
        )
    graph = load_sparse_npz_manual(graph_path)
    return GraphVariant(
        canonical,
        graph,
        np.load(graph_dir / "jp_ids.npy", allow_pickle=True).astype(str),
        np.load(graph_dir / "article_ids.npy", allow_pickle=True).astype(str),
        np.load(graph_dir / "article_codes.npy", allow_pickle=True).astype(str),
    )


def load_canonical_graph_arrays() -> tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    graph = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return (
        graph,
        z["jp_ids"].astype(str),
        z["article_ids"].astype(str),
        z["article_codes"].astype(str),
    )


def _build_g1_article_keep_mask(article_ids: np.ndarray, deg_art: np.ndarray) -> np.ndarray:
    isolated_mask = deg_art == 0
    keep_mask = np.ones(article_ids.shape[0], dtype=bool)
    by_slug: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx, pair_key in enumerate(article_ids.tolist()):
        slug, compact = parse_pair_key(str(pair_key))
        by_slug[slug].append((idx, compact))

    with sqlite3.connect(config.LEGI_SQLITE) as cx:
        for slug, items in by_slug.items():
            titre = config.ALL_CODES.get(slug)
            if titre is None:
                for idx, _compact in items:
                    keep_mask[idx] = False
                continue
            cand_to_idx: dict[str, list[int]] = defaultdict(list)
            for idx, compact in items:
                for cand in legi_num_candidates(compact):
                    cand_to_idx[cand].append(idx)
            seen_vigueur: set[int] = set()
            seen_any: set[int] = set()
            all_cands = list(cand_to_idx.keys())
            for start in range(0, len(all_cands), 400):
                chunk = all_cands[start : start + 400]
                placeholders = ",".join("?" for _ in chunk)
                sql = f"""
                SELECT a.num, a.etat
                FROM articles a
                JOIN textes_versions tv ON tv.id = a.cid
                WHERE tv.titre = ? AND a.num IN ({placeholders})
                """
                for num, etat in cx.execute(sql, [titre, *chunk]).fetchall():
                    idxs = cand_to_idx.get(str(num), [])
                    seen_any.update(idxs)
                    if (etat or "") == "VIGUEUR":
                        seen_vigueur.update(idxs)
            for idx, _compact in items:
                if idx in (seen_any - seen_vigueur):
                    keep_mask[idx] = False
    return ~(isolated_mask | (~keep_mask))


def _remove_cols_and_drop_isolated_rows(
    graph: sp.csr_matrix,
    keep_cols: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    graph_pre = graph[:, keep_cols]
    keep_rows = np.asarray(graph_pre.sum(axis=1)).ravel().astype(int) > 0
    return graph_pre[keep_rows], keep_rows


@lru_cache(maxsize=None)
def load_graph_variant(graph_version: str) -> GraphVariant:
    canonical = canonical_graph_version(graph_version)
    if canonical.startswith("G4-knn"):
        return load_g4_embedding_graph(canonical)
    if canonical.startswith("G5-citation-knn"):
        return load_g5_hybrid_graph(canonical)
    if canonical.startswith(("G6-citation-", "G6U-citation-", "G7-citation-")):
        return load_g6_typed_hybrid_graph(canonical)
    graph_g0, jp_ids_g0, article_ids_g0, article_codes_g0 = load_canonical_graph_arrays()
    if canonical == "G0":
        return GraphVariant(canonical, graph_g0, jp_ids_g0, article_ids_g0, article_codes_g0)

    deg_g0 = np.asarray(graph_g0.sum(axis=0)).ravel().astype(int)
    keep_cols_g1 = _build_g1_article_keep_mask(article_ids_g0, deg_g0)
    graph_g1_pre = graph_g0[:, keep_cols_g1]
    keep_rows_g1 = np.asarray(graph_g1_pre.sum(axis=1)).ravel().astype(int) > 0
    graph_g1 = graph_g1_pre[keep_rows_g1]
    jp_ids_g1 = jp_ids_g0[keep_rows_g1]
    article_ids_g1 = article_ids_g0[keep_cols_g1]
    article_codes_g1 = article_codes_g0[keep_cols_g1]
    if canonical == "G1":
        return GraphVariant(canonical, graph_g1, jp_ids_g1, article_ids_g1, article_codes_g1)

    deg_g1 = np.asarray(graph_g1.sum(axis=0)).ravel().astype(int)
    top14_local = np.argsort(-deg_g1)[:14]
    keep_cols_g2 = np.ones(article_ids_g1.shape[0], dtype=bool)
    keep_cols_g2[top14_local] = False
    graph_g2, keep_rows_g2 = _remove_cols_and_drop_isolated_rows(graph_g1, keep_cols_g2)
    jp_ids_g2 = jp_ids_g1[keep_rows_g2]
    article_ids_g2 = article_ids_g1[keep_cols_g2]
    article_codes_g2 = article_codes_g1[keep_cols_g2]
    if canonical == "G2":
        return GraphVariant(canonical, graph_g2, jp_ids_g2, article_ids_g2, article_codes_g2)

    keep_cols_g3 = ~np.isin(article_ids_g2, np.array(sorted(G3_REMOVE_PAIR_KEYS), dtype=object))
    graph_g3, keep_rows_g3 = _remove_cols_and_drop_isolated_rows(graph_g2, keep_cols_g3)
    jp_ids_g3 = jp_ids_g2[keep_rows_g3]
    article_ids_g3 = article_ids_g2[keep_cols_g3]
    article_codes_g3 = article_codes_g2[keep_cols_g3]
    return GraphVariant(canonical, graph_g3, jp_ids_g3, article_ids_g3, article_codes_g3)


@lru_cache(maxsize=None)
def load_retrieval_view(graph_version: str) -> RetrievalView:
    variant = load_graph_variant(graph_version)
    art_emb_full = np.load(config.EMB_ARTICLES_ALL).astype(np.float32)
    art_order_full = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True).astype(str)
    jp_emb_full = np.load(config.EMB_JP_SYNTHESE).astype(np.float32)
    jp_order_full = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True).astype(str)

    art_index = variant.article_index_by_key
    jp_index = variant.jp_index_by_id

    art_keep = np.array([pk in art_index for pk in art_order_full.tolist()], dtype=bool)
    jp_keep = np.array([jid in jp_index for jid in jp_order_full.tolist()], dtype=bool)

    art_order = art_order_full[art_keep]
    art_emb = art_emb_full[art_keep]
    p2col = np.asarray([art_index[str(pk)] for pk in art_order.tolist()], dtype=np.int64)

    jp_order = jp_order_full[jp_keep]
    jp_emb = jp_emb_full[jp_keep]
    jp_to_row = np.asarray([jp_index[str(jid)] for jid in jp_order.tolist()], dtype=np.int64)

    return RetrievalView(
        graph_version=variant.graph_version,
        graph=variant.graph,
        jp_ids_graph=variant.jp_ids,
        article_ids_graph=variant.article_ids,
        article_codes_graph=variant.article_codes,
        art_emb=art_emb,
        art_order=art_order,
        p2col=p2col,
        jp_emb=jp_emb,
        jp_order=jp_order,
        jp_to_row=jp_to_row,
    )
