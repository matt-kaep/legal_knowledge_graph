"""The ordered, representation-backed candidates that may be returned by retrieval."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from . import config


RETRIEVAL_REFERENCE_GRAPH_DIR = config.DATA / "hybrid_graphs" / "G6-citation-AA-knn5"


def stable_unique_with_indices(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep the first occurrence of every identifier in its stored order."""
    seen: set[str] = set()
    ids: list[str] = []
    indices: list[int] = []
    for index, value in enumerate(values.tolist()):
        identifier = str(value)
        if identifier not in seen:
            seen.add(identifier)
            ids.append(identifier)
            indices.append(index)
    return np.asarray(ids, dtype=str), np.asarray(indices, dtype=np.int64)


@dataclass(frozen=True)
class EffectiveRetrievalCandidateUniverse:
    """Candidate lists shared by every returnable ranking, independent of graph nodes."""

    article_ids: np.ndarray
    article_embeddings: np.ndarray
    jp_ids: np.ndarray
    jp_embeddings: np.ndarray

    @property
    def article_count(self) -> int:
        return int(self.article_ids.shape[0])

    @property
    def jp_count(self) -> int:
        return int(self.jp_ids.shape[0])


def _load_embeddings_and_order(embedding_path, order_path, label: str) -> tuple[np.ndarray, np.ndarray]:
    embeddings = np.load(embedding_path).astype(np.float32)
    order = np.load(order_path, allow_pickle=True).astype(str)
    if embeddings.shape[0] != order.shape[0]:
        raise ValueError(
            f"{label} embedding/order length mismatch: "
            f"embeddings={embeddings.shape[0]} order={order.shape[0]}"
        )
    return embeddings, order


@lru_cache(maxsize=1)
def load_effective_retrieval_candidate_universe() -> EffectiveRetrievalCandidateUniverse:
    """Load the fixed graph-backed universe and deduplicate decisions once."""
    article_embeddings, article_order = _load_embeddings_and_order(
        config.EMB_ARTICLES_ALL,
        config.ARTICLES_ORDER_ALL,
        "article",
    )
    jp_embeddings, jp_order = _load_embeddings_and_order(
        config.EMB_JP_SYNTHESE,
        config.JP_SUMMARY_ORDER,
        "jurisprudence",
    )
    reference_articles = set(
        np.load(RETRIEVAL_REFERENCE_GRAPH_DIR / "article_ids.npy", allow_pickle=True)
        .astype(str)
        .tolist()
    )
    reference_jp = set(
        np.load(RETRIEVAL_REFERENCE_GRAPH_DIR / "jp_ids.npy", allow_pickle=True)
        .astype(str)
        .tolist()
    )
    article_ids_raw, article_indices_raw = stable_unique_with_indices(article_order)
    jp_ids_raw, jp_indices_raw = stable_unique_with_indices(jp_order)
    article_keep = np.asarray(
        [identifier in reference_articles for identifier in article_ids_raw], dtype=bool
    )
    jp_keep = np.asarray([identifier in reference_jp for identifier in jp_ids_raw], dtype=bool)
    article_ids = article_ids_raw[article_keep]
    article_indices = article_indices_raw[article_keep]
    jp_ids = jp_ids_raw[jp_keep]
    jp_indices = jp_indices_raw[jp_keep]
    return EffectiveRetrievalCandidateUniverse(
        article_ids=article_ids,
        article_embeddings=article_embeddings[article_indices],
        jp_ids=jp_ids,
        jp_embeddings=jp_embeddings[jp_indices],
    )


def require_graph_covers_effective_retrieval_universe(
    *,
    graph_version: str,
    graph_article_ids: np.ndarray,
    graph_jp_ids: np.ndarray,
    universe: EffectiveRetrievalCandidateUniverse,
) -> None:
    """Reject a graph that would make the shared retrieval universe graph-specific."""
    graph_articles = {str(value) for value in graph_article_ids.tolist()}
    graph_jps = {str(value) for value in graph_jp_ids.tolist()}
    missing_articles = [identifier for identifier in universe.article_ids if identifier not in graph_articles]
    missing_jp = [identifier for identifier in universe.jp_ids if identifier not in graph_jps]
    if missing_articles or missing_jp:
        details = []
        if missing_articles:
            details.append(f"articles={missing_articles[:5]}")
        if missing_jp:
            details.append(f"jurisprudence={missing_jp[:5]}")
        raise ValueError(
            f"graph_version={graph_version}: missing official retrieval candidates: "
            + ", ".join(details)
        )
