"""Recall@K, K*, et résolution short_ref → pourvoi pour le côté JP."""
from __future__ import annotations
from collections.abc import Sequence
import re
from typing import Iterable

# Format CC : 18-82.829, 90-83.786 (XX-XX.XXX)
_POURVOI_CC = re.compile(r"\b(\d{2}-\d{2}\.\d{3})\b")


def recall_at_k(ranked: Sequence[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hit = sum(1 for x in ranked[:k] if x in gold)
    return hit / len(gold)


def kstar(ranked: Sequence[str], gold: set[str],
          ks: Iterable[int], threshold: float) -> int | None:
    """Plus petit k ∈ ks tel que recall@k ≥ threshold, ou None."""
    for k in sorted(ks):
        if recall_at_k(ranked, gold, k) >= threshold:
            return k
    return None


def extract_pourvoi_numbers(text: str) -> list[str]:
    """Format CC uniquement (cf. journal 2026-05-05, JP-side fragilité connue)."""
    return _POURVOI_CC.findall(text or "")
