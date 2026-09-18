"""Distribution de longueur en tokens d'un corpus."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np


def compute_token_stats(texts: Sequence[str], tokenizer, max_ctx: int) -> dict:
    """Tokenise (batché) et renvoie p50/p90/p99/p100 + dépassements `max_ctx`."""
    BATCH = 512
    lengths: list[int] = []
    for i in range(0, len(texts), BATCH):
        batch = list(texts[i : i + BATCH])
        enc = tokenizer(batch, add_special_tokens=False, truncation=False,
                        return_attention_mask=False, return_token_type_ids=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    arr = np.array(lengths, dtype=np.int32)
    over = int((arr > max_ctx).sum())
    return {
        "n":            int(arr.size),
        "p50":          int(np.percentile(arr, 50)),
        "p90":          int(np.percentile(arr, 90)),
        "p99":          int(np.percentile(arr, 99)),
        "p100":         int(arr.max()) if arr.size else 0,
        "mean":         float(arr.mean()) if arr.size else 0.0,
        "max_ctx":      int(max_ctx),
        "n_over_ctx":   over,
        "pct_over_ctx": over / max(arr.size, 1),
    }
