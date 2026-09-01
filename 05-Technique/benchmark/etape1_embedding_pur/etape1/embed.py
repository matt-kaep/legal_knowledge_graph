"""Embedding BGE-M3, sans troncature (sauf overflow → chunk+mean-pool).

L2-normalisé. Output : (N, EMB_DIM) float32. Aligné sur l'ordre passé en entrée.
"""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from . import config


def _detect_device(override: str | None) -> str:
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _chunk_and_mean(model: SentenceTransformer, tokenizer, text: str,
                    max_ctx: int, batch: int) -> np.ndarray:
    """Pour un texte > max_ctx : tokens → chunks contigus de max_ctx → mean-pool L2."""
    ids = tokenizer(text, add_special_tokens=False, truncation=False,
                    return_attention_mask=False)["input_ids"]
    chunks = [ids[i : i + max_ctx] for i in range(0, len(ids), max_ctx)]
    decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
    embs = model.encode(decoded, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    pooled = embs.mean(axis=0)
    n = np.linalg.norm(pooled)
    return (pooled / n).astype(np.float32) if n > 0 else pooled


def embed_corpus(texts: Sequence[str], device: str | None = None,
                 batch: int = 32) -> np.ndarray:
    """Renvoie (len(texts), EMB_DIM) float32 L2-normalisé, dans l'ordre d'entrée."""
    dev = _detect_device(device)
    print(f"  device={dev}, batch={batch}, max_len={config.BATCH_MAX_LEN}")
    model = SentenceTransformer(config.MODEL_ID, device=dev)
    # Sur MPS, on plafonne à BATCH_MAX_LEN pour éviter l'OOM (attention quadratique
    # sans Flash-Attention). Les textes plus longs sont gérés via _chunk_and_mean.
    model.max_seq_length = config.BATCH_MAX_LEN
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)

    over_idx: list[int] = []
    SCAN_BATCH = 256
    for i in range(0, len(texts), SCAN_BATCH):
        enc = tokenizer(list(texts[i : i + SCAN_BATCH]),
                        add_special_tokens=False, truncation=False,
                        return_attention_mask=False)["input_ids"]
        for j, ids in enumerate(enc):
            if len(ids) > config.BATCH_MAX_LEN:
                over_idx.append(i + j)
    if over_idx:
        print(f"  {len(over_idx)} textes > {config.BATCH_MAX_LEN} tokens → chunk+mean-pool sur ceux-là")

    out = np.zeros((len(texts), config.EMB_DIM), dtype=np.float32)
    over_set = set(over_idx)
    keep_idx = [i for i in range(len(texts)) if i not in over_set]
    keep_texts = [texts[i] for i in keep_idx]
    embs = model.encode(keep_texts, batch_size=batch, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    for k, i in enumerate(keep_idx):
        out[i] = embs[k]
    for i in over_idx:
        out[i] = _chunk_and_mean(model, tokenizer, texts[i],
                                  max_ctx=config.BATCH_MAX_LEN, batch=batch)
    return out
