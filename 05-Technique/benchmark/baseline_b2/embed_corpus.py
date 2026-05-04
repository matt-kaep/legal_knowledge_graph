#!/usr/bin/env python3
"""Embed le corpus JP par chunking + mean pooling → jp_embeddings.npy.

Stratégie :
  - Tokeniser le texte avec le tokenizer du modèle
  - Découper en chunks de CHUNK_SIZE tokens avec OVERLAP tokens de recouvrement
  - Embedder tous les chunks en batch
  - Pour chaque document : moyenner les vecteurs de ses chunks (mean pooling)

Produit :
  jp_embeddings.npy — (N_jp × 384) float32, L2-normalisé
  jp_order.npy      — (N_jp,) jp_ids dans l'ordre des lignes
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
PARQUET   = HERE / "jp_index.parquet"
OUT_EMB   = HERE / "jp_embeddings.npy"
OUT_IDS   = HERE / "jp_order.npy"

MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 512    # tokens
OVERLAP    = 64     # tokens
BATCH_SIZE = 128    # chunks par batch (réduire à 64 si OOM)
PREFIX     = "passage: "


def chunk_token_ids(token_ids: list[int], size: int, overlap: int) -> list[list[int]]:
    if len(token_ids) <= size:
        return [token_ids]
    chunks = []
    step = size - overlap
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start : start + size]
        chunks.append(chunk)
        if start + size >= len(token_ids):
            break
    return chunks


def main() -> None:
    df = pd.read_parquet(PARQUET, columns=["id", "text"])
    ids   = df["id"].to_numpy()
    texts = df["text"].fillna("").tolist()
    n_jp  = len(texts)
    print(f"JP à embedder : {n_jp}")

    model     = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    print("Tokenisation + chunking…")
    doc_chunk_counts: list[int] = []
    all_chunks: list[str]       = []

    for text in tqdm(texts, desc="tokenise"):
        # Encoder le texte brut (sans préfixe) pour obtenir tous les token IDs
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = chunk_token_ids(token_ids, CHUNK_SIZE - 2, OVERLAP)
        # Ajouter "passage: " à chaque chunk individuellement (requis par E5)
        decoded = [PREFIX + tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
        doc_chunk_counts.append(len(decoded))
        all_chunks.extend(decoded)

    print(f"Total chunks : {len(all_chunks)} (moy. {len(all_chunks)/n_jp:.1f} par JP)")

    print("Embedding des chunks…")
    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,
        convert_to_numpy=True,
    ).astype(np.float32)

    print("Mean pooling par document…")
    embeddings = np.zeros((n_jp, chunk_embeddings.shape[1]), dtype=np.float32)
    cursor = 0
    for i, n_chunks in enumerate(doc_chunk_counts):
        block = chunk_embeddings[cursor : cursor + n_chunks]
        embeddings[i] = block.mean(axis=0)
        cursor += n_chunks

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings /= norms

    np.save(OUT_EMB, embeddings)
    np.save(OUT_IDS, ids)
    print(f"✓ embeddings : {embeddings.shape} → {OUT_EMB} ({OUT_EMB.stat().st_size/1e9:.2f} GB)")
    print(f"✓ ordre ids  : {ids.shape} → {OUT_IDS}")


if __name__ == "__main__":
    main()
