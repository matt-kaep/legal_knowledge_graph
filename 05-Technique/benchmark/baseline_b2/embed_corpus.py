#!/usr/bin/env python3
"""Embed le corpus JP par chunking + mean pooling — streaming, resumable, parallèle.

Architecture :
  - Producer thread : lit le parquet par slices, tokenise + chunke (CPU)
  - Consumer (main) : pull du queue, embed sur MPS (GPU), mean pool, écrit memmap
  - Overlap CPU/GPU via une queue de slices pré-tokenisés

Garde-fous mémoire (~2-3 GB peak) :
  - DOC_BATCH=2000 docs par slice
  - MAX_CHUNKS_PER_DOC=4 → cap les longs docs (couvre ~1800 tokens)
  - np.memmap → embeddings écrits sur disque incrémentalement
  - State JSON → reprise après crash/Ctrl+C

Filtres optionnels :
  --juris CC          → embed uniquement la juridiction Cour de cassation
  --model small|base  → e5-small (384d, rapide) ou e5-base (768d, qualité)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path
from queue import Queue

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pyarrow.parquet as pq
import torch

torch.set_num_threads(4)

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

HERE    = Path(__file__).parent.resolve()
PARQUET = HERE / "jp_index.parquet"

MODELS = {
    "small": ("intfloat/multilingual-e5-small", 384),
    "base":  ("intfloat/multilingual-e5-base",  768),
}

CHUNK_SIZE         = 510
OVERLAP            = 64
MAX_CHUNKS_PER_DOC = 4
DOC_BATCH          = 2_000
EMB_BATCH          = 256
QUEUE_MAXSIZE      = 3        # pré-charger 3 slices d'avance
PREFIX             = "passage: "


def chunk_token_ids(tids, size, overlap, max_chunks):
    if len(tids) <= size:
        return [tids]
    chunks = []
    step = size - overlap
    for start in range(0, len(tids), step):
        chunks.append(tids[start : start + size])
        if len(chunks) >= max_chunks or start + size >= len(tids):
            break
    return chunks


def producer_loop(
    parquet_path: Path,
    juris_filter: str | None,
    tokenizer,
    queue: Queue,
    start_offset: int,
    n_total: int,
    valid_indices: np.ndarray,
):
    """Lit le parquet, tokenise par slice, push (offset, doc_chunk_counts, all_chunks) dans la queue."""
    table = pq.read_table(parquet_path, columns=["text", "juris"])
    texts_all = table["text"]
    juris_all = table["juris"]

    pos = start_offset
    while pos < n_total:
        end_pos = min(pos + DOC_BATCH, n_total)
        # Indices originaux dans le parquet pour ce slice
        orig_idx = valid_indices[pos : end_pos]

        all_chunks = []
        doc_chunk_counts = []
        emb_indices = []  # offset dans la matrice de sortie pour chaque doc

        for local_i, parquet_i in enumerate(orig_idx):
            text = texts_all[int(parquet_i)].as_py() or ""
            tids = tokenizer.encode(text, add_special_tokens=False)
            chunks = chunk_token_ids(tids, CHUNK_SIZE, OVERLAP, MAX_CHUNKS_PER_DOC)
            decoded = [PREFIX + tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
            doc_chunk_counts.append(len(decoded))
            all_chunks.extend(decoded)
            emb_indices.append(pos + local_i)

        queue.put({
            "offset":      pos,
            "n_docs":      end_pos - pos,
            "all_chunks":  all_chunks,
            "doc_chunk_counts": doc_chunk_counts,
        })
        pos = end_pos

    queue.put(None)  # sentinel de fin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--juris", choices=["CC", "CA", "TJ"], default=None,
                    help="Restreindre à une juridiction")
    ap.add_argument("--model", choices=["small", "base"], default="small")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limiter à N premiers docs (debug)")
    args = ap.parse_args()

    model_name, emb_dim = MODELS[args.model]

    suffix = f"_{args.juris.lower()}" if args.juris else ""
    suffix += f"_{args.model}"
    OUT_EMB = HERE / f"jp_embeddings{suffix}.npy"
    OUT_IDS = HERE / f"jp_order{suffix}.npy"
    STATE   = HERE / f"embed_state{suffix}.json"

    print(f"Modèle : {model_name} ({emb_dim} dims)", flush=True)
    print(f"Filtre : juris={args.juris or 'all'}, limit={args.limit or 'none'}", flush=True)
    print(f"Sortie : {OUT_EMB.name}", flush=True)

    # 1) Construire l'index des indices à traiter
    print("Lecture du parquet (filtrage)…", flush=True)
    table = pq.read_table(PARQUET, columns=["id", "juris"])
    juris_arr = np.array(table["juris"].to_pylist())
    ids_arr   = np.array(table["id"].to_pylist(), dtype=object)

    if args.juris:
        valid_idx = np.where(juris_arr == args.juris)[0]
    else:
        valid_idx = np.arange(len(juris_arr))

    if args.limit:
        valid_idx = valid_idx[: args.limit]

    n_total = len(valid_idx)
    print(f"Documents à traiter : {n_total}", flush=True)
    if n_total == 0:
        sys.exit("Aucun document à traiter.")

    # 2) Reprise éventuelle
    start_offset = 0
    if STATE.exists() and OUT_EMB.exists():
        s = json.loads(STATE.read_text())
        if s.get("n_total") == n_total:
            start_offset = s.get("offset", 0)
            print(f"↻ Reprise depuis offset {start_offset} ({100*start_offset/n_total:.1f}%)", flush=True)

    # 3) Init memmap + ids
    if start_offset == 0:
        print("Création du memmap…", flush=True)
        m = np.memmap(OUT_EMB, dtype=np.float32, mode="w+", shape=(n_total, emb_dim))
        m[:] = 0.0
        m.flush()
        del m

        np.save(OUT_IDS, ids_arr[valid_idx])
        del ids_arr, juris_arr
        gc.collect()

    # 4) Charger le modèle
    print(f"Chargement du modèle…", flush=True)
    model     = SentenceTransformer(model_name)
    tokenizer = model.tokenizer
    print(f"Device : {model.device}", flush=True)

    # 5) Lancer le producer (thread séparé pour overlap CPU/GPU)
    queue = Queue(maxsize=QUEUE_MAXSIZE)
    producer = threading.Thread(
        target=producer_loop,
        args=(PARQUET, args.juris, tokenizer, queue, start_offset, n_total, valid_idx),
        daemon=True,
    )
    producer.start()

    # 6) Consumer loop : embed + write
    pbar = tqdm(total=n_total, initial=start_offset, desc="docs", unit="doc",
                ncols=80, mininterval=2.0)

    try:
        while True:
            item = queue.get()
            if item is None:
                break

            offset = item["offset"]
            n_docs = item["n_docs"]
            chunks = item["all_chunks"]
            counts = item["doc_chunk_counts"]

            chunk_embs = model.encode(
                chunks,
                batch_size=EMB_BATCH,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype(np.float32)

            # Mean pool par doc + L2-normalize
            doc_embs = np.zeros((n_docs, emb_dim), dtype=np.float32)
            cur = 0
            for i, n_c in enumerate(counts):
                doc_embs[i] = chunk_embs[cur : cur + n_c].mean(axis=0)
                cur += n_c
            norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            doc_embs /= norms

            # Écrire dans le memmap
            m = np.memmap(OUT_EMB, dtype=np.float32, mode="r+", shape=(n_total, emb_dim))
            m[offset : offset + n_docs] = doc_embs
            m.flush()
            del m

            del chunks, chunk_embs, doc_embs
            gc.collect()

            pbar.update(n_docs)
            STATE.write_text(json.dumps({"offset": offset + n_docs, "n_total": n_total}))

    except KeyboardInterrupt:
        print(f"\n⚠ Interruption — état sauvé.", flush=True)
        sys.exit(130)
    finally:
        pbar.close()

    print(f"\n✓ {n_total} embeddings → {OUT_EMB} ({OUT_EMB.stat().st_size/1e9:.2f} GB)", flush=True)
    print(f"✓ IDs → {OUT_IDS}", flush=True)


if __name__ == "__main__":
    main()
