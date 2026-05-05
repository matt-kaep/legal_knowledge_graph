#!/usr/bin/env python
"""Phase B — Embedding multi-modèles avec full text + chunks gardés.

Différences vs run_cluster.py initial :
  1. Texte UNIFORME : `text` brut pour tous (CC inclus, pas de summary privilégié)
     → comparaison propre inter-modèles
  2. CHUNKS gardés en plus du mean pool → permet chunk retrieval (max pool) en local
  3. Plusieurs modèles séquentiels avec download → embed → purge
  4. PAS de scoring CRFPA ici — analyses comparatives faites en local après

Sortie pour chaque modèle <alias> :
  embeddings/mean_<alias>.npy             (N_jp × dim)        ~363 MB pour 768d
  embeddings/chunks_<alias>.npy           (N_chunks × dim)    ~2 GB pour 768d
  embeddings/chunk_to_jp_<alias>.npy      (N_chunks,) int32   ~3 MB
  embeddings/jp_order_<alias>.npy         (N_jp,) object      ~5 MB

Stockage total attendu sur cluster : ~3 GB par modèle × 3-4 modèles = ~10-12 GB.

Usage (depuis penal_bundle/) :
    export HF_TOKEN=hf_xxx
    export HF_HOME=/scratch/hf_cache
    python run_cluster_phaseB.py
    python run_cluster_phaseB.py --only e5-large bge-m3
    python run_cluster_phaseB.py --max-chunks 8

Durée estimée sur L40S, 118k JP :
    e5-base       ~6 min
    e5-large      ~12 min
    bge-m3        ~15 min
    camembert-l   ~10 min
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, tuple[str, int, str]] = {
    "e5-base":        ("intfloat/multilingual-e5-base",        768, "Multilingual E5 base"),
    "e5-large":       ("intfloat/multilingual-e5-large",      1024, "Multilingual E5 large"),
    "bge-m3":         ("BAAI/bge-m3",                          1024, "BGE-M3 long context"),
    "camembert-large":("dangvantuan/sentence-camembert-large", 1024, "Sentence-CamemBERT FR large"),
}

DEFAULT_MODELS = ["e5-base", "e5-large", "bge-m3", "camembert-large"]

HERE       = Path(__file__).parent.resolve()
PARQUET    = HERE / "jp_index_penal.parquet"
EMB_DIR    = HERE / "embeddings"; EMB_DIR.mkdir(exist_ok=True)
LOG_DIR    = HERE / "logs";       LOG_DIR.mkdir(exist_ok=True)

CHUNK_SIZE         = 510
OVERLAP            = 64
DOC_BATCH          = 4_000
EMB_BATCH          = 512
PREFIX_PASSAGE     = "passage: "


# ═══════════════════════════════════════════════════════════════════════
# UTILS
# ═══════════════════════════════════════════════════════════════════════

def hf_cache_dir() -> Path:
    base = os.environ.get("HF_HUB_CACHE") or \
           (os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")) + "/hub")
    return Path(base)


def purge_model(hf_id: str) -> None:
    slug = "models--" + hf_id.replace("/", "--")
    target = hf_cache_dir() / slug
    if not target.exists():
        return
    try:
        size_gb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e9
        shutil.rmtree(target)
        print(f"  ↳ purge {slug} ({size_gb:.1f} Go)")
    except Exception as e:
        print(f"  ⚠ purge {slug} échouée : {e}")


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chunk_token_ids(tids, size: int, overlap: int, max_chunks: int):
    if len(tids) <= size:
        return [tids]
    chunks, step = [], size - overlap
    for start in range(0, len(tids), step):
        chunks.append(tids[start : start + size])
        if len(chunks) >= max_chunks or start + size >= len(tids):
            break
    return chunks


# ═══════════════════════════════════════════════════════════════════════
# EMBEDDING d'1 modèle — produit mean pool + chunks
# ═══════════════════════════════════════════════════════════════════════

def embed_one_model(alias: str, hf_id: str, emb_dim: int, args) -> dict:
    """Embed le corpus avec ce modèle, garde les chunks + mean pool.

    Force `text` brut (pas le summary CC) pour cohérence inter-modèles.

    Stratégie de stockage :
      - mean_<alias>.npy          (N_jp × dim) écrit incrémentalement (memmap)
      - chunks_<alias>.npy        (N_chunks_total × dim) écrit incrémentalement
      - chunk_to_jp_<alias>.npy   (N_chunks_total,) int32, jp index dans jp_order
      - jp_order_<alias>.npy      (N_jp,) object, jp_ids dans l'ordre du parquet
      - embed_state_<alias>.json  pour reprise
    """
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    out_mean    = EMB_DIR / f"mean_{alias}.npy"
    out_chunks  = EMB_DIR / f"chunks_{alias}.npy"
    out_c2jp    = EMB_DIR / f"chunk_to_jp_{alias}.npy"
    out_ids     = EMB_DIR / f"jp_order_{alias}.npy"
    state       = EMB_DIR / f"embed_state_{alias}.json"

    pf = pq.ParquetFile(PARQUET)
    n_jp = pf.metadata.num_rows
    print(f"  Documents pénaux : {n_jp}")

    # Reprise éventuelle
    start_offset = 0
    n_chunks_done = 0
    if state.exists() and out_mean.exists() and not args.force:
        s = json.loads(state.read_text())
        if s.get("model") == hf_id and s.get("n_jp") == n_jp:
            start_offset = s.get("offset", 0)
            n_chunks_done = s.get("n_chunks", 0)
            if start_offset >= n_jp:
                print(f"  ✓ déjà embarqué ({n_jp}/{n_jp}) → skip")
                return {"alias": alias, "skipped": True}
            print(f"  ↻ Reprise depuis offset {start_offset} ({100*start_offset/n_jp:.1f}%)")

    # Pré-passe : compter le nombre total de chunks à allouer
    # (nécessaire pour ouvrir le memmap chunks à la bonne taille)
    if start_offset == 0:
        print("  Pré-passe : tokenisation pour compter les chunks…")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(hf_id)
        n_chunks_total = 0
        chunks_per_doc = np.zeros(n_jp, dtype=np.int32)
        texts_iter = pq.read_table(PARQUET, columns=["text"])["text"].to_pylist()
        for i, text in enumerate(tqdm(texts_iter, desc="count chunks", ncols=80)):
            tids = tokenizer.encode(text or "", add_special_tokens=False, truncation=False)
            n_c = max(1, len(chunk_token_ids(tids, CHUNK_SIZE, OVERLAP, args.max_chunks)))
            chunks_per_doc[i] = n_c
            n_chunks_total += n_c
        print(f"  Total chunks à embedder : {n_chunks_total} (moy. {n_chunks_total/n_jp:.1f}/doc)")

        # Calculer offsets de chunks par doc (cumulatif)
        chunk_offsets = np.zeros(n_jp + 1, dtype=np.int64)
        chunk_offsets[1:] = np.cumsum(chunks_per_doc)
        np.save(EMB_DIR / f"chunk_offsets_{alias}.npy", chunk_offsets)

        # Init memmaps
        np.memmap(out_mean,   dtype=np.float32, mode="w+", shape=(n_jp, emb_dim))[:] = 0
        np.memmap(out_chunks, dtype=np.float32, mode="w+", shape=(n_chunks_total, emb_dim))[:] = 0
        np.memmap(out_c2jp,   dtype=np.int32,   mode="w+", shape=(n_chunks_total,))[:] = -1

        # jp_ids dans l'ordre du parquet
        ids = pq.read_table(PARQUET, columns=["id"])["id"].to_numpy()
        np.save(out_ids, ids)
        del ids; gc.collect()
    else:
        # En cas de reprise on relit chunks_per_doc + chunk_offsets
        chunk_offsets = np.load(EMB_DIR / f"chunk_offsets_{alias}.npy")
        n_chunks_total = int(chunk_offsets[-1])

    # Charger le modèle
    print(f"  Chargement de {hf_id}…")
    device = detect_device()
    print(f"  Device : {device}")
    model = SentenceTransformer(hf_id, device=device)
    tokenizer = model.tokenizer

    # Boucle par batch
    pbar = tqdm(total=n_jp, initial=start_offset, desc=f"embed/{alias}",
                unit="doc", ncols=80, mininterval=2.0)
    offset = start_offset
    t0 = time.time()

    while offset < n_jp:
        batch_n = min(DOC_BATCH, n_jp - offset)
        slice_tbl = pq.read_table(PARQUET, columns=["text"]).slice(offset, batch_n)
        texts = slice_tbl["text"].to_pylist()
        del slice_tbl

        # Tokeniser + chunker
        doc_chunk_counts, all_chunks = [], []
        for text in texts:
            tids = tokenizer.encode(text or "", add_special_tokens=False, truncation=False)
            chunks = chunk_token_ids(tids, CHUNK_SIZE, OVERLAP, args.max_chunks)
            decoded = [PREFIX_PASSAGE + tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
            doc_chunk_counts.append(len(decoded))
            all_chunks.extend(decoded)

        # Embedder
        chunk_embs = model.encode(
            all_chunks,
            batch_size=EMB_BATCH,
            normalize_embeddings=True,  # ON normalise au niveau chunk
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        # Mean pool par doc + L2-normalize après mean
        doc_embs = np.zeros((batch_n, emb_dim), dtype=np.float32)
        cur = 0
        for i, n_c in enumerate(doc_chunk_counts):
            block = chunk_embs[cur : cur + n_c]
            # Mean des chunks normalisés → on re-normalise après
            mean_v = block.mean(axis=0)
            n = np.linalg.norm(mean_v)
            doc_embs[i] = mean_v / (n if n > 0 else 1.0)
            cur += n_c

        # Écrire mean
        m_mean = np.memmap(out_mean, dtype=np.float32, mode="r+", shape=(n_jp, emb_dim))
        m_mean[offset : offset + batch_n] = doc_embs
        m_mean.flush()
        del m_mean

        # Écrire chunks via chunk_offsets
        c_start = int(chunk_offsets[offset])
        c_end   = int(chunk_offsets[offset + batch_n])
        m_chunks = np.memmap(out_chunks, dtype=np.float32, mode="r+",
                              shape=(n_chunks_total, emb_dim))
        m_chunks[c_start : c_end] = chunk_embs
        m_chunks.flush()
        del m_chunks

        # Écrire chunk_to_jp
        c2jp_local = np.zeros(c_end - c_start, dtype=np.int32)
        cur = 0
        for i, n_c in enumerate(doc_chunk_counts):
            c2jp_local[cur : cur + n_c] = offset + i
            cur += n_c
        m_c2jp = np.memmap(out_c2jp, dtype=np.int32, mode="r+", shape=(n_chunks_total,))
        m_c2jp[c_start : c_end] = c2jp_local
        m_c2jp.flush()
        del m_c2jp

        del all_chunks, chunk_embs, doc_embs, texts, doc_chunk_counts, c2jp_local
        gc.collect()

        offset += batch_n
        n_chunks_done = c_end
        pbar.update(batch_n)
        state.write_text(json.dumps({
            "model": hf_id, "n_jp": n_jp,
            "offset": offset, "n_chunks": n_chunks_done,
        }))

    pbar.close()
    elapsed = time.time() - t0
    print(f"  ✓ Embedding fini en {elapsed/60:.1f} min ({n_jp/max(elapsed,1):.0f} doc/s)")
    print(f"  ✓ {n_chunks_total} chunks au total")
    print(f"  Stockage : mean={out_mean.stat().st_size/1e9:.2f}GB, "
          f"chunks={out_chunks.stat().st_size/1e9:.2f}GB")

    del model
    gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "alias":     alias,
        "model_id":  hf_id,
        "emb_dim":   emb_dim,
        "n_jp":      n_jp,
        "n_chunks":  n_chunks_total,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=DEFAULT_MODELS, metavar="ALIAS",
                    help=f"Modèles à embedder (défaut: {DEFAULT_MODELS})")
    ap.add_argument("--max-chunks", type=int, default=8,
                    help="Cap chunks/doc (défaut: 8 ≈ 3500 tokens couverts)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run même si artefacts existent")
    ap.add_argument("--no-purge", action="store_true",
                    help="Ne pas supprimer les modèles HF après usage")
    args = ap.parse_args()

    print(f"Device     : {detect_device()}")
    print(f"HF cache   : {hf_cache_dir()}")
    print(f"Bundle     : {HERE}")
    print(f"Embeddings : {EMB_DIR}")
    print(f"Modèles    : {args.only}")
    print(f"Max chunks : {args.max_chunks}")

    aliases = args.only
    unknown = [a for a in aliases if a not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERREUR] Alias inconnus : {unknown}")
        print(f"Dispo : {list(MODEL_REGISTRY.keys())}")
        return 1

    summary = []
    for alias in aliases:
        hf_id, emb_dim, note = MODEL_REGISTRY[alias]
        print(f"\n{'='*70}\n[{alias}] {hf_id}  |  {note}\n{'='*70}")

        try:
            from huggingface_hub import snapshot_download
            print("  ↳ download…"); t0 = time.time()
            snapshot_download(repo_id=hf_id, ignore_patterns=["*.md", "*.txt", "original/*"])
            print(f"  ↳ download OK ({int(time.time()-t0)}s)")

            result = embed_one_model(alias, hf_id, emb_dim, args)
            summary.append(result)
        except Exception as e:
            print(f"  ✗ ERREUR : {e}")
            import traceback; traceback.print_exc()
            summary.append({"alias": alias, "error": str(e)})
        finally:
            if not args.no_purge:
                purge_model(hf_id)

    print(f"\n{'='*70}")
    print("RÉCAP")
    print(f"{'='*70}")
    for r in summary:
        if "error" in r:
            print(f"  {r['alias']:<20s}  ✗ {r['error'][:60]}")
        elif r.get("skipped"):
            print(f"  {r['alias']:<20s}  ⊘ skipped (déjà fait)")
        else:
            print(f"  {r['alias']:<20s}  ✓ {r['n_jp']} JP, {r['n_chunks']} chunks, "
                  f"{r['elapsed_s']/60:.1f} min")

    print(f"\n✓ Embeddings produits dans {EMB_DIR}")
    print(f"  À rapatrier sur Mac :  scp -r cluster:~/penal_bundle/embeddings ./baseline_b2/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
