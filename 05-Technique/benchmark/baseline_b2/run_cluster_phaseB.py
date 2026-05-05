#!/usr/bin/env python
"""Phase B — Embedding multi-modèles avec full text + chunks gardés (CLUSTER L40S).

Optimisé pour L40S 48 GB VRAM + ~40 GB RAM système :
  - EMB_BATCH 2048 par défaut (vs 512 avant)  → ~4× plus rapide GPU
  - DOC_BATCH 16k par défaut (vs 4k)           → moins d'I/O répétitifs
  - Pas de pré-passe : memmap chunks alloué worst-case puis tronqué à la fin
  - Tokenisation batchée (`tokenizer(batch)`) au lieu de doc-par-doc

Différences vs baseline B2 initiale :
  1. Texte UNIFORME : `text` brut pour tous (CC inclus, pas de summary privilégié)
  2. CHUNKS gardés en plus du mean pool → permet chunk retrieval (max pool)
  3. Plusieurs modèles séquentiels avec download → embed → purge
  4. PAS de scoring CRFPA ici — analyses comparatives en local après

Sortie pour chaque modèle <alias> dans embeddings/ :
  mean_<alias>.npy             (N_jp × dim)        ~363 MB pour 768d
  chunks_<alias>.npy           (N_chunks × dim)    ~2 GB pour 768d
  chunk_to_jp_<alias>.npy      (N_chunks,) int32   ~3 MB
  jp_order_<alias>.npy         (N_jp,) object      ~5 MB
  chunk_offsets_<alias>.npy    (N_jp+1,) int64     ~1 MB

Usage (depuis penal_bundle/) :
    export HF_TOKEN=hf_xxx
    export HF_HOME=/scratch/hf_cache
    python run_cluster_phaseB.py
    python run_cluster_phaseB.py --only e5-large bge-m3
    python run_cluster_phaseB.py --emb-batch 4096 --doc-batch 32000
    python run_cluster_phaseB.py --emb-batch 1024     # si OOM VRAM

Durée estimée sur L40S, 118k JP :
    e5-base       ~2-3 min  (vs 6 min en config conservative)
    e5-large      ~5 min
    bge-m3        ~6 min
    camembert-l   ~4 min
    Total 4 modèles : ~17 min
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")  # ← batching tokenizer = on autorise

import numpy as np
import pyarrow.parquet as pq


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, tuple[str, int, str, int]] = {
    # alias → (hf_id, emb_dim, note, recommended_emb_batch_for_40GB)
    "e5-base":         ("intfloat/multilingual-e5-base",         768,  "Multilingual E5 base",         512),
    "e5-large":        ("intfloat/multilingual-e5-large",       1024,  "Multilingual E5 large",        128),
    "bge-m3":          ("BAAI/bge-m3",                           1024, "BGE-M3 long context",          128),
    "camembert-large": ("dangvantuan/sentence-camembert-large",  1024, "Sentence-CamemBERT FR large",  128),
}

DEFAULT_MODELS = ["e5-base", "e5-large", "bge-m3", "camembert-large"]

HERE       = Path(__file__).parent.resolve()
PARQUET    = HERE / "jp_index_penal.parquet"
EMB_DIR    = HERE / "embeddings"; EMB_DIR.mkdir(exist_ok=True)
LOG_DIR    = HERE / "logs";       LOG_DIR.mkdir(exist_ok=True)

CHUNK_SIZE         = 510
OVERLAP            = 64
MAX_CHUNKS_PER_DOC = 8
DOC_BATCH_DEFAULT  = 4_000      # conservateur si VRAM partagée
EMB_BATCH_DEFAULT  = 128        # conservateur ; tune via --emb-batch si VRAM dispo
PREFIX_PASSAGE     = "passage: "
MAX_SEQ_LENGTH     = 512        # clamp côté model.encode() pour éviter re-tok > 510


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


def gpu_memory_info() -> tuple[float, float] | None:
    """Renvoie (free_gb, total_gb) pour le GPU 0, ou None si pas de CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info(0)
        return free_b / 1e9, total_b / 1e9
    except Exception:
        return None


def warn_if_low_vram(threshold_gb: float = 15.0) -> None:
    info = gpu_memory_info()
    if info is None:
        return
    free, total = info
    print(f"  GPU mem : {free:.1f} GB libre / {total:.1f} GB total")
    if free < threshold_gb:
        print(f"  ⚠ VRAM libre < {threshold_gb} GB — réduisez --emb-batch si OOM")


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
# EMBEDDING d'1 modèle — pas de pré-passe, allocation worst-case
# ═══════════════════════════════════════════════════════════════════════

def embed_one_model(alias: str, hf_id: str, emb_dim: int, args) -> dict:
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    out_mean    = EMB_DIR / f"mean_{alias}.npy"
    out_chunks  = EMB_DIR / f"chunks_{alias}.npy"
    out_c2jp    = EMB_DIR / f"chunk_to_jp_{alias}.npy"
    out_ids     = EMB_DIR / f"jp_order_{alias}.npy"
    out_offsets = EMB_DIR / f"chunk_offsets_{alias}.npy"
    state       = EMB_DIR / f"embed_state_{alias}.json"

    pf = pq.ParquetFile(PARQUET)
    n_jp = pf.metadata.num_rows
    print(f"  Documents pénaux : {n_jp}")

    # Reprise éventuelle
    start_offset = 0
    chunks_written = 0
    if state.exists() and out_mean.exists() and not args.force:
        s = json.loads(state.read_text())
        if (s.get("model") == hf_id and s.get("n_jp") == n_jp
                and s.get("max_chunks") == args.max_chunks):
            start_offset = s.get("offset", 0)
            chunks_written = s.get("chunks_written", 0)
            if start_offset >= n_jp:
                print(f"  ✓ déjà embarqué ({n_jp}/{n_jp}) → skip")
                return {"alias": alias, "skipped": True}
            print(f"  ↻ Reprise depuis offset {start_offset} ({100*start_offset/n_jp:.1f}%)")

    # Allocation worst-case (n_jp × max_chunks)
    n_chunks_max = n_jp * args.max_chunks
    if start_offset == 0:
        print(f"  Allocation memmaps (worst-case n_chunks_max={n_chunks_max}, "
              f"~{n_chunks_max * emb_dim * 4 / 1e9:.1f} GB)…")
        np.memmap(out_mean,    dtype=np.float32, mode="w+", shape=(n_jp, emb_dim))[:] = 0
        np.memmap(out_chunks,  dtype=np.float32, mode="w+", shape=(n_chunks_max, emb_dim))[:] = 0
        np.memmap(out_c2jp,    dtype=np.int32,   mode="w+", shape=(n_chunks_max,))[:] = -1
        np.memmap(out_offsets, dtype=np.int64,   mode="w+", shape=(n_jp + 1,))[:] = 0

        ids = pq.read_table(PARQUET, columns=["id"])["id"].to_numpy()
        np.save(out_ids, ids)
        del ids; gc.collect()

    # Charger le modèle
    print(f"  Chargement de {hf_id}…")
    device = detect_device()
    print(f"  Device : {device}")
    warn_if_low_vram()
    model = SentenceTransformer(hf_id, device=device)
    # Force la troncature côté model.encode() : même si le re-tokenize après decode
    # produit > MAX_SEQ_LENGTH tokens, ils seront tronqués proprement
    model.max_seq_length = MAX_SEQ_LENGTH
    tokenizer = model.tokenizer

    pbar = tqdm(total=n_jp, initial=start_offset, desc=f"embed/{alias}",
                unit="doc", ncols=80, mininterval=2.0)
    offset = start_offset
    t0 = time.time()

    while offset < n_jp:
        batch_n = min(args.doc_batch, n_jp - offset)
        slice_tbl = pq.read_table(PARQUET, columns=["text"]).slice(offset, batch_n)
        texts = slice_tbl["text"].to_pylist()
        del slice_tbl

        # Tokeniser en BATCH (rapide, fast tokenizer)
        encoded_batch = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]

        # Chunker chaque doc
        doc_chunk_counts = []
        all_chunks = []
        for tids in encoded_batch:
            chunks = chunk_token_ids(tids, CHUNK_SIZE, OVERLAP, args.max_chunks)
            decoded = [PREFIX_PASSAGE + tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
            doc_chunk_counts.append(len(decoded))
            all_chunks.extend(decoded)

        # Embedder avec retry on OOM (divise le batch par 2)
        import torch
        current_batch = args.emb_batch
        chunk_embs = None
        while chunk_embs is None:
            try:
                chunk_embs = model.encode(
                    all_chunks,
                    batch_size=current_batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                ).astype(np.float32)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if current_batch <= 16:
                    print(f"\n  ✗ OOM même à batch={current_batch}, abandon")
                    raise
                current_batch = max(16, current_batch // 2)
                print(f"\n  ⚠ OOM → retry avec emb_batch={current_batch}")

        # Mean pool par doc + L2-normalize après mean
        doc_embs = np.zeros((batch_n, emb_dim), dtype=np.float32)
        cur = 0
        for i, n_c in enumerate(doc_chunk_counts):
            block = chunk_embs[cur : cur + n_c]
            mean_v = block.mean(axis=0)
            n = np.linalg.norm(mean_v)
            doc_embs[i] = mean_v / (n if n > 0 else 1.0)
            cur += n_c

        # Écrire mean pool
        m_mean = np.memmap(out_mean, dtype=np.float32, mode="r+", shape=(n_jp, emb_dim))
        m_mean[offset : offset + batch_n] = doc_embs
        m_mean.flush()
        del m_mean

        # Écrire chunks séquentiellement
        n_new = len(all_chunks)
        m_chunks = np.memmap(out_chunks, dtype=np.float32, mode="r+",
                              shape=(n_chunks_max, emb_dim))
        m_chunks[chunks_written : chunks_written + n_new] = chunk_embs
        m_chunks.flush()
        del m_chunks

        # Construire offsets et chunk_to_jp localement
        c2jp_local    = np.zeros(n_new, dtype=np.int32)
        offsets_local = np.zeros(batch_n, dtype=np.int64)
        cur = 0
        running = chunks_written
        for i, n_c in enumerate(doc_chunk_counts):
            offsets_local[i] = running
            c2jp_local[cur : cur + n_c] = offset + i
            cur += n_c
            running += n_c

        m_c2jp = np.memmap(out_c2jp, dtype=np.int32, mode="r+", shape=(n_chunks_max,))
        m_c2jp[chunks_written : chunks_written + n_new] = c2jp_local
        m_c2jp.flush()
        del m_c2jp

        m_off = np.memmap(out_offsets, dtype=np.int64, mode="r+", shape=(n_jp + 1,))
        m_off[offset : offset + batch_n] = offsets_local
        m_off[offset + batch_n] = chunks_written + n_new
        m_off.flush()
        del m_off

        del all_chunks, chunk_embs, doc_embs, texts, doc_chunk_counts, encoded_batch
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        chunks_written += n_new
        offset += batch_n
        pbar.update(batch_n)
        state.write_text(json.dumps({
            "model":          hf_id,
            "n_jp":           n_jp,
            "max_chunks":     args.max_chunks,
            "offset":         offset,
            "chunks_written": chunks_written,
        }))

    pbar.close()
    elapsed = time.time() - t0
    print(f"  ✓ Embedding fini en {elapsed/60:.1f} min ({n_jp/max(elapsed,1):.0f} doc/s)")
    print(f"  ✓ {chunks_written} chunks écrits (sur {n_chunks_max} alloués, "
          f"~{100*chunks_written/n_chunks_max:.0f}% utilisés)")

    # Tronquer chunks et c2jp à la taille réelle
    print(f"  Troncature des memmaps à la taille réelle…")
    actual_chunks = np.array(np.memmap(out_chunks, dtype=np.float32, mode="r",
                                         shape=(n_chunks_max, emb_dim))[:chunks_written])
    actual_c2jp   = np.array(np.memmap(out_c2jp, dtype=np.int32, mode="r",
                                         shape=(n_chunks_max,))[:chunks_written])
    out_chunks.unlink()
    out_c2jp.unlink()
    np.save(EMB_DIR / f"chunks_{alias}.npy",      actual_chunks)
    np.save(EMB_DIR / f"chunk_to_jp_{alias}.npy", actual_c2jp)

    print(f"  Stockage final : "
          f"mean={out_mean.stat().st_size/1e9:.2f}GB, "
          f"chunks={(EMB_DIR / f'chunks_{alias}.npy').stat().st_size/1e9:.2f}GB")

    del model, actual_chunks, actual_c2jp
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
        "n_chunks":  chunks_written,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=DEFAULT_MODELS, metavar="ALIAS",
                    help=f"Modèles à embedder (défaut: {DEFAULT_MODELS})")
    ap.add_argument("--max-chunks", type=int, default=MAX_CHUNKS_PER_DOC,
                    help=f"Cap chunks/doc (défaut: {MAX_CHUNKS_PER_DOC} ≈ 3500 tokens)")
    ap.add_argument("--doc-batch", type=int, default=DOC_BATCH_DEFAULT,
                    help=f"Docs par batch I/O (défaut: {DOC_BATCH_DEFAULT})")
    ap.add_argument("--emb-batch", type=int, default=None,
                    help="Chunks par forward GPU (défaut: auto par modèle "
                         "— e5-base 512, e5-large/bge-m3/camembert 128)")
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
    print(f"Doc batch  : {args.doc_batch}")
    print(f"Emb batch  : {args.emb_batch if args.emb_batch is not None else 'auto par modèle'}")
    print(f"Max chunks : {args.max_chunks}")
    info = gpu_memory_info()
    if info:
        free, total = info
        print(f"VRAM       : {free:.1f} GB libre / {total:.1f} GB total")

    aliases = args.only
    unknown = [a for a in aliases if a not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERREUR] Alias inconnus : {unknown}")
        print(f"Dispo : {list(MODEL_REGISTRY.keys())}")
        return 1

    user_emb_batch = args.emb_batch  # None = auto, int = forcé
    summary = []
    for alias in aliases:
        hf_id, emb_dim, note, recommended_batch = MODEL_REGISTRY[alias]
        if user_emb_batch is None:
            args.emb_batch = recommended_batch
            print(f"\n→ emb_batch auto pour {alias} = {recommended_batch}")
        else:
            args.emb_batch = user_emb_batch
        print(f"{'='*70}\n[{alias}] {hf_id}  |  {note}\n{'='*70}")

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
    print(f"  À rapatrier sur Mac :  scp -r cluster:~/penal_bundle/embeddings ./baseline_b2/penal_bundle/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
