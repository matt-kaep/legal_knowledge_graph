#!/usr/bin/env python
"""Baseline B2 — Embedding naïf sur le benchmark CRFPA pénal (cluster L40S).

Pour chaque modèle d'embedding du registre, séquentiellement :
    download HF → embed les 118k JP pénales (chunking + mean pool) →
    run sur les 8 questions CRFPA pénales (K ∈ {3, 5, 10}) →
    score via eval_rubric → purge cache HF

Usage (depuis penal_bundle/) :
    export HF_TOKEN=hf_xxx
    export HF_HOME=/scratch/hf_cache
    python run_cluster.py

Flags :
    --only e5-base e5-large    # ne run que ces alias (défaut : tous)
    --force                    # re-run même si results/<alias>.json existe
    --k 3 5 10                 # valeurs de K à tester
    --min-freq 1               # articles cités par ≥ N JP voisines
    --no-purge                 # garder les modèles HF après usage
    --max-chunks 8             # cap du nombre de chunks par doc

Artefacts produits :
    results/<alias>.json                 # détail par config K
    results/comparison_b2_penal.csv      # 1 ligne par (modèle, K)
    results/per_question/<qid>__<alias>__k<K>.json
    logs/embed_<alias>.log               # log embedding
"""
from __future__ import annotations

import argparse
import csv
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
from scipy.sparse import csr_matrix


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — registre des modèles d'embedding
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, tuple[str, int, str]] = {
    "e5-small":      ("intfloat/multilingual-e5-small",  384,  "Multilingual E5 small (~118M, 384d), rapide"),
    "e5-base":       ("intfloat/multilingual-e5-base",   768,  "Multilingual E5 base (~278M, 768d), équilibré"),
    "e5-large":      ("intfloat/multilingual-e5-large", 1024,  "Multilingual E5 large (~560M, 1024d), qualité"),
    "bge-m3":        ("BAAI/bge-m3",                    1024,  "BGE-M3 (long context 8k tokens, 1024d)"),
    "camembert-base":("dangvantuan/sentence-camembert-base", 768, "Sentence-CamemBERT FR base (768d)"),
}

HERE       = Path(__file__).parent.resolve()
PARQUET    = HERE / "jp_index_penal.parquet"
GRAPH_NPZ  = HERE / "graph_penal.npz"
RUBRICS    = HERE / "rubrics_penal.json"
RESULTS    = HERE / "results";          RESULTS.mkdir(exist_ok=True)
PER_Q_DIR  = RESULTS / "per_question";  PER_Q_DIR.mkdir(exist_ok=True)
EMB_DIR    = HERE / "embeddings";       EMB_DIR.mkdir(exist_ok=True)
LOG_DIR    = HERE / "logs";             LOG_DIR.mkdir(exist_ok=True)

CSV_PATH   = RESULTS / "comparison_b2_penal.csv"
CSV_HEADER = [
    "alias", "model_id", "k", "min_freq", "n_q",
    "S_retrieval_mean", "S_e2e_mean",
    "S_bar_art_mean", "S_bar_jp_mean",
    "art_core_mean", "art_expected_mean", "art_expert_mean",
    "jp_core_mean",  "jp_expected_mean",  "jp_expert_mean",
    "n_articles_mean", "n_jp_mean",
    "embed_time_s", "query_time_s", "status",
]

CHUNK_SIZE         = 510
OVERLAP            = 64
DOC_BATCH          = 4_000
EMB_BATCH          = 512        # L40S 48 GB VRAM
PREFIX_PASSAGE     = "passage: "
PREFIX_QUERY       = "query: "

sys.path.insert(0, str(HERE))
from eval_rubric import evaluate


# ═══════════════════════════════════════════════════════════════════════
# UTILS — cache HF, GPU, IO
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
        print(f"  ↳ purge {slug} ({size_gb:.1f} Go libérés)")
    except Exception as e:
        print(f"  ⚠ purge {slug} échouée : {e}")


def detect_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def detect_gpus() -> int:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
        return len([l for l in out.strip().split("\n") if l])
    except Exception:
        return 0


# ═══════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════

def init_csv_if_needed() -> None:
    if CSV_PATH.exists():
        return
    with open(CSV_PATH, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)


def append_csv(row: dict) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row.get(k, "") for k in CSV_HEADER])


# ═══════════════════════════════════════════════════════════════════════
# CHUNKING
# ═══════════════════════════════════════════════════════════════════════

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
# PHASE 1 — EMBEDDING DU CORPUS (resumable)
# ═══════════════════════════════════════════════════════════════════════

def embed_corpus(
    alias: str,
    hf_id: str,
    emb_dim: int,
    args,
) -> tuple[Path, Path, float]:
    """Embed les 118k JP pénales avec ce modèle. Resumable via state JSON."""
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    out_emb = EMB_DIR / f"jp_embeddings_{alias}.npy"
    out_ids = EMB_DIR / f"jp_order_{alias}.npy"
    state   = EMB_DIR / f"embed_state_{alias}.json"

    pf      = pq.ParquetFile(PARQUET)
    n_total = pf.metadata.num_rows
    print(f"  Documents pénaux à embedder : {n_total}")

    start_offset = 0
    if state.exists() and out_emb.exists() and not args.force:
        s = json.loads(state.read_text())
        if s.get("model") == hf_id and s.get("n_total") == n_total:
            start_offset = s.get("offset", 0)
            if start_offset >= n_total:
                print(f"  ✓ déjà embarqué ({n_total}/{n_total}) → skip")
                return out_emb, out_ids, 0.0
            print(f"  ↻ Reprise depuis offset {start_offset} ({100*start_offset/n_total:.1f}%)")

    if start_offset == 0:
        m = np.memmap(out_emb, dtype=np.float32, mode="w+", shape=(n_total, emb_dim))
        m[:] = 0.0
        m.flush()
        del m
        ids = pq.read_table(PARQUET, columns=["id"])["id"].to_numpy()
        np.save(out_ids, ids)
        del ids
        gc.collect()

    print(f"  Chargement de {hf_id}…")
    device = detect_device()
    print(f"  Device : {device}")
    model = SentenceTransformer(hf_id, device=device)
    tokenizer = model.tokenizer

    pbar = tqdm(total=n_total, initial=start_offset, desc=f"embed/{alias}",
                unit="doc", ncols=80, mininterval=2.0)
    offset = start_offset
    t0 = time.time()

    while offset < n_total:
        batch_n = min(DOC_BATCH, n_total - offset)
        slice_tbl = pq.read_table(PARQUET, columns=["text"]).slice(offset, batch_n)
        texts = slice_tbl["text"].to_pylist()
        del slice_tbl

        doc_chunk_counts, all_chunks = [], []
        for text in texts:
            tids = tokenizer.encode(text or "", add_special_tokens=False)
            chunks = chunk_token_ids(tids, CHUNK_SIZE, OVERLAP, args.max_chunks)
            decoded = [PREFIX_PASSAGE + tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
            doc_chunk_counts.append(len(decoded))
            all_chunks.extend(decoded)

        chunk_embs = model.encode(
            all_chunks,
            batch_size=EMB_BATCH,
            normalize_embeddings=False,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        doc_embs = np.zeros((batch_n, emb_dim), dtype=np.float32)
        cur = 0
        for i, n_c in enumerate(doc_chunk_counts):
            doc_embs[i] = chunk_embs[cur : cur + n_c].mean(axis=0)
            cur += n_c
        norms = np.linalg.norm(doc_embs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        doc_embs /= norms

        m = np.memmap(out_emb, dtype=np.float32, mode="r+", shape=(n_total, emb_dim))
        m[offset : offset + batch_n] = doc_embs
        m.flush()
        del m, all_chunks, chunk_embs, doc_embs, texts, doc_chunk_counts
        gc.collect()

        offset += batch_n
        pbar.update(batch_n)
        state.write_text(json.dumps({
            "offset": offset, "n_total": n_total, "model": hf_id,
        }))

    pbar.close()
    elapsed = time.time() - t0
    print(f"  ✓ Embedding fini en {elapsed/60:.1f} min ({n_total/max(elapsed,1):.0f} doc/s)")

    # Libérer le modèle de la VRAM
    del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return out_emb, out_ids, elapsed


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — RETRIEVER
# ═══════════════════════════════════════════════════════════════════════

class NaiveRetriever:
    def __init__(self, hf_id: str, emb_dim: int, emb_path: Path, ids_path: Path):
        from sentence_transformers import SentenceTransformer

        print("  Chargement du graphe pénal…")
        g = np.load(GRAPH_NPZ, allow_pickle=True)
        graph_jp_ids = g["jp_ids"]
        article_ids  = g["article_ids"]
        mat = csr_matrix(
            (g["data"], g["indices"], g["indptr"]),
            shape=tuple(g["shape"]),
        )

        print("  Chargement des embeddings…")
        emb_ids = np.load(ids_path, allow_pickle=True)
        n_total = len(emb_ids)
        emb = np.memmap(emb_path, dtype=np.float32, mode="r", shape=(n_total, emb_dim))

        emb_id2pos   = {uid: i for i, uid in enumerate(emb_ids)}
        graph_id2pos = {uid: i for i, uid in enumerate(graph_jp_ids)}
        common = [uid for uid in graph_jp_ids if uid in emb_id2pos]
        print(f"  JP utilisables (graphe ∩ embeddings) : {len(common)}")

        graph_rows = [graph_id2pos[uid] for uid in common]
        emb_rows   = [emb_id2pos[uid]   for uid in common]

        self._mat_sub     = mat[graph_rows, :]
        self._sub_emb     = np.array(emb[emb_rows, :])
        self._sub_ids     = np.array(common)
        self._article_ids = article_ids

        df = pq.read_table(PARQUET, columns=["id", "number"]).to_pandas()
        self._jp_id2pourvoi = dict(zip(df["id"], df["number"]))

        device = detect_device()
        self._model = SentenceTransformer(hf_id, device=device)

    def query(self, question: str, k: int = 5, min_freq: int = 1) -> dict:
        q_vec = self._model.encode(
            [PREFIX_QUERY + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        scores = self._sub_emb @ q_vec
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        jp_ids_topk = self._sub_ids[top_k_idx]
        jp_scores   = scores[top_k_idx]

        article_counts = np.zeros(self._article_ids.shape[0], dtype=np.int32)
        for idx in top_k_idx:
            row = self._mat_sub.getrow(idx)
            article_counts[row.indices] += 1

        retained_idx = np.where(article_counts >= min_freq)[0]
        retained_art = self._article_ids[retained_idx]

        articles = [{"pair_key": pk} for pk in retained_art]
        jurisprudences = [
            {"pourvoi": self._jp_id2pourvoi.get(uid, ""), "score": float(s)}
            for uid, s in zip(jp_ids_topk, jp_scores)
            if self._jp_id2pourvoi.get(uid, "")
        ]
        return {
            "articles":       articles,
            "jurisprudences": jurisprudences,
            "arguments":      [],
            "_meta": {"k": k, "min_freq": min_freq,
                      "n_articles": len(articles), "n_jp": len(jurisprudences)},
        }


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — ÉVALUATION
# ═══════════════════════════════════════════════════════════════════════

def _mean(values):
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def per_question_path(qid: str, alias: str, k: int) -> Path:
    safe = qid.replace("/", "_")
    return PER_Q_DIR / f"{safe}__{alias}__k{k}.json"


def evaluate_one_config(retriever: NaiveRetriever, alias: str,
                          questions: list[dict], k: int, min_freq: int) -> dict:
    from tqdm import tqdm

    per_q = []
    for q in tqdm(questions, desc=f"{alias} k={k}", ncols=80):
        t0 = time.time()
        canon  = retriever.query(q["question"], k=k, min_freq=min_freq)
        scores = evaluate(canon, q)
        latency = round(time.time() - t0, 3)
        record = {
            "qid":     q["id"],
            "branche": q.get("branche"),
            "alias":   alias,
            "k":       k,
            "canon":   canon,
            "scores":  scores,
            "latency": latency,
        }
        per_question_path(q["id"], alias, k).write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        per_q.append(record)

    means = {
        "S_retrieval":  _mean([r["scores"]["regime"]["retrieval"]                   for r in per_q]),
        "S_e2e":        _mean([r["scores"]["regime"]["e2e"]                         for r in per_q]),
        "S_bar_art":    _mean([r["scores"]["articles"]["S_bar"]                     for r in per_q]),
        "S_bar_jp":     _mean([r["scores"]["jurisprudences"]["S_bar"]               for r in per_q]),
        "art_core":     _mean([r["scores"]["articles"]["per_strate"]["core"]            for r in per_q]),
        "art_expected": _mean([r["scores"]["articles"]["per_strate"]["expected"]        for r in per_q]),
        "art_expert":   _mean([r["scores"]["articles"]["per_strate"]["expert"]          for r in per_q]),
        "jp_core":      _mean([r["scores"]["jurisprudences"]["per_strate"]["core"]      for r in per_q]),
        "jp_expected":  _mean([r["scores"]["jurisprudences"]["per_strate"]["expected"]  for r in per_q]),
        "jp_expert":    _mean([r["scores"]["jurisprudences"]["per_strate"]["expert"]    for r in per_q]),
        "n_articles":   _mean([r["canon"]["_meta"]["n_articles"] for r in per_q]),
        "n_jp":         _mean([r["canon"]["_meta"]["n_jp"]       for r in per_q]),
    }
    return {"k": k, "min_freq": min_freq, "means": means, "per_question": per_q}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ALIAS",
                    help="Ne run que ces alias (défaut : tous)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run même si embedding/JSON existent")
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 10],
                    help="Valeurs de K à tester (défaut: 3 5 10)")
    ap.add_argument("--min-freq", type=int, default=1,
                    help="Articles cités par ≥ N JP voisines (défaut: 1)")
    ap.add_argument("--max-chunks", type=int, default=8,
                    help="Cap chunks/doc (défaut: 8 ≈ 3500 tokens couverts)")
    ap.add_argument("--no-purge", action="store_true",
                    help="Ne pas supprimer le modèle HF après usage")
    ap.add_argument("--list-questions", action="store_true",
                    help="Liste les question IDs et quitte")
    args = ap.parse_args()

    n_gpus = detect_gpus()
    print(f"GPUs       : {n_gpus} (device détecté : {detect_device()})")
    print(f"HF cache   : {hf_cache_dir()}")
    print(f"Bundle     : {HERE}")
    print(f"Résultats  : {RESULTS}")
    print(f"CSV        : {CSV_PATH}")

    questions = json.loads(RUBRICS.read_text(encoding="utf-8"))["questions"]
    if args.list_questions:
        for q in questions:
            print(f"  {q['id']:30s}  [{q.get('branche'):16s}]  {q.get('specialisation','')[:60]}")
        print(f"\nTotal : {len(questions)} questions")
        return 0
    print(f"Questions  : {len(questions)} (branches : "
          f"{sorted(set(q.get('branche') for q in questions))})")

    aliases = args.only if args.only else list(MODEL_REGISTRY.keys())
    unknown = [a for a in aliases if a not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERREUR] Alias inconnus : {unknown}")
        print(f"Dispo : {list(MODEL_REGISTRY.keys())}")
        return 1

    init_csv_if_needed()

    # ─── BOUCLE PRINCIPALE ───────────────────────────────────────────
    for alias in aliases:
        hf_id, emb_dim, note = MODEL_REGISTRY[alias]
        out_json = RESULTS / f"{alias}.json"

        # Skip si déjà fait pour TOUS les K demandés
        skip = False
        if out_json.exists() and not args.force:
            try:
                done = json.loads(out_json.read_text())
                done_ks = {c["k"] for c in done.get("configs", [])}
                if set(args.k).issubset(done_ks):
                    skip = True
            except Exception:
                pass
        if skip:
            print(f"\n[{alias}] déjà fait → skip ({out_json})")
            continue

        print(f"\n{'='*70}\n[{alias}] {hf_id}  |  {note}\n{'='*70}")
        try:
            from huggingface_hub import snapshot_download

            # 1) Download
            print("  ↳ download…"); t0 = time.time()
            snapshot_download(repo_id=hf_id, ignore_patterns=["*.md", "*.txt", "original/*"])
            print(f"  ↳ download OK ({int(time.time()-t0)}s)")

            # 2) Embed
            emb_path, ids_path, embed_time = embed_corpus(alias, hf_id, emb_dim, args)

            # 3) Build retriever
            print("  ↳ build retriever…")
            retriever = NaiveRetriever(hf_id, emb_dim, emb_path, ids_path)

            # 4) Run for each K
            configs = []
            t_query = time.time()
            for k in args.k:
                print(f"\n  ── Config k={k}, min_freq={args.min_freq} ─────")
                result = evaluate_one_config(retriever, alias, questions,
                                              k=k, min_freq=args.min_freq)
                configs.append(result)
                m = result["means"]
                append_csv({
                    "alias":           alias,
                    "model_id":        hf_id,
                    "k":               k,
                    "min_freq":        args.min_freq,
                    "n_q":             len(result["per_question"]),
                    "S_retrieval_mean":  m["S_retrieval"],
                    "S_e2e_mean":        m["S_e2e"],
                    "S_bar_art_mean":    m["S_bar_art"],
                    "S_bar_jp_mean":     m["S_bar_jp"],
                    "art_core_mean":     m["art_core"],
                    "art_expected_mean": m["art_expected"],
                    "art_expert_mean":   m["art_expert"],
                    "jp_core_mean":      m["jp_core"],
                    "jp_expected_mean":  m["jp_expected"],
                    "jp_expert_mean":    m["jp_expert"],
                    "n_articles_mean":   m["n_articles"],
                    "n_jp_mean":         m["n_jp"],
                    "embed_time_s":      round(embed_time, 1) if k == args.k[0] else "",
                    "query_time_s":      round(time.time() - t_query, 1),
                    "status":            "ok",
                })
                print(f"    S_retrieval = {m['S_retrieval']}  (plafond LLM ≈ 0.095)")
                print(f"    S̄_art={m['S_bar_art']}  S̄_jp={m['S_bar_jp']}  "
                      f"N_art={m['n_articles']}  N_jp={m['n_jp']}")

            # 5) Save aggregate
            out_json.write_text(json.dumps({
                "alias":      alias,
                "model_id":   hf_id,
                "embed_time_s": embed_time,
                "configs":    configs,
            }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

        except Exception as e:
            print(f"  ✗ ERREUR : {e}")
            import traceback; traceback.print_exc()
            append_csv({"alias": alias, "model_id": hf_id,
                        "status": f"error: {e}"[:200]})
        finally:
            if not args.no_purge:
                purge_model(hf_id)

    print(f"\n✓ Terminé — {CSV_PATH}")
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        cols = ["alias", "k", "S_retrieval_mean", "S_bar_art_mean", "S_bar_jp_mean",
                "n_articles_mean", "n_jp_mean", "status"]
        print("\n" + df[cols].to_string(index=False))
    except ImportError:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
