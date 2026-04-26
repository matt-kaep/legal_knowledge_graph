#!/usr/bin/env python
"""Benchmark séquentiel : 1 modèle à la fois, download → serve → eval → purge.

Usage (depuis llm_benchmark/) :
    export HF_TOKEN=hf_xxx
    export HF_HOME=/scratch/hf_cache      # ← volume avec du disque
    python run_all_models.py

Flags :
    --only gemma4-E2B qwen3.5-2B      # ne run que ces alias
    --skip-existing                   # skip si results/<alias>.json existe (défaut)
    --force                           # re-run même si JSON existe
    --max-len 16384                   # contexte max vLLM
    --gpu-util 0.90
    --port 8000

Artefacts produits :
    results/<alias>.json              # détail par arrêt
    results/comparison.csv            # tableau cumulatif (regex V3 + 1 ligne/modèle)
    logs/vllm_<alias>.log             # logs serveur
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION — registre des modèles à benchmarker
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "gemma4-E2B":       ("google/gemma-4-E2B-it",                 "Gemma 4 E2B-IT (~2B effectif)"),
    "qwen3.5-2B":       ("Qwen/Qwen3.5-2B-Instruct",              "Qwen3.5 dense 2B"),
    "gemma4-E4B":       ("google/gemma-4-E4B-it",                 "Gemma 4 E4B-IT (~4B effectif)"),
    "ministral-8B":     ("mistralai/Ministral-8B-Instruct-2410",  "Ministral 8B FR"),
    "qwen3.5-9B":       ("Qwen/Qwen3.5-9B-Instruct",              "Qwen3.5 dense 9B"),
    "gemma4-26B-A4B":   ("google/gemma-4-26B-A4B-it",             "Gemma 4 MoE 26B/4B"),
    "gemma4-31B":       ("google/gemma-4-31B-it",                 "Gemma 4 dense 31B"),
}

HERE         = Path(__file__).parent.resolve()
RESULTS_DIR  = HERE / "results";  RESULTS_DIR.mkdir(exist_ok=True)
LOG_DIR      = HERE / "logs";     LOG_DIR.mkdir(exist_ok=True)
VLLM_PID_F   = LOG_DIR / "vllm.pid"

CSV_PATH = RESULTS_DIR / "comparison.csv"
CSV_HEADER = ["alias", "model_id", "TP", "FP", "FN",
              "precision", "recall", "f1",
              "latency_mean_s", "tokens_mean", "status"]


# ═══════════════════════════════════════════════════════════════════════
# UTILS vLLM
# ═══════════════════════════════════════════════════════════════════════

def kill_vllm() -> None:
    if not VLLM_PID_F.exists(): return
    try:
        pid = int(VLLM_PID_F.read_text())
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        time.sleep(5)
    except (ProcessLookupError, ValueError, PermissionError):
        pass
    VLLM_PID_F.unlink(missing_ok=True)


def start_vllm(model_id: str, log_path: Path, args) -> subprocess.Popen:
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_id,
        "--tensor-parallel-size", str(args.num_gpus),
        "--max-model-len", str(args.max_len),
        "--gpu-memory-utilization", str(args.gpu_util),
        "--port", str(args.port),
    ]
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
    VLLM_PID_F.write_text(str(proc.pid))
    return proc


def wait_vllm(proc: subprocess.Popen, port: int, timeout_s: int = 900) -> bool:
    health = f"http://localhost:{port}/health"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(health, timeout=2) as r:
                if r.status == 200: return True
        except Exception:
            pass
        time.sleep(5)
    return False


# ═══════════════════════════════════════════════════════════════════════
# PURGE cache HF
# ═══════════════════════════════════════════════════════════════════════

def hf_cache_dir() -> Path:
    base = os.environ.get("HF_HUB_CACHE") or \
           (os.environ.get("HF_HOME", str(Path.home()/".cache/huggingface")) + "/hub")
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


# ═══════════════════════════════════════════════════════════════════════
# CSV
# ═══════════════════════════════════════════════════════════════════════

def init_csv_if_needed(regex_results: dict) -> None:
    if CSV_PATH.exists(): return
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f); w.writerow(CSV_HEADER)
        m = regex_results["metrics"]; t = regex_results["totals"]
        w.writerow(["regex-v3", "regex-v3-frozen", t["tp"], t["fp"], t["fn"],
                    round(m["precision"], 3), round(m["recall"], 3), round(m["f1"], 3),
                    "", "", "ok"])


def append_csv(row: dict) -> None:
    with open(CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([row.get(k, "") for k in CSV_HEADER])


# ═══════════════════════════════════════════════════════════════════════
# BASELINE regex V3
# ═══════════════════════════════════════════════════════════════════════

def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p  = tp/(tp+fp) if tp+fp else 0.0
    r  = tp/(tp+fn) if tp+fn else 0.0
    f1 = 2*p*r/(p+r) if p+r else 0.0
    return p, r, f1


def run_regex_baseline(gt_pairs: dict, recs: dict) -> dict:
    from iterate_regex import extract_pairs_v3
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for rid, gt in gt_pairs.items():
        pred = extract_pairs_v3(recs[rid]["text"])
        totals["tp"] += len(gt & pred)
        totals["fp"] += len(pred - gt)
        totals["fn"] += len(gt - pred)
    p, r, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    return {"totals": totals,
            "metrics": {"precision": p, "recall": r, "f1": f1}}


# ═══════════════════════════════════════════════════════════════════════
# ÉVALUATION D'UN MODÈLE
# ═══════════════════════════════════════════════════════════════════════

def evaluate_one_model(
    alias: str, hf_id: str, args,
    gt_pairs: dict, ann: dict, recs: dict,
) -> dict:
    from huggingface_hub import snapshot_download
    from openai import OpenAI
    from tqdm import tqdm
    from llm_extract_articles import extract_pairs_llm, CODE_SLUGS

    # 1) Download
    print("  ↳ download…"); t0 = time.time()
    snapshot_download(repo_id=hf_id,
                      ignore_patterns=["*.md", "*.txt", "original/*"])
    print(f"  ↳ download OK ({int(time.time()-t0)}s)")

    # 2) Start vLLM
    log_path = LOG_DIR / f"vllm_{alias}.log"
    print("  ↳ start vLLM…"); t0 = time.time()
    proc = start_vllm(hf_id, log_path, args)
    if not wait_vllm(proc, args.port, timeout_s=args.wait_timeout):
        raise RuntimeError(f"vLLM KO — voir {log_path}")
    print(f"  ↳ vLLM prêt ({int(time.time()-t0)}s)")

    # 3) Extraction
    client = OpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY")
    per_arret, latencies, tokens_counts = [], [], []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    slug_known = set(CODE_SLUGS.keys())

    for rid, gt in tqdm(list(gt_pairs.items()), desc=alias):
        pred, meta = extract_pairs_llm(
            recs[rid]["text"], client, hf_id, max_tokens=args.max_tokens_out)
        tp, fp, fn = gt & pred, pred - gt, gt - pred
        totals["tp"] += len(tp); totals["fp"] += len(fp); totals["fn"] += len(fn)
        if meta.get("latency_s") is not None: latencies.append(meta["latency_s"])
        if meta.get("tokens_used"): tokens_counts.append(meta["tokens_used"])
        fp_bad = sum(1 for pk in fp if pk.split(":", 1)[0] not in slug_known)
        per_arret.append({
            "id": rid, "jur": ann[rid]["jur"],
            "n_gt": len(gt), "n_pred": len(pred),
            "tp": len(tp), "fp": len(fp), "fn": len(fn),
            "missed": sorted(fn), "extra": sorted(fp),
            "fp_bad_slug": fp_bad, "fp_hallucination": len(fp) - fp_bad,
            "latency_s": meta.get("latency_s"),
            "tokens_used": meta.get("tokens_used"),
            "finish_reason": meta.get("finish_reason"),
            "error": meta.get("error"),
        })

    p, r, f1 = prf(totals["tp"], totals["fp"], totals["fn"])
    lat_mean = sum(latencies)/len(latencies) if latencies else None
    tok_mean = sum(tokens_counts)/len(tokens_counts) if tokens_counts else None

    return {
        "alias": alias, "model_id": hf_id,
        "metrics": {"precision": p, "recall": r, "f1": f1},
        "totals": totals,
        "latency_mean_s": lat_mean, "tokens_mean": tok_mean,
        "per_arret": per_arret,
    }


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", metavar="ALIAS",
                    help="Ne run que ces alias (défaut : tous)")
    ap.add_argument("--force", action="store_true",
                    help="Re-run même si results/<alias>.json existe")
    ap.add_argument("--max-len", type=int, default=32768)
    ap.add_argument("--max-tokens-out", type=int, default=2048)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-gpus", type=int, default=None,
                    help="Auto-détecté via nvidia-smi si absent")
    ap.add_argument("--wait-timeout", type=int, default=900,
                    help="Timeout démarrage vLLM (s)")
    ap.add_argument("--no-purge", action="store_true",
                    help="Ne pas supprimer le modèle après usage")
    args = ap.parse_args()

    # Auto-détection GPUs
    if args.num_gpus is None:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True)
            args.num_gpus = len([l for l in out.strip().split("\n") if l])
        except Exception:
            args.num_gpus = 1
    print(f"GPUs       : {args.num_gpus}")
    print(f"HF cache   : {hf_cache_dir()}")
    print(f"Résultats  : {RESULTS_DIR}")
    print(f"CSV        : {CSV_PATH}")

    # Imports des modules locaux
    sys.path.insert(0, str(HERE))

    # Charge GT + sample (chemins robustes local / parent)
    def _resolve(*candidates):
        for c in candidates:
            if c.exists(): return c
        raise FileNotFoundError(f"Aucun chemin : {candidates}")

    sample_path = _resolve(
        HERE / "cluster_data" / "regex_validation" / "sample_100.jsonl",
        HERE.parent / "cluster_data" / "regex_validation" / "sample_100.jsonl",
    )
    ann_path = _resolve(
        HERE / "manual_annotations.json",
        HERE.parent / "regex_v3" / "manual_annotations.json",
    )

    recs = {}
    with open(sample_path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line); recs[r["id"]] = r
    ann = json.loads(ann_path.read_text())["annotations"]
    gt_pairs = {rid: set(v["gt"]) for rid, v in ann.items()}
    print(f"Sample     : {len(recs)} arrêts  ·  GT : {len(ann)} annotés "
          f"({sum(len(s) for s in gt_pairs.values())} pair_keys)")

    # Baseline regex V3 (pour CSV + sanity)
    regex_results = run_regex_baseline(gt_pairs, recs)
    m = regex_results["metrics"]; t = regex_results["totals"]
    print(f"Regex V3   : TP={t['tp']} FP={t['fp']} FN={t['fn']} "
          f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")
    init_csv_if_needed(regex_results)

    # Sélection modèles
    aliases = args.only if args.only else list(MODEL_REGISTRY.keys())
    unknown = [a for a in aliases if a not in MODEL_REGISTRY]
    if unknown:
        print(f"[ERREUR] Alias inconnus : {unknown}")
        print(f"Dispo : {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    # ─── BOUCLE PRINCIPALE ───────────────────────────────────────────
    for alias in aliases:
        hf_id, note = MODEL_REGISTRY[alias]
        out_json = RESULTS_DIR / f"{alias}.json"
        if out_json.exists() and not args.force:
            print(f"\n[{alias}] déjà fait → skip ({out_json})")
            continue

        print(f"\n{'='*70}\n[{alias}] {hf_id}  |  {note}\n{'='*70}")
        try:
            kill_vllm()
            report = evaluate_one_model(alias, hf_id, args, gt_pairs, ann, recs)
            out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2))
            m = report["metrics"]; t = report["totals"]
            append_csv({
                "alias": alias, "model_id": hf_id,
                "TP": t["tp"], "FP": t["fp"], "FN": t["fn"],
                "precision": round(m["precision"], 3),
                "recall":    round(m["recall"], 3),
                "f1":        round(m["f1"], 3),
                "latency_mean_s": round(report["latency_mean_s"], 2) if report["latency_mean_s"] else "",
                "tokens_mean":    int(report["tokens_mean"]) if report["tokens_mean"] else "",
                "status": "ok",
            })
            lat = report["latency_mean_s"]
            print(f"  ✓ F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}"
                  + (f"  lat={lat:.2f}s" if lat else ""))
        except Exception as e:
            print(f"  ✗ ERREUR : {e}")
            append_csv({"alias": alias, "model_id": hf_id,
                        "status": f"error: {e}"[:200]})
        finally:
            kill_vllm()
            if not args.no_purge:
                purge_model(hf_id)

    print(f"\n✓ Terminé — {CSV_PATH}")
    # Affichage final du CSV
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        print("\n" + df.to_string(index=False))
    except ImportError:
        print(CSV_PATH.read_text())


if __name__ == "__main__":
    main()
