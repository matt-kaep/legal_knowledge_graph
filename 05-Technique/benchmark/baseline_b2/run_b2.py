#!/usr/bin/env python3
"""Évalue la baseline B2 (embedding naïf) sur les 41 questions CRFPA.

Usage (depuis 05-Technique/benchmark/) :
    python baseline_b2/run_b2.py
    python baseline_b2/run_b2.py --k 3 5 10
    python baseline_b2/run_b2.py --k 5 --min-freq 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
BENCH_DIR   = HERE.parent
RUBRICS_DIR = BENCH_DIR / "data" / "rubrics"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BENCH_DIR / "crfpa_benchmark"))
from eval_rubric import evaluate

CSV_PATH   = RESULTS_DIR / "comparison_b2.csv"
CSV_HEADER = [
    "config", "k", "min_freq",
    "n_q",
    "S_retrieval_mean", "S_e2e_mean",
    "S_bar_art_mean", "S_bar_jp_mean",
    "art_core_mean", "art_expected_mean", "art_expert_mean",
    "jp_core_mean",  "jp_expected_mean",  "jp_expert_mean",
    "n_articles_mean", "n_jp_mean",
]


def load_all_questions() -> list[dict]:
    questions = []
    for f in sorted(RUBRICS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        questions.extend(data.get("questions", []))
    return questions


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_one_config(retriever, questions: list[dict], k: int, min_freq: int) -> dict:
    per_q = []
    for q in questions:
        t0 = time.time()
        canon  = retriever.query(q["question"], k=k, min_freq=min_freq)
        scores = evaluate(canon, q)
        per_q.append({
            "qid":     q["id"],
            "branche": q.get("branche"),
            "canon":   canon,
            "scores":  scores,
            "latency": round(time.time() - t0, 3),
        })

    means = {
        "S_retrieval":  _mean([r["scores"]["regime"]["retrieval"]             for r in per_q]),
        "S_e2e":        _mean([r["scores"]["regime"]["e2e"]                   for r in per_q]),
        "S_bar_art":    _mean([r["scores"]["articles"]["S_bar"]               for r in per_q]),
        "S_bar_jp":     _mean([r["scores"]["jurisprudences"]["S_bar"]         for r in per_q]),
        "art_core":     _mean([r["scores"]["articles"]["per_strate"]["core"]      for r in per_q]),
        "art_expected": _mean([r["scores"]["articles"]["per_strate"]["expected"]  for r in per_q]),
        "art_expert":   _mean([r["scores"]["articles"]["per_strate"]["expert"]    for r in per_q]),
        "jp_core":      _mean([r["scores"]["jurisprudences"]["per_strate"]["core"]     for r in per_q]),
        "jp_expected":  _mean([r["scores"]["jurisprudences"]["per_strate"]["expected"] for r in per_q]),
        "jp_expert":    _mean([r["scores"]["jurisprudences"]["per_strate"]["expert"]   for r in per_q]),
        "n_articles":   _mean([r["canon"]["_meta"]["n_articles"] for r in per_q]),
        "n_jp":         _mean([r["canon"]["_meta"]["n_jp"]       for r in per_q]),
    }
    return {"k": k, "min_freq": min_freq, "means": means, "per_question": per_q}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k",        nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--min-freq", type=int,  default=1)
    args = ap.parse_args()

    questions = load_all_questions()
    print(f"Questions CRFPA : {len(questions)}")

    sys.path.insert(0, str(HERE))
    from query_naive import NaiveRetriever
    retriever = NaiveRetriever()

    init_csv = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as cf:
        writer = csv.writer(cf)
        if init_csv:
            writer.writerow(CSV_HEADER)

        for k in args.k:
            config  = f"b2_k{k}_mf{args.min_freq}"
            out_json = RESULTS_DIR / f"{config}.json"

            print(f"\n{'='*60}\nConfig : k={k}, min_freq={args.min_freq}\n{'='*60}")
            result = run_one_config(retriever, questions, k=k, min_freq=args.min_freq)
            out_json.write_text(
                json.dumps(result, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

            m = result["means"]
            writer.writerow([
                config, k, args.min_freq,
                len(result["per_question"]),
                m["S_retrieval"], m["S_e2e"],
                m["S_bar_art"],   m["S_bar_jp"],
                m["art_core"],    m["art_expected"], m["art_expert"],
                m["jp_core"],     m["jp_expected"],  m["jp_expert"],
                m["n_articles"],  m["n_jp"],
            ])
            cf.flush()

            print(f"  S_retrieval = {m['S_retrieval']}  (plafond LLM ≈ 0.095)")
            print(f"  S̄_art={m['S_bar_art']}  S̄_jp={m['S_bar_jp']}")
            print(f"  N articles moy.={m['n_articles']}  N JP={m['n_jp']}")
            print(f"  → {out_json}")

    print(f"\n✓ Résultats → {CSV_PATH}")
    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        cols = ["config", "S_retrieval_mean", "S_bar_art_mean", "S_bar_jp_mean",
                "n_articles_mean", "n_jp_mean"]
        print("\n" + df[cols].to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
