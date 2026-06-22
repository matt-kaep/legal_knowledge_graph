from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from graph_protocol import load_bench_questions, resolve_graph_bench_dir  # noqa: E402


def _group_seed(seed: int, n_articles_strict: int, n_jp_resolues: int) -> int:
    return (seed * 1_000_003) + (n_articles_strict * 1_009) + (n_jp_resolues * 9_173)


def build_fold_assignments(
    questions: list[dict], n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for q in questions:
        grouped[(int(q["n_articles_strict"]), int(q["n_jp_resolues"]))].append(q)

    rows: list[dict] = []
    fold_cursor = 0
    for signature in sorted(grouped):
        bucket = sorted(grouped[signature], key=lambda q: q["qid"])
        rng = random.Random(_group_seed(seed, *signature))
        rng.shuffle(bucket)
        for q in bucket:
            rows.append({"qid": q["qid"], "fold": fold_cursor % n_folds})
            fold_cursor += 1

    rows.sort(key=lambda row: row["qid"])
    return pd.DataFrame(rows, columns=["qid", "fold"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    bench_dir = resolve_graph_bench_dir(args.graph_version, args.split)
    questions = load_bench_questions(bench_dir)
    df = build_fold_assignments(questions, n_folds=args.n_folds, seed=args.seed)
    bench_dir.mkdir(parents=True, exist_ok=True)
    out_csv = bench_dir / "fold_assignments.csv"
    df.to_csv(out_csv, index=False)
    meta = {
        "graph_version": args.graph_version,
        "split": args.split,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_questions": int(len(df)),
    }
    (bench_dir / "fold_assignments_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
