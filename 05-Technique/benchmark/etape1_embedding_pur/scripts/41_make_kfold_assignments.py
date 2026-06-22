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

import graph_protocol  # noqa: E402


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--graph-version",
        default="canonical",
        help="Conserve pour compatibilite, mais les folds officiels sont partages et independants du graphe.",
    )
    parser.add_argument("--split", default=graph_protocol.OFFICIAL_TRAIN_SPLIT)
    parser.add_argument("--n-folds", type=int, default=graph_protocol.OFFICIAL_N_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.split != graph_protocol.OFFICIAL_TRAIN_SPLIT:
        parser.error(
            f"official folds must use split={graph_protocol.OFFICIAL_TRAIN_SPLIT}"
        )
    if args.n_folds != graph_protocol.OFFICIAL_N_FOLDS:
        parser.error(
            f"official folds must use n_folds={graph_protocol.OFFICIAL_N_FOLDS}"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    bench_dir = graph_protocol.resolve_official_train_bench_dir()
    questions = graph_protocol.load_bench_questions(bench_dir)
    df = build_fold_assignments(questions, n_folds=args.n_folds, seed=args.seed)
    out_csv, out_meta = graph_protocol.resolve_shared_fold_paths(args.split)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    meta = {
        "graph_version": args.graph_version,
        "split": graph_protocol.OFFICIAL_TRAIN_SPLIT,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "n_questions": int(len(df)),
        "source_bench_dir": str(bench_dir),
        "output_dir": str(out_csv.parent),
        "is_canonical_shared_protocol": True,
    }
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
