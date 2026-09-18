"""G8 — prepare LLM-verified JP-JP link jobs from G4 candidates.

This script is intentionally CPU/DB-only. It extracts JP-JP candidate pairs from
G4-knn30, fetches compact Step1 decision cards from OVH PostgreSQL, and writes
JSONL shards ready for a vLLM/OpenAI-compatible judge job on the cluster.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from psycopg2.extras import execute_values

CODE_REPO = Path(
    os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4]))
).expanduser().resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).expanduser().resolve()
ETAPE1 = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DATA_ETAPE1 = DATA_REPO / "05-Technique/benchmark/etape1_embedding_pur"
if str(ETAPE1) not in sys.path:
    sys.path.insert(0, str(ETAPE1))

from etape1.db import connect  # noqa: E402

G8_VERSION = "G8-llm-JJ-knn30-issue-rule"
DEFAULT_SOURCE = DATA_ETAPE1 / "data/embedding_graphs/G4-knn30"
DEFAULT_OUT = DATA_ETAPE1 / "data/llm_verified_graphs" / G8_VERSION
PROMPT_VERSION = "g8_llm_jp_link_judge_v3"
PROMPT_PATH = ETAPE1 / "prompts/g8_llm_jp_link_judge_v3.txt"
DEFAULT_JUDGE_MODEL = "QuantTrio/gemma-4-31B-it-AWQ"


@dataclass(frozen=True)
class CandidatePair:
    left_id: str
    right_id: str
    left_rank: int | None
    right_rank: int | None
    similarity: float
    reciprocal: bool

    @property
    def pair_id(self) -> str:
        return f"{self.left_id}__{self.right_id}"


def compact_text(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def compact_articles(value: object, max_items: int) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(compact_text(item, 120))
        elif isinstance(item, dict):
            label = item.get("article") or item.get("text") or item.get("id") or item.get("code")
            if label:
                out.append(compact_text(label, 120))
        if len(out) >= max_items:
            break
    return out


def compact_arguments(raw_args: object, max_args: int, max_field_chars: int) -> list[dict[str, str]]:
    if not isinstance(raw_args, list):
        return []
    rows: list[dict[str, str]] = []
    for arg in raw_args:
        if not isinstance(arg, dict):
            continue
        argument = compact_text(arg.get("argument"), max_field_chars)
        reponse = compact_text(arg.get("reponse_juge"), max_field_chars)
        if not argument and not reponse:
            continue
        rows.append({"argument": argument, "reponse_juge": reponse})
        if len(rows) >= max_args:
            break
    return rows


def make_decision_card(
    source_id: str,
    synthese: str | None,
    raw: dict | None,
    *,
    max_args: int,
    max_arg_chars: int,
    max_field_chars: int,
) -> dict:
    raw = raw or {}
    return {
        "jp_id": source_id,
        "synthese_pour_avocat": compact_text(
            raw.get("synthese_pour_avocat") or synthese, max_field_chars
        ),
        "fondements_retenus": compact_text(raw.get("fondements_retenus"), max_field_chars),
        "cited_articles": compact_articles(raw.get("cited_articles"), max_items=8),
        "solution_resume": compact_text(raw.get("solution_resume"), max_field_chars),
        "arguments_parties": compact_arguments(
            raw.get("arguments_parties"), max_args=max_args, max_field_chars=max_arg_chars
        ),
    }


def select_pairs(
    pairs: list[CandidatePair],
    *,
    max_pairs: int | None,
    selection: str,
    seed: int,
) -> list[CandidatePair]:
    if max_pairs is None:
        return pairs
    if selection == "top":
        return pairs[:max_pairs]
    if selection != "rank-stratified":
        raise ValueError(f"Unsupported selection={selection!r}")
    buckets: list[tuple[int, int]] = [(1, 5), (6, 10), (11, 20), (21, 30)]
    rng = np.random.default_rng(seed)
    selected: list[CandidatePair] = []
    per_bucket = max(1, max_pairs // len(buckets))
    for low, high in buckets:
        bucket = [
            pair
            for pair in pairs
            if low <= min(pair.left_rank, pair.right_rank or pair.left_rank) <= high
        ]
        if len(bucket) <= per_bucket:
            selected.extend(bucket)
        else:
            idx = rng.choice(len(bucket), size=per_bucket, replace=False)
            selected.extend(bucket[int(i)] for i in idx.tolist())
    if len(selected) < max_pairs:
        seen = {pair.pair_id for pair in selected}
        rest = [pair for pair in pairs if pair.pair_id not in seen]
        selected.extend(rest[: max_pairs - len(selected)])
    return selected[:max_pairs]


def min_pair_rank(pair: CandidatePair) -> int:
    ranks = [rank for rank in (pair.left_rank, pair.right_rank) if rank is not None]
    if not ranks:
        raise ValueError(f"Candidate pair has no directed rank: {pair.pair_id}")
    return min(ranks)


def max_pair_rank(pair: CandidatePair) -> int:
    ranks = [rank for rank in (pair.left_rank, pair.right_rank) if rank is not None]
    if not ranks:
        raise ValueError(f"Candidate pair has no directed rank: {pair.pair_id}")
    return max(ranks)


def filter_pairs(
    pairs: list[CandidatePair],
    *,
    reciprocal_only: bool,
    min_min_rank: int | None,
    max_min_rank: int | None,
    excluded_pair_ids: set[str] | None = None,
) -> list[CandidatePair]:
    out = pairs
    if reciprocal_only:
        out = [pair for pair in out if pair.reciprocal]
    if min_min_rank is not None:
        out = [pair for pair in out if min_pair_rank(pair) >= min_min_rank]
    if max_min_rank is not None:
        out = [pair for pair in out if min_pair_rank(pair) <= max_min_rank]
    if excluded_pair_ids:
        out = [pair for pair in out if pair.pair_id not in excluded_pair_ids]
    return out


def extract_candidate_pairs(
    g4_dir: Path,
    *,
    max_pairs: int | None = None,
    selection: str = "top",
    seed: int = 42,
    reciprocal_only: bool = False,
    min_min_rank: int | None = None,
    max_min_rank: int | None = None,
    excluded_pair_ids: set[str] | None = None,
) -> list[CandidatePair]:
    directed_path = g4_dir / "directed_neighbors.npz"
    if not directed_path.exists():
        raise FileNotFoundError(
            f"Missing {directed_path}. Generate it with "
            "54_build_g4_embedding_graphs.py --knn 30 --export-directed-neighbors. "
            "The symmetrized G4 graph cannot recover directional ranks."
        )
    neighbors = np.load(directed_path, allow_pickle=False)
    embedded_node_indices = neighbors["embedded_node_indices"].astype(np.int64)
    neighbor_nodes = neighbors["neighbor_nodes"].astype(np.int64)
    neighbor_sims = neighbors["neighbor_sims"].astype(np.float32)
    if neighbor_nodes.shape != neighbor_sims.shape:
        raise ValueError("directed_neighbors has mismatched nodes/similarities")
    if neighbor_nodes.shape[0] != len(embedded_node_indices):
        raise ValueError("directed_neighbors row count does not match embedded_node_indices")
    if neighbor_nodes.shape[1] < 30:
        raise ValueError("directed_neighbors must contain at least 30 ranked neighbors")
    node_ids = np.load(g4_dir / "node_ids.npy", allow_pickle=True).astype(str)
    node_types = np.load(g4_dir / "node_types.npy", allow_pickle=True).astype(str)
    jp_ids = np.load(g4_dir / "jp_ids.npy", allow_pickle=True).astype(str)
    n_jp = len(jp_ids)
    jp_mask = node_types == "jp"
    if not np.all(jp_mask[:n_jp]):
        raise ValueError("Expected JP nodes first in G4 node order")

    directed_rank: dict[tuple[int, int], tuple[int, float]] = {}
    for local_src, src in enumerate(embedded_node_indices.tolist()):
        if src >= n_jp:
            continue
        for rank, (dst, sim) in enumerate(
            zip(neighbor_nodes[local_src, :30].tolist(), neighbor_sims[local_src, :30].tolist()),
            start=1,
        ):
            if dst < 0 or dst == src or dst >= n_jp:
                continue
            directed_rank[(int(src), int(dst))] = (rank, float(sim))

    by_pair: dict[tuple[str, str], CandidatePair] = {}
    for (src, dst), (rank, sim) in directed_rank.items():
        source_id = str(node_ids[src])
        target_id = str(node_ids[dst])
        left_id, right_id = sorted((source_id, target_id))
        if left_id == right_id:
            continue
        left_node = src if source_id == left_id else dst
        right_node = dst if target_id == right_id else src
        rev = directed_rank.get((dst, src))
        reciprocal = rev is not None
        left_rank = directed_rank.get((left_node, right_node), (None, None))[0]
        right_rank = directed_rank.get((right_node, left_node), (None, None))[0]
        pair_sim = max(sim, rev[1] if rev else sim)
        pair = CandidatePair(
            left_id=left_id,
            right_id=right_id,
            left_rank=int(left_rank) if left_rank is not None else None,
            right_rank=int(right_rank) if right_rank is not None else None,
            similarity=float(pair_sim),
            reciprocal=reciprocal,
        )
        key = (left_id, right_id)
        existing = by_pair.get(key)
        if existing is None or min_pair_rank(pair) < min_pair_rank(existing):
            by_pair[key] = pair

    pairs = sorted(
        by_pair.values(),
        key=lambda p: (not p.reciprocal, min_pair_rank(p), -p.similarity, p.pair_id),
    )
    pairs = filter_pairs(
        pairs,
        reciprocal_only=reciprocal_only,
        min_min_rank=min_min_rank,
        max_min_rank=max_min_rank,
        excluded_pair_ids=excluded_pair_ids,
    )
    return select_pairs(pairs, max_pairs=max_pairs, selection=selection, seed=seed)


def load_pair_ids(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "pair_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing pair_id column in {path}")
        return {str(row["pair_id"]) for row in reader if row.get("pair_id")}


def batched(items: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def fetch_decision_cards(
    jp_ids: list[str],
    *,
    max_args: int,
    max_arg_chars: int,
    max_field_chars: int,
    batch_size: int,
) -> dict[str, dict]:
    cards: dict[str, dict] = {}
    with connect() as conn, conn.cursor() as cur:
        for chunk in batched(jp_ids, batch_size):
            cur.execute("CREATE TEMP TABLE tmp_g8_jp_ids(source_id text PRIMARY KEY) ON COMMIT DROP")
            execute_values(
                cur,
                "INSERT INTO tmp_g8_jp_ids(source_id) VALUES %s ON CONFLICT DO NOTHING",
                [(item,) for item in chunk],
                page_size=5000,
            )
            cur.execute(
                """
                SELECT d.source_id, d.synthese_pour_avocat, d.step1_raw
                FROM tmp_g8_jp_ids ids
                JOIN jp_decisions d
                  ON d.source = 'judilibre' AND d.source_id = ids.source_id
                WHERE d.step1_raw IS NOT NULL
                """
            )
            for source_id, synthese, raw in cur.fetchall():
                cards[str(source_id)] = make_decision_card(
                    str(source_id),
                    synthese,
                    raw,
                    max_args=max_args,
                    max_arg_chars=max_arg_chars,
                    max_field_chars=max_field_chars,
                )
            cur.execute("DROP TABLE tmp_g8_jp_ids")
    return cards


def approx_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))


def write_jsonl_shards(jobs: list[dict], out_dir: Path, shard_size: int) -> tuple[list[str], list[str]]:
    jobs_dir = out_dir / "jobs"
    ids_dir = out_dir / "job_ids"
    if any(jobs_dir.glob("jobs-*.jsonl")) or any(ids_dir.glob("job-ids-*.jsonl")):
        raise FileExistsError(
            "Refusing to reuse existing job shards or job-id indexes; choose a fresh G8 Large output directory."
        )
    jobs_dir.mkdir(parents=True, exist_ok=True)
    ids_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    id_indexes: list[str] = []
    for idx, start in enumerate(range(0, len(jobs), shard_size)):
        path = jobs_dir / f"jobs-{idx:05d}.jsonl"
        ids_path = ids_dir / f"job-ids-{idx:05d}.jsonl"
        with path.open("w", encoding="utf-8") as fh, ids_path.open("w", encoding="utf-8") as ids_fh:
            for job in jobs[start : start + shard_size]:
                fh.write(json.dumps(job, ensure_ascii=False, separators=(",", ":")) + "\n")
                ids_fh.write(str(job["job_id"]) + "\n")
        written.append(str(path.relative_to(out_dir)))
        id_indexes.append(str(ids_path.relative_to(out_dir)))
    return written, id_indexes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g4-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--selection", choices=["top", "rank-stratified"], default="top")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reciprocal-only", action="store_true")
    parser.add_argument("--min-min-rank", type=int, default=None)
    parser.add_argument("--max-min-rank", type=int, default=None)
    parser.add_argument(
        "--exclude-pairs-csv",
        type=Path,
        default=None,
        help="Candidate CSV containing pair_id values that must not be regenerated.",
    )
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--db-batch-size", type=int, default=5000)
    parser.add_argument("--max-args", type=int, default=6)
    parser.add_argument("--max-arg-chars", type=int, default=800)
    parser.add_argument("--max-field-chars", type=int, default=900)
    parser.add_argument("--judge-model-id", default=DEFAULT_JUDGE_MODEL)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    protected_outputs = [
        args.out_dir / "candidate_pairs.csv",
        args.out_dir / "manifest.json",
        args.out_dir / "responses.jsonl",
        args.out_dir / "responses",
        args.out_dir / "jobs",
    ]
    existing = [path for path in protected_outputs if path.exists() and (not path.is_dir() or any(path.iterdir()))]
    if existing:
        raise FileExistsError(
            "Refusing to mix a new campaign with existing artifacts: "
            + ", ".join(str(path) for path in existing)
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    judge_contract = {
        "model_id": args.judge_model_id,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest(),
    }
    excluded_pair_ids = load_pair_ids(args.exclude_pairs_csv)
    pairs = extract_candidate_pairs(
        args.g4_dir,
        max_pairs=args.limit_pairs,
        selection=args.selection,
        seed=args.seed,
        reciprocal_only=args.reciprocal_only,
        min_min_rank=args.min_min_rank,
        max_min_rank=args.max_min_rank,
        excluded_pair_ids=excluded_pair_ids,
    )
    jp_ids = sorted({p.left_id for p in pairs} | {p.right_id for p in pairs})
    print(f"[g8-prepare] candidate_pairs={len(pairs)} unique_jp={len(jp_ids)}")
    cards = fetch_decision_cards(
        jp_ids,
        max_args=args.max_args,
        max_arg_chars=args.max_arg_chars,
        max_field_chars=args.max_field_chars,
        batch_size=args.db_batch_size,
    )
    print(f"[g8-prepare] cards_from_db={len(cards)}")

    candidate_csv = args.out_dir / "candidate_pairs.csv"
    with candidate_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "pair_id",
                "left_id",
                "right_id",
                "left_rank",
                "right_rank",
                "min_rank",
                "max_rank",
                "similarity",
                "reciprocal",
                "has_both_cards",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "pair_id": pair.pair_id,
                    "left_id": pair.left_id,
                    "right_id": pair.right_id,
                    "left_rank": pair.left_rank,
                    "right_rank": pair.right_rank or "",
                    "min_rank": min_pair_rank(pair),
                    "max_rank": max_pair_rank(pair),
                    "similarity": f"{pair.similarity:.8f}",
                    "reciprocal": int(pair.reciprocal),
                    "has_both_cards": int(pair.left_id in cards and pair.right_id in cards),
                }
            )

    jobs: list[dict] = []
    token_estimates: list[int] = []
    for pair in pairs:
        left = cards.get(pair.left_id)
        right = cards.get(pair.right_id)
        if left is None or right is None:
            continue
        payload = {
            "job_id": pair.pair_id,
            "prompt_version": PROMPT_VERSION,
            "candidate_source": "G4-knn30",
            "judge_contract": judge_contract,
            "left_id": pair.left_id,
            "right_id": pair.right_id,
            "candidate_metadata": {
                "left_rank": pair.left_rank,
                "right_rank": pair.right_rank,
                "min_rank": min_pair_rank(pair),
                "max_rank": max_pair_rank(pair),
                "similarity": round(pair.similarity, 8),
                "reciprocal": pair.reciprocal,
            },
            "decision_a": left,
            "decision_b": right,
        }
        token_estimates.append(approx_tokens(json.dumps(payload, ensure_ascii=False)))
        jobs.append(payload)

    shards, job_id_shards = write_jsonl_shards(jobs, args.out_dir, args.shard_size)
    stats = {
        "graph_version": G8_VERSION,
        "prompt_version": PROMPT_VERSION,
        "judge_contract": judge_contract,
        "candidate_source": str(args.g4_dir),
        "selection": args.selection,
        "seed": args.seed,
        "reciprocal_only": args.reciprocal_only,
        "min_min_rank": args.min_min_rank,
        "max_min_rank": args.max_min_rank,
        "excluded_pair_ids": len(excluded_pair_ids),
        "n_candidate_pairs": len(pairs),
        "n_jobs": len(jobs),
        "n_missing_card_pairs": len(pairs) - len(jobs),
        "n_unique_jp": len(jp_ids),
        "n_cards": len(cards),
        "shard_size": args.shard_size,
        "job_shards": shards,
        "job_id_shards": job_id_shards,
        "compact_fields": [
            "synthese_pour_avocat",
            "fondements_retenus",
            "cited_articles",
            "solution_resume",
            "arguments_parties.argument",
            "arguments_parties.reponse_juge",
        ],
        "limits": {
            "max_args": args.max_args,
            "max_arg_chars": args.max_arg_chars,
            "max_field_chars": args.max_field_chars,
        },
        "approx_input_tokens_per_pair": {
            "mean": float(np.mean(token_estimates)) if token_estimates else 0.0,
            "p50": float(np.percentile(token_estimates, 50)) if token_estimates else 0.0,
            "p95": float(np.percentile(token_estimates, 95)) if token_estimates else 0.0,
            "max": int(max(token_estimates)) if token_estimates else 0,
        },
        "build_seconds": round(time.time() - t0, 3),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
