#!/usr/bin/env python3
"""Prepare the deduplicated E017 graded-JP pool and exact E016 cache reuse."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


CODE_REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).expanduser().resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DATA = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
DEFAULT_TASKS = DATA / "_protocol/e017_intergraph_graded_jp/tasks.jsonl"
DEFAULT_RANKINGS_ROOT = DATA / "_final_e017_intergraph_graded_jp"
DEFAULT_BENCH = DATA / "eval_rich_retrievable_strict/bench_global.json"
DEFAULT_OUT = DATA / "E017-intergraph-graded-jp-v1"
DEFAULT_E016 = (
    DATA
    / "G7-citation-JJ-cit1-sem025-knn5"
    / "eval_rich_retrievable_strict/E016-g7-graded-jp-v1"
)
EXPECTED_TASKS = 33
EXPECTED_QUESTIONS = 754
DEFAULT_K = 10
CARD_FIELDS = (
    "synthese_pour_avocat",
    "fondements_retenus",
    "cited_articles",
    "solution_resume",
    "arguments_parties",
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _e016_preparation_module():
    return _load_module(
        ROOT / "scripts/75_prepare_g7_graded_jp_eval.py",
        "e016_graded_jp_preparation_for_e017",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _blind_card(card: dict) -> dict:
    return {
        field: card.get(
            field, [] if field in {"cited_articles", "arguments_parties"} else ""
        )
        for field in CARD_FIELDS
    }


def select_positions(
    rankings: pd.DataFrame,
    *,
    graph_id: str,
    seed: int,
    k: int,
    expected_questions: int,
) -> pd.DataFrame:
    required = {"qid", "modality", "selection_target", "rank", "item_id"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"rankings missing columns: {sorted(missing)}")
    if k <= 0:
        raise ValueError("K must be positive")

    selected = rankings.loc[
        (rankings["modality"].astype(str) == "jp")
        & (rankings["selection_target"].astype(str) == "jp")
        & (rankings["rank"].astype(int) <= k)
    ].copy()
    selected["qid"] = selected["qid"].astype(str)
    selected["rank"] = selected["rank"].astype(int)
    selected["jp_id"] = selected.pop("item_id").astype(str)
    selected["graph_id"] = str(graph_id)
    selected["seed"] = int(seed)
    selected = selected.sort_values(["qid", "rank"], kind="stable").reset_index(
        drop=True
    )

    if selected["qid"].nunique() != expected_questions:
        raise ValueError(
            f"{graph_id}/seed_{seed}: expected {expected_questions} questions, "
            f"found {selected['qid'].nunique()}"
        )
    expected_ranks = list(range(1, k + 1))
    for qid, group in selected.groupby("qid", sort=False):
        ranks = group["rank"].tolist()
        if ranks != expected_ranks:
            raise ValueError(
                f"{graph_id}/seed_{seed}/{qid}: expected ranks 1..{k}, found {ranks}"
            )
    selected["duplicate_position"] = selected.duplicated(
        ["qid", "jp_id"], keep="first"
    )
    return selected


def build_unique_jobs(
    *,
    positions: pd.DataFrame,
    questions: dict[str, dict],
    cards: dict[str, dict],
    judge_contract: dict,
) -> tuple[list[dict], pd.DataFrame]:
    frozen = positions.copy()
    frozen["job_id"] = [
        hashlib.sha256(f"{qid}\x1f{jp_id}".encode("utf-8")).hexdigest()
        for qid, jp_id in zip(frozen["qid"].astype(str), frozen["jp_id"].astype(str))
    ]
    frozen["card_status"] = frozen["jp_id"].map(
        lambda jp_id: "available" if str(jp_id) in cards else "missing"
    )

    jobs: list[dict] = []
    unique_pairs = (
        frozen[["qid", "jp_id", "job_id"]]
        .drop_duplicates(["qid", "jp_id"], keep="first")
        .sort_values(["qid", "jp_id"], kind="stable")
    )
    for row in unique_pairs.itertuples(index=False):
        qid = str(row.qid)
        jp_id = str(row.jp_id)
        question = questions.get(qid)
        if question is None:
            raise ValueError(f"missing benchmark question: {qid}")
        card = cards.get(jp_id)
        if card is None:
            continue
        jobs.append(
            {
                "job_id": str(row.job_id),
                "qid": qid,
                "jp_id": jp_id,
                "question": str(question.get("enonce") or "").strip(),
                "decision_card": _blind_card(card),
                "judge_contract": dict(judge_contract),
            }
        )
    return jobs, frozen


def split_exact_cache(
    jobs: list[dict], cached_jobs: list[dict], cached_responses: list[dict]
) -> tuple[list[dict], list[dict]]:
    cached_job_by_id = {
        str(job["job_id"]): job for job in cached_jobs if job.get("job_id")
    }
    ok_response_by_id: dict[str, dict] = {}
    for response in cached_responses:
        if response.get("job_id") and response.get("status") == "ok":
            ok_response_by_id[str(response["job_id"])] = response

    new_jobs: list[dict] = []
    reused: list[dict] = []
    for job in jobs:
        job_id = str(job["job_id"])
        cached_job = cached_job_by_id.get(job_id)
        response = ok_response_by_id.get(job_id)
        response_matches = bool(
            response
            and response.get("qid") == job.get("qid")
            and response.get("jp_id") == job.get("jp_id")
            and response.get("judge_contract") == job.get("judge_contract")
        )
        if cached_job == job and response_matches:
            reused.append(response)
        else:
            new_jobs.append(job)
    return new_jobs, reused


def load_questions(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(questions, list):
        raise ValueError("benchmark must contain a questions list")
    result = {str(row["qid"]): row for row in questions}
    if len(result) != len(questions):
        raise ValueError("duplicate qid in benchmark")
    return result


def load_tasks(path: Path) -> list[dict]:
    tasks = _read_jsonl(path)
    identities = {(str(row["graph_id"]), int(row["seed"])) for row in tasks}
    if len(tasks) != EXPECTED_TASKS or len(identities) != EXPECTED_TASKS:
        raise ValueError(
            f"expected {EXPECTED_TASKS} unique graph/seed tasks, found "
            f"{len(tasks)} rows and {len(identities)} identities"
        )
    return tasks


def collect_positions(
    *,
    tasks: list[dict],
    rankings_root: Path,
    k: int,
    expected_questions: int,
) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    for task in tasks:
        graph_id = str(task["graph_id"])
        seed = int(task["seed"])
        path = rankings_root / graph_id / f"seed_{seed}" / "rankings.parquet"
        if not path.is_file():
            raise FileNotFoundError(f"missing E017 ranking: {path}")
        selected = select_positions(
            pd.read_parquet(path),
            graph_id=graph_id,
            seed=seed,
            k=k,
            expected_questions=expected_questions,
        )
        selected["ranking_path"] = str(path)
        frames.append(selected)
        sources.append(
            {
                "graph_id": graph_id,
                "seed": seed,
                "path": str(path),
                "sha256": sha256(path),
            }
        )
    return pd.concat(frames, ignore_index=True), sources


def prepare_cards(jp_ids: set[str], e016_dir: Path) -> tuple[dict[str, dict], int]:
    cards_path = e016_dir / "decision_cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    cards = {str(key): value for key, value in cards.items() if str(key) in jp_ids}
    missing = sorted(jp_ids - set(cards))
    if missing:
        fetch_cards = _e016_preparation_module()._load_card_fetcher()
        cards.update(
            fetch_cards(
                missing,
                max_args=12,
                max_arg_chars=1_500,
                max_field_chars=4_000,
                batch_size=5_000,
            )
        )
    return cards, len(missing)


def write_outputs(
    *,
    out_dir: Path,
    positions: pd.DataFrame,
    all_jobs: list[dict],
    new_jobs: list[dict],
    reused_responses: list[dict],
    cards: dict[str, dict],
    manifest: dict,
    force: bool,
) -> None:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite frozen campaign: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    positions_path = out_dir / "rankings_topk.parquet"
    catalog_path = out_dir / "judgment_catalog.jsonl"
    jobs_path = out_dir / "judge_jobs.jsonl"
    reused_path = out_dir / "reused_responses.jsonl"
    responses_path = out_dir / "judge_responses.jsonl"
    cards_path = out_dir / "decision_cards.json"

    positions.to_parquet(positions_path, index=False)
    _write_jsonl(catalog_path, all_jobs)
    _write_jsonl(jobs_path, new_jobs)
    _write_jsonl(reused_path, reused_responses)
    _write_jsonl(responses_path, reused_responses)
    cards_path.write_text(
        json.dumps(cards, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest["artifacts"] = {
        path.name: sha256(path)
        for path in (
            positions_path,
            catalog_path,
            jobs_path,
            reused_path,
            cards_path,
        )
    }
    manifest["mutable_outputs"] = {
        "judge_responses.jsonl": "append_only_seeded_from_reused_responses"
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--rankings-root", type=Path, default=DEFAULT_RANKINGS_ROOT)
    parser.add_argument("--bench", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--e016-cache", type=Path, default=DEFAULT_E016)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.bench)
    if len(questions) != EXPECTED_QUESTIONS:
        raise ValueError(
            f"expected {EXPECTED_QUESTIONS} benchmark questions, found {len(questions)}"
        )
    tasks = load_tasks(args.tasks)
    positions, ranking_sources = collect_positions(
        tasks=tasks,
        rankings_root=args.rankings_root,
        k=args.k,
        expected_questions=len(questions),
    )
    if set(positions["qid"]) != set(questions):
        raise ValueError("E017 ranking question set does not match the benchmark")

    jp_ids = set(positions["jp_id"].astype(str))
    cards, initially_missing_cards = prepare_cards(jp_ids, args.e016_cache)
    missing_cards = sorted(jp_ids - set(cards))
    if missing_cards:
        raise ValueError(
            f"missing {len(missing_cards)} decision cards after local preparation: "
            f"{missing_cards[:5]}"
        )

    e016 = _e016_preparation_module()
    judge_contract = e016.make_judge_contract(
        model_id=e016.DEFAULT_MODEL,
        model_revision=e016.DEFAULT_MODEL_REVISION,
    )
    all_jobs, frozen_positions = build_unique_jobs(
        positions=positions,
        questions=questions,
        cards=cards,
        judge_contract=judge_contract,
    )
    cached_jobs = _read_jsonl(args.e016_cache / "judge_jobs.jsonl")
    cached_responses = _read_jsonl(args.e016_cache / "judge_responses.jsonl")
    new_jobs, reused_responses = split_exact_cache(
        all_jobs, cached_jobs, cached_responses
    )

    expected_positions = len(tasks) * len(questions) * args.k
    if len(frozen_positions) != expected_positions:
        raise ValueError(
            f"expected {expected_positions} fixed-K positions, found {len(frozen_positions)}"
        )
    if len(all_jobs) != frozen_positions["job_id"].nunique():
        raise ValueError("unique job catalog does not cover every position job_id")

    manifest = {
        "experiment_id": "E017",
        "campaign_id": "e017-intergraph-graded-jp-v1-2026-08-11",
        "scientific_status": "exploratory_internal_evaluation",
        "split": "eval_rich_retrievable_strict",
        "k": args.k,
        "n_graphs": len({str(task["graph_id"]) for task in tasks}),
        "n_seeds": len({int(task["seed"]) for task in tasks}),
        "n_rankings": len(tasks),
        "n_questions_per_ranking": len(questions),
        "n_positions": len(frozen_positions),
        "n_duplicate_positions": int(frozen_positions["duplicate_position"].sum()),
        "n_unique_pairs": len(all_jobs),
        "n_unique_jp": len(jp_ids),
        "n_cards_reused_from_e016_or_available": len(cards) - initially_missing_cards,
        "n_cards_requested_from_local_database": initially_missing_cards,
        "n_cards_available": len(cards),
        "n_cache_reused": len(reused_responses),
        "n_new_jobs": len(new_jobs),
        "judge_contract": judge_contract,
        "sources": {
            "tasks": {"path": str(args.tasks), "sha256": sha256(args.tasks)},
            "benchmark": {"path": str(args.bench), "sha256": sha256(args.bench)},
            "e016_manifest": {
                "path": str(args.e016_cache / "manifest.json"),
                "sha256": sha256(args.e016_cache / "manifest.json"),
            },
            "rankings": ranking_sources,
            "preparation_script": {
                "path": str(Path(__file__)),
                "sha256": sha256(Path(__file__)),
            },
        },
    }
    write_outputs(
        out_dir=args.out_dir,
        positions=frozen_positions,
        all_jobs=all_jobs,
        new_jobs=new_jobs,
        reused_responses=reused_responses,
        cards=cards,
        manifest=manifest,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "out_dir": str(args.out_dir),
                "n_positions": len(frozen_positions),
                "n_unique_pairs": len(all_jobs),
                "n_cache_reused": len(reused_responses),
                "n_new_jobs": len(new_jobs),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
