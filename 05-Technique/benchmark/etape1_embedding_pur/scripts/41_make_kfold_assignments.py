from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import graph_protocol  # noqa: E402


def normalize_question_text(text: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(text)).casefold()
    alphanumeric = "".join(char if char.isalnum() else " " for char in normalized)
    return " ".join(alphanumeric.split())


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _question_text(question: dict) -> str:
    for field in ("enonce", "question"):
        value = question.get(field)
        if value is not None and str(value).strip():
            return normalize_question_text(value)
    return ""


def _provenance_key(question: dict) -> tuple[str, str, str] | None:
    values = tuple(question.get(field) for field in ("source", "doc_id", "section_id"))
    if any(value is None or not str(value).strip() for value in values):
        return None
    return tuple(str(value) for value in values)


def _bucket(value: int) -> str:
    if value == 1:
        return "1"
    if 2 <= value <= 3:
        return "2-3"
    if value >= 4:
        return "4+"
    return "0"


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _build_groups(questions: list[dict]) -> list[dict]:
    indexed_questions = {str(question["qid"]): question for question in questions}
    if len(indexed_questions) != len(questions):
        raise ValueError("questions must have unique qid values")

    union_find = UnionFind(sorted(indexed_questions))
    provenance_members: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    text_members: dict[str, list[str]] = defaultdict(list)
    annotations: dict[str, dict[str, str | None]] = {}
    for qid, question in indexed_questions.items():
        provenance_key = _provenance_key(question)
        normalized_text = _question_text(question)
        annotations[qid] = {
            "provenance_fingerprint": _fingerprint(provenance_key) if provenance_key else None,
            "text_fingerprint": _fingerprint(normalized_text) if normalized_text else None,
        }
        if provenance_key:
            provenance_members[provenance_key].append(qid)
        if normalized_text:
            text_members[normalized_text].append(qid)

    for members in [*provenance_members.values(), *text_members.values()]:
        for qid in members[1:]:
            union_find.union(members[0], qid)

    members_by_root: dict[str, list[str]] = defaultdict(list)
    for qid in indexed_questions:
        members_by_root[union_find.find(qid)].append(qid)

    groups: list[dict] = []
    for members in members_by_root.values():
        member_qids = sorted(members)
        group_id = f"group_{_fingerprint(member_qids)}"
        group_questions = [indexed_questions[qid] for qid in member_qids]
        groups.append(
            {
                "group_id": group_id,
                "member_qids": member_qids,
                "questions": group_questions,
                "group_size": len(member_qids),
                "article_buckets": Counter(
                    _bucket(int(question.get("n_articles_strict", 0)))
                    for question in group_questions
                ),
                "jp_buckets": Counter(
                    _bucket(int(question.get("n_jp_resolues", 0)))
                    for question in group_questions
                ),
                "question_types": Counter(
                    str(question["question_type"])
                    for question in group_questions
                    if question.get("question_type") is not None
                ),
                "granularities": Counter(
                    str(question["granularity"])
                    for question in group_questions
                    if question.get("granularity") is not None
                ),
                "annotations": {qid: annotations[qid] for qid in member_qids},
            }
        )
    return groups


def _distribution_loss(
    counters: list[Counter[str]], target: dict[str, float], addition: Counter[str], fold: int
) -> float:
    return sum(
        (
            counters[candidate_fold][key]
            + (addition[key] if candidate_fold == fold else 0)
            - target[key]
        ) ** 2
        for candidate_fold in range(len(counters))
        for key in target
    )


def build_fold_assignments(
    questions: list[dict], n_folds: int = 5, seed: int = 42
) -> pd.DataFrame:
    if n_folds < 1:
        raise ValueError("n_folds must be positive")

    groups = _build_groups(questions)
    totals = {
        "article_buckets": sum((group["article_buckets"] for group in groups), Counter()),
        "jp_buckets": sum((group["jp_buckets"] for group in groups), Counter()),
        "question_types": sum((group["question_types"] for group in groups), Counter()),
        "granularities": sum((group["granularities"] for group in groups), Counter()),
    }
    targets = {
        name: {key: count / n_folds for key, count in counter.items()}
        for name, counter in totals.items()
    }
    fold_question_counts = [0] * n_folds
    fold_distributions = {
        name: [Counter() for _ in range(n_folds)] for name in totals
    }
    target_question_count = len(questions) / n_folds
    rows: list[dict] = []

    groups.sort(
        key=lambda group: (
            -group["group_size"],
            _fingerprint([seed, group["group_id"]]),
        )
    )
    for group in groups:
        def allocation_cost(fold: int) -> tuple[float, float, float, float, float, int, int]:
            question_loss = sum(
                (
                    fold_question_counts[candidate_fold]
                    + (group["group_size"] if candidate_fold == fold else 0)
                    - target_question_count
                ) ** 2
                for candidate_fold in range(n_folds)
            )
            article_loss = _distribution_loss(
                fold_distributions["article_buckets"],
                targets["article_buckets"],
                group["article_buckets"],
                fold,
            )
            jp_loss = _distribution_loss(
                fold_distributions["jp_buckets"],
                targets["jp_buckets"],
                group["jp_buckets"],
                fold,
            )
            question_type_loss = _distribution_loss(
                fold_distributions["question_types"],
                targets["question_types"],
                group["question_types"],
                fold,
            )
            granularity_loss = _distribution_loss(
                fold_distributions["granularities"],
                targets["granularities"],
                group["granularities"],
                fold,
            )
            return (
                question_loss,
                article_loss,
                jp_loss,
                question_type_loss,
                granularity_loss,
                fold_question_counts[fold],
                fold,
            )

        fold = min(range(n_folds), key=allocation_cost)
        fold_question_counts[fold] += group["group_size"]
        for name in totals:
            fold_distributions[name][fold].update(group[name])
        for qid in group["member_qids"]:
            annotations = group["annotations"][qid]
            rows.append(
                {
                    "qid": qid,
                    "fold": fold,
                    "group_id": group["group_id"],
                    "group_size": group["group_size"],
                    "provenance_fingerprint": annotations["provenance_fingerprint"],
                    "text_fingerprint": annotations["text_fingerprint"],
                }
            )

    rows.sort(key=lambda row: row["qid"])
    return pd.DataFrame(
        rows,
        columns=[
            "qid",
            "fold",
            "group_id",
            "group_size",
            "provenance_fingerprint",
            "text_fingerprint",
        ],
    )


def build_fold_metadata(
    assignments: pd.DataFrame, questions: list[dict], n_folds: int
) -> dict:
    fold_question_counts = {
        str(fold): int((assignments["fold"] == fold).sum())
        for fold in range(n_folds)
    }
    distribution_summary: dict[str, dict] = {}
    question_by_qid = {str(question["qid"]): question for question in questions}
    for fold in range(n_folds):
        fold_questions = [
            question_by_qid[qid]
            for qid in assignments.loc[assignments["fold"] == fold, "qid"]
        ]
        distribution_summary[str(fold)] = {
            "n_questions": fold_question_counts[str(fold)],
            "n_articles_strict": _counter_dict(Counter(
                _bucket(int(question.get("n_articles_strict", 0)))
                for question in fold_questions
            )),
            "n_jp_resolues": _counter_dict(Counter(
                _bucket(int(question.get("n_jp_resolues", 0)))
                for question in fold_questions
            )),
            "question_type": _counter_dict(Counter(
                str(question["question_type"])
                for question in fold_questions
                if question.get("question_type") is not None
            )),
            "granularity": _counter_dict(Counter(
                str(question["granularity"])
                for question in fold_questions
                if question.get("granularity") is not None
            )),
        }

    def crossing_fingerprints(column: str) -> int:
        grouped = assignments.dropna(subset=[column]).groupby(column)["fold"].nunique()
        return int((grouped > 1).sum())

    return {
        "n_questions": int(len(assignments)),
        "n_groups": int(assignments["group_id"].nunique()),
        "largest_group_size": int(assignments["group_size"].max()) if not assignments.empty else 0,
        "provenance_groups_crossing_folds": crossing_fingerprints("provenance_fingerprint"),
        "normalized_text_groups_crossing_folds": crossing_fingerprints("text_fingerprint"),
        "fold_question_counts": fold_question_counts,
        "fold_distribution_summary": distribution_summary,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--protocol-version", default=graph_protocol.PROTOCOL_VERSION)
    args = parser.parse_args(argv)
    if args.split != graph_protocol.OFFICIAL_TRAIN_SPLIT:
        parser.error(
            f"official folds must use split={graph_protocol.OFFICIAL_TRAIN_SPLIT}"
        )
    if args.n_folds != graph_protocol.OFFICIAL_N_FOLDS:
        parser.error(
            f"official folds must use n_folds={graph_protocol.OFFICIAL_N_FOLDS}"
        )
    if args.protocol_version != graph_protocol.PROTOCOL_VERSION:
        parser.error(
            f"official folds must use protocol-version={graph_protocol.PROTOCOL_VERSION}"
        )
    if args.seed != 42:
        parser.error("official grouped_v2 folds must use seed=42")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bench_dir = graph_protocol.resolve_official_train_bench_dir()
    questions = graph_protocol.load_bench_questions(bench_dir)
    assignments = build_fold_assignments(questions, n_folds=args.n_folds, seed=args.seed)
    out_csv, out_meta = graph_protocol.resolve_shared_fold_paths(
        args.split, args.protocol_version
    )
    out_dir = out_csv.parent
    parent = out_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    lock_path = parent / f".{out_dir.name}.publish.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError(f"official fold publication already running: {lock_path}") from exc
    tmp_dir: Path | None = None
    try:
        os.write(lock_fd, json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}).encode())
        if out_dir.exists():
            raise FileExistsError(
                f"official fold artifacts already exist: {out_dir}; use a new protocol version"
            )
        tmp_dir = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.", dir=parent))
        tmp_csv = tmp_dir / out_csv.name
        tmp_meta = tmp_dir / out_meta.name
        assignments.to_csv(tmp_csv, index=False)
        metadata = build_fold_metadata(assignments, questions, args.n_folds)
        metadata.update(
            {
                "protocol_version": args.protocol_version,
                "dataset_split": args.split,
                "dataset_sha256": _sha256_file(bench_dir / "bench_global.json"),
                "fold_assignment_sha256": _sha256_file(tmp_csv),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "n_folds": args.n_folds,
                "seed": args.seed,
                "graph_version": args.graph_version,
                "source_bench_dir": str(bench_dir),
                "output_dir": str(out_dir),
            }
        )
        tmp_meta.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        os.replace(tmp_dir, out_dir)
        tmp_dir = None
    finally:
        os.close(lock_fd)
        lock_path.unlink(missing_ok=True)
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir)
    print(out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
