"""Explicit candidate-coverage contracts shared by benchmark runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence


LIGHTGCN_PROJECTION_FILENAME = "lightgcn_article_positive_projection.json"
LIGHTGCN_PROJECTION_SCHEMA = "lightgcn-article-positive-projection.v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_unique_strings(values: Iterable[object]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def stable_sequence_sha256(values: Iterable[object]) -> str:
    payload = json.dumps(
        stable_unique_strings(values),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _qid(question: Mapping[str, object]) -> str:
    qid = question.get("qid", question.get("id"))
    if qid is None:
        raise ValueError("question is missing qid/id")
    return str(qid)


def _labels(question: Mapping[str, object], raw_name: str, normalized_name: str) -> list[str]:
    values = question.get(raw_name)
    if values is None:
        values = question.get(normalized_name, [])
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{_qid(question)}: {raw_name} must be a sequence")
    return stable_unique_strings(values)


def strict_candidate_coverage_issues(
    questions: Sequence[Mapping[str, object]],
    *,
    article_candidate_ids: Iterable[object],
    jp_candidate_ids: Iterable[object],
) -> list[dict[str, object]]:
    article_set = set(stable_unique_strings(article_candidate_ids))
    jp_set = set(stable_unique_strings(jp_candidate_ids))
    issues: list[dict[str, object]] = []
    for question in questions:
        missing_articles = sorted(
            set(_labels(question, "articles_attendus", "gt_strict")) - article_set
        )
        missing_jp = sorted(
            set(_labels(question, "gold_jp_ids", "gold_jp_ids")) - jp_set
        )
        if missing_articles or missing_jp:
            issues.append(
                {
                    "qid": _qid(question),
                    "missing_articles_attendus": missing_articles,
                    "missing_gold_jp_ids": missing_jp,
                }
            )
    return sorted(issues, key=lambda item: str(item["qid"]))


def require_strict_candidate_coverage(
    questions: Sequence[Mapping[str, object]],
    *,
    article_candidate_ids: Iterable[object],
    jp_candidate_ids: Iterable[object],
    context: str,
) -> None:
    issues = strict_candidate_coverage_issues(
        questions,
        article_candidate_ids=article_candidate_ids,
        jp_candidate_ids=jp_candidate_ids,
    )
    if issues:
        raise ValueError(
            f"{context}: strict labels absent from candidate spaces: {issues[:5]}"
        )


def _projection_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "questions": len(rows),
        "extended_label_occurrences": sum(
            len(row["extended_article_ids"]) for row in rows
        ),
        "extended_labels_present": sum(
            len(row["retrievable_positive_article_ids"]) for row in rows
        ),
        "extended_labels_absent": sum(
            len(row["excluded_extended_article_ids"]) for row in rows
        ),
        "questions_with_absent_extended_labels": sum(
            bool(row["excluded_extended_article_ids"]) for row in rows
        ),
        "questions_without_retrievable_positive": sum(
            not row["retrievable_positive_article_ids"] for row in rows
        ),
    }


def build_lightgcn_article_positive_projection(
    questions: Sequence[Mapping[str, object]],
    *,
    article_candidate_ids: Iterable[object],
    bench_sha256: str,
) -> dict[str, object]:
    article_candidates = stable_unique_strings(article_candidate_ids)
    article_set = set(article_candidates)
    rows: list[dict[str, object]] = []
    for question in questions:
        strict = _labels(question, "articles_attendus", "gt_strict")
        extended = _labels(question, "articles_attendus_etendu", "gt_ext") or strict
        present = [article_id for article_id in extended if article_id in article_set]
        absent = [article_id for article_id in extended if article_id not in article_set]
        if not present:
            raise ValueError(
                f"{_qid(question)} has no retrievable LightGCN article positive"
            )
        rows.append(
            {
                "qid": _qid(question),
                "extended_article_ids": extended,
                "retrievable_positive_article_ids": present,
                "excluded_extended_article_ids": absent,
            }
        )
    rows.sort(key=lambda row: str(row["qid"]))
    return {
        "schema_version": LIGHTGCN_PROJECTION_SCHEMA,
        "policy": {
            "input_labels": "articles_attendus_etendu, falling back to articles_attendus only when the extended field is empty",
            "deduplication": "stable first occurrence per question",
            "positive_projection": "retain only labels present in the stable candidate sequence",
            "require_at_least_one_positive_per_training_question": True,
        },
        "source_bench_sha256": bench_sha256,
        "article_candidate_sequence_sha256": stable_sequence_sha256(article_candidates),
        "article_candidate_count": len(article_candidates),
        "counts": _projection_counts(rows),
        "rows": rows,
    }


def write_lightgcn_article_positive_projection(
    bench_dir: Path,
    questions: Sequence[Mapping[str, object]],
    *,
    article_candidate_ids: Iterable[object],
) -> dict[str, object]:
    projection = build_lightgcn_article_positive_projection(
        questions,
        article_candidate_ids=article_candidate_ids,
        bench_sha256=sha256_file(bench_dir / "bench_global.json"),
    )
    (bench_dir / LIGHTGCN_PROJECTION_FILENAME).write_text(
        json.dumps(projection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return projection


def write_subset_lightgcn_article_positive_projection(
    source_bench_dir: Path,
    destination_bench_dir: Path,
    *,
    qids: set[str],
) -> dict[str, object]:
    """Carry a projection into a CV subset while retaining its provenance hash."""
    source_path = source_bench_dir / LIGHTGCN_PROJECTION_FILENAME
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source LightGCN projection: {source_path}")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    rows = [row for row in source.get("rows", []) if str(row.get("qid")) in qids]
    destination_questions = json.loads(
        (destination_bench_dir / "bench_global.json").read_text(encoding="utf-8")
    )["questions"]
    destination_qids = {_qid(question) for question in destination_questions}
    if {str(row["qid"]) for row in rows} != destination_qids:
        raise ValueError("subset projection QIDs do not match subset benchmark")
    projection = dict(source)
    projection["source_bench_sha256"] = sha256_file(destination_bench_dir / "bench_global.json")
    projection["rows"] = sorted(rows, key=lambda row: str(row["qid"]))
    projection["counts"] = _projection_counts(projection["rows"])
    projection["derived_from_projection_sha256"] = sha256_file(source_path)
    (destination_bench_dir / LIGHTGCN_PROJECTION_FILENAME).write_text(
        json.dumps(projection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return projection


def load_verified_lightgcn_article_positive_projection(
    bench_dir: Path,
    *,
    article_candidate_ids: Iterable[object],
) -> tuple[dict[str, set[str]], dict[str, object], str]:
    projection_path = bench_dir / LIGHTGCN_PROJECTION_FILENAME
    if not projection_path.exists():
        raise FileNotFoundError(
            f"Missing explicit LightGCN positive projection: {projection_path}"
        )
    questions = json.loads((bench_dir / "bench_global.json").read_text(encoding="utf-8"))["questions"]
    expected = build_lightgcn_article_positive_projection(
        questions,
        article_candidate_ids=article_candidate_ids,
        bench_sha256=sha256_file(bench_dir / "bench_global.json"),
    )
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    comparable_keys = {
        "schema_version",
        "policy",
        "source_bench_sha256",
        "article_candidate_sequence_sha256",
        "article_candidate_count",
        "counts",
        "rows",
    }
    if {key: projection.get(key) for key in comparable_keys} != {
        key: expected[key] for key in comparable_keys
    }:
        raise ValueError(
            f"LightGCN positive projection does not match bench/candidate inputs: {projection_path}"
        )
    positives = {
        str(row["qid"]): set(row["retrievable_positive_article_ids"])
        for row in projection["rows"]
    }
    return positives, projection, sha256_file(projection_path)
