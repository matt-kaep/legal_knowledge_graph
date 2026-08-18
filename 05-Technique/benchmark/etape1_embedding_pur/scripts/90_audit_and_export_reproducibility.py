"""Audit recovered benchmark artifacts and build lightweight reproducibility exports.

The command is read-only with respect to the data checkout. It writes only the
small, versioned audit and table outputs under ``results/`` in the code
checkout. The internal evaluation is deliberately labelled exploratory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CODE_REPO = Path(
    os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4]))
).expanduser().resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).expanduser().resolve()
BENCH = DATA_REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
DEFAULT_MANIFEST = (
    CODE_REPO
    / "05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json"
)
DEFAULT_OUTPUT = CODE_REPO / "results/benchmark-repro-v1"


def classify_evidence(*, exists: bool, complete: bool, scientific_status: str) -> str:
    """Map operational evidence to a non-ambiguous scientific state."""

    if not exists:
        return "missing"
    if scientific_status == "exploratory":
        return "exploratory"
    return "complete" if complete else "incomplete"


def portable_relative_path(path: Path, root: Path) -> str:
    """Return a repository-relative path, never an absolute local checkout path."""

    return str(path.resolve().relative_to(root.resolve()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_sha256(path: Path) -> str | None:
    return sha256_file(path) if path.is_file() else None


def manifest_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path(root: Path, raw: str) -> Path:
    candidate = Path(raw)
    return candidate if candidate.is_absolute() else root / candidate


def _hashed_record(path: Path, root: Path, expected: str | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": portable_relative_path(path, root) if path.exists() else str(path),
        "exists": path.is_file(),
    }
    if path.is_file():
        record["size_bytes"] = path.stat().st_size
        record["sha256"] = sha256_file(path)
        if expected is not None:
            record["expected_sha256"] = expected
            record["hash_matches"] = record["sha256"] == expected
    return record


def audit_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = load_json(manifest_path)
    records: list[dict[str, Any]] = []
    for entry in payload.get("datasets", {}).values():
        records.append(_hashed_record(_path(DATA_REPO, entry["path"]), DATA_REPO, entry["sha256"]))
    for entry in payload.get("shared_hybrid_ids", {}).values():
        records.append(_hashed_record(_path(DATA_REPO, entry["path"]), DATA_REPO, entry["sha256"]))
    for entry in payload.get("immutable_inputs", {}).values():
        records.append(_hashed_record(_path(DATA_REPO, entry["path"]), DATA_REPO, entry["sha256"]))
    for entry in payload.get("graphs", []):
        records.append(_hashed_record(_path(DATA_REPO, entry["matrix_path"]), DATA_REPO, entry["matrix_sha256"]))
    code_records = []
    for entry in payload.get("code_bundle", {}).values():
        code_records.append(_hashed_record(_path(CODE_REPO, entry["path"]), CODE_REPO, entry["sha256"]))
    all_records = records + code_records
    complete = all(record.get("hash_matches", False) for record in all_records)
    parent_path = _path(CODE_REPO, payload["parent_manifest"])
    parent_ok = parent_path.is_file() and sha256_file(parent_path) == payload["parent_manifest_sha256"]
    return {
        "campaign_id": payload.get("campaign_id"),
        "manifest_sha256": manifest_sha256(payload),
        "manifest_path": portable_relative_path(manifest_path, CODE_REPO),
        "manifest_status": "complete" if complete and parent_ok else "invalide",
        "parent_manifest": {
            "path": portable_relative_path(parent_path, CODE_REPO) if parent_path.exists() else str(parent_path),
            "hash_matches": parent_ok,
        },
        "data_inputs": records,
        "code_bundle": code_records,
        "verified_input_count": len(records),
        "verified_code_count": len(code_records),
    }


def _coverage(row: dict[str, Any]) -> bool:
    return (
        bool(row.get("eligible_champion"))
        and int(row.get("n_folds_covered", 0)) == int(row.get("expected_folds", 5))
        and int(row.get("n_questions_covered", 0)) == int(row.get("n_questions_expected", 5603))
    )


def audit_ppr_cv(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for graph in manifest["graphs"]:
        graph_id = graph["graph_id"]
        root = BENCH / "_cv_grouped_v2" / graph_id / "ppr"
        summary_path = root / "summary.csv"
        champion_path = root / "champions.json"
        complete = summary_path.is_file() and champion_path.is_file()
        champions = load_json(champion_path) if champion_path.is_file() else {}
        targets_complete = all(_coverage(champions.get(target, {})) for target in ("art", "jp"))
        complete = complete and targets_complete
        graph_rows.append({
            "graph_id": graph_id,
            "summary": portable_relative_path(summary_path, DATA_REPO) if summary_path.exists() else str(summary_path),
            "folds": champions.get("art", {}).get("n_folds_covered"),
            "questions": champions.get("art", {}).get("n_questions_covered"),
            "complete": complete,
        })
        if champion_path.is_file():
            for target in ("art", "jp"):
                row = dict(champions.get(target, {}))
                row.update({
                    "method_family": "PPR",
                    "graph_id": graph_id,
                    "target": "articles" if target == "art" else "jp",
                    "split": "train_augmented_retrievable_strict",
                    "source_artifact": portable_relative_path(champion_path, DATA_REPO),
                    "source_sha256": sha256_file(champion_path),
                    "scientific_status": "confirmatoire_en_cours",
                })
                table_rows.append(row)
    return {
        "status": "complete" if len(graph_rows) == len(manifest["graphs"]) and all(row["complete"] for row in graph_rows) else "incomplete",
        "n_graphs_expected": len(manifest["graphs"]),
        "n_graphs_complete": sum(row["complete"] for row in graph_rows),
        "graphs": graph_rows,
    }, table_rows


def _source_hashes_match(root: Path, expected: dict[str, str]) -> bool:
    return all((root / name).is_file() and sha256_file(root / name) == digest for name, digest in expected.items())


def _ranking_coverage(frame: pd.DataFrame, *, questions: int, k: int) -> bool:
    """Check exact question and rank coverage for one replay."""

    if len(frame) < questions * k or frame["qid"].nunique() != questions:
        return False
    expected_ranks = set(range(1, k + 1))
    return all(expected_ranks <= set(group["rank"].astype(int)) for _, group in frame.groupby("qid"))


def _e017_replay_configuration(graph_id: str, seed: int) -> dict[str, Any]:
    """Read the frozen replay contract without inferring it from filenames."""

    status_path = BENCH / "_final_e017_intergraph_graded_jp" / graph_id / f"seed_{seed}" / "status.json"
    champions_path = BENCH / "_final_e017_intergraph_graded_jp" / graph_id / f"seed_{seed}" / "champions.json"
    status = load_json(status_path) if status_path.is_file() else {}
    champions = load_json(champions_path) if champions_path.is_file() else {}
    task = status.get("task", {})
    return {
        "configuration": (
            f"LightGCN-trained_K{task.get('train_k')}-lr{task.get('learning_rate')}"
            f"-e{task.get('epochs')}-la{task.get('lambda_anchor')}"
            f"-neg-{task.get('negative_sampling_strategy')}"
        ),
        "train_k": task.get("train_k"),
        "learning_rate": task.get("learning_rate"),
        "epochs": task.get("epochs"),
        "lambda_anchor": task.get("lambda_anchor"),
        "negative_sampling_strategy": task.get("negative_sampling_strategy"),
        "replay_epochs_art": champions.get("art", {}).get("replay_epochs"),
        "replay_epochs_jp": champions.get("jp", {}).get("replay_epochs"),
        "epoch_selection_metric_art": champions.get("art", {}).get("epoch_selection_metric"),
        "epoch_selection_metric_jp": champions.get("jp", {}).get("epoch_selection_metric"),
        "configuration_artifact": portable_relative_path(status_path, DATA_REPO),
        "configuration_sha256": sha256_file(status_path) if status_path.is_file() else None,
        "champions_artifact": portable_relative_path(champions_path, DATA_REPO),
        "champions_sha256": sha256_file(champions_path) if champions_path.is_file() else None,
    }


def audit_e017() -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = BENCH / "E017-intergraph-graded-jp-v1"
    manifest_path = BENCH / "_protocol/e017_intergraph_graded_jp/campaign_manifest.json"
    summary_path = root / "e017_summary.json"
    metrics_path = root / "e017_graph_metrics.csv"
    seed_path = root / "e017_graph_seed_metrics.csv"
    per_question_path = root / "e017_per_question_metrics.csv"
    expected = {}
    summary = load_json(summary_path) if summary_path.is_file() else {}
    for key, filename in (("graph_seed_metrics", seed_path.name), ("graph_metrics", metrics_path.name), ("per_question_metrics", per_question_path.name)):
        if key in summary.get("artifact_hashes", {}):
            expected[filename] = summary["artifact_hashes"][key]
    hashes_ok = _source_hashes_match(root, expected)
    graph_metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    seed_metrics = pd.read_csv(seed_path) if seed_path.is_file() else pd.DataFrame()
    per_question = pd.read_csv(per_question_path) if per_question_path.is_file() else pd.DataFrame()
    replay_root = BENCH / "_final_e017_intergraph_graded_jp"
    replay_dirs = [p for p in replay_root.glob("*/seed_*") if p.is_dir()]
    complete_replays = 0
    for replay in replay_dirs:
        status_path = replay / "status.json"
        ranking_path = replay / "rankings.parquet"
        if status_path.is_file() and ranking_path.is_file() and load_json(status_path).get("status") == "complete":
            frame = pd.read_parquet(ranking_path, columns=["qid", "rank"])
            if _ranking_coverage(frame, questions=754, k=10):
                complete_replays += 1
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    complete = (
        manifest.get("experiment_id") == "E017"
        and len(graph_metrics) == 11
        and len(seed_metrics) == 33
        and len(per_question) == 33 * 754
        and len(replay_dirs) == 33
        and complete_replays == 33
        and hashes_ok
    )
    status = classify_evidence(exists=root.is_dir(), complete=complete, scientific_status="exploratory")
    audit = {
        "experiment_id": "E017",
        "status": status,
        "scientific_status": manifest.get("scientific_status", "exploratory_internal_evaluation"),
        "hashes_ok": hashes_ok,
        "graphs": int(graph_metrics["graph_id"].nunique()) if not graph_metrics.empty else 0,
        "seeds": int(seed_metrics["seed"].nunique()) if not seed_metrics.empty else 0,
        "replays_expected": 33,
        "replays_complete": complete_replays,
        "questions_per_replay": 754,
        "positions_per_replay": 7540,
        "source_artifact": portable_relative_path(metrics_path, DATA_REPO) if metrics_path.exists() else str(metrics_path),
        "source_sha256": sha256_file(metrics_path) if metrics_path.is_file() else None,
    }
    return audit, graph_metrics, seed_metrics, per_question


def _official_metrics(rankings: pd.DataFrame, benchmark_path: Path) -> dict[str, float]:
    benchmark = load_json(benchmark_path)
    gold = {str(q["qid"]): set(map(str, q.get("gold_jp_ids") or [])) for q in benchmark["questions"]}
    values = {"hit_at_10": [], "ndcg_at_10": [], "mrr_at_10": []}
    for qid, expected in gold.items():
        if not expected:
            continue
        ordered = rankings.loc[rankings["qid"].astype(str).eq(qid)].sort_values("rank")["jp_id"].astype(str).tolist()
        unique: list[str] = []
        for item in ordered:
            if item not in unique:
                unique.append(item)
            if len(unique) == 10:
                break
        values["hit_at_10"].append(len(set(unique) & expected) / min(len(expected), 10))
        first = next((index for index, item in enumerate(unique, 1) if item in expected), None)
        values["mrr_at_10"].append(1.0 / first if first is not None else 0.0)
        dcg = sum(1.0 / math.log2(index + 2) for index, item in enumerate(unique) if item in expected)
        idcg = sum(1.0 / math.log2(index + 2) for index in range(min(len(expected), 10)))
        values["ndcg_at_10"].append(dcg / idcg if idcg else float("nan"))
    return {f"{key}_mean": float(pd.Series(items).mean()) for key, items in values.items()}


def _unique_judged_pairs(
    summary: dict[str, Any], detail_rows: int, manifest: dict[str, Any] | None = None
) -> int:
    """Use the source summary for unique jobs, not repeated ranking positions."""

    for source in (summary, manifest or {}):
        for key in ("n_unique_judged_pairs", "n_unique_jobs", "n_jobs"):
            if source.get(key) is not None:
                return int(source[key])
    return int(detail_rows)


def audit_e016() -> tuple[dict[str, Any], dict[str, Any]]:
    root = BENCH / "G7-citation-JJ-cit1-sem025-knn5/eval_rich_retrievable_strict/E016-g7-graded-jp-v1"
    summary_path = root / "graded_jp_summary.json"
    manifest_path = root / "manifest.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    manifest = load_json(manifest_path) if manifest_path.is_file() else {}
    detail_path = root / "graded_jp_detail.csv"
    per_question_path = root / "graded_jp_per_question.csv"
    hashes_ok = (
        detail_path.is_file()
        and per_question_path.is_file()
        and sha256_file(detail_path) == summary.get("artifact_hashes", {}).get("detail")
        and sha256_file(per_question_path) == summary.get("artifact_hashes", {}).get("per_question")
    )
    audit_dir = root / "lawyer_audit"
    agreement_path = audit_dir / "lawyer_agreement.json"
    detail = pd.read_csv(detail_path) if detail_path.is_file() else pd.DataFrame()
    per_question = pd.read_csv(per_question_path) if per_question_path.is_file() else pd.DataFrame()
    complete = (
        len(detail) == int(summary.get("n_positions", len(detail)))
        and len(per_question) == int(summary.get("n_questions", len(per_question)))
        and int(manifest.get("n_jobs", len(detail))) == _unique_judged_pairs(summary, len(detail), manifest)
        and hashes_ok
    )
    status = classify_evidence(exists=root.is_dir(), complete=complete, scientific_status="exploratory")
    audit = {
        "experiment_id": "E016",
        "status": status,
        "scientific_status": "exploratory_en_attente_audit_avocat",
        "hashes_ok": hashes_ok,
        "questions": len(per_question),
        "positions": int(summary.get("n_positions", 0)),
        "unique_judged_pairs": _unique_judged_pairs(summary, len(detail), manifest),
        "exact_any_gold_at_10_mean": summary.get("exact_hit_at_10"),
        "lawyer_sample_exists": (audit_dir / "lawyer_audit_sample.csv").is_file(),
        "lawyer_key_exists": (audit_dir / "lawyer_audit_key.csv").is_file(),
        "lawyer_agreement_exists": agreement_path.is_file(),
        "blocking_reason": "human_blind_lawyer_audit_not_completed",
        "source_artifact": portable_relative_path(summary_path, DATA_REPO),
        "source_sha256": optional_sha256(summary_path),
    }
    exact = {}
    rankings_path = root / "rankings_topk.parquet"
    benchmark_path = BENCH / "eval_rich_retrievable_strict/bench_global.json"
    if rankings_path.is_file() and benchmark_path.is_file():
        exact = _official_metrics(pd.read_parquet(rankings_path), benchmark_path)
        audit.update(exact)
    return audit, {**summary, **exact}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def build_outputs(manifest_path: Path, output_dir: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    manifest_audit = audit_manifest(manifest_path)
    ppr_audit, ppr_rows = audit_ppr_cv(manifest)
    e016_audit, e016_summary = audit_e016()
    e017_audit, e017_graphs, e017_seeds, _ = audit_e017()
    output_dir.mkdir(parents=True, exist_ok=True)

    ppr_frame = pd.DataFrame(ppr_rows)
    ppr_frame.to_csv(output_dir / "train_cv_retrieval.csv", index=False)
    if not e017_graphs.empty:
        configurations = pd.DataFrame(
            [_e017_replay_configuration(graph_id, 42) for graph_id in e017_graphs["graph_id"]]
        )
        e017_graphs = pd.concat([e017_graphs.reset_index(drop=True), configurations], axis=1)
        internal_articles = e017_graphs.assign(
            experiment_id="E017",
            method_family="LightGCN",
            target="articles",
            split="eval_rich_retrievable_strict",
            folds="5 (selection only)",
            seeds="42;43;44",
            questions=754,
            metric_primary="Recall@10",
            mean=e017_graphs["m1_recall_at_10_mean"],
            dispersion=e017_graphs["m1_recall_at_10_std"],
            recall_at_10=e017_graphs["m1_recall_at_10_mean"],
            ndcg_at_10=e017_graphs["ndcg_at_10_mean"],
            mrr_at_10=e017_graphs["mrr_at_10_mean"],
            exact_any_gold_at_10=e017_graphs["exact_any_gold_at_10_mean"],
            source_artifact=portable_relative_path(BENCH / "E017-intergraph-graded-jp-v1/e017_graph_metrics.csv", DATA_REPO),
            source_sha256=sha256_file(BENCH / "E017-intergraph-graded-jp-v1/e017_graph_metrics.csv"),
            scientific_status="exploratoire_agrege_en_attente_audit_avocat",
        )
        internal_articles[["experiment_id", "method_family", "graph_id", "target", "split", "folds", "seeds", "questions", "metric_primary", "mean", "dispersion", "recall_at_10", "ndcg_at_10", "mrr_at_10", "exact_any_gold_at_10", "configuration", "train_k", "learning_rate", "epochs", "lambda_anchor", "negative_sampling_strategy", "replay_epochs_art", "epoch_selection_metric_art", "configuration_artifact", "configuration_sha256", "source_artifact", "source_sha256", "scientific_status"]].to_csv(output_dir / "internal_eval_articles.csv", index=False)
        internal_jp = e017_graphs.assign(
            experiment_id="E017",
            method_family="LightGCN",
            target="jp",
            split="eval_rich_retrievable_strict",
            folds="5 (selection only)",
            seeds="42;43;44",
            questions=754,
            metric_primary="Hit@10",
            mean=e017_graphs["hit_at_10_mean"],
            dispersion=e017_graphs["hit_at_10_std"],
            hit_at_10=e017_graphs["hit_at_10_mean"],
            ndcg_at_10=e017_graphs["ndcg_at_10_mean"],
            mrr_at_10=e017_graphs["mrr_at_10_mean"],
            exact_any_gold_at_10=e017_graphs["exact_any_gold_at_10_mean"],
            source_artifact=portable_relative_path(BENCH / "E017-intergraph-graded-jp-v1/e017_graph_metrics.csv", DATA_REPO),
            source_sha256=sha256_file(BENCH / "E017-intergraph-graded-jp-v1/e017_graph_metrics.csv"),
            scientific_status="exploratoire_agrege_en_attente_audit_avocat",
        )
        internal_jp[["experiment_id", "method_family", "graph_id", "target", "split", "folds", "seeds", "questions", "metric_primary", "mean", "dispersion", "hit_at_10", "ndcg_at_10", "mrr_at_10", "exact_any_gold_at_10", "configuration", "train_k", "learning_rate", "epochs", "lambda_anchor", "negative_sampling_strategy", "replay_epochs_jp", "epoch_selection_metric_jp", "configuration_artifact", "configuration_sha256", "source_artifact", "source_sha256", "scientific_status"]].to_csv(output_dir / "internal_eval_jp_exact.csv", index=False)
        e017_llm = e017_graphs.assign(
            experiment_id="E017",
            method_family="LLM-as-a-Judge",
            target="jp",
            split="eval_rich_retrievable_strict",
            folds="5 (selection only)",
            seeds="42;43;44",
            questions=754,
            metric_primary="LLM-as-a-Judge@10",
            mean=e017_graphs["score_gradue_at_10_mean"],
            dispersion=e017_graphs["score_gradue_at_10_std"],
            denominator_k=10,
            label_contract="A=1;B=0.5;C/D/E/non_jugeable=0;repeat_gain=0",
            source_artifact=portable_relative_path(BENCH / "E017-intergraph-graded-jp-v1/e017_graph_metrics.csv", DATA_REPO),
            source_sha256=sha256_file(BENCH / "E017-intergraph-graded-jp-v1/e017_graph_metrics.csv"),
            scientific_status="exploratoire_agrege_en_attente_audit_avocat",
        )
        e017_llm[["experiment_id", "method_family", "graph_id", "target", "split", "folds", "seeds", "questions", "metric_primary", "mean", "dispersion", "denominator_k", "label_contract", "configuration", "train_k", "learning_rate", "epochs", "lambda_anchor", "negative_sampling_strategy", "replay_epochs_jp", "epoch_selection_metric_jp", "configuration_artifact", "configuration_sha256", "source_artifact", "source_sha256", "scientific_status"]].to_csv(output_dir / "internal_eval_jp_llm_as_a_judge.csv", index=False)

    e016_row = pd.DataFrame([{
        "experiment_id": "E016",
        "method_family": "LLM-as-a-Judge",
        "graph_id": "G7-citation-JJ-cit1-sem025-knn5",
        "target": "jp",
        "split": "eval_rich_retrievable_strict",
        "questions": e016_audit["questions"],
        "positions": e016_audit["positions"],
        "metric_primary": "LLM-as-a-Judge@10",
        "llm_score_mean": e016_summary.get("macro_score_gradue_at_10"),
        "exact_any_gold_at_10_mean": e016_summary.get("exact_hit_at_10"),
        "official_hit_at_10_mean": e016_summary.get("hit_at_10_mean"),
        "official_ndcg_at_10_mean": e016_summary.get("ndcg_at_10_mean"),
        "official_mrr_at_10_mean": e016_summary.get("mrr_at_10_mean"),
        "denominator_k": 10,
        "label_contract": "A=1;B=0.5;C/D/E/non_jugeable=0;repeat_gain=0",
        "source_artifact": e016_audit["source_artifact"],
        "source_sha256": e016_audit["source_sha256"],
        "scientific_status": e016_audit["scientific_status"],
    }])
    e016_row.to_csv(output_dir / "e016_jp_llm_and_exact_context.csv", index=False)

    audit = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": manifest_audit,
        "ppr_cv": ppr_audit,
        "e016": e016_audit,
        "e017": e017_audit,
        "final_grouped_v2": {"status": "missing", "path": "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_final_grouped_v2"},
        "scientific_notes": [
            "eval_rich_retrievable_strict is an already consulted internal evaluation, not an unseen lockbox.",
            "exact_any_gold_at_10 is retained as a separate binary diagnostic and is not official Hit@10.",
            "E016 and E017 LLM-as-a-Judge outputs remain exploratory until the blind lawyer audit is completed.",
            "PPR CV is complete on train-only folds; its internal-evaluation replay is missing and was not rerun under the local resource gate.",
        ],
    }
    write_json(output_dir / "audit.json", audit)
    write_json(output_dir / "manifest_snapshot.json", manifest)
    write_json(
        output_dir / "data-manifest.json",
        {
            "schema_version": 1,
            "campaign_id": manifest.get("campaign_id"),
            "provenance": "Inputs sealed by the campaign manifest and verified from the local data checkout.",
            "redistribution": {
                "status": "not_cleared",
                "license": "Do not redistribute benchmark text, legal decisions, or embeddings until each source license is verified.",
                "recovery": "Reconstruct or retrieve the files listed here, then run the preflight before computation.",
            },
            "artifacts": manifest_audit["data_inputs"],
        },
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    audit = build_outputs(args.manifest.resolve(), args.output_dir.resolve())
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "statuses": {key: value.get("status") for key, value in audit.items() if isinstance(value, dict)}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
