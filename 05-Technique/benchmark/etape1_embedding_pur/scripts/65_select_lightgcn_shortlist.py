"""Freeze the deterministic LightGCN graph shortlist from grouped-v2 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


REPO = Path(
    os.environ.get(
        "LKG_REPO",
        str(Path(__file__).resolve().parents[4]),
    )
)
BENCH = REPO / "05-Technique/benchmark/etape1_embedding_pur"
DEFAULT_MANIFEST = BENCH / "configs/confirmatory_campaign_grouped_v2_repro_v1.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_frozen_shortlist(path: Path, expected_manifest_sha256: str) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("manifest_sha256") != expected_manifest_sha256:
        raise ValueError("shortlist manifest_sha256 does not match the campaign manifest")
    graph_ids = payload.get("graph_ids")
    if not isinstance(graph_ids, list) or not graph_ids or len(graph_ids) != len(set(graph_ids)):
        raise ValueError("shortlist graph_ids must be a non-empty unique list")
    return [str(graph_id) for graph_id in graph_ids]


def _best_graph(rows: pd.DataFrame, modality: str) -> str:
    if modality == "art":
        order = ["article_recall_at_10_mean", "article_ndcg_at_10_mean", "article_mrr_at_10_mean"]
    elif modality == "jp":
        order = ["jp_hit_at_10_mean", "jp_ndcg_at_10_mean", "jp_mrr_at_10_mean"]
    else:
        raise ValueError(f"Unsupported modality={modality}")
    candidates = rows[
        rows["eligible_champion"].eq(True)
        & rows["modality"].eq(modality)
        & rows["graph_id"].astype(str).str.startswith("G7")
    ].copy()
    if candidates.empty:
        raise ValueError(f"No eligible G7 screening result for modality={modality}")
    missing = [column for column in order if column not in candidates.columns]
    if missing:
        raise KeyError(missing[0])
    return str(candidates.sort_values(order, ascending=False, kind="stable").iloc[0]["graph_id"])


def select_shortlist(
    rows: pd.DataFrame, *, always_include: list[str], max_graphs: int = 5
) -> list[str]:
    eligible_graphs = set(rows.loc[rows["eligible_champion"].eq(True), "graph_id"].astype(str))
    missing_controls = [graph for graph in always_include if graph not in eligible_graphs]
    if missing_controls:
        raise ValueError(f"Missing eligible LightGCN controls: {missing_controls}")
    selected = [*always_include, _best_graph(rows, "art"), _best_graph(rows, "jp")]
    deduplicated = list(dict.fromkeys(selected))
    if len(deduplicated) > max_graphs:
        raise ValueError(f"Shortlist exceeds max_graphs={max_graphs}")
    return deduplicated


def validate_screening_summary(frame: pd.DataFrame, manifest: dict, graph_id: str) -> None:
    required = {
        "modality", "eligible_champion", "protocol_version", "dataset_sha256",
        "fold_assignment_sha256", "n_folds_covered", "question_coverage",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"screening summary {graph_id} missing {missing[0]}")
    checks = {
        "protocol_version": frame["protocol_version"].eq(manifest["protocol_version"]).all(),
        "dataset_sha256": frame["dataset_sha256"].eq(manifest["datasets"]["train"]["sha256"]).all(),
        "fold_assignment_sha256": frame["fold_assignment_sha256"].eq(manifest["folds"]["sha256"]).all(),
        "modalities": {"art", "jp"}.issubset(set(frame["modality"])),
        "eligible_modalities": {
            "art", "jp"
        }.issubset(set(frame.loc[frame["eligible_champion"].eq(True), "modality"])),
        "coverage": not frame.loc[frame["eligible_champion"].eq(True)].empty and frame.loc[frame["eligible_champion"].eq(True), "question_coverage"].eq(1.0).all(),
        "folds": not frame.loc[frame["eligible_champion"].eq(True)].empty and frame.loc[frame["eligible_champion"].eq(True), "n_folds_covered"].eq(manifest["folds"]["count"]).all(),
    }
    failed = [name for name, valid in checks.items() if not valid]
    if failed:
        raise ValueError(f"screening summary {graph_id} failed {failed[0]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cv_root = REPO / manifest["outputs"]["cv_root"]
    frames = []
    sources = []
    for graph in manifest["graphs"]:
        summary_path = cv_root / graph["graph_id"] / "lightgcn_screen" / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        frame = pd.read_csv(summary_path)
        validate_screening_summary(frame, manifest, graph["graph_id"])
        frame["graph_id"] = graph["graph_id"]
        frames.append(frame)
        sources.append({"path": str(summary_path), "sha256": sha256_file(summary_path)})
    screening = pd.concat(frames, ignore_index=True)
    always_include = [str(value) for value in manifest["lightgcn"]["shortlist"]["always_include"]]
    known = {graph["graph_id"] for graph in manifest["graphs"]}
    unknown_controls = sorted(set(always_include) - known)
    if unknown_controls:
        raise ValueError(f"shortlist always_include contains unknown graphs: {unknown_controls}")
    graph_ids = select_shortlist(
        screening,
        always_include=always_include,
        max_graphs=int(manifest["lightgcn"]["shortlist"]["max_graphs"]),
    )
    payload = {
        "campaign_id": manifest["campaign_id"],
        "protocol_version": manifest["protocol_version"],
        "manifest_sha256": manifest_sha256(manifest),
        "graph_ids": graph_ids,
        "selection": {
            "always_include": always_include,
            "article": "Recall@10 > NDCG@10 > MRR@10",
            "jp": "Hit@10 > NDCG@10 > MRR@10",
        },
        "sources": sources,
    }
    out_path = (REPO / manifest["outputs"]["status_root"]).parent / "lightgcn_shortlist.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
