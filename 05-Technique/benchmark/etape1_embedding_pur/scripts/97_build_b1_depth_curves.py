"""Derive B1 Hit@K curves from frozen top-100 rankings without retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


_configured_repo = os.environ.get("LKG_REPO")
CODE_REPO = Path(_configured_repo).resolve() if _configured_repo else Path(__file__).resolve().parents[4]
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _data_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_REPO / path


def _stable_unique(values: list[object]) -> list[str]:
    seen: set[str] = set()
    return [
        identifier
        for value in values
        if (identifier := str(value)) not in seen and not seen.add(identifier)
    ]


def _stable_sequence_sha256(values: list[str]) -> str:
    payload = json.dumps(_stable_unique(values), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _effective_candidate_order(representation_order_path: Path, graph_ids_path: Path) -> list[str]:
    """Rebuild A3's stable, graph-backed returnable order from sealed inputs."""
    representation_order = np.load(representation_order_path, allow_pickle=True).tolist()
    graph_ids = set(map(str, np.load(graph_ids_path, allow_pickle=True).tolist()))
    return _stable_unique([value for value in representation_order if str(value) in graph_ids])


def _deduplicated_prefix(ranking: list[str], k: int) -> list[str]:
    """B1 semantics: deduplicate the returned positions 1..K, not later ranks."""
    seen: set[str] = set()
    return [item for item in ranking[:k] if not (item in seen or seen.add(item))]


def hit_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    if not gold:
        raise ValueError("B1 Hit@K requires at least one strict gold label")
    return len(set(_deduplicated_prefix(ranking, k)) & gold) / float(min(len(gold), k))


def mrr_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    """MRR with the same returned-position and deduplication convention as Hit@K."""
    if not gold:
        raise ValueError("B1 MRR@K requires at least one strict gold label")
    for position, item in enumerate(_deduplicated_prefix(ranking, k), start=1):
        if item in gold:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranking: list[str], gold: set[str], k: int) -> float:
    """Binary NDCG with the same returned-position and deduplication convention as Hit@K."""
    if not gold:
        raise ValueError("B1 NDCG@K requires at least one strict gold label")
    ranked = _deduplicated_prefix(ranking, k)
    dcg = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked, start=1)
        if item in gold
    )
    ideal_count = min(len(gold), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_count + 1))
    return dcg / idcg


def _ranking_groups(frame: pd.DataFrame, *, source: str) -> list[tuple[str, str, str, pd.DataFrame]]:
    required = {"qid", "modality", "rank", "item_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source}: ranking file missing {missing}")
    rows: list[tuple[str, str, str, pd.DataFrame]] = []
    for modality, target in (("art", "articles"), ("jp", "jurisprudence")):
        sub = frame.loc[frame["modality"].astype(str).eq(modality)].copy()
        if "selected_target" in sub.columns:
            sub = sub.loc[sub["selected_target"].astype(str).eq("art" if modality == "art" else "jp")]
        if sub.empty:
            raise ValueError(f"{source}: no frozen ranking rows for target={target}")
        seed_column = "replay_seed" if "replay_seed" in sub.columns else None
        if seed_column is None:
            rows.append((source, target, "single", sub))
            continue
        for seed, seed_rows in sub.groupby(seed_column, dropna=False, sort=True):
            label = "single" if pd.isna(seed) else str(int(seed))
            rows.append((source, target, label, seed_rows.copy()))
    return rows


def score_ranking_group(
    frame: pd.DataFrame,
    *,
    questions: dict[str, dict],
    candidate_ids: set[str],
    target: str,
    max_k: int = 100,
) -> pd.DataFrame:
    """Validate a complete top-100 ranking and return per-question Hit@K values."""
    expected_qids = set(questions)
    observed_qids = set(frame["qid"].astype(str))
    if observed_qids != expected_qids:
        raise ValueError(
            f"ranking question coverage mismatch: missing={len(expected_qids - observed_qids)} "
            f"extra={len(observed_qids - expected_qids)}"
        )
    gold_field = "articles_attendus" if target == "articles" else "gold_jp_ids"
    records: list[dict] = []
    for qid, group in frame.groupby(frame["qid"].astype(str), sort=False):
        group = group.sort_values("rank", kind="stable")
        ranks = group["rank"].astype(int).tolist()
        if ranks != list(range(1, max_k + 1)):
            raise ValueError(f"qid={qid}: expected exactly ranks 1..{max_k}, got {ranks[:3]}…{ranks[-3:]}")
        ranked = group["item_id"].astype(str).tolist()
        if len(set(ranked)) != len(ranked):
            raise ValueError(f"qid={qid}: duplicate item in frozen top-{max_k}")
        outside = set(ranked) - candidate_ids
        if outside:
            raise ValueError(f"qid={qid}: candidate outside official universe: {sorted(outside)[:3]}")
        gold = {str(item) for item in questions[qid].get(gold_field, [])}
        mrr_at_10 = mrr_at_k(ranked, gold, 10)
        ndcg_at_10 = ndcg_at_k(ranked, gold, 10)
        for k in range(1, max_k + 1):
            records.append({
                "qid": qid,
                "k": k,
                "hit_at_k": hit_at_k(ranked, gold, k),
                "mrr_at_10": mrr_at_10 if k == 10 else None,
                "ndcg_at_10": ndcg_at_10 if k == 10 else None,
            })
    return pd.DataFrame(records)


def derive_curves(
    payload: dict,
    sources: dict[str, Path],
    out_dir: Path,
    *,
    render_plots: bool = False,
) -> dict[str, Path]:
    eval_path = _data_path(payload["datasets"]["evaluation"]["path"])
    questions_list = json.loads(eval_path.read_text(encoding="utf-8"))["questions"]
    questions = {str(question["qid"]): question for question in questions_list}
    if len(questions) != int(payload["datasets"]["evaluation"]["questions"]):
        raise ValueError("evaluation question count differs from B1 manifest")
    universe_paths = payload["candidate_inputs"]
    candidate_orders = {
        "articles": _effective_candidate_order(
            _data_path(universe_paths["articles_order"]["path"]),
            _data_path(universe_paths["shared_article_ids"]["path"]),
        ),
        "jurisprudence": _effective_candidate_order(
            _data_path(universe_paths["jurisprudence_order"]["path"]),
            _data_path(universe_paths["shared_jp_ids"]["path"]),
        ),
    }
    candidate_ids = {target: set(order) for target, order in candidate_orders.items()}
    expected_counts = payload["candidate_universe"]
    for target, label in (("articles", "Article"), ("jurisprudence", "JP")):
        if len(candidate_orders[target]) != int(expected_counts[target]["count"]):
            raise ValueError(f"{label} candidate count differs from B1 manifest")
        expected_hash = expected_counts[target].get("order_sha256")
        actual_hash = _stable_sequence_sha256(candidate_orders[target])
        if expected_hash is not None and actual_hash != expected_hash:
            raise ValueError(f"{label} candidate order differs from B1 manifest")

    per_seed: list[pd.DataFrame] = []
    source_hashes: dict[str, str] = {}
    for source, path in sources.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        source_hashes[source] = _sha256(path)
        frame = pd.read_parquet(path)
        for source_name, target, seed, group in _ranking_groups(frame, source=source):
            scored = score_ranking_group(
                group,
                questions=questions,
                candidate_ids=candidate_ids[target],
                target=target,
            )
            scored.insert(0, "source", source_name)
            scored.insert(1, "target", target)
            scored.insert(2, "seed", seed)
            per_seed.append(scored)
    per_seed_df = pd.concat(per_seed, ignore_index=True)
    curves = (
        per_seed_df.groupby(["source", "target", "k"], as_index=False)["hit_at_k"]
        .agg(mean="mean", seed_std="std", seeds="count")
        .sort_values(["target", "source", "k"], kind="stable")
    )
    curves["seed_std"] = curves["seed_std"].fillna(0.0)
    per_seed_metrics = (
        per_seed_df.loc[per_seed_df["k"].eq(10)]
        .groupby(["source", "target", "seed"], as_index=False)
        .agg(
            hit_at_10=("hit_at_k", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            mrr_at_10=("mrr_at_10", "mean"),
        )
        .sort_values(["target", "source", "seed"], kind="stable")
    )
    metrics_at_10 = (
        per_seed_metrics.groupby(["source", "target"], as_index=False)
        .agg(
            hit_at_10=("hit_at_10", "mean"),
            hit_at_10_seed_std=("hit_at_10", "std"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            ndcg_at_10_seed_std=("ndcg_at_10", "std"),
            mrr_at_10=("mrr_at_10", "mean"),
            mrr_at_10_seed_std=("mrr_at_10", "std"),
            seeds=("seed", "count"),
        )
        .sort_values(["target", "source"], kind="stable")
    )
    for column in ("hit_at_10_seed_std", "ndcg_at_10_seed_std", "mrr_at_10_seed_std"):
        metrics_at_10[column] = metrics_at_10[column].fillna(0.0)
    out_dir.mkdir(parents=True, exist_ok=False)
    per_seed_path = out_dir / "depth_curves_per_seed.csv"
    curves_path = out_dir / "depth_curves.csv"
    per_seed_metrics_path = out_dir / "ranking_metrics_at_10_per_seed.csv"
    metrics_path = out_dir / "ranking_metrics_at_10.csv"
    report_path = out_dir / "depth_curves_manifest.json"
    per_seed_df.to_csv(per_seed_path, index=False)
    curves.to_csv(curves_path, index=False)
    per_seed_metrics.to_csv(per_seed_metrics_path, index=False)
    metrics_at_10.to_csv(metrics_path, index=False)
    if render_plots:
        _plot(curves, out_dir)
    artifact_paths = {
        "depth_curves_per_seed.csv": per_seed_path,
        "depth_curves.csv": curves_path,
        "ranking_metrics_at_10_per_seed.csv": per_seed_metrics_path,
        "ranking_metrics_at_10.csv": metrics_path,
    }
    if render_plots:
        artifact_paths |= {
            "depth_curves.png": out_dir / "depth_curves.png",
            "depth_curves.pdf": out_dir / "depth_curves.pdf",
        }
    report_path.write_text(json.dumps({
        "campaign_id": payload["campaign_id"],
        "campaign_manifest_sha256": payload.get("_manifest_sha256"),
        "a3_sha256": payload.get("a3", {}).get("sha256"),
        "derivation_script_sha256": _sha256(Path(__file__)),
        "evaluation_sha256": payload["datasets"]["evaluation"]["sha256"],
        "sources": source_hashes,
        "artifacts": {name: _sha256(path) for name, path in artifact_paths.items()},
        "plots_rendered": render_plots,
        "max_k": 100,
        "metric": "Hit@K = |dedup(R_K) intersect Y| / min(|Y|, K)",
        "secondary_metrics": ["NDCG@10", "MRR@10"],
        "coverage": {"questions": len(questions), "articles": len(candidate_ids["articles"]), "jurisprudence": len(candidate_ids["jurisprudence"])},
        "candidate_order_sha256": {
            target: _stable_sequence_sha256(order) for target, order in candidate_orders.items()
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "curves": curves_path,
        "per_seed": per_seed_path,
        "metrics_at_10": metrics_path,
        "metrics_at_10_per_seed": per_seed_metrics_path,
        "manifest": report_path,
    }


def _plot(curves: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, target, title in zip(axes, ("articles", "jurisprudence"), ("Articles", "Jurisprudence"), strict=True):
        for source, part in curves.loc[curves["target"].eq(target)].groupby("source", sort=True):
            ax.plot(part["k"], part["mean"], label=source)
        for k in (10, 50, 100):
            ax.axvline(k, color="0.75", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel("Hit@K")
        ax.set_xlim(1, 100)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "depth_curves.png", dpi=200)
    fig.savefig(out_dir / "depth_curves.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=RANKINGS_PARQUET")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--render-plots", action="store_true")
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    sources: dict[str, Path] = {}
    for raw in args.source:
        name, sep, path = raw.partition("=")
        if not sep or not name or not path:
            parser.error("--source must have NAME=RANKINGS_PARQUET form")
        if name in sources:
            parser.error(f"duplicate --source name: {name}")
        sources[name] = Path(path)
    payload["_manifest_sha256"] = _sha256(args.manifest)
    out_dir = args.out_dir or _data_path(payload["outputs"]["depth_curves"])
    outputs = derive_curves(payload, sources, out_dir, render_plots=args.render_plots)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
