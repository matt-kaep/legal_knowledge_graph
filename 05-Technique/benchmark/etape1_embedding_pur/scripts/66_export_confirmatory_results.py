"""Generate diagnostics and paper-facing exports from completed grouped-v2 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


CODE_REPO = Path(
    os.environ.get(
        "LKG_CODE_REPO",
        os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4])),
    )
)
REPO = Path(
    os.environ.get(
        "LKG_DATA_ROOT",
        os.environ.get("LKG_REPO", str(Path(__file__).resolve().parents[4])),
    )
)
BENCH = REPO / "05-Technique/benchmark/etape1_embedding_pur"
CODE_BENCH = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
DEFAULT_MANIFEST = CODE_BENCH / "configs/confirmatory_campaign_grouped_v2_repro_v1.json"
EXPERIMENT_REGISTRY = CODE_REPO / "01-Projet/paper-control/REGISTRE-EXPERIENCES.csv"
RESULT_REGISTRY = CODE_REPO / "01-Projet/paper-control/REGISTRE-RESULTATS.csv"
DEFAULT_KS = [5, 10, 15, 20, 30, 50, 80, 100, 150, 200, 300, 400, 500, 800, 1000]


def portable_artifact_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def expected_coverage_rows(
    rankings: pd.DataFrame,
    expected_by_qid_modality: dict[tuple[str, str], set[str]],
    *,
    ks: Iterable[int] = DEFAULT_KS,
) -> pd.DataFrame:
    required = {"qid", "method", "modality", "rank", "item_id"}
    missing = required - set(rankings.columns)
    if missing:
        raise KeyError(f"Missing ranking columns: {sorted(missing)}")
    rows = []
    group_columns = [
        column
        for column in [
            "graph_version", "family", "run_id", "selection_target", "champion_id",
            "method", "k_in", "modality", "qid",
        ]
        if column in rankings.columns
    ]
    for keys, group in rankings.groupby(group_columns, dropna=False, sort=False):
        keys = (keys,) if len(group_columns) == 1 else keys
        identity = dict(zip(group_columns, keys, strict=True))
        qid = str(identity["qid"])
        modality = str(identity["modality"])
        expected = expected_by_qid_modality.get((qid, modality), set())
        if not expected:
            continue
        ordered = group.sort_values("rank", kind="stable")
        for k in ks:
            retrieved = set(ordered.loc[ordered["rank"] <= int(k), "item_id"].astype(str))
            overlap = retrieved & expected
            rows.append(
                {
                    **identity,
                    "k": int(k),
                    "n_expected": len(expected),
                    "n_expected_retrieved": len(overlap),
                    "expected_coverage_at_k": len(overlap) / len(expected),
                    "any_expected_answer_at_k": float(bool(overlap)),
                }
            )
    return pd.DataFrame(rows)


def validate_paper_rows(rows: pd.DataFrame, manifest: dict | None = None) -> None:
    required = {
        "family",
        "graph_version",
        "target",
        "protocol_version",
        "dataset_sha256",
        "fold_assignment_sha256",
        "source_artifact",
        "experiment_id",
        "scientific_status",
        "manifest_sha256",
        "internal_eval_sha256",
        "graph_matrix_sha256",
        "eligible_champion",
        "question_coverage",
    }
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"Missing paper provenance column: {missing[0]}")
    if rows.empty:
        raise ValueError("No confirmatory rows available for paper export")
    if not rows["protocol_version"].eq("grouped_v2").all():
        raise ValueError("paper export contains a non-grouped_v2 row")
    if not rows["scientific_status"].isin(["confirmee_interne", "refutee"]).all():
        raise ValueError("paper export contains an unauthorized scientific_status")
    for column in required - {"eligible_champion", "question_coverage"}:
        if rows[column].isna().any() or rows[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"paper export contains empty provenance: {column}")
    if not rows["eligible_champion"].eq(True).all() or not rows["question_coverage"].eq(1.0).all():
        raise ValueError("paper export contains an incomplete champion")
    if manifest is not None:
        import hashlib
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        expected_manifest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        graph_hashes = {row["graph_id"]: row["matrix_sha256"] for row in manifest["graphs"]}
        if not rows["manifest_sha256"].eq(expected_manifest).all():
            raise ValueError("paper export manifest_sha256 mismatch")
        if not rows["internal_eval_sha256"].eq(manifest["datasets"]["internal_eval"]["sha256"]).all():
            raise ValueError("paper export internal_eval_sha256 mismatch")
        expected_graph_hashes = rows["graph_version"].map(graph_hashes)
        if expected_graph_hashes.isna().any() or not rows["graph_matrix_sha256"].eq(expected_graph_hashes).all():
            raise ValueError("paper export graph_matrix_sha256 mismatch")


def load_authorized_experiment_statuses(path: Path = EXPERIMENT_REGISTRY) -> dict[str, str]:
    registry = pd.read_csv(path)
    required = {"experiment_id", "statut", "artefact_principal"}
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"experiment registry missing {missing[0]}")
    authorized: dict[str, str] = {}
    for row in registry.to_dict(orient="records"):
        if row["statut"] not in {"confirmee_interne", "refutee"}:
            continue
        artifact_raw = row.get("artefact_principal")
        if pd.isna(artifact_raw) or not str(artifact_raw).strip():
            raise ValueError(f"authorized experiment lacks evidence: {row['experiment_id']}")
        artifact = Path(str(artifact_raw))
        artifact = artifact if artifact.is_absolute() else CODE_REPO / artifact
        if not artifact.is_file():
            raise ValueError(f"authorized experiment evidence is missing: {row['experiment_id']}")
        authorized[str(row["experiment_id"])] = str(row["statut"])
    return authorized


def load_authorized_result_verdicts(
    path: Path = RESULT_REGISTRY,
) -> dict[tuple[str, str, str, str], str]:
    registry = pd.read_csv(path)
    required = {
        "result_id", "experiment_id", "graph_version", "family", "target",
        "verdict", "source_artifact", "source_sha256",
    }
    missing = sorted(required - set(registry.columns))
    if missing:
        raise ValueError(f"result registry missing {missing[0]}")
    verdicts: dict[tuple[str, str, str, str], str] = {}
    for row in registry.to_dict(orient="records"):
        if row["verdict"] not in {"confirmee_interne", "refutee"}:
            continue
        artifact = Path(str(row["source_artifact"]))
        artifact = artifact if artifact.is_absolute() else CODE_REPO / artifact
        if not artifact.is_file():
            raise ValueError(f"authorized result evidence is missing: {row['result_id']}")
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != str(row["source_sha256"]):
            raise ValueError(f"authorized result evidence hash mismatch: {row['result_id']}")
        key = tuple(str(row[column]) for column in ("experiment_id", "graph_version", "family", "target"))
        if key in verdicts:
            raise ValueError(f"duplicate result classification: {key}")
        verdicts[key] = str(row["verdict"])
    return verdicts


def summarize_expected_coverage(detailed: pd.DataFrame) -> pd.DataFrame:
    identity_columns = [
        column for column in (
            "graph_version", "family", "run_id", "selection_target", "champion_id",
            "method", "modality", "k",
        ) if column in detailed.columns
    ]
    return (
        detailed.groupby(identity_columns, dropna=False)[
            ["expected_coverage_at_k", "any_expected_answer_at_k"]
        ]
        .mean()
        .reset_index()
    )


def primary_metric_plot_rows(rows: pd.DataFrame) -> pd.DataFrame:
    output = rows[["graph_version", "family", "target", "m1", "hit"]].copy()
    output["primary_metric"] = output["m1"].where(
        output["target"].eq("articles_strict"), output["hit"]
    )
    output["metric_name"] = output["target"].map(
        {"articles_strict": "Recall@10 Articles", "jp": "Hit@10 JP"}
    )
    return output


def write_primary_metric_figure(rows: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    plot_rows = primary_metric_plot_rows(rows).sort_values(
        ["metric_name", "primary_metric"], kind="stable"
    )
    labels = (
        plot_rows["graph_version"].astype(str)
        + " · "
        + plot_rows["family"].astype(str)
        + " · "
        + plot_rows["metric_name"].astype(str)
    )
    height = max(4.0, 0.28 * len(plot_rows))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(labels, plot_rows["primary_metric"])
    ax.set_xlabel("Métrique primaire sur évaluation interne")
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _expected_answers(eval_path: Path) -> dict[tuple[str, str], set[str]]:
    questions = json.loads(eval_path.read_text(encoding="utf-8"))["questions"]
    expected = {}
    for question in questions:
        qid = str(question["qid"])
        expected[(qid, "art")] = set(map(str, question.get("articles_attendus") or []))
        expected[(qid, "jp")] = set(map(str, question.get("gold_jp_ids") or []))
    return expected


def _write_diagnostics(manifest: dict, out_root: Path) -> None:
    final_root = REPO / manifest["outputs"]["final_root"]
    eval_path = REPO / manifest["datasets"]["internal_eval"]["path"]
    expected = _expected_answers(eval_path)
    frames = []
    for graph in manifest["graphs"]:
        path = final_root / graph["graph_id"] / "rankings.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        rankings = pd.read_parquet(path)
        rankings["graph_version"] = graph["graph_id"]
        frames.append(expected_coverage_rows(rankings, expected))
    detailed = pd.concat(frames, ignore_index=True)
    summary = summarize_expected_coverage(detailed)
    out_root.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(out_root / "expected_coverage_by_question.csv", index=False)
    summary.to_csv(out_root / "expected_coverage_by_k.csv", index=False)


def _write_paper_exports(manifest: dict, out_root: Path) -> None:
    final_root = REPO / manifest["outputs"]["final_root"]
    frames = []
    experiment_ids = {"ppr": "E002", "lightgcn": "E003", "b3_b4": "E014"}
    protocol_complete = load_authorized_experiment_statuses()
    result_verdicts = load_authorized_result_verdicts()
    for graph in manifest["graphs"]:
        path = final_root / graph["graph_id"] / "final_champions_summary.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["source_artifact"] = portable_artifact_path(path)
        frame["experiment_id"] = frame["family"].map(experiment_ids)
        if frame["experiment_id"].isna().any():
            raise ValueError("paper export contains an unmapped family")
        if not frame["experiment_id"].isin(protocol_complete).all():
            pending = sorted(frame.loc[~frame["experiment_id"].isin(protocol_complete), "experiment_id"].unique())
            raise ValueError(f"paper export awaits protocol evidence: {pending}")
        keys = zip(frame["experiment_id"], frame["graph_version"], frame["family"], frame["target"], strict=True)
        frame["scientific_status"] = [
            result_verdicts.get(tuple(map(str, key))) for key in keys
        ]
        if frame["scientific_status"].isna().any():
            raise ValueError("paper export awaits per-result scientific classification")
        frames.append(frame)
    rows = pd.concat(frames, ignore_index=True)
    validate_paper_rows(rows, manifest)
    out_root.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_root / "internal_eval_results.csv", index=False)
    rows[rows["target"] == "articles_strict"].to_csv(out_root / "internal_eval_articles.csv", index=False)
    rows[rows["target"] == "jp"].to_csv(out_root / "internal_eval_jp.csv", index=False)
    write_primary_metric_figure(rows, out_root / "internal_eval_primary_metrics.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--mode", choices=("diagnostics", "paper-exports"), required=True)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    out_root = CODE_REPO / manifest["outputs"]["export_root"]
    if args.mode == "diagnostics":
        _write_diagnostics(manifest, out_root / "diagnostics")
    else:
        _write_paper_exports(manifest, out_root / "paper")
    print(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
