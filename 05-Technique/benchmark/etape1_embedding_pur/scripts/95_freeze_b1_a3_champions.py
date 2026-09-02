"""Freeze one B1 champion per task from train/CV summaries only."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


CODE_REPO = Path(os.environ.get("LKG_REPO", Path(__file__).resolve().parents[4])).resolve()
DATA_REPO = Path(os.environ.get("LKG_DATA_ROOT", str(CODE_REPO))).resolve()
ROOT = CODE_REPO / "05-Technique/benchmark/etape1_embedding_pur"
SCRIPTS = ROOT / "scripts"
DEFAULT_MANIFEST = ROOT / "configs/confirmatory_campaign_b1_a3.json"


def _load_script(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _data_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else DATA_REPO / path


def _metric_column(frame: pd.DataFrame, names: list[str]) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"CV summary is missing one of {names}")


def _standardize_summary(
    frame: pd.DataFrame,
    *,
    graph_version: str,
    family: str,
    train_split: str,
) -> pd.DataFrame:
    if "modality" not in frame.columns:
        raise ValueError("CV summary is missing modality")
    rows: list[pd.DataFrame] = []
    for modality, target in (("art", "art"), ("jp", "jp")):
        sub = frame.loc[frame["modality"].astype(str).eq(modality)].copy()
        if sub.empty:
            continue
        if target == "art":
            primary = _metric_column(sub, ["article_recall_at_10_mean", "recall_at_10_mean"])
            ndcg = _metric_column(sub, ["article_ndcg_at_10_mean", "ndcg_at_10_mean"])
            mrr = _metric_column(sub, ["article_mrr_at_10_mean", "mrr_at_10_mean"])
        else:
            primary = _metric_column(sub, ["jp_hit_at_10_mean", "hit_at_10_mean"])
            ndcg = _metric_column(sub, ["jp_ndcg_at_10_mean", "ndcg_at_10_mean"])
            mrr = _metric_column(sub, ["jp_mrr_at_10_mean", "mrr_at_10_mean"])
        sub["target"] = target
        sub["graph_version"] = graph_version
        sub["family"] = family
        sub["dataset_split"] = train_split
        sub["primary_mean"] = sub[primary]
        sub["ndcg_at_10_mean"] = sub[ndcg]
        sub["mrr_at_10_mean"] = sub[mrr]
        rows.append(sub)
    if not rows:
        raise ValueError(f"No task rows in {family} CV summary for graph={graph_version}")
    return pd.concat(rows, ignore_index=True)


def _lightgcn_method(row: dict[str, Any]) -> str:
    return (
        f"LightGCN-{row['variant']}-K{int(row['train_k'])}-s{int(row['seed'])}"
        f"-lr{float(row['lr']):g}-e{int(row['epochs'])}-la{float(row['lambda_anchor']):g}"
        f"-neg{row.get('negative_sampling_strategy', 'random')}"
    )


def _ppr_method(row: dict[str, Any]) -> str:
    return f"PPR-sweep-k{int(row['k_in'])}-{row['seed_variant']}-a{float(row['alpha']):g}"


def freeze_family(payload: dict[str, Any], family: str) -> Path:
    if family not in {"ppr", "lightgcn"}:
        raise ValueError(f"Unsupported B1 family={family}")
    contract = _load_script("b1_campaign_contract.py", "b1_contract")
    cv_root = _data_path(payload["outputs"][f"{family}_cv"])
    source_rows: list[pd.DataFrame] = []
    source_files: list[dict[str, str]] = []
    histories: list[pd.DataFrame] = []
    for graph_id in payload["graphs"]:
        summary_path = cv_root / graph_id / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing B1 CV summary: {summary_path}")
        source_rows.append(
            _standardize_summary(
                pd.read_csv(summary_path),
                graph_version=graph_id,
                family=family,
                train_split=payload["datasets"]["train"]["split"],
            )
        )
        source_files.append({"path": str(summary_path), "sha256": _sha256(summary_path)})
        if family == "lightgcn":
            history_path = cv_root / graph_id / "lightgcn_history_all.csv"
            if not history_path.is_file():
                raise FileNotFoundError(f"Missing LightGCN CV history: {history_path}")
            histories.append(pd.read_csv(history_path))
            source_files.append({"path": str(history_path), "sha256": _sha256(history_path)})
    selected = contract.select_complete_cv_champions(
        pd.concat(source_rows, ignore_index=True),
        expected_train_split=payload["datasets"]["train"]["split"],
    )
    for target, champion in selected.items():
        champion["selection_target"] = target
        champion["modality"] = target
        champion["eligible_champion"] = True
        champion["protocol_version"] = payload["protocol_version"]
        champion["dataset_sha256"] = payload["datasets"]["train"]["sha256"]
        champion["fold_assignment_sha256"] = payload["folds"]["sha256"]
        if family == "ppr":
            champion["method"] = _ppr_method(champion)
        else:
            champion["method"] = _lightgcn_method(champion)
    if family == "lightgcn":
        cv = _load_script("44_run_cv_lightgcn.py", "b1_lightgcn_cv")
        history = pd.concat(histories, ignore_index=True)
        selected = cv.attach_replay_epochs(selected, history)
    frozen = {
        "campaign_id": payload["campaign_id"],
        "manifest_sha256": _manifest_hash(payload),
        "a3_sha256": payload["a3"]["sha256"],
        "family": family,
        "selection_data": "train_cv_only",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "sources": source_files,
        "champions": selected,
    }
    frozen_root = _data_path(payload["outputs"]["root"]) / "frozen"
    frozen_root.mkdir(parents=True, exist_ok=True)
    out_path = frozen_root / f"{family}_champions.json"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen B1 champions: {out_path}")
    out_path.write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--family", choices=("ppr", "lightgcn"), required=True)
    args = parser.parse_args(argv)
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(freeze_family(payload, args.family))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
