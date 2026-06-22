from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import graph_protocol  # noqa: E402


def _load_script_module(script_name: str, module_name: str):
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ppr_sweep = _load_script_module("25_ppr_kin_sweep.py", "ppr_sweep")


def enforce_official_split(split: str) -> None:
    if split != graph_protocol.OFFICIAL_TRAIN_SPLIT:
        raise ValueError(
            "CV wrappers support only "
            f"split={graph_protocol.OFFICIAL_TRAIN_SPLIT}; got {split}"
        )


def load_fold_assignments() -> pd.DataFrame:
    fold_csv, _ = graph_protocol.resolve_shared_fold_paths()
    df = pd.read_csv(fold_csv)
    expected = set(range(graph_protocol.OFFICIAL_N_FOLDS))
    found = set(df["fold"].astype(int).unique().tolist())
    if found != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, got {sorted(found)}")
    return df


def run_fold_subset(
    question_ids: set[str],
    bench_dir: Path,
    config_specs: list[str] | None = None,
) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="cv_ppr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        for filename in ["bench_global.json", "questions_emb.npy", "questions_ids.npy"]:
            src = bench_dir / filename
            if not src.exists():
                raise FileNotFoundError(f"Missing required bench artifact: {src}")
            dst = tmp_path / filename
            if src.suffix == ".json":
                dst.write_text(src.read_text())
            else:
                dst.write_bytes(src.read_bytes())
        ppr_sweep.main(
            tmp_path,
            config_specs=config_specs,
            qid_filter=question_ids,
        )
        return pd.read_csv(tmp_path / "ppr_kin_sweep_eval.csv")


def summarize_cv_results(df: pd.DataFrame, modality: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if modality == "art":
        metric_map = {
            "hit_strict": "hit_strict_art",
            "ndcg_strict": "ndcg_strict_art",
            "mrr_strict": "mrr_strict_art",
            "m1_strict": "m1_strict_art",
            "m2_strict": "m2_strict_art",
            "hit_ext": "hit_ext_art",
            "ndcg_ext": "ndcg_ext_art",
            "mrr_ext": "mrr_ext_art",
            "m1_ext": "m1_ext_art",
            "m2_ext": "m2_ext_art",
        }
    else:
        metric_map = {
            "hit": "hit_jp",
            "ndcg": "ndcg_jp",
            "mrr": "mrr_jp",
            "m1": "m1_jp",
            "m2": "m2_jp",
        }
    available = {out: src for out, src in metric_map.items() if src in df.columns}
    if not available:
        return pd.DataFrame()
    summary = (
        df.groupby(["k_in", "seed_variant", "alpha"], dropna=False)[list(available.values())]
        .mean()
        .reset_index()
        .rename(columns={src: out for out, src in available.items()})
    )
    coverage = (
        df.groupby(["k_in", "seed_variant", "alpha"], dropna=False)
        .agg(
            n_questions_covered=("qid", "nunique"),
            n_folds_covered=("fold", "nunique"),
        )
        .reset_index()
    )
    coverage["fold_coverage"] = (
        coverage["n_folds_covered"] / graph_protocol.OFFICIAL_N_FOLDS
    )
    summary = (
        summary.merge(coverage, on=["k_in", "seed_variant", "alpha"], how="left")
        .sort_values(["k_in", "seed_variant", "alpha"])
        .reset_index(drop=True)
    )
    summary.insert(0, "method", summary.apply(
        lambda row: f"PPR-sweep-k{int(row['k_in'])}-{row['seed_variant']}-a{float(row['alpha'])}",
        axis=1,
    ))
    summary.insert(1, "modality", modality)
    return summary


def select_champion(summary_df: pd.DataFrame, modality: str) -> dict:
    if summary_df.empty:
        raise ValueError(f"No CV results available for modality={modality}")
    records = summary_df.to_dict(orient="records")
    return max(records, key=lambda row: graph_protocol.metric_rank_tuple(row, modality))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version", default="canonical")
    parser.add_argument("--split", default=graph_protocol.OFFICIAL_TRAIN_SPLIT)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--config", action="append")
    args = parser.parse_args(argv)

    enforce_official_split(args.split)
    bench_dir = graph_protocol.resolve_graph_bench_dir(args.graph_version, args.split)
    folds = load_fold_assignments()
    rows = []
    for fold in sorted(folds["fold"].astype(int).unique()):
        qids = set(folds.loc[folds["fold"] == fold, "qid"].astype(str))
        fold_df = run_fold_subset(qids, bench_dir, config_specs=args.config)
        fold_df.insert(0, "fold", fold)
        rows.append(fold_df)

    raw_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    summary_parts = [
        summarize_cv_results(raw_df, "art"),
        summarize_cv_results(raw_df, "jp"),
    ]
    summary_df = pd.concat(
        [part for part in summary_parts if not part.empty],
        ignore_index=True,
    )
    champions = {}
    if not summary_df.empty:
        for modality in ["art", "jp"]:
            sub = summary_df[summary_df["modality"] == modality]
            if not sub.empty:
                champions[modality] = select_champion(sub, modality)

    out_dir = args.out_dir or (bench_dir / "_cv" / "ppr")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(out_dir / "cv_results_raw.csv", index=False)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    (out_dir / "champions.json").write_text(json.dumps(champions, ensure_ascii=False, indent=2))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
