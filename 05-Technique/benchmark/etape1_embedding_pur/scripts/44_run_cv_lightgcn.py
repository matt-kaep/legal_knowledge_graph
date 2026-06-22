from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
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


lightgcn = _load_script_module("32_lightgcn_strict.py", "lightgcn_strict")


def enforce_official_split(split: str) -> None:
    if split != graph_protocol.OFFICIAL_TRAIN_SPLIT:
        raise ValueError(
            "CV wrappers support only "
            f"split={graph_protocol.OFFICIAL_TRAIN_SPLIT}; got {split}"
        )


def validate_fold_assignments(df: pd.DataFrame, bench_qids: set[str]) -> pd.DataFrame:
    qids = df["qid"].astype(str)
    duplicate_qids = sorted(qids[qids.duplicated()].unique().tolist())
    if duplicate_qids:
        raise ValueError(f"duplicate qids in fold assignments: {duplicate_qids}")
    assigned_qids = set(qids.tolist())
    missing_qids = sorted(bench_qids - assigned_qids)
    extra_qids = sorted(assigned_qids - bench_qids)
    if missing_qids or extra_qids:
        raise ValueError(
            "fold assignments mismatch: "
            f"missing qids={missing_qids}; extra qids={extra_qids}"
        )
    return df


def load_fold_assignments(bench_qids: set[str]) -> pd.DataFrame:
    fold_csv, _ = graph_protocol.resolve_shared_fold_paths()
    df = pd.read_csv(fold_csv)
    expected = set(range(graph_protocol.OFFICIAL_N_FOLDS))
    found = set(df["fold"].astype(int).unique().tolist())
    if found != expected:
        raise ValueError(f"Expected folds {sorted(expected)}, got {sorted(found)}")
    return validate_fold_assignments(df, bench_qids)


def build_subset_bench(src_dir: Path, qids: set[str], dst_dir: Path) -> None:
    payload = json.loads((src_dir / "bench_global.json").read_text())
    questions = payload["questions"]
    ids = np.load(src_dir / "questions_ids.npy", allow_pickle=True)
    emb = np.load(src_dir / "questions_emb.npy")
    keep_mask = np.array([str(qid) in qids for qid in ids.tolist()], dtype=bool)
    filtered_questions = [q for q in questions if str(q["qid"]) in qids]
    filtered_qids = ids[keep_mask]
    filtered_emb = emb[keep_mask]
    if len(filtered_questions) != len(filtered_qids):
        qids_from_questions = {str(q["qid"]) for q in filtered_questions}
        qids_from_arrays = {str(qid) for qid in filtered_qids.tolist()}
        raise ValueError(
            "subset bench mismatch: "
            f"questions_only={sorted(qids_from_questions - qids_from_arrays)} "
            f"arrays_only={sorted(qids_from_arrays - qids_from_questions)}"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    (dst_dir / "bench_global.json").write_text(
        json.dumps({"questions": filtered_questions}, ensure_ascii=False, indent=2)
    )
    np.save(dst_dir / "questions_ids.npy", filtered_qids)
    np.save(dst_dir / "questions_emb.npy", filtered_emb)


def run_lightgcn_config(
    train_bench_dir: Path,
    val_bench_dir: Path,
    *,
    graph_version: str,
    fold: int,
    train_k: int,
    seed: int,
    lr: float,
    epochs: int,
    lambda_anchor: float,
    include_baselines: bool,
    train_variant: bool,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    suffix = (
        f"fold{fold}_k{train_k}_s{seed}_lr{lr:g}_e{epochs}_la{lambda_anchor:g}"
        .replace(".", "p")
    )
    args = [
        "--train-bench-dir",
        str(train_bench_dir),
        "--eval-bench-dir",
        str(val_bench_dir),
        "--graph-version",
        graph_version,
        "--train-k",
        str(train_k),
        "--seed",
        str(seed),
        "--lr",
        str(lr),
        "--epochs",
        str(epochs),
        "--lambda-anchor",
        str(lambda_anchor),
        "--output-suffix",
        suffix,
    ]
    if not train_variant:
        args.append("--notrain")
    if not include_baselines:
        args.append("--trained-only")
    rc = lightgcn.main(args)
    if rc != 0:
        raise RuntimeError(f"LightGCN run failed for suffix={suffix} rc={rc}")
    raw_df = pd.read_csv(val_bench_dir / f"lightgcn_eval_{suffix}.csv")
    history_path = val_bench_dir / f"lightgcn_history_{suffix}.csv"
    history_df = pd.read_csv(history_path) if history_path.exists() else None
    return raw_df, history_df


def summarize_cv_results(
    df: pd.DataFrame,
    modality: str,
    n_questions_benchmark: int | None = None,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    if n_questions_benchmark is None:
        n_questions_benchmark = int(df["qid"].nunique())
    metric_map = (
        {
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
        if modality == "art"
        else {
            "hit": "hit_jp",
            "ndcg": "ndcg_jp",
            "mrr": "mrr_jp",
            "m1": "m1_jp",
            "m2": "m2_jp",
        }
    )
    available = {out: src for out, src in metric_map.items() if src in df.columns}
    if not available:
        return pd.DataFrame()
    group_cols = [
        "variant",
        "train_k",
        "seed",
        "lr",
        "epochs",
        "lambda_anchor",
        "graph_version",
    ]
    summary = (
        df.groupby(group_cols, dropna=False)[list(available.values())]
        .mean()
        .reset_index()
        .rename(columns={src: out for out, src in available.items()})
    )
    coverage = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_questions_covered=("qid", "nunique"),
            n_folds_covered=("fold", "nunique"),
        )
        .reset_index()
    )
    coverage["n_questions_benchmark"] = int(n_questions_benchmark)
    coverage["question_coverage"] = (
        coverage["n_questions_covered"] / coverage["n_questions_benchmark"]
    )
    coverage["fold_coverage"] = (
        coverage["n_folds_covered"] / graph_protocol.OFFICIAL_N_FOLDS
    )
    summary = summary.merge(coverage, on=group_cols, how="left")
    summary.insert(
        0,
        "method",
        summary.apply(
            lambda row: (
                f"LightGCN-{row['variant']}"
                if row["variant"].startswith("untrained_")
                else (
                    f"LightGCN-{row['variant']}"
                    f"-s{int(row['seed'])}"
                    f"-lr{float(row['lr']):g}"
                    f"-e{int(row['epochs'])}"
                    f"-la{float(row['lambda_anchor']):g}"
                )
            ),
            axis=1,
        ),
    )
    summary.insert(1, "modality", modality)
    return summary.sort_values(["method"]).reset_index(drop=True)


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
    parser.add_argument("--train-k", type=int, action="append", dest="train_ks")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--lr", type=float, action="append", dest="lrs")
    parser.add_argument("--epochs", type=int, action="append", dest="epochs_list")
    parser.add_argument(
        "--lambda-anchor",
        type=float,
        action="append",
        dest="lambda_anchors",
    )
    args = parser.parse_args(argv)

    enforce_official_split(args.split)
    bench_dir = graph_protocol.resolve_graph_bench_dir(args.graph_version, args.split)
    bench_questions = graph_protocol.load_bench_questions(bench_dir)
    bench_qids = {str(question["qid"]) for question in bench_questions}
    folds = load_fold_assignments(bench_qids)

    train_ks = args.train_ks or [2]
    seeds = args.seeds or [42]
    lrs = args.lrs or [1e-3]
    epochs_list = args.epochs_list or [30]
    lambda_anchors = args.lambda_anchors or [1.0]
    trained_configs = list(itertools.product(train_ks, seeds, lrs, epochs_list, lambda_anchors))
    out_dir = args.out_dir or (bench_dir / "_cv" / "lightgcn")
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_parts: list[pd.DataFrame] = []
    history_parts: list[pd.DataFrame] = []

    for fold in sorted(folds["fold"].astype(int).unique()):
        val_qids = set(folds.loc[folds["fold"] == fold, "qid"].astype(str))
        train_qids = bench_qids - val_qids
        with tempfile.TemporaryDirectory(prefix=f"cv_lightgcn_fold{fold}_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            train_dir = tmp_path / "train"
            val_dir = tmp_path / "val"
            build_subset_bench(bench_dir, train_qids, train_dir)
            build_subset_bench(bench_dir, val_qids, val_dir)

            baseline_df, _ = run_lightgcn_config(
                train_dir,
                val_dir,
                graph_version=args.graph_version,
                fold=fold,
                train_k=train_ks[0],
                seed=seeds[0],
                lr=lrs[0],
                epochs=epochs_list[0],
                lambda_anchor=lambda_anchors[0],
                include_baselines=True,
                train_variant=False,
            )
            baseline_df = baseline_df[baseline_df["variant"] != "cosine_raw"].copy()
            baseline_df.insert(0, "fold", fold)
            baseline_df["train_k"] = np.where(
                baseline_df["variant"].str.contains("_K"),
                baseline_df["variant"].str.rsplit("K", n=1).str[-1].astype(int),
                train_ks[0],
            )
            baseline_df["seed"] = np.nan
            baseline_df["lr"] = np.nan
            baseline_df["epochs"] = np.nan
            baseline_df["lambda_anchor"] = np.nan
            baseline_df["graph_version"] = args.graph_version
            raw_parts.append(baseline_df)

            for train_k, seed, lr, epochs, lambda_anchor in trained_configs:
                trained_df, history_df = run_lightgcn_config(
                    train_dir,
                    val_dir,
                    graph_version=args.graph_version,
                    fold=fold,
                    train_k=train_k,
                    seed=seed,
                    lr=lr,
                    epochs=epochs,
                    lambda_anchor=lambda_anchor,
                    include_baselines=False,
                    train_variant=True,
                )
                trained_df.insert(0, "fold", fold)
                trained_df["train_k"] = train_k
                trained_df["seed"] = seed
                trained_df["lr"] = lr
                trained_df["epochs"] = epochs
                trained_df["lambda_anchor"] = lambda_anchor
                trained_df["graph_version"] = args.graph_version
                raw_parts.append(trained_df)
                if history_df is not None and not history_df.empty:
                    history_df.insert(0, "fold", fold)
                    history_df["train_k"] = train_k
                    history_df["seed"] = seed
                    history_df["lr"] = lr
                    history_df["epochs"] = epochs
                    history_df["lambda_anchor"] = lambda_anchor
                    history_parts.append(history_df)
                    history_suffix = (
                        f"fold{fold}_k{train_k}_s{seed}_lr{lr:g}_e{epochs}_la{lambda_anchor:g}"
                        .replace(".", "p")
                    )
                    history_df.to_csv(
                        out_dir / f"lightgcn_history_{history_suffix}.csv",
                        index=False,
                    )

    raw_df = pd.concat(raw_parts, ignore_index=True) if raw_parts else pd.DataFrame()
    summary_parts = [
        summarize_cv_results(raw_df, "art", n_questions_benchmark=len(bench_qids)),
        summarize_cv_results(raw_df, "jp", n_questions_benchmark=len(bench_qids)),
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

    raw_df.to_csv(out_dir / "cv_results_raw.csv", index=False)
    summary_df.to_csv(out_dir / "cv_results_summary.csv", index=False)
    if history_parts:
        history_df = pd.concat(history_parts, ignore_index=True)
        history_df.to_csv(out_dir / "lightgcn_history_all.csv", index=False)
    (out_dir / "champions.json").write_text(json.dumps(champions, ensure_ascii=False, indent=2))
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
