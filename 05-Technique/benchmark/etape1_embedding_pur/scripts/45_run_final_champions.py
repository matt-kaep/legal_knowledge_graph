from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import graph_protocol  # noqa: E402


COVERAGE_SUMMARY = (
    REPO / "01-Projet/specs/graph-g0-g1-g2-g3-dataset-coverage-2026-06-22/summary.json"
)
FINAL_DIRNAME = "_final_champions"
CV_FAMILIES = {
    "b3_b4": "b3_b4",
    "ppr": "ppr",
    "lightgcn": "lightgcn",
}


def _load_script_module(script_name: str, module_name: str):
    script_path = SCRIPT_DIR / script_name
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


build_global_table = _load_script_module("24_build_global_table.py", "build_global_table")


def _copy_bench_artifacts(src_dir: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["bench_global.json", "questions_emb.npy", "questions_ids.npy"]:
        src = src_dir / filename
        if not src.exists():
            raise FileNotFoundError(f"Missing required bench artifact: {src}")
        dst = dst_dir / filename
        if src.suffix == ".json":
            dst.write_text(src.read_text(), encoding="utf-8")
        else:
            dst.write_bytes(src.read_bytes())


def load_champions(cv_root: Path) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    for family, dirname in CV_FAMILIES.items():
        path = cv_root / dirname / "champions.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing champions file: {path}")
        bundle[family] = json.loads(path.read_text(encoding="utf-8"))
    return bundle


def format_method_label(method: str, k_in: Any) -> str:
    if pd.isna(k_in):
        return str(method)
    if str(method).startswith("PPR-") or str(method).startswith("LightGCN-"):
        return str(method)
    return f"{method} (k={int(k_in)})"


def baseline_kins(champions: dict[str, Any]) -> list[int]:
    out = {
        int(row["k_in"])
        for row in champions.values()
        if row and not pd.isna(row.get("k_in"))
    }
    return sorted(out)


def ppr_configs(champions: dict[str, Any]) -> list[str]:
    configs = set()
    for row in champions.values():
        configs.add(
            f"{int(row['k_in'])}:{row['seed_variant']}:{float(row['alpha']):g}"
        )
    return sorted(configs)


def unique_lightgcn_champions(champions: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out = []
    for row in champions.values():
        key = (
            row.get("variant"),
            row.get("train_k"),
            row.get("seed"),
            row.get("lr"),
            row.get("epochs"),
            row.get("lambda_anchor"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def replay_b3_b4(eval_bench_dir: Path, champions: dict[str, Any]) -> pd.DataFrame:
    baseline_eval = _load_script_module("26_eval_doctrine_v3plus_m1_m2.py", "eval_b3b4_final")
    questions = graph_protocol.load_bench_questions(eval_bench_dir)
    with tempfile.TemporaryDirectory(prefix="final_b3b4_") as tmp_dir:
        out_dir = Path(tmp_dir)
        baseline_eval.eval_m1_m2(
            questions,
            out_dir,
            ks_in=baseline_kins(champions),
        )
        df = pd.read_csv(out_dir / "eval_m1_m2.csv")
    masks = []
    for row in champions.values():
        mask = (
            (df["method"] == row["method"])
            & (df["modality"] == row["modality"])
        )
        if pd.isna(row.get("k_in")):
            mask &= df["k_in"].isna()
        else:
            mask &= df["k_in"] == row["k_in"]
        masks.append(mask)
    selected = df[pd.concat(masks, axis=1).any(axis=1)].copy() if masks else df.iloc[0:0].copy()
    return selected.reset_index(drop=True)


def replay_ppr(eval_bench_dir: Path, champions: dict[str, Any]) -> pd.DataFrame:
    ppr_sweep = _load_script_module("25_ppr_kin_sweep.py", "ppr_final")
    with tempfile.TemporaryDirectory(prefix="final_ppr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        _copy_bench_artifacts(eval_bench_dir, tmp_path)
        ppr_sweep.main(tmp_path, config_specs=ppr_configs(champions))
        df = pd.read_csv(tmp_path / "ppr_kin_sweep_eval.csv")
    masks = [
        (
            (df["k_in"] == row["k_in"])
            & (df["seed_variant"] == row["seed_variant"])
            & (df["alpha"] == row["alpha"])
        )
        for row in champions.values()
    ]
    selected = df[pd.concat(masks, axis=1).any(axis=1)].copy() if masks else df.iloc[0:0].copy()
    return selected.reset_index(drop=True)


def replay_lightgcn(
    train_bench_dir: Path,
    eval_bench_dir: Path,
    graph_version: str,
    champions: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    lightgcn = _load_script_module("32_lightgcn_strict.py", "lightgcn_final")
    eval_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="final_lightgcn_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_tmp = tmp_path / "train"
        eval_tmp = tmp_path / "eval"
        _copy_bench_artifacts(train_bench_dir, train_tmp)
        _copy_bench_artifacts(eval_bench_dir, eval_tmp)
        for row in unique_lightgcn_champions(champions):
            suffix = (
                f"{row['variant']}_k{int(row.get('train_k', row.get('k_in', 2)))}"
                .replace(".", "p")
                .replace("-", "_")
            )
            args = [
                "--train-bench-dir",
                str(train_tmp),
                "--eval-bench-dir",
                str(eval_tmp),
                "--graph-version",
                graph_version,
                "--train-k",
                str(int(row.get("train_k", row.get("k_in", 2)))),
                "--output-suffix",
                suffix,
                "--dump-rankings",
            ]
            variant = str(row.get("variant", ""))
            if variant.startswith("trained_"):
                args.extend(
                    [
                        "--seed",
                        str(int(row.get("seed", 42))),
                        "--lr",
                        str(float(row.get("lr", 1e-3))),
                        "--epochs",
                        str(int(row.get("epochs", 30))),
                        "--lambda-anchor",
                        str(float(row.get("lambda_anchor", 1.0))),
                        "--trained-only",
                    ]
                )
            else:
                args.append("--notrain")
            rc = lightgcn.main(args)
            if rc != 0:
                raise RuntimeError(f"LightGCN final replay failed for variant={variant}")
            eval_path = eval_tmp / f"lightgcn_eval_{suffix}.csv"
            history_path = eval_tmp / f"lightgcn_history_{suffix}.csv"
            run_df = pd.read_csv(eval_path)
            run_df["train_k"] = row.get("train_k", row.get("k_in", 2))
            run_df["seed"] = row.get("seed")
            run_df["lr"] = row.get("lr")
            run_df["epochs"] = row.get("epochs")
            run_df["lambda_anchor"] = row.get("lambda_anchor")
            run_df["graph_version"] = graph_version
            run_df = run_df[run_df["variant"] == variant].copy()
            eval_frames.append(run_df)
            if history_path.exists():
                hist_df = pd.read_csv(history_path)
                hist_df["graph_version"] = graph_version
                hist_df = hist_df[hist_df["variant"] == variant].copy()
                history_frames.append(hist_df)
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    return eval_df, history_df


def load_coverage_summary(path: Path, graph_version: str) -> dict[str, float]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("datasets", {}).get("eval_rich_retrievable_strict", {})
    graph_key = str(graph_version).lower()
    row = split.get(graph_key, {})
    return {
        "coverage_articles": row.get("strict_q_any_pct"),
        "coverage_articles_occ_pct": row.get("strict_occ_pct"),
        "coverage_articles_unique_pct": row.get("strict_unique_pct"),
        "coverage_jp": row.get("jp_q_any_pct"),
        "coverage_jp_occ_pct": row.get("jp_occ_pct"),
        "coverage_jp_unique_pct": row.get("jp_unique_pct"),
    }


def _final_metric_columns(modality: str, df: pd.DataFrame) -> dict[str, str]:
    if modality == "art":
        for prefix in ["", "_art"]:
            hit_col = f"hit_strict{prefix}"
            if hit_col in df.columns:
                return {
                    "m1": f"m1_strict{prefix}",
                    "hit": hit_col,
                    "mrr": f"mrr_strict{prefix}",
                    "ndcg": f"ndcg_strict{prefix}",
                    "m2": f"m2_strict{prefix}",
                    "m1_ext": f"m1_ext{prefix}",
                    "hit_ext": f"hit_ext{prefix}",
                    "mrr_ext": f"mrr_ext{prefix}",
                    "ndcg_ext": f"ndcg_ext{prefix}",
                    "m2_ext": f"m2_ext{prefix}",
                }
    for prefix in ["", "_jp"]:
        hit_col = f"hit{prefix}"
        if hit_col in df.columns:
            return {
                "m1": f"m1{prefix}",
                "hit": hit_col,
                "mrr": f"mrr{prefix}",
                "ndcg": f"ndcg{prefix}",
                "m2": f"m2{prefix}",
            }
    raise ValueError(f"Could not resolve final metric columns for modality={modality}")


def summarize_final_slice(
    df: pd.DataFrame,
    *,
    graph_version: str,
    family: str,
    champion: dict[str, Any],
    n_questions_benchmark: int,
    coverage: dict[str, float],
) -> dict[str, Any]:
    modality = champion["modality"]
    metrics = _final_metric_columns(modality, df)
    row = {
        "graph_version": graph_version,
        "family": family,
        "target": "articles_strict" if modality == "art" else "jp",
        "modality": modality,
        "method": champion["method"],
        "method_label": format_method_label(champion["method"], champion.get("k_in")),
        "k_in": champion.get("k_in"),
        "question_coverage": df["qid"].nunique() / float(n_questions_benchmark or 1),
        "n_questions_covered": int(df["qid"].nunique()),
        "n_questions_benchmark": int(n_questions_benchmark),
        "coverage_articles": coverage.get("coverage_articles"),
        "coverage_articles_occ_pct": coverage.get("coverage_articles_occ_pct"),
        "coverage_articles_unique_pct": coverage.get("coverage_articles_unique_pct"),
        "coverage_jp": coverage.get("coverage_jp"),
        "coverage_jp_occ_pct": coverage.get("coverage_jp_occ_pct"),
        "coverage_jp_unique_pct": coverage.get("coverage_jp_unique_pct"),
        "m3_display": "—",
    }
    for out_col, src_col in metrics.items():
        if src_col in df.columns:
            row[out_col] = float(df[src_col].mean())
    for extra in ["seed_variant", "alpha", "variant", "train_k", "seed", "lr", "epochs", "lambda_anchor"]:
        if extra in champion:
            row[extra] = champion[extra]
    return row


def build_final_summary(
    graph_version: str,
    eval_bench_dir: Path,
    champion_bundle: dict[str, dict[str, Any]],
    b3_b4_df: pd.DataFrame,
    ppr_df: pd.DataFrame,
    lightgcn_df: pd.DataFrame,
    coverage: dict[str, float],
) -> pd.DataFrame:
    questions = graph_protocol.load_bench_questions(eval_bench_dir)
    n_questions_benchmark = len(questions)
    rows: list[dict[str, Any]] = []

    for champion in champion_bundle["b3_b4"].values():
        mask = (b3_b4_df["method"] == champion["method"]) & (b3_b4_df["modality"] == champion["modality"])
        if pd.isna(champion.get("k_in")):
            mask &= b3_b4_df["k_in"].isna()
        else:
            mask &= b3_b4_df["k_in"] == champion["k_in"]
        sub = b3_b4_df[mask].copy()
        if not sub.empty:
            rows.append(
                summarize_final_slice(
                    sub,
                    graph_version=graph_version,
                    family="b3_b4",
                    champion=champion,
                    n_questions_benchmark=n_questions_benchmark,
                    coverage=coverage,
                )
            )

    for champion in champion_bundle["ppr"].values():
        mask = (
            (ppr_df["k_in"] == champion["k_in"])
            & (ppr_df["seed_variant"] == champion["seed_variant"])
            & (ppr_df["alpha"] == champion["alpha"])
        )
        sub = ppr_df[mask].copy()
        if not sub.empty:
            rows.append(
                summarize_final_slice(
                    sub,
                    graph_version=graph_version,
                    family="ppr",
                    champion=champion,
                    n_questions_benchmark=n_questions_benchmark,
                    coverage=coverage,
                )
            )

    for champion in champion_bundle["lightgcn"].values():
        mask = lightgcn_df["variant"] == champion["variant"]
        for col in ["train_k", "seed", "lr", "epochs", "lambda_anchor"]:
            value = champion.get(col)
            if value is None or pd.isna(value):
                continue
            if col in lightgcn_df.columns:
                mask &= lightgcn_df[col] == value
        sub = lightgcn_df[mask].copy()
        if not sub.empty:
            rows.append(
                summarize_final_slice(
                    sub,
                    graph_version=graph_version,
                    family="lightgcn",
                    champion=champion,
                    n_questions_benchmark=n_questions_benchmark,
                    coverage=coverage,
                )
            )

    return pd.DataFrame(rows)


def write_final_tables(summary_df: pd.DataFrame, out_dir: Path) -> None:
    articles = build_global_table.build_final_target_table(summary_df, "articles_strict")
    jp = build_global_table.build_final_target_table(summary_df, "jp")
    comparison = build_global_table.build_graph_comparison_table(summary_df)
    articles.to_csv(out_dir / "global_table_articles.csv", index=False)
    jp.to_csv(out_dir / "global_table_jp.csv", index=False)
    comparison.to_csv(out_dir / "global_table_graph_comparison.csv", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-version", default="G0")
    parser.add_argument(
        "--cv-root",
        type=Path,
        help="Root folder containing per-family CV outputs; defaults to train strict bench/_cv.",
    )
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--coverage-summary", type=Path, default=COVERAGE_SUMMARY)
    parser.add_argument(
        "--skip-replay",
        action="store_true",
        help="Reuse already materialized final replay CSVs from --out-dir.",
    )
    args = parser.parse_args(argv)

    train_bench_dir = graph_protocol.resolve_graph_bench_dir(
        args.graph_version,
        graph_protocol.OFFICIAL_TRAIN_SPLIT,
    )
    eval_bench_dir = graph_protocol.resolve_graph_bench_dir(
        args.graph_version,
        "eval_rich_retrievable_strict",
    )
    cv_root = args.cv_root or (train_bench_dir / "_cv")
    out_dir = args.out_dir or (eval_bench_dir / FINAL_DIRNAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    champion_bundle = load_champions(cv_root)
    coverage = load_coverage_summary(args.coverage_summary, args.graph_version)

    if args.skip_replay:
        b3_b4_df = pd.read_csv(out_dir / "eval_m1_m2.csv")
        ppr_df = pd.read_csv(out_dir / "ppr_kin_sweep_eval.csv")
        lightgcn_df = pd.read_csv(out_dir / "lightgcn_eval.csv")
        history_df = pd.read_csv(out_dir / "lightgcn_history.csv") if (out_dir / "lightgcn_history.csv").exists() else pd.DataFrame()
    else:
        b3_b4_df = replay_b3_b4(eval_bench_dir, champion_bundle["b3_b4"])
        ppr_df = replay_ppr(eval_bench_dir, champion_bundle["ppr"])
        lightgcn_df, history_df = replay_lightgcn(
            train_bench_dir,
            eval_bench_dir,
            args.graph_version,
            champion_bundle["lightgcn"],
        )
        b3_b4_df.to_csv(out_dir / "eval_m1_m2.csv", index=False)
        ppr_df.to_csv(out_dir / "ppr_kin_sweep_eval.csv", index=False)
        lightgcn_df.to_csv(out_dir / "lightgcn_eval.csv", index=False)
        if not history_df.empty:
            history_df.to_csv(out_dir / "lightgcn_history.csv", index=False)

    summary_df = build_final_summary(
        args.graph_version,
        eval_bench_dir,
        champion_bundle,
        b3_b4_df,
        ppr_df,
        lightgcn_df,
        coverage,
    )
    summary_df.to_csv(out_dir / "final_champions_summary.csv", index=False)
    (out_dir / "champions_manifest.json").write_text(
        json.dumps(champion_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_final_tables(summary_df, out_dir)

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
