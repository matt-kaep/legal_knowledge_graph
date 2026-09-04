from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(os.environ.get(
    "LKG_REPO",
    str(Path(__file__).resolve().parents[4]),
))
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
confirmatory_runner = _load_script_module("64_run_confirmatory_campaign.py", "confirmatory_campaign_validation")


def validate_campaign_provenance(campaign: dict[str, Any]) -> dict[str, Any]:
    confirmatory_runner.validate_manifest_payload(campaign)
    report = confirmatory_runner.preflight(campaign, verify_hashes=True)
    if not report.get("scientific_inputs_ok"):
        raise ValueError("campaign scientific inputs failed provenance validation")
    if not report.get("resource_assessment", {}).get("compatible"):
        insufficient = report.get("resource_assessment", {}).get("insufficient", [])
        raise RuntimeError(
            "campaign execution resources are not validated: " + ",".join(map(str, insufficient))
        )
    return report


def refresh_intergraph_report() -> None:
    try:
        report = _load_script_module("49_build_intergraph_results_report.py", "intergraph_results_report")
        report.build_report(report.DEFAULT_OUT_DIR)
    except Exception as exc:  # pragma: no cover - best effort refresh
        print(f"[warn] inter-graph report refresh failed: {exc}")


def refresh_week13_snippets() -> None:
    try:
        snippets = _load_script_module("50_build_week13_intergraph_snippets.py", "week13_intergraph_snippets")
        snippets.main([])
    except Exception as exc:  # pragma: no cover - best effort refresh
        print(f"[warn] week13 snippet refresh failed: {exc}")


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


def _m3_cache_dir(out_dir: Path) -> Path:
    return out_dir.parent / f".{out_dir.name}_m3_cache"


def _copy_m3_inputs(
    eval_bench_dir: Path,
    out_dir: Path,
    tmp_dir: Path,
    manifest: dict[str, dict[str, Any]],
) -> None:
    bench_global = eval_bench_dir / "bench_global.json"
    rankings = out_dir / "rankings.parquet"
    if not bench_global.exists():
        raise FileNotFoundError(f"Missing M3 input bench file: {bench_global}")
    if not rankings.exists():
        raise FileNotFoundError(f"Missing M3 input rankings file: {rankings}")
    (tmp_dir / "bench_global.json").write_text(bench_global.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_dir / "rankings.parquet").write_bytes(rankings.read_bytes())
    (tmp_dir / "champions_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    cache_dir = _m3_cache_dir(out_dir)
    if cache_dir.exists():
        for cache_path in cache_dir.glob("m3_judge_cache_*.csv"):
            (tmp_dir / cache_path.name).write_bytes(cache_path.read_bytes())


def _m3_group_key(method: str, modality: str, k_in: Any) -> str:
    return build_global_table._m3_group_key(method, modality, k_in)


def _champion_method_aliases(champion: dict[str, Any]) -> set[str]:
    aliases = {str(champion["method"])}
    variant = champion.get("variant")
    if pd.notna(variant):
        aliases.add(f"LightGCN-{variant}")
    return aliases


def _champion_mask(df: pd.DataFrame, champion: dict[str, Any]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    if "method" in df.columns:
        mask &= df["method"].astype(str).isin(_champion_method_aliases(champion))
    if "modality" in df.columns:
        mask &= df["modality"] == champion["modality"]
    if "variant" in champion and "variant" in df.columns:
        mask &= df["variant"] == champion["variant"]
    if "seed_variant" in champion and "seed_variant" in df.columns:
        mask &= df["seed_variant"] == champion["seed_variant"]
    if "alpha" in champion and "alpha" in df.columns:
        mask &= df["alpha"] == champion["alpha"]
    if "k_in" in df.columns:
        champion_k = champion.get("k_in", champion.get("train_k"))
        if pd.isna(champion_k):
            mask &= df["k_in"].isna()
        else:
            mask &= df["k_in"] == champion_k
    for col in ["train_k", "seed", "lr", "lambda_anchor", "selection_target"]:
        if col in champion and col in df.columns:
            value = champion.get(col)
            if value is None or pd.isna(value):
                continue
            mask &= df[col] == value
    if "epochs" in champion and "cv_epochs" in df.columns:
        mask &= df["cv_epochs"] == champion["epochs"]
    if "replay_epochs" in champion and "replay_epochs" in df.columns:
        mask &= df["replay_epochs"] == champion["replay_epochs"]
    if "run_id" in df.columns:
        mask &= df["run_id"] == lightgcn_run_id(champion)
    if "negative_sampling_strategy" in champion and "negative_sampling_strategy" in df.columns:
        mask &= df["negative_sampling_strategy"] == champion["negative_sampling_strategy"]
    return mask


def _select_champion_rows(df: pd.DataFrame, champions: dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    masks = [_champion_mask(df, row) for row in champions.values()]
    if not masks:
        return df.iloc[0:0].copy()
    return df[pd.concat(masks, axis=1).any(axis=1)].copy().reset_index(drop=True)


def _load_m3_lookup(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    agg = build_global_table.aggregate_m3(pd.read_csv(path)).reset_index()
    return dict(zip(agg["m3_key"], agg["m3_display"]))


def _m3_display_for_champion(
    m3_lookup: dict[str, str],
    champion: dict[str, Any],
    modality: str,
) -> str:
    champion_k = champion.get("k_in", champion.get("train_k"))
    for method in _champion_method_aliases(champion):
        key = _m3_group_key(method, modality, champion_k)
        if key in m3_lookup:
            return m3_lookup[key]
    return "—"


def _run_m3_judge(
    eval_bench_dir: Path,
    out_dir: Path,
    champion_bundle: dict[str, dict[str, Any]],
    *,
    mock: bool = False,
    pilot: int | None = None,
    workers: int = 32,
    model_id: str | None = None,
    port: int | None = None,
) -> Path:
    m3_judge = _load_script_module("23_eval_m3_llm_judge.py", "m3_judge_final")
    suffix = "_mock" if mock else ("_pilot" if pilot is not None else "")
    with tempfile.TemporaryDirectory(prefix="final_m3_") as tmp_root:
        tmp_dir = Path(tmp_root)
        _copy_m3_inputs(eval_bench_dir, out_dir, tmp_dir, champion_bundle)
        args = [
            "--bench-dir",
            str(tmp_dir),
            "--workers",
            str(workers),
        ]
        if mock:
            args.append("--mock")
        if pilot is not None:
            args.extend(["--pilot", str(pilot)])
        if model_id:
            args.extend(["--model-id", model_id])
        if port is not None:
            args.extend(["--port", str(port)])
        rc = m3_judge.main(args)
        if rc != 0:
            raise RuntimeError("M3 judge failed during final champion replay")

        eval_name = f"eval_m3{suffix}.csv"
        summary_name = f"eval_m3_summary{suffix}.json"
        eval_src = tmp_dir / eval_name
        summary_src = tmp_dir / summary_name
        if eval_src.exists():
            (out_dir / eval_name).write_bytes(eval_src.read_bytes())
        if summary_src.exists():
            (out_dir / summary_name).write_text(summary_src.read_text(encoding="utf-8"), encoding="utf-8")
        cache_dir = _m3_cache_dir(out_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for cache_path in tmp_dir.glob("m3_judge_cache_*.csv"):
            (cache_dir / cache_path.name).write_bytes(cache_path.read_bytes())
    return out_dir / eval_name


def load_champions(
    cv_root: Path,
    families: list[str] | None = None,
    *,
    frozen_lightgcn: bool = False,
) -> dict[str, dict[str, Any]]:
    bundle: dict[str, dict[str, Any]] = {}
    selected = families or list(CV_FAMILIES)
    unknown = sorted(set(selected) - set(CV_FAMILIES))
    if unknown:
        raise ValueError(f"Unknown replay families: {unknown}")
    for family in selected:
        dirname = CV_FAMILIES[family]
        filename = "frozen_champions.json" if family == "lightgcn" and frozen_lightgcn else "champions.json"
        path = cv_root / dirname / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing champions file: {path}")
        bundle[family] = json.loads(path.read_text(encoding="utf-8"))
    return bundle


def resolve_replay_roots(graph_version: str, protocol_version: str) -> tuple[Path, Path]:
    if protocol_version != graph_protocol.PROTOCOL_VERSION:
        raise ValueError(f"Unsupported isolated replay protocol: {protocol_version}")
    return (
        graph_protocol.cv_root(graph_protocol.BENCH_ROOT, protocol_version) / graph_version,
        graph_protocol.final_root(graph_protocol.BENCH_ROOT, protocol_version) / graph_version,
    )


def canonical_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


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
            row.get("replay_epochs"),
            row.get("selection_target"),
            row.get("modality"),
            row.get("lambda_anchor"),
            row.get("negative_sampling_strategy"),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def lightgcn_run_id(row: dict[str, Any]) -> str:
    """Stable identity for one frozen target-specific LightGCN replay."""
    parts = [
        row.get("selection_target", row.get("modality", "shared")),
        row.get("variant"),
        f"k{row.get('train_k', row.get('k_in', 2))}",
        f"s{row.get('seed')}",
        f"lr{row.get('lr')}",
        f"cv{row.get('epochs')}",
        f"replay{row.get('replay_epochs')}",
        f"la{row.get('lambda_anchor')}",
        f"neg{row.get('negative_sampling_strategy', 'random')}",
    ]
    return "__".join(str(part).replace(".", "p").replace("-", "_") for part in parts)


def validate_grouped_champion_bundle(
    champion_bundle: dict[str, dict[str, Any]],
    *,
    dataset_sha256: str,
    fold_assignment_sha256: str,
) -> None:
    """Reject any champion that cannot prove complete grouped-v2 selection."""
    for family, champions in champion_bundle.items():
        for target, champion in champions.items():
            context = f"family={family} target={target}"
            checks = {
                "eligible_champion": champion.get("eligible_champion") is True,
                "n_folds_covered": int(champion.get("n_folds_covered", -1))
                == graph_protocol.OFFICIAL_N_FOLDS,
                "question_coverage": float(champion.get("question_coverage", -1.0)) == 1.0,
                "protocol_version": champion.get("protocol_version")
                == graph_protocol.PROTOCOL_VERSION,
                "dataset_sha256": champion.get("dataset_sha256") == dataset_sha256,
                "fold_assignment_sha256": champion.get("fold_assignment_sha256")
                == fold_assignment_sha256,
            }
            failed = [field for field, valid in checks.items() if not valid]
            if failed:
                raise ValueError(f"Invalid grouped champion {context}: {failed[0]}")
            if (
                family == "lightgcn"
                and str(champion.get("variant", "")).startswith("trained_")
                and "replay_epochs" not in champion
            ):
                raise ValueError(f"Invalid grouped champion {context}: replay_epochs")
            if (
                family == "lightgcn"
                and str(champion.get("variant", "")).startswith("trained_")
                and not champion.get("robustness_summary_sha256")
            ):
                raise ValueError(f"Invalid grouped champion {context}: robustness_summary_sha256")
            if family == "lightgcn" and str(champion.get("variant", "")).startswith("trained_"):
                robustness_path = Path(str(champion.get("robustness_summary_path", "")))
                if (
                    not robustness_path.is_file()
                    or hashlib.sha256(robustness_path.read_bytes()).hexdigest()
                    != champion["robustness_summary_sha256"]
                ):
                    raise ValueError(f"Invalid grouped champion {context}: robustness_summary_path")


def build_lightgcn_replay_args(
    champion: dict[str, Any],
    *,
    train_bench_dir: Path,
    eval_bench_dir: Path,
    graph_version: str,
    suffix: str,
    top_k_out: int,
) -> list[str]:
    """Build immutable final-replay arguments from one frozen CV champion."""
    variant = str(champion.get("variant", ""))
    args = [
        "--train-bench-dir", str(train_bench_dir),
        "--eval-bench-dir", str(eval_bench_dir),
        "--graph-version", graph_version,
        "--train-k", str(int(champion.get("train_k", champion.get("k_in", 2)))),
        "--output-suffix", suffix,
        "--dump-rankings",
        "--top-k-out", str(top_k_out),
        "--history-top-k-out", "10",
        "--negative-sampling-strategy",
        str(champion.get("negative_sampling_strategy", "random")),
    ]
    if variant.startswith("trained_"):
        if "replay_epochs" not in champion:
            raise ValueError("LightGCN trained champion is missing replay_epochs")
        args.extend(
            [
                "--seed", str(int(champion.get("seed", 42))),
                "--lr", str(float(champion.get("lr", 1e-3))),
                "--epochs", str(int(champion["replay_epochs"])),
                "--checkpoint-selection", "fixed_final_epoch",
                "--lambda-anchor", str(float(champion.get("lambda_anchor", 1.0))),
                "--trained-only",
            ]
        )
    else:
        args.append("--notrain")
    return args


def replay_b3_b4(
    eval_bench_dir: Path,
    champions: dict[str, Any],
    graph_version: str,
    top_k_out: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_eval = _load_script_module("26_eval_doctrine_v3plus_m1_m2.py", "eval_b3b4_final")
    questions = graph_protocol.load_bench_questions(eval_bench_dir)
    ks_in = baseline_kins(champions)
    with tempfile.TemporaryDirectory(prefix="final_b3b4_") as tmp_dir:
        out_dir = Path(tmp_dir)
        baseline_eval.eval_m1_m2(
            questions,
            out_dir,
            ks_in=ks_in or None,
            question_cache_dir=eval_bench_dir,
            graph_version=graph_version,
            top_k_out=top_k_out,
        )
        df = pd.read_csv(out_dir / "eval_m1_m2.csv")
        rankings_path = out_dir / "rankings.parquet"
        rankings_df = pd.read_parquet(rankings_path) if rankings_path.exists() else pd.DataFrame()
    return _select_champion_rows(df, champions), _select_champion_rows(rankings_df, champions)


def replay_ppr(
    eval_bench_dir: Path,
    champions: dict[str, Any],
    graph_version: str,
    top_k_out: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ppr_sweep = _load_script_module("25_ppr_kin_sweep.py", "ppr_final")
    with tempfile.TemporaryDirectory(prefix="final_ppr_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        _copy_bench_artifacts(eval_bench_dir, tmp_path)
        ppr_sweep.main(
            tmp_path,
            config_specs=ppr_configs(champions),
            dump_rankings=True,
            graph_version=graph_version,
            top_k_out=top_k_out,
        )
        df = pd.read_csv(tmp_path / "ppr_kin_sweep_eval.csv")
        rankings_df = pd.read_parquet(tmp_path / "rankings.parquet")
    return _select_champion_rows(df, champions), _select_champion_rows(rankings_df, champions)


def replay_lightgcn(
    train_bench_dir: Path,
    eval_bench_dir: Path,
    graph_version: str,
    champions: dict[str, Any],
    top_k_out: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lightgcn = _load_script_module("32_lightgcn_strict.py", "lightgcn_final")
    eval_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    ranking_frames: list[pd.DataFrame] = []
    with tempfile.TemporaryDirectory(prefix="final_lightgcn_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        train_tmp = tmp_path / "train"
        eval_tmp = tmp_path / "eval"
        _copy_bench_artifacts(train_bench_dir, train_tmp)
        _copy_bench_artifacts(eval_bench_dir, eval_tmp)
        for row in unique_lightgcn_champions(champions):
            suffix = lightgcn_run_id(row)
            variant = str(row.get("variant", ""))
            args = build_lightgcn_replay_args(
                row,
                train_bench_dir=train_tmp,
                eval_bench_dir=eval_tmp,
                graph_version=graph_version,
                suffix=suffix,
                top_k_out=top_k_out,
            )
            rc = lightgcn.main(args)
            if rc != 0:
                raise RuntimeError(f"LightGCN final replay failed for variant={variant}")
            eval_path = eval_tmp / f"lightgcn_eval_{suffix}.csv"
            history_path = eval_tmp / f"lightgcn_history_{suffix}.csv"
            run_df = pd.read_csv(eval_path)
            run_df["train_k"] = row.get("train_k", row.get("k_in", 2))
            run_df["seed"] = row.get("seed")
            run_df["lr"] = row.get("lr")
            run_df["cv_epochs"] = row.get("epochs")
            run_df["replay_epochs"] = row.get("replay_epochs", row.get("epochs"))
            run_df["selected_epoch_index"] = row.get("selected_epoch_index")
            run_df["lambda_anchor"] = row.get("lambda_anchor")
            run_df["negative_sampling_strategy"] = row.get("negative_sampling_strategy", "random")
            run_df["selection_target"] = row.get("selection_target", row.get("modality"))
            run_df["run_id"] = suffix
            run_df["graph_version"] = graph_version
            run_df = run_df[run_df["variant"] == variant].copy()
            eval_frames.append(run_df)
            if history_path.exists():
                hist_df = pd.read_csv(history_path)
                hist_df["graph_version"] = graph_version
                hist_df["negative_sampling_strategy"] = row.get("negative_sampling_strategy", "random")
                hist_df["selection_target"] = row.get("selection_target", row.get("modality"))
                hist_df["run_id"] = suffix
                hist_df = hist_df[hist_df["variant"] == variant].copy()
                history_frames.append(hist_df)
            rankings_path = eval_tmp / "rankings.parquet"
            if rankings_path.exists():
                method_name = f"LightGCN-{variant}"
                rk_df = pd.read_parquet(rankings_path)
                rk_df = rk_df[rk_df["method"] == method_name].copy()
                rk_df["negative_sampling_strategy"] = row.get("negative_sampling_strategy", "random")
                rk_df["train_k"] = row.get("train_k", row.get("k_in", 2))
                rk_df["seed"] = row.get("seed")
                rk_df["lr"] = row.get("lr")
                rk_df["cv_epochs"] = row.get("epochs")
                rk_df["replay_epochs"] = row.get("replay_epochs", row.get("epochs"))
                rk_df["lambda_anchor"] = row.get("lambda_anchor")
                rk_df["selection_target"] = row.get("selection_target", row.get("modality"))
                rk_df["run_id"] = suffix
                ranking_frames.append(rk_df)
    eval_df = pd.concat(eval_frames, ignore_index=True) if eval_frames else pd.DataFrame()
    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    rankings_df = pd.concat(ranking_frames, ignore_index=True) if ranking_frames else pd.DataFrame()
    return eval_df, history_df, _select_champion_rows(rankings_df, champions)


def load_coverage_summary(path: Path, graph_version: str) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Missing coverage summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    split = payload.get("datasets", {}).get("eval_rich_retrievable_strict", {})
    graph_key = coverage_source_graph_key(graph_version)
    if graph_key not in split:
        raise KeyError(f"Missing coverage row for graph_version={graph_version} in {path}")
    row = split[graph_key]
    return {
        "coverage_questions": row.get("questions"),
        "coverage_articles": row.get("strict_q_any_pct"),
        "coverage_articles_occ_pct": row.get("strict_occ_pct"),
        "coverage_articles_occ_present": row.get("strict_occ_present"),
        "coverage_articles_occ_total": row.get("strict_occ_total"),
        "coverage_articles_unique_pct": row.get("strict_unique_pct"),
        "coverage_articles_unique_present": row.get("strict_unique_present"),
        "coverage_articles_unique_total": row.get("strict_unique_total"),
        "coverage_articles_q_all_pct": row.get("strict_q_all_pct"),
        "coverage_articles_q_any_pct": row.get("strict_q_any_pct"),
        "coverage_articles_extended_occ_pct": row.get("ext_occ_pct"),
        "coverage_articles_extended_occ_present": row.get("ext_occ_present"),
        "coverage_articles_extended_occ_total": row.get("ext_occ_total"),
        "coverage_articles_extended_unique_pct": row.get("ext_unique_pct"),
        "coverage_articles_extended_unique_present": row.get("ext_unique_present"),
        "coverage_articles_extended_unique_total": row.get("ext_unique_total"),
        "coverage_jp": row.get("jp_q_any_pct"),
        "coverage_jp_occ_pct": row.get("jp_occ_pct"),
        "coverage_jp_occ_present": row.get("jp_occ_present"),
        "coverage_jp_occ_total": row.get("jp_occ_total"),
        "coverage_jp_unique_pct": row.get("jp_unique_pct"),
        "coverage_jp_unique_present": row.get("jp_unique_present"),
        "coverage_jp_unique_total": row.get("jp_unique_total"),
        "coverage_jp_q_all_pct": row.get("jp_q_all_pct"),
        "coverage_jp_q_any_pct": row.get("jp_q_any_pct"),
    }


def coverage_source_graph_key(graph_version: str) -> str:
    """Map derived G4--G7 graph variants to their G1 coverage source row."""
    lowered = str(graph_version).lower()
    if lowered.startswith(("g4-", "g5-", "g6-", "g6u-", "g7-")):
        return "g1"
    return lowered


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
    jp_metric_variants = [
        {
            "m1": "m1",
            "hit": "hit",
            "mrr": "mrr",
            "ndcg": "ndcg",
            "m2": "m2",
        },
        {
            "m1": "m1_jp",
            "hit": "hit_jp",
            "mrr": "mrr_jp",
            "ndcg": "ndcg_jp",
            "m2": "m2_jp",
        },
        {
            "m1": "m1_strict",
            "hit": "hit_strict",
            "mrr": "mrr_strict",
            "ndcg": "ndcg_strict",
            "m2": "m2_strict",
        },
    ]
    for metric_cols in jp_metric_variants:
        if metric_cols["hit"] in df.columns:
            return metric_cols
    raise ValueError(f"Could not resolve final metric columns for modality={modality}")


def summarize_final_slice(
    df: pd.DataFrame,
    *,
    graph_version: str,
    family: str,
    champion: dict[str, Any],
    n_questions_benchmark: int,
    coverage: dict[str, float],
    m3_lookup: dict[str, str],
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
        "m3_display": _m3_display_for_champion(m3_lookup, champion, modality),
    }
    row.update(coverage)
    for out_col, src_col in metrics.items():
        if src_col in df.columns:
            row[out_col] = float(df[src_col].mean())
    for extra in [
        "seed_variant",
        "alpha",
        "variant",
        "train_k",
        "seed",
        "lr",
        "epochs",
        "selected_epoch_index",
        "replay_epochs",
        "epoch_selection_metric",
        "lambda_anchor",
        "protocol_version",
        "dataset_sha256",
        "fold_assignment_sha256",
        "eligible_champion",
        "n_folds_covered",
    ]:
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
    m3_lookup: dict[str, str],
) -> pd.DataFrame:
    questions = graph_protocol.load_bench_questions(eval_bench_dir)
    n_questions_benchmark = len(questions)
    rows: list[dict[str, Any]] = []

    for champion in champion_bundle.get("b3_b4", {}).values():
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
                    m3_lookup=m3_lookup,
                )
            )

    for champion in champion_bundle.get("ppr", {}).values():
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
                    m3_lookup=m3_lookup,
                )
            )

    for champion in champion_bundle.get("lightgcn", {}).values():
        mask = lightgcn_df["variant"] == champion["variant"]
        for col in ["train_k", "seed", "lr", "lambda_anchor", "selection_target"]:
            value = champion.get(col)
            if value is None or pd.isna(value):
                continue
            if col in lightgcn_df.columns:
                mask &= lightgcn_df[col] == value
        if "epochs" in champion and "cv_epochs" in lightgcn_df.columns:
            mask &= lightgcn_df["cv_epochs"] == champion["epochs"]
        if "replay_epochs" in champion and "replay_epochs" in lightgcn_df.columns:
            mask &= lightgcn_df["replay_epochs"] == champion["replay_epochs"]
        if "run_id" in lightgcn_df.columns:
            mask &= lightgcn_df["run_id"] == lightgcn_run_id(champion)
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
                    m3_lookup=m3_lookup,
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
    parser.add_argument("--campaign-manifest", type=Path)
    parser.add_argument(
        "--authorize-internal-eval",
        help="Must exactly equal campaign_id before grouped_v2 may read internal evaluation.",
    )
    parser.add_argument(
        "--protocol-version",
        choices=("legacy", graph_protocol.PROTOCOL_VERSION),
        default="legacy",
    )
    parser.add_argument(
        "--families",
        default="b3_b4,ppr,lightgcn",
        help="Comma-separated replay families.",
    )
    parser.add_argument("--coverage-summary", type=Path, default=COVERAGE_SUMMARY)
    parser.add_argument("--run-m3", action="store_true")
    parser.add_argument("--m3-mock", action="store_true")
    parser.add_argument("--m3-pilot", type=int)
    parser.add_argument("--m3-workers", type=int, default=32)
    parser.add_argument("--m3-model-id")
    parser.add_argument("--m3-port", type=int)
    parser.add_argument(
        "--top-k-out",
        type=int,
        default=10,
        help="Nombre de résultats matérialisés et évalués par méthode. Défaut 10; utiliser 100 pour diagnostic profond.",
    )
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
    families = [family.strip() for family in args.families.split(",") if family.strip()]
    campaign = None
    if args.protocol_version == graph_protocol.PROTOCOL_VERSION:
        if args.campaign_manifest is None:
            raise ValueError("grouped_v2 replay requires --campaign-manifest")
        campaign = json.loads(args.campaign_manifest.read_text(encoding="utf-8"))
        validate_campaign_provenance(campaign)
        if args.authorize_internal_eval != campaign.get("campaign_id"):
            raise ValueError("internal evaluation authorization must exactly match campaign_id")
        grouped_cv_root, grouped_out_dir = resolve_replay_roots(
            args.graph_version, args.protocol_version
        )
        cv_root = args.cv_root or grouped_cv_root
        out_dir = args.out_dir or grouped_out_dir
    else:
        cv_root = args.cv_root or (train_bench_dir / "_cv")
        out_dir = args.out_dir or (eval_bench_dir / FINAL_DIRNAME)
    out_dir.mkdir(parents=True, exist_ok=True)

    champion_bundle = load_champions(
        cv_root,
        families,
        frozen_lightgcn=args.protocol_version == graph_protocol.PROTOCOL_VERSION,
    )
    if args.protocol_version == graph_protocol.PROTOCOL_VERSION:
        _folds, fold_metadata = graph_protocol.load_verified_grouped_fold_assignments(
            train_bench_dir
        )
        validate_grouped_champion_bundle(
            champion_bundle,
            dataset_sha256=fold_metadata["dataset_sha256"],
            fold_assignment_sha256=fold_metadata["fold_assignment_sha256"],
        )
    coverage = load_coverage_summary(args.coverage_summary, args.graph_version)
    manifest_path = out_dir / (
        "selected_champions.json"
        if args.protocol_version == graph_protocol.PROTOCOL_VERSION
        else "champions_manifest.json"
    )
    if args.skip_replay and manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != champion_bundle:
            raise ValueError(
                "Existing final bundle was produced with a different champions manifest. "
                "Rerun without --skip-replay or use a fresh --out-dir."
            )
    manifest_path.write_text(
        json.dumps(champion_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    selected_champions_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    if args.skip_replay:
        b3_b4_df = pd.read_csv(out_dir / "eval_m1_m2.csv") if "b3_b4" in champion_bundle else pd.DataFrame()
        ppr_df = pd.read_csv(out_dir / "ppr_kin_sweep_eval.csv") if "ppr" in champion_bundle else pd.DataFrame()
        lightgcn_df = pd.read_csv(out_dir / "lightgcn_eval.csv") if "lightgcn" in champion_bundle else pd.DataFrame()
        rankings_df = pd.read_parquet(out_dir / "rankings.parquet") if (out_dir / "rankings.parquet").exists() else pd.DataFrame()
        history_df = pd.read_csv(out_dir / "lightgcn_history.csv") if (out_dir / "lightgcn_history.csv").exists() else pd.DataFrame()
    else:
        b3_b4_df, b3_b4_rankings = (
            replay_b3_b4(eval_bench_dir, champion_bundle["b3_b4"], args.graph_version, top_k_out=args.top_k_out)
            if "b3_b4" in champion_bundle else (pd.DataFrame(), pd.DataFrame())
        )
        ppr_df, ppr_rankings = (
            replay_ppr(eval_bench_dir, champion_bundle["ppr"], args.graph_version, top_k_out=args.top_k_out)
            if "ppr" in champion_bundle else (pd.DataFrame(), pd.DataFrame())
        )
        lightgcn_df, history_df, lightgcn_rankings = (
            replay_lightgcn(train_bench_dir, eval_bench_dir, args.graph_version, champion_bundle["lightgcn"], top_k_out=args.top_k_out)
            if "lightgcn" in champion_bundle else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        )
        rankings_df = pd.concat(
            [b3_b4_rankings, ppr_rankings, lightgcn_rankings],
            ignore_index=True,
        )
        if not b3_b4_df.empty:
            b3_b4_df.to_csv(out_dir / "eval_m1_m2.csv", index=False)
        if not ppr_df.empty:
            ppr_df.to_csv(out_dir / "ppr_kin_sweep_eval.csv", index=False)
        if not lightgcn_df.empty:
            lightgcn_df.to_csv(out_dir / "lightgcn_eval.csv", index=False)
        rankings_df.to_parquet(out_dir / "rankings.parquet", index=False)
        if not history_df.empty:
            history_df.to_csv(out_dir / "lightgcn_history.csv", index=False)

    m3_path = out_dir / "eval_m3.csv"
    if args.run_m3:
        m3_path = _run_m3_judge(
            eval_bench_dir,
            out_dir,
            champion_bundle,
            mock=args.m3_mock,
            pilot=args.m3_pilot,
            workers=args.m3_workers,
            model_id=args.m3_model_id,
            port=args.m3_port,
        )

    m3_lookup = _load_m3_lookup(m3_path)
    summary_df = build_final_summary(
        args.graph_version,
        eval_bench_dir,
        champion_bundle,
        b3_b4_df,
        ppr_df,
        lightgcn_df,
        coverage,
        m3_lookup,
    )
    if args.protocol_version == graph_protocol.PROTOCOL_VERSION:
        assert campaign is not None
        validate_campaign_provenance(campaign)
        if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != selected_champions_sha256:
            raise ValueError("selected_champions.json changed during internal replay")
        graph_rows = [row for row in campaign["graphs"] if row["graph_id"] == args.graph_version]
        if len(graph_rows) != 1:
            raise ValueError(f"campaign manifest missing graph={args.graph_version}")
        summary_df["manifest_sha256"] = canonical_json_sha256(campaign)
        summary_df["internal_eval_sha256"] = campaign["datasets"]["internal_eval"]["sha256"]
        summary_df["graph_matrix_sha256"] = graph_rows[0]["matrix_sha256"]
    summary_df.to_csv(out_dir / "final_champions_summary.csv", index=False)
    write_final_tables(summary_df, out_dir)
    if args.protocol_version == "legacy":
        refresh_intergraph_report()
        refresh_week13_snippets()

    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
