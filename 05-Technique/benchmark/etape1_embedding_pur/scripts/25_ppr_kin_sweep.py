"""Sweep PPR sur K_IN × SEED_VARIANT × ALPHA (norme row uniquement).

Tâche #27 (Week-10). On reprend la structure du script 20_ppr_naive.py et on
étend le sweep selon trois dimensions :
  - K_IN ∈ {5, 10, 20, 50}
  - SEED_VARIANT ∈ {"art_only", "jp_only", "both"}
  - ALPHA ∈ {0.5, 0.7, 0.85, 0.95}

→ 48 variantes / question × 971 q.

Sym-norm déjà confirmé collapsé sur cosine (cf. décisions Week-9), donc on
n'évalue ici que row-norm. Power iteration identique au script 20 :
20 itérations max, tol 1e-7, r ← α P^T r + (1-α) s.

Sortie dans --bench-dir :
  - ppr_kin_sweep_eval.csv   (1 ligne / qid / variante)
  - ppr_kin_sweep_summary.json (agrégats par triplet)
  - rankings.parquet mis à jour si --dump-rankings
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq

REPO = Path(os.environ.get(
    "LKG_REPO",
    str(Path(__file__).resolve().parents[4]),
))
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
sys.path.insert(0, str(Path(__file__).parent))
from etape1 import config  # noqa: E402
from etape1 import graph_versions  # noqa: E402
import metrics as M  # noqa: E402

DEFAULT_BENCH_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/global_bench"
_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

K_OUT = 10
K_INS = [5, 10, 20, 50]
SEED_VARIANTS = ["art_only", "jp_only", "both"]
ALPHAS = [0.5, 0.7, 0.85, 0.95]
N_ITER = 20
TOL = 1e-7


def write_progress(progress_path: Path | None, payload: dict) -> None:
    if progress_path is None:
        return
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_questions(bench_path: Path, qid_filter: set[str] | None = None) -> list[dict]:
    d = json.loads(bench_path.read_text())
    out = []
    for q in d["questions"]:
        if qid_filter is not None and q["qid"] not in qid_filter:
            continue
        arts = q.get("articles_attendus") or []
        gold_jp_ids = q.get("gold_jp_ids") or []
        pourvois = q.get("pourvois_cc") or []
        if not arts or not (gold_jp_ids or pourvois):
            continue
        if q.get("n_jp_resolues", 0) < 1 and not pourvois:
            continue
        out.append({
            "id": q["qid"],
            "gt_strict": set(arts),
            "gt_ext": set(q.get("articles_attendus_etendu") or arts),
            "gold_jp_ids": set(gold_jp_ids),
            "pourvois": set(pourvois),
        })
    return out


def build_pourvoi_map() -> dict[str, list[str]]:
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def resolve_question_cache_paths(bench_dir: Path) -> tuple[Path, Path]:
    q_emb_cache = bench_dir / "questions_emb.npy"
    q_ids_cache = bench_dir / "questions_ids.npy"
    if q_emb_cache.exists() and q_ids_cache.exists():
        return q_emb_cache, q_ids_cache

    legacy_q_emb_cache = bench_dir / "questions_977_emb.npy"
    legacy_q_ids_cache = bench_dir / "questions_977_ids.npy"
    if legacy_q_emb_cache.exists() and legacy_q_ids_cache.exists():
        return legacy_q_emb_cache, legacy_q_ids_cache

    return q_emb_cache, q_ids_cache


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "method_key",
                "k_in",
                "seed_variant",
                "alpha",
                "n_iter_avg",
                "n_rows",
            ]
        )
    cols = (
        [f"{m}_strict_art" for m in M.METRIC_NAMES]
        + [f"{m}_ext_art" for m in M.METRIC_NAMES]
        + [f"{m}_jp" for m in M.METRIC_NAMES]
    )
    rows = []
    for (k_in, variant, alpha), sub in df.groupby(["k_in", "seed_variant", "alpha"]):
        rows.append(
            {
                "method_key": f"PPR-sweep-k{int(k_in)}-{variant}-a{float(alpha)}",
                "k_in": int(k_in),
                "seed_variant": variant,
                "alpha": float(alpha),
                "n_iter_avg": float(sub["n_iter_used"].mean()),
                "n_rows": int(len(sub)),
                **{c: float(sub[c].mean()) for c in cols if c in sub.columns},
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["k_in", "seed_variant", "alpha"]
    ).reset_index(drop=True)


def row_normalize(Mat: sp.csr_matrix) -> sp.csr_matrix:
    rs = np.asarray(Mat.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return sp.diags(1.0 / rs) @ Mat


def ppr_power_iteration(PT: sp.csr_matrix, s: np.ndarray, alpha: float,
                        n_iter: int = N_ITER, tol: float = TOL) -> tuple[np.ndarray, int]:
    """r ← α P^T r + (1-α) s  (row-norm => PageRank standard)."""
    r = s.copy()
    for t in range(n_iter):
        r_new = alpha * (PT @ r) + (1 - alpha) * s
        if np.abs(r_new - r).sum() < tol:
            return r_new, t + 1
        r = r_new
    return r, n_iter


def ppr_power_iteration_batch(PT: sp.csr_matrix, S: np.ndarray, alphas: np.ndarray,
                              n_iter: int = N_ITER, tol: float = TOL) -> tuple[np.ndarray, np.ndarray]:
    """Version batched de PPR pour plusieurs seeds/alphas d'une même question."""
    R = S.copy()
    alpha_row = alphas.reshape(1, -1)
    teleport_row = (1.0 - alphas).reshape(1, -1)
    n_used = np.zeros(len(alphas), dtype=np.int16)
    for t in range(n_iter):
        R_new = (PT @ R) * alpha_row + S * teleport_row
        delta = np.abs(R_new - R).sum(axis=0)
        newly_converged = (n_used == 0) & (delta < tol)
        n_used[newly_converged] = t + 1
        R = R_new
        if np.all(n_used):
            break
    n_used[n_used == 0] = n_iter
    return R, n_used


def build_seed(variant: str, n_jp: int, N_total: int,
               top_art_pks, art_sims, top_jp_ids, jp_sims,
               artid_to_graphcol, jpid_to_graphrow) -> np.ndarray | None:
    """Construit s normalisé (None si somme = 0)."""
    s = np.zeros(N_total)
    if variant in ("art_only", "both"):
        for pk, sim in zip(top_art_pks, art_sims):
            col = artid_to_graphcol.get(pk)
            if col is not None:
                s[n_jp + col] += max(float(sim), 0.0)
    if variant in ("jp_only", "both"):
        for jid, sim in zip(top_jp_ids, jp_sims):
            row = jpid_to_graphrow.get(jid)
            if row is not None:
                s[row] += max(float(sim), 0.0)
    ssum = s.sum()
    if ssum == 0:
        return None
    s /= ssum
    return s


def parse_config_specs(specs: list[str] | None) -> set[tuple[int, str, float]] | None:
    if not specs:
        return None
    out = set()
    for spec in specs:
        try:
            k_raw, variant, alpha_raw = spec.split(":")
            k_in = int(k_raw)
            alpha = float(alpha_raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid --config {spec!r}; expected format k_in:seed_variant:alpha "
                "for example 20:both:0.5"
            ) from exc
        if k_in not in K_INS:
            raise ValueError(f"Invalid k_in={k_in}; expected one of {K_INS}")
        if variant not in SEED_VARIANTS:
            raise ValueError(f"Invalid seed_variant={variant!r}; expected one of {SEED_VARIANTS}")
        if alpha not in ALPHAS:
            raise ValueError(f"Invalid alpha={alpha}; expected one of {ALPHAS}")
        out.add((k_in, variant, alpha))
    return out


def build_seed_configs(n_jp: int, N_total: int, top_art_full_q, top_jp_full_q,
                       sim_art_q, sim_jp_q, art_order, jp_order,
                       artid_to_graphcol, jpid_to_graphrow,
                       allowed_configs: set[tuple[int, str, float]] | None = None):
    configs = []
    seeds = []
    for k_in in K_INS:
        top_art_emb_idx = top_art_full_q[:k_in]
        top_art_pks = art_order[top_art_emb_idx]
        art_sims = sim_art_q[top_art_emb_idx]
        top_jp_emb_idx = top_jp_full_q[:k_in]
        top_jp_ids_arr = jp_order[top_jp_emb_idx]
        jp_sims = sim_jp_q[top_jp_emb_idx]

        for variant in SEED_VARIANTS:
            wanted_alphas = [
                alpha for alpha in ALPHAS
                if allowed_configs is None or (k_in, variant, alpha) in allowed_configs
            ]
            if not wanted_alphas:
                continue
            s = build_seed(variant, n_jp, N_total,
                           top_art_pks, art_sims, top_jp_ids_arr, jp_sims,
                           artid_to_graphcol, jpid_to_graphrow)
            if s is None:
                continue
            for alpha in wanted_alphas:
                configs.append((k_in, variant, alpha))
                seeds.append(s)
    if not seeds:
        return [], None, None
    S = np.column_stack(seeds)
    alphas = np.array([cfg[2] for cfg in configs], dtype=np.float64)
    return configs, S, alphas


def ranking_rows(qid, method, k_in, modality, ranked, k):
    return [
        {
            "qid": qid,
            "method": method,
            "k_in": k_in,
            "modality": modality,
            "rank": rank + 1,
            "item_id": str(item),
        }
        for rank, item in enumerate(ranked[:k])
    ]


def top_k_labels(scores: np.ndarray, labels: np.ndarray, k: int) -> list:
    """Retourne les labels des k meilleurs scores, triés décroissant.

    On évite argsort complet : les pools JP dépassent 100k items et le sweep
    appelle ce top-k des dizaines de milliers de fois.
    """
    if len(scores) <= k:
        order = np.argsort(-scores)
    else:
        candidates = np.argpartition(-scores, k - 1)[:k]
        order = candidates[np.argsort(-scores[candidates])]
    return list(labels[order])


def main(
    bench_dir: Path,
    limit_q: int | None = None,
    config_specs: list[str] | None = None,
    dump_rankings: bool = False,
    qid_filter: set[str] | None = None,
    graph_version: str = "canonical",
    progress_path: Path | None = None,
    progress_label: str | None = None,
    top_k_out: int = K_OUT,
) -> int:
    t0 = time.time()
    bench_path = bench_dir / "bench_global.json"
    q_emb_cache, q_ids_cache = resolve_question_cache_paths(bench_dir)
    out_csv = bench_dir / "ppr_kin_sweep_eval.csv"
    out_summary = bench_dir / "ppr_kin_sweep_summary.json"
    allowed_configs = parse_config_specs(config_specs)
    if not bench_path.exists():
        raise FileNotFoundError(f"Missing bench file: {bench_path}")
    if not q_emb_cache.exists() or not q_ids_cache.exists():
        raise FileNotFoundError(
            f"Missing question embedding cache in {bench_dir}: "
            "expected questions_emb.npy/questions_ids.npy or "
            "legacy questions_977_emb.npy/questions_977_ids.npy"
        )
    write_progress(
        progress_path,
        {
            "status": "starting",
            "label": progress_label,
            "graph_version": graph_version,
            "bench_dir": str(bench_dir),
            "started_at": t0,
        },
    )

    print("══ Chargement graphe + embeddings ─────────────────────────")
    view = graph_versions.load_retrieval_view(graph_version)
    art_emb = view.art_emb
    art_order = view.art_order
    jp_emb = view.jp_emb
    jp_order = view.jp_order
    G = view.graph
    jp_ids_graph = view.jp_ids_graph
    article_ids_graph = view.article_ids_graph
    n_jp = len(jp_ids_graph)
    n_art = len(article_ids_graph)
    print(f"  G {G.shape}  nnz {G.nnz:,}")

    jpid_to_graphrow = {jid: i for i, jid in enumerate(jp_ids_graph)}
    artid_to_graphcol = {aid: i for i, aid in enumerate(article_ids_graph)}
    pool_articles_set = set(art_order.tolist())
    pool_jp_set = set(jp_order.tolist())

    print("\n══ Block-bipartite + row-normalize ────────────────────────")
    N_total = n_jp + n_art
    if G.shape == (n_jp, n_art):
        G_full = sp.bmat([[None, G], [G.T, None]], format="csr")
    elif G.shape == (N_total, N_total):
        G_full = G.tocsr()
    else:
        raise ValueError(
            f"Unsupported graph shape for PPR: graph={G.shape}, "
            f"n_jp={n_jp}, n_art={n_art}"
        )
    print(f"  G_full {G_full.shape}  nnz {G_full.nnz:,}")
    P_row = row_normalize(G_full)
    PT_row = P_row.T.tocsr()
    print(f"  P_row ready  (t={time.time()-t0:.1f}s)")

    art_pool_graph_idx = np.array(
        [n_jp + artid_to_graphcol[pk] for pk in art_order if pk in artid_to_graphcol],
        dtype=np.int64
    )
    art_pool_pks = np.array(
        [pk for pk in art_order if pk in artid_to_graphcol]
    )
    jp_pool_graph_idx = np.array(
        [jpid_to_graphrow[j] for j in jp_order if j in jpid_to_graphrow],
        dtype=np.int64
    )
    jp_pool_ids = np.array(
        [j for j in jp_order if j in jpid_to_graphrow]
    )

    print("\n══ Chargement cohorte ─────────────────────────────────────")
    questions = load_questions(bench_path, qid_filter=qid_filter)
    pourvoi_map = build_pourvoi_map() if any(q["pourvois"] for q in questions) else {}
    Q_emb = np.load(q_emb_cache)
    cached_qids = np.load(q_ids_cache, allow_pickle=True).tolist()
    qid_set = set(cached_qids)
    questions = [q for q in questions if q["id"] in qid_set]
    qid_to_emb = {qid: i for i, qid in enumerate(cached_qids)}
    Q = np.asarray([Q_emb[qid_to_emb[q["id"]]] for q in questions])
    if limit_q is not None:
        questions = questions[:limit_q]
        Q = Q[:limit_q]
    print(f"  questions évaluées : {len(questions)}")
    write_progress(
        progress_path,
        {
            "status": "running",
            "label": progress_label,
            "graph_version": graph_version,
            "bench_dir": str(bench_dir),
            "phase": "loaded_questions",
            "questions_total": len(questions),
            "rows_written": 0,
            "elapsed_seconds": time.time() - t0,
        },
    )
    if allowed_configs is not None:
        print("  configs PPR filtrées : " + ", ".join(
            f"k{k}|{variant}|a{alpha}" for k, variant, alpha in sorted(allowed_configs)
        ))

    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T
    sim_jp = Q @ jp_emb.T
    print(f"  sim shapes {sim_art.shape} / {sim_jp.shape}  (t={time.time()-t0:.1f}s)")

    K_MAX = max(K_INS)
    print(f"\n══ Pré-calcul top-{K_MAX} cosine seeds ─────────────────")
    top_art_full = np.argpartition(-sim_art, K_MAX, axis=1)[:, :K_MAX]
    top_jp_full = np.argpartition(-sim_jp, K_MAX, axis=1)[:, :K_MAX]
    # Tri pour que top-K_in (K_in < K_MAX) = prefix correct des K_in meilleurs.
    for qi in range(len(questions)):
        order = np.argsort(-sim_art[qi, top_art_full[qi]])
        top_art_full[qi] = top_art_full[qi][order]
        order_j = np.argsort(-sim_jp[qi, top_jp_full[qi]])
        top_jp_full[qi] = top_jp_full[qi][order_j]
    print(f"  pré-calcul fait (t={time.time()-t0:.1f}s)")

    print("\n══ Boucle PPR sweep ───────────────────────────────────────")
    rows = []
    rankings = []
    n_skip = 0
    for qi, q in enumerate(questions):
        if qi < 20 or qi % 100 == 0:
            print(f"  q {qi}/{len(questions)}  (t={time.time()-t0:.1f}s, rows={len(rows)})")
            write_progress(
                progress_path,
                {
                    "status": "running",
                    "label": progress_label,
                    "graph_version": graph_version,
                    "bench_dir": str(bench_dir),
                    "phase": "ppr_sweep",
                    "questions_total": len(questions),
                    "question_index": qi,
                    "question_id": q["id"],
                    "rows_written": len(rows),
                    "elapsed_seconds": time.time() - t0,
                },
            )

        gt_s = q["gt_strict"] & pool_articles_set
        gt_e = q["gt_ext"] & pool_articles_set
        gold_jp = (
            q["gold_jp_ids"]
            | {jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}
        ) & pool_jp_set
        if not gt_s and not gold_jp:
            n_skip += 1
            continue

        configs, S, alphas = build_seed_configs(
            n_jp, N_total,
            top_art_full[qi], top_jp_full[qi],
            sim_art[qi], sim_jp[qi],
            art_order, jp_order,
            artid_to_graphcol, jpid_to_graphrow,
            allowed_configs,
        )
        if S is None:
            continue
        R, n_used_by_col = ppr_power_iteration_batch(PT_row, S, alphas)
        R_art_pool = R[art_pool_graph_idx, :]
        R_jp_pool = R[jp_pool_graph_idx, :]

        for col, (k_in, variant, alpha) in enumerate(configs):
            method_name = f"PPR-sweep-k{k_in}-{variant}-a{alpha}"

            ranked_art = top_k_labels(R_art_pool[:, col], art_pool_pks, top_k_out)
            ranked_jp = top_k_labels(R_jp_pool[:, col], jp_pool_ids, top_k_out)
            if dump_rankings:
                rankings.extend(
                    ranking_rows(q["id"], method_name, k_in, "art", ranked_art, top_k_out)
                )
                rankings.extend(
                    ranking_rows(q["id"], method_name, k_in, "jp", ranked_jp, top_k_out)
                )

            art_metrics = M.panel_strict_ext(ranked_art, gt_s, gt_e, top_k_out)
            art_metrics = {f"{k}_art": v for k, v in art_metrics.items()}
            jp_metrics = {f"{k}_jp": v for k, v in
                          M.all_metrics(ranked_jp, gold_jp, top_k_out).items()}
            rows.append({
                "qid": q["id"],
                "k_in": k_in,
                "seed_variant": variant,
                "alpha": alpha,
                "n_iter_used": int(n_used_by_col[col]),
                **art_metrics,
                **jp_metrics,
            })

    print(f"\n  skipped (GT vide) : {n_skip}")
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}  ({len(df)} lignes)")

    if dump_rankings:
        rank_path = bench_dir / "rankings.parquet"
        rk = pd.DataFrame(rankings)
        if rank_path.exists():
            prev = pd.read_parquet(rank_path)
            prev = prev[~prev["method"].astype(str).str.startswith("PPR-sweep-")]
            rk = pd.concat([prev, rk], ignore_index=True)
        rk.to_parquet(rank_path, index=False)
        print(f"✓ {rank_path}  ({len(rk)} lignes rankings)")

    print("\n══ Agrégats par (k_in, seed, α) ───────────────────────────")
    cols = (
        [f"{m}_strict_art" for m in M.METRIC_NAMES] +
        [f"{m}_ext_art"    for m in M.METRIC_NAMES] +
        [f"{m}_jp"         for m in M.METRIC_NAMES]
    )
    summary_df = summarize_results(df)
    summary = {}
    for row in summary_df.to_dict(orient="records"):
        key = row["method_key"]
        summary[key] = {k: v for k, v in row.items() if k != "method_key"}
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✓ {out_summary}")
    write_progress(
        progress_path,
        {
            "status": "completed",
            "label": progress_label,
            "graph_version": graph_version,
            "bench_dir": str(bench_dir),
            "phase": "completed",
            "questions_total": len(questions),
            "rows_written": len(df),
            "elapsed_seconds": time.time() - t0,
            "out_csv": str(out_csv),
            "out_summary": str(out_summary),
        },
    )

    s_df = summary_df.drop(columns=["method_key"], errors="ignore")
    for metric in ["m1_ext_art", "ndcg_ext_art", "mrr_strict_art", "ndcg_jp", "hit_ext_art"]:
        if metric not in s_df.columns:
            continue
        top = s_df.sort_values(metric, ascending=False).head(5)
        print(f"\n  TOP-5 sur {metric}:")
        for _, r in top.iterrows():
            print(f"    k_in={int(r['k_in']):>2} {r['seed_variant']:>9} α={r['alpha']:.2f}  "
                  f"{metric}={r[metric]:.4f}")

    print(f"\n  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, default=DEFAULT_BENCH_DIR)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Restrict sweep to one config k_in:seed_variant:alpha, "
            "for example --config 20:both:0.5. Repeatable."
        ),
    )
    parser.add_argument("--dump-rankings", action="store_true")
    parser.add_argument("--graph-version", default="canonical")
    parser.add_argument("--progress-path", type=Path)
    parser.add_argument("--progress-label")
    parser.add_argument("--top-k-out", type=int, default=K_OUT)
    args = parser.parse_args()
    if args.limit is not None:
        print(f"[mode sanity check : limit={args.limit}]")
    sys.exit(
        main(
            args.bench_dir,
            args.limit,
            args.config,
            args.dump_rankings,
            graph_version=args.graph_version,
            progress_path=args.progress_path,
            progress_label=args.progress_label,
            top_k_out=args.top_k_out,
        )
    )
