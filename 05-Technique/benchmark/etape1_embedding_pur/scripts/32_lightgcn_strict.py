"""LightGCN strict -- train/tuning strict, eval finale stricte.

Version branchee sur les deux datasets officiels :
  - train : train_augmented_retrievable_strict
  - eval  : eval_rich_retrievable_strict

Sorties dans --eval-bench-dir :
  - lightgcn_eval.csv
  - lightgcn_summary.json

Cette version garde le design A de 31_lightgcn.py :
  - items = JP + articles du graphe ;
  - questions hors graphe, embeddings BGE-M3 figes ;
  - propagation LightGCN sur le graphe Art<->JP ;
  - entrainement BPR cosine sur les articles gold etendus du train strict.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etape1 import config  # noqa: E402
import metrics as M  # noqa: E402

DEFAULT_BASE = ROOT / "data/doctrine_v3plus_bench"
DEFAULT_TRAIN = DEFAULT_BASE / "train_augmented_retrievable_strict"
DEFAULT_EVAL = DEFAULT_BASE / "eval_rich_retrievable_strict"

K_OUT = 10
N_LAYERS_TO_EVAL = (1, 2, 3)
TAU = 0.1
DEVICE = "cpu"


def rowcol_top_k(scores: np.ndarray, labels: np.ndarray, k: int) -> list:
    if len(scores) <= k:
        order = np.argsort(-scores)
    else:
        cand = np.argpartition(-scores, k - 1)[:k]
        order = cand[np.argsort(-scores[cand])]
    return list(labels[order])


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


def sym_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    return sp.diags(dinv) @ A @ sp.diags(dinv)


def sparse_scipy_to_torch(A: sp.csr_matrix) -> torch.Tensor:
    A = A.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape)).coalesce()


def load_split(bench_dir: Path, limit: int | None = None) -> tuple[list[dict], np.ndarray]:
    bench = json.loads((bench_dir / "bench_global.json").read_text())["questions"]
    qids = np.load(bench_dir / "questions_ids.npy", allow_pickle=True).tolist()
    emb = np.load(bench_dir / "questions_emb.npy").astype(np.float32)
    qid_to_i = {qid: i for i, qid in enumerate(qids)}
    rows = []
    q_embs = []
    for q in bench:
        qid = q["qid"]
        if qid not in qid_to_i:
            continue
        arts = set(q.get("articles_attendus") or [])
        arts_ext = set(q.get("articles_attendus_etendu") or arts)
        jp = set(q.get("gold_jp_ids") or [])
        if not arts or not jp:
            continue
        rows.append({
            "id": qid,
            "gt_strict": arts,
            "gt_ext": arts_ext,
            "gold_jp_ids": jp,
        })
        q_embs.append(emb[qid_to_i[qid]])
        if limit is not None and len(rows) >= limit:
            break
    return rows, np.asarray(q_embs, dtype=np.float32)


class LightGCN(nn.Module):
    def __init__(self, e0_init: torch.Tensor, n_layers: int):
        super().__init__()
        self.emb = nn.Parameter(e0_init.clone())
        self.n_layers = n_layers

    def propagate(self, adj: torch.Tensor) -> torch.Tensor:
        outs = [self.emb]
        x = self.emb
        for _ in range(self.n_layers):
            x = torch.sparse.mm(adj, x)
            outs.append(x)
        return sum(outs) / (self.n_layers + 1)


def l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-9)


def bpr_loss(q_vec, item_final, pos_idx, neg_idx, tau=TAU):
    qn = l2(q_vec)
    pos = (qn * l2(item_final[pos_idx])).sum(1) / tau
    neg = (qn * l2(item_final[neg_idx])).sum(1) / tau
    return -torch.log(torch.sigmoid(pos - neg) + 1e-10).mean()


def summarize_eval_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    summary = {}
    for column in [
        "hit_strict_art",
        "ndcg_strict_art",
        "mrr_strict_art",
        "m1_strict_art",
        "m2_strict_art",
        "hit_jp",
        "ndcg_jp",
        "mrr_jp",
        "m1_jp",
        "m2_jp",
    ]:
        if column in frame.columns:
            summary[column] = float(frame[column].mean())
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bench-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-bench-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-k", type=int, default=int(os.environ.get("TRAIN_K", 2)))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--notrain", action="store_true")
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Autorise les qid communs train/eval, uniquement pour scorer le train en phase tuning.",
    )
    parser.add_argument(
        "--trained-only",
        action="store_true",
        help="Ne calcule que la variante entraînée, utile pour sweeps sans écraser les baselines.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Suffixe ajouté aux artefacts lightgcn_eval/summary, ex: _s1_k2_e30.",
    )
    parser.add_argument(
        "--graph-version",
        default="canonical",
        help="Label de graphe reporte dans lightgcn_history_<suffix>.csv.",
    )
    parser.add_argument("--dump-rankings", action="store_true")
    args = parser.parse_args(argv)

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("══ Chargement graphe + embeddings ─────────────────────────")
    art_emb = np.load(config.EMB_ARTICLES_ALL).astype(np.float32)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    jp_emb = np.load(config.EMB_JP_SYNTHESE).astype(np.float32)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    D = art_emb.shape[1]

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]
    n_jp, n_art = G.shape
    n_total = n_jp + n_art
    print(f"  G {G.shape}  nnz {G.nnz:,}  N={n_total:,}")

    G_full = sp.bmat([[None, G], [G.T, None]], format="csr")
    adj = sparse_scipy_to_torch(sym_normalize(G_full)).to(DEVICE)
    print(f"  adj sym ready  (t={time.time()-t0:.1f}s)")

    artid_to_graphcol = {aid: i for i, aid in enumerate(article_ids_graph)}
    jpid_to_graphrow = {jid: i for i, jid in enumerate(jp_ids_graph)}
    pk_to_emb = {pk: i for i, pk in enumerate(art_order)}
    jid_to_emb = {jid: i for i, jid in enumerate(jp_order)}

    e0 = np.zeros((n_total, D), dtype=np.float32)
    has_init = np.zeros(n_total, dtype=bool)
    for row, jid in enumerate(jp_ids_graph):
        k = jid_to_emb.get(jid)
        if k is not None:
            e0[row] = jp_emb[k]
            has_init[row] = True
    for col, aid in enumerate(article_ids_graph):
        k = pk_to_emb.get(aid)
        if k is not None:
            e0[n_jp + col] = art_emb[k]
            has_init[n_jp + col] = True

    med_norm = float(np.median(np.linalg.norm(e0[has_init], axis=1)))
    missing = ~has_init
    rand = rng.standard_normal((missing.sum(), D)).astype(np.float32)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9
    e0[missing] = rand * med_norm
    e0_zero = e0.copy()
    e0_zero[missing] = 0.0
    print(f"  E0 init : {has_init.sum():,}/{n_total:,} nodes BGE, missing {missing.sum():,}")

    art_pool_graph_idx = np.array(
        [n_jp + artid_to_graphcol[pk] for pk in art_order if pk in artid_to_graphcol],
        dtype=np.int64,
    )
    art_pool_pks = np.array([pk for pk in art_order if pk in artid_to_graphcol])
    jp_pool_graph_idx = np.array(
        [jpid_to_graphrow[j] for j in jp_order if j in jpid_to_graphrow],
        dtype=np.int64,
    )
    jp_pool_ids = np.array([j for j in jp_order if j in jpid_to_graphrow])
    pool_art_set = set(art_pool_pks.tolist())
    pool_jp_set = set(jp_pool_ids.tolist())
    art_pk_to_itemidx = {
        pk: n_jp + artid_to_graphcol[pk]
        for pk in art_order
        if pk in artid_to_graphcol
    }
    print(f"  pool art {len(art_pool_pks):,}  pool JP {len(jp_pool_ids):,}")

    train_q, q_train_np = load_split(args.train_bench_dir, args.limit_train)
    eval_q, q_eval_np = load_split(args.eval_bench_dir, args.limit_eval)
    eval_qids = {q["id"] for q in eval_q}
    overlap = {q["id"] for q in train_q} & eval_qids
    if overlap and args.allow_overlap:
        print(f"  ⚠ overlap train/eval autorisé pour tuning : {len(overlap)}")
    elif overlap:
        kept = [(q, emb) for q, emb in zip(train_q, q_train_np) if q["id"] not in overlap]
        train_q = [q for q, _ in kept]
        q_train_np = np.asarray([emb for _, emb in kept], dtype=np.float32)
        print(f"  ⚠ overlap train/eval qid exclu du train : {len(overlap)}")
    print(f"  train questions {len(train_q)}  eval questions {len(eval_q)}")

    train_q_rows = []
    train_pos = []
    for qi, q in enumerate(train_q):
        gts = (q["gt_ext"] or q["gt_strict"]) & pool_art_set
        if not gts:
            continue
        local_qi = len(train_q_rows)
        train_q_rows.append(q_train_np[qi])
        for pk in gts:
            train_pos.append((local_qi, art_pk_to_itemidx[pk]))
    if not train_pos:
        raise RuntimeError("No train positives after pool filtering")
    print(f"  train positives {len(train_pos):,} over {len(train_q_rows):,} questions")

    q_train_t = torch.tensor(np.asarray(train_q_rows), dtype=torch.float32, device=DEVICE)
    pos_q = torch.tensor([p[0] for p in train_pos], dtype=torch.long, device=DEVICE)
    pos_item = torch.tensor([p[1] for p in train_pos], dtype=torch.long, device=DEVICE)
    q_eval_t = torch.tensor(q_eval_np, dtype=torch.float32, device=DEVICE)
    e0_t = torch.tensor(e0, dtype=torch.float32, device=DEVICE)
    e0_zero_t = torch.tensor(e0_zero, dtype=torch.float32, device=DEVICE)
    art_pool_idx_t = torch.tensor(art_pool_graph_idx, dtype=torch.long, device=DEVICE)
    jp_pool_idx_t = torch.tensor(jp_pool_graph_idx, dtype=torch.long, device=DEVICE)
    anchor_idx = torch.tensor(np.where(has_init)[0], dtype=torch.long, device=DEVICE)

    def evaluate(item_final: torch.Tensor, variant: str) -> tuple[list[dict], list[dict]]:
        rows = []
        rankings = []
        with torch.no_grad():
            qn = l2(q_eval_t)
            art_final = l2(item_final[art_pool_idx_t])
            jp_final = l2(item_final[jp_pool_idx_t])
            sc_art = (qn @ art_final.T).cpu().numpy()
            sc_jp = (qn @ jp_final.T).cpu().numpy()
        method = f"LightGCN-{variant}"
        kin = 0 if variant == "cosine_raw" else int(variant.rsplit("K", 1)[-1])
        for i, q in enumerate(eval_q):
            gt_s = q["gt_strict"] & pool_art_set
            gt_e = q["gt_ext"] & pool_art_set
            gold_jp = q["gold_jp_ids"] & pool_jp_set
            ranked_art = rowcol_top_k(sc_art[i], art_pool_pks, K_OUT)
            ranked_jp = rowcol_top_k(sc_jp[i], jp_pool_ids, K_OUT)
            rankings.extend(ranking_rows(q["id"], method, kin, "art", ranked_art, K_OUT))
            rankings.extend(ranking_rows(q["id"], method, kin, "jp", ranked_jp, K_OUT))
            am = {f"{k}_art": v for k, v in M.panel_strict_ext(ranked_art, gt_s, gt_e, K_OUT).items()}
            jm = {f"{k}_jp": v for k, v in M.all_metrics(ranked_jp, gold_jp, K_OUT).items()}
            rows.append({"qid": q["id"], "variant": variant, **am, **jm})
        return rows, rankings

    history_rows: list[dict] = []

    def train_model(n_layers: int) -> torch.Tensor:
        model = LightGCN(e0_t, n_layers).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        n_pos = pos_q.shape[0]
        e0_ref = e0_t[anchor_idx].detach().clone()
        for ep in range(args.epochs):
            item_final = model.propagate(adj)
            neg = torch.tensor(
                rng.integers(0, len(art_pool_graph_idx), n_pos),
                dtype=torch.long,
                device=DEVICE,
            )
            neg_item = art_pool_idx_t[neg]
            bpr = bpr_loss(q_train_t[pos_q], item_final, pos_item, neg_item)
            anchor = ((model.emb[anchor_idx] - e0_ref) ** 2).sum(1).mean()
            loss = bpr + args.lambda_anchor * anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                current_final = model.propagate(adj)
            val_rows, _ = evaluate(current_final, f"trained_K{n_layers}")
            val_summary = summarize_eval_rows(val_rows)
            history_rows.append(
                {
                    "epoch": ep,
                    "graph_version": args.graph_version,
                    "variant": f"trained_K{n_layers}",
                    "train_loss": float(loss.item()),
                    "bpr_loss": float(bpr.item()),
                    "anchor_loss": float(anchor.item()),
                    "val_hit": float(val_summary.get("hit_strict_art", np.nan)),
                    "val_ndcg": float(val_summary.get("ndcg_strict_art", np.nan)),
                    "val_mrr": float(val_summary.get("mrr_strict_art", np.nan)),
                    "val_recall": float(val_summary.get("m1_strict_art", np.nan)),
                    "val_norm_rank": float(val_summary.get("m2_strict_art", np.nan)),
                    "val_hit_jp": float(val_summary.get("hit_jp", np.nan)),
                    "val_ndcg_jp": float(val_summary.get("ndcg_jp", np.nan)),
                }
            )
            if ep % 10 == 0 or ep == args.epochs - 1:
                print(
                    f"    K{n_layers} ep {ep:>3d}/{args.epochs} "
                    f"bpr {bpr.item():.4f} anchor {anchor.item():.4f} "
                    f"(t={time.time()-t0:.1f}s)"
                )
        return current_final

    all_rows = []
    all_rankings = []
    if not args.trained_only:
        print("\n══ Eval cosine_raw ────────────────────────────────────────")
        rows, rk = evaluate(e0_t, "cosine_raw")
        all_rows += rows
        all_rankings += rk

        print("══ Eval untrained propagation ─────────────────────────────")
        for k in N_LAYERS_TO_EVAL:
            model = LightGCN(e0_zero_t, k).to(DEVICE)
            with torch.no_grad():
                final = model.propagate(adj)
            rows, rk = evaluate(final, f"untrained_K{k}")
            all_rows += rows
            all_rankings += rk
            print(f"  untrained_K{k} done  (t={time.time()-t0:.1f}s)")

    if not args.notrain:
        print(f"══ Train LightGCN K={args.train_k} ────────────────────────")
        rows, rk = evaluate(train_model(args.train_k), f"trained_K{args.train_k}")
        all_rows += rows
        all_rankings += rk

    df = pd.DataFrame(all_rows)
    suffix = args.output_suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    out_csv = args.eval_bench_dir / f"lightgcn_eval{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv} ({len(df)} lignes)")

    if args.dump_rankings:
        rank_path = args.eval_bench_dir / "rankings.parquet"
        rk = pd.DataFrame(all_rankings)
        if rank_path.exists():
            prev = pd.read_parquet(rank_path)
            prev = prev[~prev["method"].astype(str).str.startswith("LightGCN-")]
            rk = pd.concat([prev, rk], ignore_index=True)
        rk.to_parquet(rank_path, index=False)
        print(f"✓ {rank_path} ({len(rk)} lignes rankings)")

    metric_cols = (
        [f"{m}_strict_art" for m in M.METRIC_NAMES]
        + [f"{m}_ext_art" for m in M.METRIC_NAMES]
        + [f"{m}_jp" for m in M.METRIC_NAMES]
    )
    summary = {
        variant: {c: float(sub[c].mean()) for c in metric_cols}
        for variant, sub in df.groupby("variant")
    }
    out_summary = args.eval_bench_dir / f"lightgcn_summary{suffix}.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"✓ {out_summary}")

    if history_rows:
        history_df = pd.DataFrame(history_rows)
        out_history = args.eval_bench_dir / f"lightgcn_history{suffix}.csv"
        history_df.to_csv(out_history, index=False)
        print(f"✓ {out_history} ({len(history_df)} lignes)")

    print("\n══ LightGCN strict summary ────────────────────────────────")
    for variant, vals in summary.items():
        print(
            f"  {variant:>12s} | art_s Hit={vals['hit_strict_art']:.3f} "
            f"MRR={vals['mrr_strict_art']:.3f} NDCG={vals['ndcg_strict_art']:.3f} | "
            f"art_e Hit={vals['hit_ext_art']:.3f} NDCG={vals['ndcg_ext_art']:.3f} | "
            f"JP Hit={vals['hit_jp']:.3f} NDCG={vals['ndcg_jp']:.3f}"
        )

    print(f"\n  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
