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

REPO = Path(os.environ.get(
    "LKG_REPO",
    str(Path(__file__).resolve().parents[4]),
))
ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from etape1 import config  # noqa: E402
from etape1 import graph_versions  # noqa: E402
import metrics as M  # noqa: E402
import benchmark_labels  # noqa: E402

DEFAULT_BASE = ROOT / "data/doctrine_v3plus_bench"
DEFAULT_TRAIN = DEFAULT_BASE / "train_augmented_retrievable_strict"
DEFAULT_EVAL = DEFAULT_BASE / "eval_rich_retrievable_strict"

K_OUT = 10
N_LAYERS_TO_EVAL = (1, 2, 3)
TAU = 0.1
DEVICE = "cpu"
NEGATIVE_RANDOM = "random"
HARD_NEGATIVE_PREFIX = "hard_negative_cosine_top"
SEMI_HARD_NEGATIVE_PREFIX = "semi_hard_cosine_rank"


def rowcol_top_k(scores: np.ndarray, labels: np.ndarray, k: int) -> list:
    if len(scores) <= k:
        order = np.argsort(-scores)
    else:
        cand = np.argpartition(-scores, k - 1)[:k]
        order = cand[np.argsort(-scores[cand])]
    return list(labels[order])


def ranking_rows(qid, method, k_in, modality, ranked, k, negative_sampling_strategy):
    return [
        {
            "qid": qid,
            "method": method,
            "k_in": k_in,
            "modality": modality,
            "rank": rank + 1,
            "item_id": str(item),
            "negative_sampling_strategy": negative_sampling_strategy,
        }
        for rank, item in enumerate(ranked[:k])
    ]


def sym_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    return sp.diags(dinv) @ A @ sp.diags(dinv)


def row_normalize(A: sp.csr_matrix) -> sp.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    return sp.diags(1.0 / deg) @ A


def normalize_adjacency(A: sp.csr_matrix, mode: str) -> sp.csr_matrix:
    if mode == "sym":
        return sym_normalize(A)
    if mode == "row":
        return row_normalize(A)
    if mode == "none":
        return A.tocsr()
    raise ValueError(f"Unsupported adjacency normalization: {mode!r}")


def sparse_scipy_to_torch(A: sp.csr_matrix) -> torch.Tensor:
    A = A.tocoo().astype(np.float32)
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long)
    val = torch.tensor(A.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, torch.Size(A.shape)).coalesce()


def iter_similarity_batches(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    *,
    batch_size: int,
) -> tuple[int, np.ndarray]:
    """Yield cosine score batches without materializing all query-item scores."""
    if batch_size <= 0:
        raise ValueError(f"eval batch size must be positive, got {batch_size}")
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        yield start, (queries[start:stop] @ candidates.T).cpu().numpy()


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


def prepare_train_lightgcn_positives(
    bench_dir: Path,
    train_questions: list[dict],
    *,
    article_candidate_ids: list[object] | np.ndarray,
    jp_candidate_ids: list[object] | np.ndarray,
) -> tuple[dict[str, set[str]], dict, str]:
    """Reject invalid strict labels before CV/loss and load sealed positives."""
    benchmark_labels.require_strict_candidate_coverage(
        train_questions,
        article_candidate_ids=article_candidate_ids,
        jp_candidate_ids=jp_candidate_ids,
        context=f"LightGCN training input {bench_dir}",
    )
    positives, projection, projection_sha256 = (
        benchmark_labels.load_verified_lightgcn_article_positive_projection(
            bench_dir,
            article_candidate_ids=article_candidate_ids,
        )
    )
    requested_qids = {str(question["id"]) for question in train_questions}
    missing_qids = sorted(requested_qids - set(positives))
    if missing_qids:
        raise ValueError(
            f"LightGCN projection is missing training questions: {missing_qids[:5]}"
        )
    return {qid: positives[qid] for qid in requested_qids}, projection, projection_sha256


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


def parse_negative_sampling_strategy(
    strategy: str,
) -> tuple[str, int | tuple[int, int] | None]:
    if strategy == NEGATIVE_RANDOM:
        return NEGATIVE_RANDOM, None
    if strategy.startswith(HARD_NEGATIVE_PREFIX):
        suffix = strategy.removeprefix(HARD_NEGATIVE_PREFIX)
        if suffix.isdigit() and int(suffix) > 0:
            return HARD_NEGATIVE_PREFIX, int(suffix)
    if strategy.startswith(SEMI_HARD_NEGATIVE_PREFIX):
        suffix = strategy.removeprefix(SEMI_HARD_NEGATIVE_PREFIX)
        bounds = suffix.split("_")
        if len(bounds) == 2 and all(bound.isdigit() for bound in bounds):
            start_rank, end_rank = (int(bound) for bound in bounds)
            if 1 <= start_rank <= end_rank:
                return SEMI_HARD_NEGATIVE_PREFIX, (start_rank, end_rank)
    raise ValueError(
        "Unsupported negative sampling strategy: "
        f"{strategy}. Expected random, hard_negative_cosine_topN, "
        "or semi_hard_cosine_rankSTART_END."
    )


def _l2_np(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)


def build_cosine_negative_pools(
    q_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_indices: np.ndarray,
    gold_item_indices_by_question: dict[int, set[int]],
    *,
    start_rank: int,
    end_rank: int,
) -> dict[int, np.ndarray]:
    if start_rank < 1 or end_rank < start_rank:
        raise ValueError(
            f"Invalid cosine rank window: start_rank={start_rank}, end_rank={end_rank}"
        )
    qn = _l2_np(q_embeddings.astype(np.float32, copy=False))
    itemn = _l2_np(item_embeddings.astype(np.float32, copy=False))
    k = min(int(end_rank), len(item_indices))
    pools: dict[int, np.ndarray] = {}
    if k <= 0:
        return {qi: np.array([], dtype=np.int64) for qi in range(len(q_embeddings))}
    scores = qn @ itemn.T
    for qi, row in enumerate(scores):
        if len(row) <= k:
            order = np.argsort(-row)
        else:
            cand = np.argpartition(-row, k - 1)[:k]
            order = cand[np.argsort(-row[cand])]
        gold = gold_item_indices_by_question.get(qi, set())
        window = order[start_rank - 1:k]
        negatives = [
            int(item_indices[idx])
            for idx in window
            if int(item_indices[idx]) not in gold
        ]
        pools[qi] = np.asarray(negatives, dtype=np.int64)
    return pools


def build_hard_negative_pools(
    q_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    item_indices: np.ndarray,
    gold_item_indices_by_question: dict[int, set[int]],
    *,
    top_n: int,
) -> dict[int, np.ndarray]:
    return build_cosine_negative_pools(
        q_embeddings,
        item_embeddings,
        item_indices,
        gold_item_indices_by_question,
        start_rank=1,
        end_rank=top_n,
    )


def _sample_random_non_gold_item(
    item_indices: np.ndarray,
    gold: set[int],
    rng: np.random.Generator,
) -> int:
    if len(item_indices) <= len(gold):
        return int(rng.choice(item_indices))
    for _ in range(32):
        candidate = int(rng.choice(item_indices))
        if candidate not in gold:
            return candidate
    candidates = [int(item) for item in item_indices if int(item) not in gold]
    return int(rng.choice(np.asarray(candidates, dtype=np.int64)))


def sample_negative_items(
    pos_q: np.ndarray,
    item_indices: np.ndarray,
    gold_item_indices_by_question: dict[int, set[int]],
    rng: np.random.Generator,
    *,
    hard_negative_pools: dict[int, np.ndarray] | None = None,
) -> np.ndarray:
    negatives = np.empty(len(pos_q), dtype=np.int64)
    for i, qi_raw in enumerate(pos_q):
        qi = int(qi_raw)
        pool = None if hard_negative_pools is None else hard_negative_pools.get(qi)
        if pool is not None and len(pool) > 0:
            negatives[i] = int(rng.choice(pool))
        else:
            negatives[i] = _sample_random_non_gold_item(
                item_indices,
                gold_item_indices_by_question.get(qi, set()),
                rng,
            )
    return negatives


def evaluate_training_epoch(
    checkpoint_selection: str,
    *,
    item_final,
    variant: str,
    top_k_out: int,
    evaluate_fn,
):
    """Evaluate an epoch only when validation is authorized for checkpoint selection."""
    if checkpoint_selection == "validation_best":
        rows, _ = evaluate_fn(item_final, variant, top_k_out=top_k_out)
        return rows
    if checkpoint_selection == "fixed_final_epoch":
        return None
    raise ValueError(f"Unsupported checkpoint_selection={checkpoint_selection!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-bench-dir", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--eval-bench-dir", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--limit-train", type=int)
    parser.add_argument("--limit-eval", type=int)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation_best", "fixed_final_epoch"),
        default="validation_best",
        help=(
            "validation_best évalue chaque epoch sur la validation; "
            "fixed_final_epoch interdit toute évaluation intermédiaire."
        ),
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Nombre de questions scorees ensemble pendant l'evaluation (defaut: 32).",
    )
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
    parser.add_argument(
        "--adj-normalization",
        choices=["sym", "row", "none"],
        default="sym",
        help="Normalisation de l'adjacence LightGCN : sym=D^-1/2 A D^-1/2, row=D^-1 A.",
    )
    parser.add_argument("--dump-rankings", action="store_true")
    parser.add_argument("--top-k-out", type=int, default=K_OUT)
    parser.add_argument(
        "--history-top-k-out",
        type=int,
        default=K_OUT,
        help="K utilisé pour les métriques de validation pendant l'entraînement. Garder 10 pour éviter de ralentir les exports top-100.",
    )
    parser.add_argument(
        "--selection-metric",
        default="val_hit",
        choices=[
            "val_hit",
            "val_ndcg",
            "val_mrr",
            "val_recall",
            "val_norm_rank",
            "val_hit_jp",
            "val_ndcg_jp",
        ],
        help="Métrique de validation utilisée pour retenir le meilleur epoch entraîné.",
    )
    parser.add_argument(
        "--negative-sampling-strategy",
        default=NEGATIVE_RANDOM,
        help=(
            "random, hard_negative_cosine_topN, ou "
            "semi_hard_cosine_rankSTART_END, ex: semi_hard_cosine_rank21_50."
        ),
    )
    args = parser.parse_args(argv)
    negative_strategy_kind, negative_strategy_param = parse_negative_sampling_strategy(
        args.negative_sampling_strategy
    )

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("══ Chargement graphe + embeddings ─────────────────────────")
    view = graph_versions.load_retrieval_view(args.graph_version)
    art_emb = view.art_emb.astype(np.float32)
    art_order = view.art_order
    jp_emb = view.jp_emb.astype(np.float32)
    jp_order = view.jp_order
    D = art_emb.shape[1]

    G = view.graph
    jp_ids_graph = view.jp_ids_graph
    article_ids_graph = view.article_ids_graph
    n_jp = len(jp_ids_graph)
    n_art = len(article_ids_graph)
    n_total = n_jp + n_art
    print(f"  G {G.shape}  nnz {G.nnz:,}  N={n_total:,}")

    if G.shape == (n_jp, n_art):
        G_full = sp.bmat([[None, G], [G.T, None]], format="csr")
    elif G.shape == (n_total, n_total):
        G_full = G.tocsr()
    else:
        raise ValueError(
            f"Unsupported graph shape for LightGCN: graph={G.shape}, "
            f"n_jp={n_jp}, n_art={n_art}"
        )
    adj = sparse_scipy_to_torch(normalize_adjacency(G_full, args.adj_normalization)).to(DEVICE)
    print(f"  adj {args.adj_normalization} ready  (t={time.time()-t0:.1f}s)")

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
    benchmark_labels.require_strict_candidate_coverage(
        eval_q,
        article_candidate_ids=art_pool_pks,
        jp_candidate_ids=jp_pool_ids,
        context=f"LightGCN evaluation input {args.eval_bench_dir}",
    )
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

    train_positives_by_qid, train_projection, train_projection_sha256 = (
        prepare_train_lightgcn_positives(
            args.train_bench_dir,
            train_q,
            article_candidate_ids=art_pool_pks,
            jp_candidate_ids=jp_pool_ids,
        )
    )

    train_q_rows = []
    train_pos = []
    train_gold_items: dict[int, set[int]] = {}
    for qi, q in enumerate(train_q):
        gts = train_positives_by_qid[q["id"]]
        if not gts <= pool_art_set:
            raise AssertionError(f"LightGCN projection escaped graph candidate space: {q['id']}")
        local_qi = len(train_q_rows)
        train_q_rows.append(q_train_np[qi])
        train_gold_items[local_qi] = {art_pk_to_itemidx[pk] for pk in gts}
        for pk in gts:
            train_pos.append((local_qi, art_pk_to_itemidx[pk]))
    if not train_pos:
        raise RuntimeError("No train positives after pool filtering")
    print(f"  train positives {len(train_pos):,} over {len(train_q_rows):,} questions")

    pos_q_np = np.asarray([p[0] for p in train_pos], dtype=np.int64)
    q_train_t = torch.tensor(np.asarray(train_q_rows), dtype=torch.float32, device=DEVICE)
    pos_q = torch.tensor(pos_q_np, dtype=torch.long, device=DEVICE)
    pos_item = torch.tensor([p[1] for p in train_pos], dtype=torch.long, device=DEVICE)
    q_eval_t = torch.tensor(q_eval_np, dtype=torch.float32, device=DEVICE)
    e0_t = torch.tensor(e0, dtype=torch.float32, device=DEVICE)
    e0_zero_t = torch.tensor(e0_zero, dtype=torch.float32, device=DEVICE)
    art_pool_idx_t = torch.tensor(art_pool_graph_idx, dtype=torch.long, device=DEVICE)
    jp_pool_idx_t = torch.tensor(jp_pool_graph_idx, dtype=torch.long, device=DEVICE)
    anchor_idx = torch.tensor(np.where(has_init)[0], dtype=torch.long, device=DEVICE)
    hard_negative_pools: dict[int, np.ndarray] | None = None
    hard_pool_stats = {
        "hard_negative_pool_mean": float("nan"),
        "hard_negative_pool_empty_pct": float("nan"),
    }
    if negative_strategy_kind == HARD_NEGATIVE_PREFIX:
        hard_negative_pools = build_hard_negative_pools(
            np.asarray(train_q_rows, dtype=np.float32),
            e0[art_pool_graph_idx],
            art_pool_graph_idx,
            train_gold_items,
            top_n=int(negative_strategy_param or 0),
        )
    elif negative_strategy_kind == SEMI_HARD_NEGATIVE_PREFIX:
        start_rank, end_rank = negative_strategy_param  # type: ignore[misc]
        hard_negative_pools = build_cosine_negative_pools(
            np.asarray(train_q_rows, dtype=np.float32),
            e0[art_pool_graph_idx],
            art_pool_graph_idx,
            train_gold_items,
            start_rank=int(start_rank),
            end_rank=int(end_rank),
        )

    if hard_negative_pools is not None:
        pool_sizes = np.asarray([len(v) for v in hard_negative_pools.values()], dtype=np.float32)
        hard_pool_stats = {
            "hard_negative_pool_mean": float(pool_sizes.mean()) if len(pool_sizes) else float("nan"),
            "hard_negative_pool_empty_pct": float((pool_sizes == 0).mean()) if len(pool_sizes) else float("nan"),
        }
        print(
            "  cosine negative pool "
            f"{args.negative_sampling_strategy}: mean_pool={hard_pool_stats['hard_negative_pool_mean']:.2f} "
            f"empty={hard_pool_stats['hard_negative_pool_empty_pct']:.1%}"
        )

    def evaluate(
        item_final: torch.Tensor,
        variant: str,
        *,
        top_k_out: int | None = None,
    ) -> tuple[list[dict], list[dict]]:
        k_out = int(top_k_out or args.top_k_out)
        rows = []
        rankings = []
        with torch.no_grad():
            qn = l2(q_eval_t)
            art_final = l2(item_final[art_pool_idx_t])
            jp_final = l2(item_final[jp_pool_idx_t])
        method = f"LightGCN-{variant}"
        kin = 0 if variant == "cosine_raw" else int(variant.rsplit("K", 1)[-1])
        for start, sc_art in iter_similarity_batches(
            qn, art_final, batch_size=args.eval_batch_size
        ):
            stop = start + len(sc_art)
            with torch.no_grad():
                sc_jp = (qn[start:stop] @ jp_final.T).cpu().numpy()
            for offset, q in enumerate(eval_q[start:stop]):
                gt_s = q["gt_strict"]
                gt_e = q["gt_ext"]
                gold_jp = q["gold_jp_ids"]
                ranked_art = rowcol_top_k(sc_art[offset], art_pool_pks, k_out)
                ranked_jp = rowcol_top_k(sc_jp[offset], jp_pool_ids, k_out)
                benchmark_labels.require_ranked_ids_within_candidate_universe(
                    ranked_art,
                    candidate_ids=art_order,
                    context=f"LightGCN ranking method={method} modality=art qid={q['id']}",
                )
                benchmark_labels.require_ranked_ids_within_candidate_universe(
                    ranked_jp,
                    candidate_ids=jp_order,
                    context=f"LightGCN ranking method={method} modality=jp qid={q['id']}",
                )
                rankings.extend(
                    ranking_rows(
                        q["id"],
                        method,
                        kin,
                        "art",
                        ranked_art,
                        k_out,
                        args.negative_sampling_strategy,
                    )
                )
                rankings.extend(
                    ranking_rows(
                        q["id"],
                        method,
                        kin,
                        "jp",
                        ranked_jp,
                        k_out,
                        args.negative_sampling_strategy,
                    )
                )
                am = {
                    f"{k}_art": v
                    for k, v in M.panel_strict_ext(ranked_art, gt_s, gt_e, k_out).items()
                }
                jm = {
                    f"{k}_jp": v
                    for k, v in M.all_metrics(ranked_jp, gold_jp, k_out).items()
                }
                rows.append({
                    "qid": q["id"],
                    "variant": variant,
                    "adj_normalization": args.adj_normalization,
                    "negative_sampling_strategy": args.negative_sampling_strategy,
                    **am,
                    **jm,
                })
        return rows, rankings

    history_rows: list[dict] = []

    def train_model(n_layers: int) -> torch.Tensor:
        model = LightGCN(e0_t, n_layers).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
        n_pos = pos_q.shape[0]
        e0_ref = e0_t[anchor_idx].detach().clone()
        best_metric = -float("inf")
        best_epoch = -1
        best_final: torch.Tensor | None = None
        for ep in range(args.epochs):
            item_final = model.propagate(adj)
            neg_item_np = sample_negative_items(
                pos_q_np,
                art_pool_graph_idx,
                train_gold_items,
                rng,
                hard_negative_pools=hard_negative_pools,
            )
            neg_item = torch.tensor(neg_item_np, dtype=torch.long, device=DEVICE)
            bpr = bpr_loss(q_train_t[pos_q], item_final, pos_item, neg_item)
            anchor = ((model.emb[anchor_idx] - e0_ref) ** 2).sum(1).mean()
            loss = bpr + args.lambda_anchor * anchor
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                current_final = model.propagate(adj)
            val_rows = evaluate_training_epoch(
                args.checkpoint_selection,
                item_final=current_final,
                variant=f"trained_K{n_layers}",
                top_k_out=args.history_top_k_out,
                evaluate_fn=evaluate,
            )
            val_summary = summarize_eval_rows(val_rows) if val_rows is not None else {}
            selection_metric_values = {
                "val_hit": float(val_summary.get("hit_strict_art", np.nan)),
                "val_ndcg": float(val_summary.get("ndcg_strict_art", np.nan)),
                "val_mrr": float(val_summary.get("mrr_strict_art", np.nan)),
                "val_recall": float(val_summary.get("m1_strict_art", np.nan)),
                "val_norm_rank": float(val_summary.get("m2_strict_art", np.nan)),
                "val_hit_jp": float(val_summary.get("hit_jp", np.nan)),
                "val_ndcg_jp": float(val_summary.get("ndcg_jp", np.nan)),
            }
            selection_value = selection_metric_values[args.selection_metric]
            is_best = bool(
                args.checkpoint_selection == "validation_best"
                and np.isfinite(selection_value)
                and selection_value > best_metric
            )
            if is_best:
                best_metric = selection_value
                best_epoch = ep
                best_final = current_final.detach().clone()
            history_rows.append(
                {
                    "epoch": ep,
                    "graph_version": args.graph_version,
                    "variant": f"trained_K{n_layers}",
                    "adj_normalization": args.adj_normalization,
                    "negative_sampling_strategy": args.negative_sampling_strategy,
                    "train_loss": float(loss.item()),
                    "bpr_loss": float(bpr.item()),
                    "anchor_loss": float(anchor.item()),
                    **hard_pool_stats,
                    "selection_metric": args.selection_metric,
                    "selection_metric_value": selection_value,
                    "is_new_best_epoch": is_best,
                    "is_best_epoch": is_best,
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
                    f"{args.selection_metric} {selection_value:.4f} "
                    f"(t={time.time()-t0:.1f}s)"
                )
        if args.checkpoint_selection == "fixed_final_epoch":
            best_final = current_final.detach().clone()
            best_epoch = args.epochs - 1
            best_metric = float("nan")
        elif best_final is None:
            best_final = current_final.detach().clone()
            best_epoch = args.epochs - 1
            best_metric = float("nan")
        print(
            f"  selected trained_K{n_layers}: epoch {best_epoch} "
            f"{args.selection_metric}={best_metric:.4f}"
        )
        return best_final

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

    out_inputs = args.eval_bench_dir / f"lightgcn_inputs{suffix}.json"
    out_inputs.write_text(
        json.dumps(
            {
                "train_bench_sha256": benchmark_labels.sha256_file(args.train_bench_dir / "bench_global.json"),
                "eval_bench_sha256": benchmark_labels.sha256_file(args.eval_bench_dir / "bench_global.json"),
                "lightgcn_article_positive_projection_sha256": train_projection_sha256,
                "lightgcn_article_positive_projection_counts": train_projection["counts"],
                "article_candidate_sequence_sha256": train_projection["article_candidate_sequence_sha256"],
                "strict_candidate_coverage_verified": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"✓ {out_inputs}")

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
