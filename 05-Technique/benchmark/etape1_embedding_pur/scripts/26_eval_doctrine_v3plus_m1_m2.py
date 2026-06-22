"""Build + evaluate Doctrine QGen v3+ strict splits with M1/M2/Hit/MRR/NDCG.

This script intentionally writes to data/doctrine_v3plus_bench/<split>/ so the
historical data/global_bench/977 artifacts remain untouched.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))
sys.path.insert(0, str(Path(__file__).parent))

from etape1 import config  # noqa: E402
import metrics as M  # noqa: E402

DATASET_DIR = (
    REPO
    / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/"
    / "dataset_v2_augmented_train_v3plus/retrievable_strict"
)
SPLIT_FILES = {
    "train_augmented_retrievable_strict": (
        DATASET_DIR / "dataset_penal_v2_train_augmented_retrievable_strict.json"
    ),
    "eval_rich_retrievable_strict": (
        DATASET_DIR / "dataset_penal_v2_eval_rich_retrievable_strict.json"
    ),
}
OUT_ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"

K_ART = 10
K_JP = 10
KS_IN = [10, 20, 50]
K_RRF = 60


def load_graph():
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    graph = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return graph, z["jp_ids"], z["article_ids"]


def build_bench(split: str, out_dir: Path) -> list[dict]:
    dataset_path = SPLIT_FILES[split]
    raw = json.loads(dataset_path.read_text())
    graph, jp_ids_graph, article_ids_graph = load_graph()

    questions = []
    for q in raw["questions"]:
        strict = sorted(set(q.get("article_keys") or q.get("central_article_keys") or []))
        jp_ids = sorted(set(q.get("gold_jp_ids") or []))
        if not strict or not jp_ids:
            raise ValueError(f"{q.get('qid')} is not retrievable-strict: empty gold")
        ext = set(strict)
        for row in q.get("gold_jp_graph_rows") or []:
            if 0 <= int(row) < graph.shape[0]:
                ext.update(article_ids_graph[graph[int(row)].indices].tolist())

        questions.append(
            {
                "qid": q["qid"],
                "source": "doctrine_qgen_v3plus",
                "split": split,
                "branche": q.get("branche", "penal"),
                "doc_id": q.get("doc_id"),
                "section_id": q.get("section_id"),
                "theme": q.get("theme"),
                "question_type": q.get("question_type"),
                "granularity": q.get("granularity"),
                "enonce": q["enonce"],
                "articles_attendus": strict,
                "articles_attendus_etendu": sorted(ext),
                "gold_jp_ids": jp_ids,
                "n_articles_strict": len(strict),
                "n_articles_etendu": len(ext),
                "n_jp_resolues": len(jp_ids),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    qids = [q["qid"] for q in questions]
    if len(qids) != len(set(qids)):
        duplicates = [qid for qid, n in Counter(qids).items() if n > 1]
        raise ValueError(f"Duplicate qid in {split}: {duplicates[:5]}")
    bench = {
        "schema_version": "doctrine_v3plus_bench.v1",
        "source_dataset": str(dataset_path),
        "split": split,
        "policy": "retrievable_strict_only",
        "k": 10,
        "questions": questions,
    }
    (out_dir / "bench_global.json").write_text(json.dumps(bench, ensure_ascii=False, indent=2))
    (out_dir / "stats.json").write_text(json.dumps(compute_stats(questions), ensure_ascii=False, indent=2))
    return questions


def compute_stats(questions: list[dict]) -> dict:
    def describe(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()) if len(arr) else 0.0,
            "median": float(np.median(arr)) if len(arr) else 0.0,
            "min": int(arr.min()) if len(arr) else 0,
            "max": int(arr.max()) if len(arr) else 0,
            "ge_2": int((arr >= 2).sum()) if len(arr) else 0,
            "histo": dict(sorted(Counter(int(v) for v in vals).items())),
        }

    return {
        "n_questions": len(questions),
        "n_docs": len({q["doc_id"] for q in questions}),
        "by_question_type": dict(Counter(q.get("question_type") for q in questions)),
        "by_granularity": dict(Counter(q.get("granularity") for q in questions)),
        "articles_strict": describe([q["n_articles_strict"] for q in questions]),
        "articles_etendu": describe([q["n_articles_etendu"] for q in questions]),
        "jp_resolues": describe([q["n_jp_resolues"] for q in questions]),
    }


def encode_questions(questions: list[dict], out_dir: Path) -> np.ndarray:
    qids = [q["qid"] for q in questions]
    emb_cache = out_dir / "questions_emb.npy"
    ids_cache = out_dir / "questions_ids.npy"
    if emb_cache.exists() and ids_cache.exists():
        cached_ids = np.load(ids_cache, allow_pickle=True).tolist()
        if cached_ids == qids:
            print(f"  cache questions HIT : {emb_cache}")
            return np.load(emb_cache)

    from sentence_transformers import SentenceTransformer
    import torch

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"  encodage questions avec {config.MODEL_ID} sur {device}")
    model = SentenceTransformer(config.MODEL_ID, device=device)
    model.max_seq_length = config.BATCH_MAX_LEN
    emb = model.encode(
        [q["enonce"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)
    np.save(emb_cache, emb)
    np.save(ids_cache, np.asarray(qids, dtype=object))
    return emb


def top_sorted(sim_row: np.ndarray, k: int) -> np.ndarray:
    if k >= len(sim_row):
        idx = np.argsort(-sim_row)
    else:
        idx = np.argpartition(-sim_row, k)[:k]
        idx = idx[np.argsort(-sim_row[idx])]
    return idx


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


def eval_m1_m2(
    questions: list[dict],
    out_dir: Path,
    limit: int | None = None,
    qid_filter: set[str] | None = None,
    ks_in: list[int] | None = None,
) -> None:
    t0 = time.time()
    if qid_filter is not None:
        questions = [q for q in questions if q["qid"] in qid_filter]
    if limit is not None:
        questions = questions[:limit]
        print(f"  mode limit : {len(questions)} questions")
    k_ins = list(ks_in or KS_IN)
    if not k_ins:
        raise ValueError("ks_in must contain at least one value")

    print("══ Chargement embeddings + graphe")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    jp_emb = np.load(config.EMB_JP_SYNTHESE)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    jp_to_row = np.load(config.JP_SUMMARY_TO_GRAPHROW)
    graph, jp_ids_graph, article_ids_graph = load_graph()

    pk_to_emb_idx = {pk: i for i, pk in enumerate(art_order)}
    jpid_to_emb_idx = {jid: i for i, jid in enumerate(jp_order)}
    pool_articles = set(art_order.tolist())
    pool_jp = set(jp_order.tolist())
    print(f"  art_emb={art_emb.shape} jp_emb={jp_emb.shape} graph={graph.shape}")

    print("══ Encodage / cache questions")
    q_emb = encode_questions(questions, out_dir)
    print(f"  Q={q_emb.shape}")

    print("══ Similarités cosine")
    sim_art = q_emb @ art_emb.T
    sim_jp = q_emb @ jp_emb.T
    print(f"  sim_art={sim_art.shape} sim_jp={sim_jp.shape} t={time.time()-t0:.1f}s")

    rows = []
    rankings = []
    max_k_in = max(k_ins)
    print("══ Évaluation M1/M2/Hit/MRR/NDCG")
    for qi, q in enumerate(questions):
        if qi % 100 == 0:
            print(f"  q {qi}/{len(questions)} rows={len(rows)} t={time.time()-t0:.1f}s")

        gt_s = set(q["articles_attendus"]) & pool_articles
        gt_e = set(q["articles_attendus_etendu"]) & pool_articles
        gold_jp = set(q["gold_jp_ids"]) & pool_jp
        if not gt_s and not gold_jp:
            continue

        top_art_max = top_sorted(sim_art[qi], max_k_in)
        top_jp_max = top_sorted(sim_jp[qi], max_k_in)

        ranked_art = list(art_order[top_art_max[:K_ART]])
        rows.append(
            {
                "qid": q["qid"],
                "method": "B2-a",
                "k_in": None,
                "k": K_ART,
                "modality": "art",
                **M.panel_strict_ext(ranked_art, gt_s, gt_e, K_ART),
            }
        )
        rankings.extend(ranking_rows(q["qid"], "B2-a", None, "art", ranked_art, K_ART))

        ranked_jp = list(jp_order[top_jp_max[:K_JP]])
        rows.append(
            {
                "qid": q["qid"],
                "method": "B3-a",
                "k_in": None,
                "k": K_JP,
                "modality": "jp",
                **M.panel_strict_ext(ranked_jp, gold_jp, gold_jp, K_JP),
            }
        )
        rankings.extend(ranking_rows(q["qid"], "B3-a", None, "jp", ranked_jp, K_JP))

        for k_in in k_ins:
            top_art_emb_idx = top_art_max[:k_in]
            top_art_cols = p2col[top_art_emb_idx]
            top_art_pks = set(art_order[top_art_emb_idx].tolist())
            top_jp_emb_idx = top_jp_max[:k_in]
            top_jp_ids = set(jp_order[top_jp_emb_idx].tolist())
            top_jp_rows = jp_to_row[top_jp_emb_idx]

            jp_count_arr = np.asarray((graph[:, top_art_cols] != 0).sum(axis=1)).ravel()
            a_jp_ids = set(jp_ids_graph[jp_count_arr >= 1].tolist())
            jp_citation_count = {
                jid: int(jp_count_arr[i])
                for i, jid in enumerate(jp_ids_graph)
                if jp_count_arr[i] >= 1
            }

            art_count = np.asarray((graph[top_jp_rows, :] != 0).sum(axis=0)).ravel()
            b_art_cols = np.where(art_count >= 1)[0]
            b_art_pks = set(article_ids_graph[b_art_cols].tolist())

            for method, candidates in [
                ("B3-e", b_art_pks),
                ("B4-a", top_art_pks | b_art_pks),
            ]:
                emb_idx = [pk_to_emb_idx[pk] for pk in candidates if pk in pk_to_emb_idx]
                if emb_idx:
                    arr = np.asarray(emb_idx, dtype=np.int64)
                    order = np.argsort(-sim_art[qi, arr])
                    ranked = list(art_order[arr[order]])
                else:
                    ranked = []
                rows.append(
                    {
                        "qid": q["qid"],
                        "method": method,
                        "k_in": k_in,
                        "k": K_ART,
                        "modality": "art",
                        **M.panel_strict_ext(ranked, gt_s, gt_e, K_ART),
                    }
                )
                rankings.extend(ranking_rows(q["qid"], method, k_in, "art", ranked, K_ART))

            jp_methods = {
                "B4-c": a_jp_ids | top_jp_ids,
                "B4-d": a_jp_ids & top_jp_ids,
            }
            for method, candidates in jp_methods.items():
                emb_idx = [jpid_to_emb_idx[j] for j in candidates if j in jpid_to_emb_idx]
                if emb_idx:
                    arr = np.asarray(emb_idx, dtype=np.int64)
                    order = np.argsort(-sim_jp[qi, arr])
                    ranked = list(jp_order[arr[order]])
                else:
                    ranked = []
                rows.append(
                    {
                        "qid": q["qid"],
                        "method": method,
                        "k_in": k_in,
                        "k": K_JP,
                        "modality": "jp",
                        **M.panel_strict_ext(ranked, gold_jp, gold_jp, K_JP),
                    }
                )
                rankings.extend(ranking_rows(q["qid"], method, k_in, "jp", ranked, K_JP))

            rank_cos = {jp_order[top_jp_emb_idx[r]]: r + 1 for r in range(k_in)}
            a_sorted = sorted(a_jp_ids, key=lambda j: (-jp_citation_count.get(j, 0), j))
            rank_graph = {j: r + 1 for r, j in enumerate(a_sorted)}
            candidates = [j for j in (a_jp_ids | top_jp_ids) if j in jpid_to_emb_idx]

            def rrf_score(j):
                score = 0.0
                if j in rank_cos:
                    score += 1.0 / (K_RRF + rank_cos[j])
                if j in rank_graph:
                    score += 1.0 / (K_RRF + rank_graph[j])
                return score

            ranked = sorted(
                candidates,
                key=lambda j: (-rrf_score(j), -float(sim_jp[qi, jpid_to_emb_idx[j]])),
            )
            rows.append(
                {
                    "qid": q["qid"],
                    "method": "B4-e",
                    "k_in": k_in,
                    "k": K_JP,
                    "modality": "jp",
                    **M.panel_strict_ext(ranked, gold_jp, gold_jp, K_JP),
                }
            )
            rankings.extend(ranking_rows(q["qid"], "B4-e", k_in, "jp", ranked, K_JP))

            ranked = sorted(
                candidates,
                key=lambda j: (
                    -jp_citation_count.get(j, 0),
                    -float(sim_jp[qi, jpid_to_emb_idx[j]]),
                ),
            )
            rows.append(
                {
                    "qid": q["qid"],
                    "method": "B4-f",
                    "k_in": k_in,
                    "k": K_JP,
                    "modality": "jp",
                    **M.panel_strict_ext(ranked, gold_jp, gold_jp, K_JP),
                }
            )
            rankings.extend(ranking_rows(q["qid"], "B4-f", k_in, "jp", ranked, K_JP))

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "eval_m1_m2.csv", index=False)
    pd.DataFrame(rankings).to_parquet(out_dir / "rankings.parquet", index=False)

    cols_metrics = [f"{m}_{r}" for m in M.METRIC_NAMES for r in ("strict", "ext")]
    summary = {}
    for (method, modality, k_in), sub in df.groupby(["method", "modality", "k_in"], dropna=False):
        kin_disp = str(int(k_in)) if pd.notna(k_in) else "-"
        summary[f"{method}|{modality}|kin={kin_disp}"] = {
            "n_q": int(len(sub)),
            **{c: float(sub[c].mean()) for c in cols_metrics},
        }
    (out_dir / "eval_m1_m2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"✓ {out_dir / 'eval_m1_m2.csv'}")
    print(f"✓ {out_dir / 'eval_m1_m2_summary.json'}")
    print(f"✓ {out_dir / 'rankings.parquet'}")
    print(f"  t total : {time.time()-t0:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        choices=sorted(SPLIT_FILES),
        default="eval_rich_retrievable_strict",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    out_dir = OUT_ROOT / args.split
    print(f"══ Build bench v3+ : {args.split}")
    questions = build_bench(args.split, out_dir)
    print(f"  questions={len(questions)} out={out_dir}")
    if not args.build_only:
        eval_m1_m2(questions, out_dir, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
