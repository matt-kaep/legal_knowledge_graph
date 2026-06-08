"""Variantes 1-hop + cross-modales sur le graphe biparti articles ↔ JP.

Étend 10_eval_doctrine_qgen.py avec :

(1) Mêmes-type 1-hop (point d'entrée → voisins du même type indirectement)
  Articles → JP (point d'entrée articles, gold = JP)
    UNION         : JP citant ≥ 1 article du top-K articles (= jp_via_graph baseline)
    INTERSECTION  : JP citant TOUS les articles du top-K
    MAJORITY      : JP citant ≥ ⌈K/2⌉ articles du top-K

  JP → Articles (point d'entrée JP, gold = articles)
    UNION         : articles cités par ≥ 1 JP du top-K JP
    INTERSECTION  : articles cités par TOUTES les JP du top-K
    MAJORITY      : articles cités par ≥ ⌈K/2⌉ JP du top-K

(2) Cross-modales (combinent les deux espaces d'embedding indépendants)

  Côté articles (gold = articles oblig) :
    A_art = top-K_a articles par cosine question × emb_articles    (signal direct)
    B_art = articles cités par top-K_j JP du cosine direct          (signal indirect)
    cross_art_union        = A_art ∪ B_art   (max rappel)
    cross_art_intersection = A_art ∩ B_art   (max précision)

  Côté JP (gold = JP pourvois) :
    A_jp = JP voisines via graphe des top-K_a articles              (jp_via_graph)
    B_jp = top-K_j JP par cosine direct sur synthèses                (jp_direct)
    cross_jp_union         = A_jp ∪ B_jp
    cross_jp_intersection  = A_jp ∩ B_jp

Pour les variantes set (sans ordre), recall = |gold ∩ set| / |gold|.
On rapporte aussi mean_returned (taille du set) pour juger précision implicite.
"""
from __future__ import annotations
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))

from etape1 import config  # noqa: E402
from etape1.eval_recall import recall_at_k  # noqa: E402

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

CORPUS_PATH = (
    REPO
    / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_strict_gemma4-26B-A4B.json"
)
OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen"

# K d'entrée à tester (top-K articles ou top-K JP utilisés comme « point d'entrée »)
KS_IN = [5, 10, 20, 50]


def load_doctrine_qgen() -> list[dict]:
    d = json.loads(CORPUS_PATH.read_text())
    out = []
    for q in d["questions"]:
        oblig = {
            f'{a["code_slug"]}:{a["article_num"]}'
            for a in q.get("articles_attendus", [])
        }
        pourvois = {
            p
            for j in q.get("jp_attendues", [])
            if (p := (j.get("pourvoi") or "").strip()) and _POURVOI_RE.fullmatch(p)
        }
        out.append(
            {
                "id": q["qid"],
                "doc_id": q.get("doc_id"),
                "enonce": q["enonce"],
                "oblig": oblig,
                "pourvois": pourvois,
            }
        )
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


def encode(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    dev = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    m = SentenceTransformer(config.MODEL_ID, device=dev)
    m.max_seq_length = config.BATCH_MAX_LEN
    return m.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=32,
    ).astype(np.float32)


def main() -> int:
    t0 = time.time()
    print("══ Chargement ─────────────────────────────────────────────")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    jp_emb = np.load(config.EMB_JP_SYNTHESE)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    jp_to_row = np.load(config.JP_SUMMARY_TO_GRAPHROW)

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    )
    G_csc = G.tocsc()  # pour slicing par colonne efficace
    jp_ids_graph = z["jp_ids"]
    article_ids_graph = z["article_ids"]  # ordre dans le graphe

    # Mappings inverses
    pk_to_col = {pk: i for i, pk in enumerate(article_ids_graph)}
    jpid_to_row = {jid: i for i, jid in enumerate(jp_ids_graph)}

    pourvoi_map = build_pourvoi_map()
    print(f"  art_emb {art_emb.shape}  jp_emb {jp_emb.shape}  G {G.shape}")

    questions = load_doctrine_qgen()
    print(f"  questions : {len(questions)}")

    # ── Encoding
    print("\n══ Encoding ───────────────────────────────────────────────")
    Q = encode([q["enonce"] for q in questions])
    print(f"  Q : {Q.shape}  (t={time.time()-t0:.1f}s)")

    # ── Cosine
    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T  # (n_q, 31357)
    sim_jp = Q @ jp_emb.T  # (n_q, 116755)
    print(f"  sim_art {sim_art.shape}  sim_jp {sim_jp.shape}")

    # Pré-trier
    print("  pré-tri argsort articles + JP …")
    art_rank = np.argsort(-sim_art, axis=1)  # (n_q, n_art)
    jp_rank = np.argsort(-sim_jp, axis=1)  # (n_q, n_jp)
    print(f"  t={time.time()-t0:.1f}s")

    rows = []
    print("\n══ Boucle d'éval ──────────────────────────────────────────")
    for qi, q in enumerate(questions):
        if qi % 200 == 0:
            print(f"  q {qi}/{len(questions)} (t={time.time()-t0:.1f}s)")
        oblig = q["oblig"]
        gold_jp = {jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}

        for k_in in KS_IN:
            # ────────────── ARTICLES → JP ──────────────
            top_art_local = art_rank[qi, :k_in]
            top_art_cols = p2col[top_art_local]  # indices colonne dans G
            # Sous-matrice G[:, top_art_cols] -> (n_jp_in_G, k_in) sparse
            sub = G[:, top_art_cols]
            # Compteur d'articles cités par chaque JP parmi le top-K_in
            counts_per_jp = np.asarray((sub != 0).sum(axis=1)).ravel()
            # UNION : counts >= 1
            jp_union = jp_ids_graph[counts_per_jp >= 1].tolist()
            # INTERSECTION : counts == k_in
            jp_inter = jp_ids_graph[counts_per_jp >= k_in].tolist()
            # MAJORITY : counts >= ceil(k_in/2)
            maj_th = math.ceil(k_in / 2)
            jp_maj = jp_ids_graph[counts_per_jp >= maj_th].tolist()

            for variant, jp_set in (
                ("art_to_jp_union", jp_union),
                ("art_to_jp_intersection", jp_inter),
                ("art_to_jp_majority", jp_maj),
            ):
                if gold_jp:
                    r = len(set(jp_set) & gold_jp) / len(gold_jp)
                else:
                    r = None
                rows.append(
                    {
                        "question_id": q["id"],
                        "doc_id": q["doc_id"],
                        "k_in": k_in,
                        "direction": "art_to_jp",
                        "variant": variant.split("_")[-1],
                        "side": "jp",
                        "n_returned": len(jp_set),
                        "n_gold": len(gold_jp),
                        "recall": r,
                    }
                )

            # ────────────── JP → ARTICLES (symétrique) ──────────────
            # top-K JP par cosine sur synthèses
            top_jp_local = jp_rank[qi, :k_in]
            # Mapper vers les lignes du graphe via jp_to_row
            # jp_to_row[i] donne la row dans G pour la i-ème JP dans jp_emb/jp_order
            top_jp_rows = jp_to_row[top_jp_local]
            # Sous-matrice G[top_jp_rows, :] → (k_in, n_art_in_G)
            sub2 = G[top_jp_rows, :]
            # Compteur de JP du top-K qui citent chaque article
            counts_per_art = np.asarray((sub2 != 0).sum(axis=0)).ravel()
            # UNION : articles cités par ≥ 1 JP
            art_cols_union = np.where(counts_per_art >= 1)[0]
            # INTER : par toutes les JP du top-K
            art_cols_inter = np.where(counts_per_art >= k_in)[0]
            # MAJORITY
            art_cols_maj = np.where(counts_per_art >= maj_th)[0]

            for variant, art_cols in (
                ("jp_to_art_union", art_cols_union),
                ("jp_to_art_intersection", art_cols_inter),
                ("jp_to_art_majority", art_cols_maj),
            ):
                art_pks = set(article_ids_graph[art_cols].tolist())
                if oblig:
                    r = len(art_pks & oblig) / len(oblig)
                else:
                    r = None
                rows.append(
                    {
                        "question_id": q["id"],
                        "doc_id": q["doc_id"],
                        "k_in": k_in,
                        "direction": "jp_to_art",
                        "variant": variant.split("_")[-1],
                        "side": "art",
                        "n_returned": len(art_pks),
                        "n_gold": len(oblig),
                        "recall": r,
                    }
                )

            # ────────────── CROSS-MODALES ──────────────
            # Côté ARTICLES : A_art (cosine direct) ∪/∩ B_art (cités par top-K JP)
            A_art_set = set(art_order[top_art_local].tolist())  # top-K_in articles cosine direct
            B_art_set = set(article_ids_graph[art_cols_union].tolist())  # articles cités par top-K JP (union)

            for variant, art_set in (
                ("cross_art_union", A_art_set | B_art_set),
                ("cross_art_intersection", A_art_set & B_art_set),
            ):
                if oblig:
                    r = len(art_set & oblig) / len(oblig)
                else:
                    r = None
                rows.append(
                    {
                        "question_id": q["id"],
                        "doc_id": q["doc_id"],
                        "k_in": k_in,
                        "direction": "cross",
                        "variant": variant.replace("cross_art_", "art_"),
                        "side": "art",
                        "n_returned": len(art_set),
                        "n_gold": len(oblig),
                        "recall": r,
                    }
                )

            # Côté JP : A_jp (via graphe depuis top-K articles) ∪/∩ B_jp (top-K JP direct)
            A_jp_set = set(jp_union)  # JP voisines top-K_in articles (union 1-hop)
            B_jp_set = set(jp_order[top_jp_local].tolist())  # top-K_in JP cosine direct

            for variant, jp_set in (
                ("cross_jp_union", A_jp_set | B_jp_set),
                ("cross_jp_intersection", A_jp_set & B_jp_set),
            ):
                if gold_jp:
                    r = len(jp_set & gold_jp) / len(gold_jp)
                else:
                    r = None
                rows.append(
                    {
                        "question_id": q["id"],
                        "doc_id": q["doc_id"],
                        "k_in": k_in,
                        "direction": "cross",
                        "variant": variant.replace("cross_jp_", "jp_"),
                        "side": "jp",
                        "n_returned": len(jp_set),
                        "n_gold": len(gold_jp) if gold_jp else 0,
                        "recall": r,
                    }
                )

    print(f"  fin boucle (t={time.time()-t0:.1f}s)")

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "recall_graph_variants.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n✓ {out_csv}")

    # ── Aggrégats ────────────────────────────────────────
    print("\n══ Aggrégats ──────────────────────────────────────────────")
    valid = df[df["recall"].notna()].copy()

    print(
        "\n  Articles → JP (gold = pourvois CC évaluables, ~971 questions) :"
    )
    print(
        f"  {'k_in':>5s}  {'variant':>15s}  {'n_q':>5s}  {'mean_recall':>11s}  "
        f"{'mean_returned':>13s}  {'pass_r≥0.5':>10s}"
    )
    for k_in in KS_IN:
        for variant in ("union", "intersection", "majority"):
            sub = valid[
                (valid["direction"] == "art_to_jp")
                & (valid["k_in"] == k_in)
                & (valid["variant"] == variant)
            ]
            n = len(sub)
            mr = sub["recall"].mean()
            mret = sub["n_returned"].mean()
            n_pass = int((sub["recall"] >= 0.5).sum())
            print(
                f"  {k_in:>5d}  {variant:>15s}  {n:>5d}  {mr:>11.3f}  "
                f"{mret:>13.0f}  {n_pass:>5d} ({100*n_pass/max(n,1):>3.0f} %)"
            )

    print(
        "\n  JP → Articles (gold = articles oblig, 1707 questions) :"
    )
    print(
        f"  {'k_in':>5s}  {'variant':>15s}  {'n_q':>5s}  {'mean_recall':>11s}  "
        f"{'mean_returned':>13s}  {'pass_r≥0.5':>10s}"
    )
    for k_in in KS_IN:
        for variant in ("union", "intersection", "majority"):
            sub = valid[
                (valid["direction"] == "jp_to_art")
                & (valid["k_in"] == k_in)
                & (valid["variant"] == variant)
            ]
            n = len(sub)
            mr = sub["recall"].mean()
            mret = sub["n_returned"].mean()
            n_pass = int((sub["recall"] >= 0.5).sum())
            print(
                f"  {k_in:>5d}  {variant:>15s}  {n:>5d}  {mr:>11.3f}  "
                f"{mret:>13.0f}  {n_pass:>5d} ({100*n_pass/max(n,1):>3.0f} %)"
            )

    print("\n  CROSS-MODAL — Articles (gold = articles oblig) :")
    print(
        f"  {'k_in':>5s}  {'variant':>20s}  {'n_q':>5s}  {'mean_recall':>11s}  "
        f"{'mean_returned':>13s}  {'pass_r≥0.5':>10s}"
    )
    for k_in in KS_IN:
        for variant in ("art_union", "art_intersection"):
            sub = valid[
                (valid["direction"] == "cross")
                & (valid["k_in"] == k_in)
                & (valid["variant"] == variant)
                & (valid["side"] == "art")
            ]
            n = len(sub)
            mr = sub["recall"].mean()
            mret = sub["n_returned"].mean()
            n_pass = int((sub["recall"] >= 0.5).sum())
            print(
                f"  {k_in:>5d}  {variant:>20s}  {n:>5d}  {mr:>11.3f}  "
                f"{mret:>13.0f}  {n_pass:>5d} ({100*n_pass/max(n,1):>3.0f} %)"
            )

    print("\n  CROSS-MODAL — JP (gold = pourvois CC) :")
    print(
        f"  {'k_in':>5s}  {'variant':>20s}  {'n_q':>5s}  {'mean_recall':>11s}  "
        f"{'mean_returned':>13s}  {'pass_r≥0.5':>10s}"
    )
    for k_in in KS_IN:
        for variant in ("jp_union", "jp_intersection"):
            sub = valid[
                (valid["direction"] == "cross")
                & (valid["k_in"] == k_in)
                & (valid["variant"] == variant)
                & (valid["side"] == "jp")
            ]
            n = len(sub)
            mr = sub["recall"].mean()
            mret = sub["n_returned"].mean()
            n_pass = int((sub["recall"] >= 0.5).sum())
            print(
                f"  {k_in:>5d}  {variant:>20s}  {n:>5d}  {mr:>11.3f}  "
                f"{mret:>13.0f}  {n_pass:>5d} ({100*n_pass/max(n,1):>3.0f} %)"
            )

    summary = {
        "ks_in": KS_IN,
        "by_setting": {
            f"{d}_{v}_kin{k}": {
                "n_q": int(((valid["direction"] == d) & (valid["k_in"] == k) & (valid["variant"] == v)).sum()),
                "mean_recall": float(valid[(valid["direction"] == d) & (valid["k_in"] == k) & (valid["variant"] == v)]["recall"].mean()),
                "mean_returned": float(valid[(valid["direction"] == d) & (valid["k_in"] == k) & (valid["variant"] == v)]["n_returned"].mean()),
                "n_pass_50": int(((valid["direction"] == d) & (valid["k_in"] == k) & (valid["variant"] == v) & (valid["recall"] >= 0.5)).sum()),
            }
            for d in ("art_to_jp", "jp_to_art")
            for v in ("union", "intersection", "majority")
            for k in KS_IN
        },
    }
    (OUT_DIR / "recall_graph_variants_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(f"\n✓ {OUT_DIR}/recall_graph_variants_summary.json")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
