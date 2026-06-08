"""Éval embeddings articles + JP sur le corpus doctrine_qgen (1707 q strict).

Réutilise les artefacts produits par les scripts 03b/07 :
  - emb_articles_all.npy   (31 357 × 1024)
  - emb_jp_synthese.npy    (116 755 × 1024)
  - graph_penal.npz        (back-edge biparti)

Pour chaque question, on calcule :
  ARTICLES side
    (open)      cosine question × 31 357 articles tous codes
    (filtered)  cosine question × articles des codes pénal+procpen+civil
  JP side (gold JP résolu via regex pourvoi CC `\\d{2}-\\d{2}\\.\\d{3}`)
    (direct)    cosine question × 116 755 synthèses JP
    (via_graph) JP citant les top-K articles
    (hybrid_max) union top-K direct + via_graph, dédup, garde l'ordre direct

Sorties :
  data/doctrine_qgen/recall_articles.csv
  data/doctrine_qgen/recall_jp.csv
  data/doctrine_qgen/summary.json
"""
from __future__ import annotations
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq

# Pointer sys.path vers le repo principal (le module etape1 n'est pas dans le worktree)
REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
sys.path.insert(0, str(REPO / "05-Technique" / "benchmark" / "etape1_embedding_pur"))

from etape1 import config  # noqa: E402
from etape1.eval_recall import recall_at_k, extract_pourvoi_numbers  # noqa: E402

_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")

# Pool « filtered » pour doctrine_qgen : tous les codes effectivement cités par
# les 1707 questions, avec une fréquence ≥ 3 (filtrage des codes anecdotiques).
# Les questions doctrine_qgen sont toutes branche=penal mais touchent aussi
# code_civil (incidence civile), code_des_douanes, code_general_des_impots, etc.
FILTERED_CODES_PENAL_EXTENDED = {
    "code_de_procedure_penale",
    "code_penal",
    "code_civil",
    "code_des_douanes",
    "code_d_instruction_criminelle",
    "code_de_procedure_civile",
    "code_general_des_impots",
    "code_de_la_securite_sociale",
    "code_de_l_organisation_judiciaire",
    "code_de_la_justice_penale_des_mineurs",
}

# Strict-strict : juste les 4 codes pénal officiels (équivalent au pool 3k initial)
FILTERED_CODES_PENAL_STRICT = set(config.PENAL_CODES.keys())

CORPUS_PATH = (
    REPO
    / "05-Technique/benchmark/llm_benchmark/doctrine_qgen/corpus_strict_gemma4-26B-A4B.json"
)

OUT_DIR = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_qgen"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def load_doctrine_qgen() -> list[dict]:
    """Charge corpus_strict et l'aplatit en liste de questions, en injectant un
    champ `articles_attendus_oblig: set[str]` au format `code_slug:article_num`."""
    d = json.loads(CORPUS_PATH.read_text())
    questions = []
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
        questions.append(
            {
                "id": q["qid"],
                "branche": "penal",
                "doc_id": q.get("doc_id"),
                "theme": q.get("theme", ""),
                "enonce": q["enonce"],
                "oblig": oblig,
                "pourvois": pourvois,
                "n_articles_gold": len(oblig),
                "n_jp_gold_pourvois": len(pourvois),
            }
        )
    return questions


def build_pourvoi_to_jpid_map() -> dict[str, list[str]]:
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def encode_questions(texts: list[str], device: str | None = None) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"  device d'encoding : {device}")
    m = SentenceTransformer(config.MODEL_ID, device=device)
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
    print("══ Chargement des embeddings ──────────────────────────────")
    art_emb = np.load(config.EMB_ARTICLES_ALL)
    art_order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    jp_emb = np.load(config.EMB_JP_SYNTHESE)
    jp_order = np.load(config.JP_SUMMARY_ORDER, allow_pickle=True)
    print(f"  art_emb       : {art_emb.shape}")
    print(f"  jp_synth_emb  : {jp_emb.shape}")

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    G = sp.csr_matrix(
        (z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"])
    )
    jp_ids_graph = z["jp_ids"]
    full_codes = z["article_codes"]
    art_codes = full_codes[p2col]  # code_slug aligné sur art_emb

    pourvoi_map = build_pourvoi_to_jpid_map()
    print(f"  pourvois CC mappables : {len(pourvoi_map)}")

    print("\n══ Chargement corpus doctrine_qgen ────────────────────────")
    questions = load_doctrine_qgen()
    print(f"  questions : {len(questions)}")
    # Smoke mode : limiter via env var ETAPE1_LIMIT
    import os
    limit = int(os.environ.get("ETAPE1_LIMIT", "0") or "0")
    if limit > 0:
        questions = questions[:limit]
        print(f"  ⚠ ETAPE1_LIMIT={limit} → tronqué à {len(questions)} questions")

    # Stats gold
    n_with_jp_resolved = sum(
        1
        for q in questions
        if any(p in pourvoi_map for p in q["pourvois"])
    )
    print(f"  questions avec JP gold résolu (pourvoi ∈ CC corpus) : {n_with_jp_resolved} / {len(questions)}")

    # Couverture des articles gold dans le pool embeddé
    pool_pk = set(art_order.tolist())
    n_q_all_art_in_pool = 0
    n_art_in_pool = n_art_total = 0
    for q in questions:
        if q["oblig"]:
            in_pool = q["oblig"] & pool_pk
            n_art_in_pool += len(in_pool)
            n_art_total += len(q["oblig"])
            if len(in_pool) == len(q["oblig"]):
                n_q_all_art_in_pool += 1
    print(
        f"  articles gold couvrant le pool : {n_art_in_pool}/{n_art_total} "
        f"({100*n_art_in_pool/max(n_art_total,1):.1f} %)"
    )
    print(
        f"  questions avec 100 % articles dans pool : "
        f"{n_q_all_art_in_pool} / {len(questions)} ({100*n_q_all_art_in_pool/len(questions):.1f} %)"
    )

    print("\n══ Encoding des 1707 questions ────────────────────────────")
    Q = encode_questions([q["enonce"] for q in questions])
    print(f"  Q : {Q.shape} (t={time.time()-t0:.1f}s)")

    print("\n══ Cosine sim ─────────────────────────────────────────────")
    sim_art = Q @ art_emb.T  # (n_q, 31357)
    sim_jp = Q @ jp_emb.T  # (n_q, 116755)
    print(f"  sim_art : {sim_art.shape}, sim_jp : {sim_jp.shape}")
    print(f"  t={time.time()-t0:.1f}s")

    # Masks de pool filtered
    mask_strict = np.array([c in FILTERED_CODES_PENAL_STRICT for c in art_codes])
    mask_ext = np.array([c in FILTERED_CODES_PENAL_EXTENDED for c in art_codes])
    print(
        f"  pool filtered strict (4 codes pénal): {mask_strict.sum()} articles | "
        f"étendu (10 codes): {mask_ext.sum()}"
    )

    KS = config.KS
    rows_art: list[dict] = []
    rows_jp: list[dict] = []
    per_question: list[dict] = []

    print("\n══ Boucle d'éval par question ─────────────────────────────")
    for qi, q in enumerate(questions):
        if qi % 200 == 0:
            print(f"  q {qi}/{len(questions)}  (t={time.time()-t0:.1f}s)")

        oblig = q["oblig"]
        gold_jp_ids = {jid for p in q["pourvois"] for jid in pourvoi_map.get(p, [])}

        # ── ARTICLES ──
        sm = {
            "question_id": q["id"],
            "branche": q["branche"],
            "doc_id": q["doc_id"],
            "n_gold_oblig": len(oblig),
            "n_gold_jp_pourvois": len(q["pourvois"]),
            "n_gold_jp_ids": len(gold_jp_ids),
        }
        # Préinitialiser toutes les clés JP à None pour éviter les KeyError
        for strat_name in ("open", "filtered_strict", "filtered_ext"):
            sm[f"r5_jp_via_graph_{strat_name}"] = None
            sm[f"r10_jp_via_graph_{strat_name}"] = None
        for m_ in ("jp_direct", "jp_hybrid_max"):
            sm[f"r5_{m_}"] = None
            sm[f"r10_{m_}"] = None

        for strat_name, mask in (
            ("open", None),
            ("filtered_strict", mask_strict),
            ("filtered_ext", mask_ext),
        ):
            if mask is None:
                sim_q = sim_art[qi]
                local_order = np.argsort(-sim_q)
                ranked_pks = art_order[local_order]
                ranked_cols = p2col[local_order]
            else:
                sim_q = sim_art[qi][mask]
                local_order = np.argsort(-sim_q)
                ranked_pks = art_order[mask][local_order]
                ranked_cols = p2col[mask][local_order]

            for k in KS:
                rows_art.append(
                    {
                        "question_id": q["id"],
                        "strategy": strat_name,
                        "k": k,
                        "recall": recall_at_k(list(ranked_pks), oblig, k),
                    }
                )

            r10 = recall_at_k(list(ranked_pks), oblig, 10)
            r20 = recall_at_k(list(ranked_pks), oblig, 20)
            sm[f"r10_art_{strat_name}"] = r10
            sm[f"r20_art_{strat_name}"] = r20

            # ── JP via graph pour cette stratégie ──
            if gold_jp_ids:
                for k in KS:
                    top_cols = ranked_cols[:k]
                    jp_mask = (G[:, top_cols].sum(axis=1) > 0).A1
                    hit = len(set(jp_ids_graph[jp_mask].tolist()) & gold_jp_ids)
                    r = hit / len(gold_jp_ids)
                    rows_jp.append(
                        {
                            "question_id": q["id"],
                            "method": f"jp_via_graph_{strat_name}",
                            "k": k,
                            "recall": r,
                        }
                    )
                top5 = ranked_cols[:5]
                top10 = ranked_cols[:10]
                jp5 = (G[:, top5].sum(axis=1) > 0).A1
                jp10 = (G[:, top10].sum(axis=1) > 0).A1
                sm[f"r5_jp_via_graph_{strat_name}"] = (
                    len(set(jp_ids_graph[jp5].tolist()) & gold_jp_ids) / len(gold_jp_ids)
                )
                sm[f"r10_jp_via_graph_{strat_name}"] = (
                    len(set(jp_ids_graph[jp10].tolist()) & gold_jp_ids) / len(gold_jp_ids)
                )

        # ── JP direct ──
        if gold_jp_ids:
            order_direct = np.argsort(-sim_jp[qi])
            ranked_direct = list(jp_order[order_direct])
            for k in KS:
                rows_jp.append(
                    {
                        "question_id": q["id"],
                        "method": "jp_direct",
                        "k": k,
                        "recall": recall_at_k(ranked_direct, gold_jp_ids, k),
                    }
                )
            sm["r5_jp_direct"] = recall_at_k(ranked_direct, gold_jp_ids, 5)
            sm["r10_jp_direct"] = recall_at_k(ranked_direct, gold_jp_ids, 10)

            # JP hybrid_max (union top-K direct + top-K via_graph open, dédupe)
            # ré-utilise top10 du dernier strat = filtered_ext ; on calcule plutôt
            # sur open pour cohérence.
            top10_open = p2col[np.argsort(-sim_art[qi])[:10]]
            jp_via10 = jp_ids_graph[(G[:, top10_open].sum(axis=1) > 0).A1].tolist()
            for k in KS:
                top_dir = list(ranked_direct[:k])
                top_via = jp_via10[:k]
                cand = list(dict.fromkeys(top_dir + top_via))[:k]
                rows_jp.append(
                    {
                        "question_id": q["id"],
                        "method": "jp_hybrid_max",
                        "k": k,
                        "recall": (
                            len(set(cand) & gold_jp_ids) / len(gold_jp_ids)
                        ),
                    }
                )
            sm["r5_jp_hybrid_max"] = (
                len(set(list(dict.fromkeys(list(ranked_direct[:5]) + jp_via10[:5]))[:5]) & gold_jp_ids)
                / len(gold_jp_ids)
            )
            sm["r10_jp_hybrid_max"] = (
                len(set(list(dict.fromkeys(list(ranked_direct[:10]) + jp_via10[:10]))[:10]) & gold_jp_ids)
                / len(gold_jp_ids)
            )
        # (les valeurs JP restent à None si gold_jp_ids vide — déjà initialisées)

        # Pass critères (open uniquement pour récap principale)
        if gold_jp_ids:
            sm["pass_hard"] = (
                sm["r10_art_open"] >= 0.5 and sm["r5_jp_via_graph_open"] >= 0.5
            )
            sm["pass_easy"] = (
                sm["r20_art_open"] >= 0.5 and sm["r10_jp_via_graph_open"] >= 0.5
            )
        else:
            sm["pass_hard"] = sm["r10_art_open"] >= 0.5
            sm["pass_easy"] = sm["r20_art_open"] >= 0.5

        per_question.append(sm)

    print(f"  fin de la boucle (t={time.time()-t0:.1f}s)")

    print("\n══ Écriture résultats ─────────────────────────────────────")
    pd.DataFrame(rows_art).to_csv(OUT_DIR / "recall_articles.csv", index=False)
    pd.DataFrame(rows_jp).to_csv(OUT_DIR / "recall_jp.csv", index=False)

    # Agrégats
    n_total = len(per_question)
    n_eval_jp = sum(1 for s in per_question if s["n_gold_jp_ids"] > 0)
    agg = {
        "n_questions": n_total,
        "n_questions_with_jp_resolved": n_eval_jp,
        "n_pass_hard": sum(1 for s in per_question if s["pass_hard"]),
        "n_pass_easy": sum(1 for s in per_question if s["pass_easy"]),
        "mean_r10_art_open": float(
            np.mean([s["r10_art_open"] for s in per_question])
        ),
        "mean_r10_art_filtered_strict": float(
            np.mean([s["r10_art_filtered_strict"] for s in per_question])
        ),
        "mean_r10_art_filtered_ext": float(
            np.mean([s["r10_art_filtered_ext"] for s in per_question])
        ),
        "mean_r20_art_open": float(
            np.mean([s["r20_art_open"] for s in per_question])
        ),
    }
    if n_eval_jp:
        for m_ in ("jp_direct", "jp_via_graph_open", "jp_hybrid_max"):
            vals_r5 = [
                s[f"r5_{m_}"] for s in per_question if s[f"r5_{m_}"] is not None
            ]
            vals_r10 = [
                s[f"r10_{m_}"] for s in per_question if s[f"r10_{m_}"] is not None
            ]
            agg[f"mean_r5_{m_}"] = float(np.mean(vals_r5)) if vals_r5 else None
            agg[f"mean_r10_{m_}"] = float(np.mean(vals_r10)) if vals_r10 else None
            agg[f"n_pass_dur_{m_}"] = sum(1 for v in vals_r5 if v >= 0.5)
            agg[f"n_pass_easy_{m_}"] = sum(1 for v in vals_r10 if v >= 0.5)

    (OUT_DIR / "summary.json").write_text(
        json.dumps(
            {"aggregate": agg, "per_question": per_question, "ks": KS},
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n══ Résumé ─────────────────────────────────────────────────")
    print(f"  n_questions                          : {agg['n_questions']}")
    print(
        f"  n_questions évaluables JP (pourvoi → id)  : {agg['n_questions_with_jp_resolved']}"
    )
    print(
        f"  mean recall@10 articles open         : {agg['mean_r10_art_open']:.3f}"
    )
    print(
        f"  mean recall@10 articles filtered_strict (4 codes): {agg['mean_r10_art_filtered_strict']:.3f}"
    )
    print(
        f"  mean recall@10 articles filtered_ext (10 codes): {agg['mean_r10_art_filtered_ext']:.3f}"
    )
    print(
        f"  mean recall@20 articles open         : {agg['mean_r20_art_open']:.3f}"
    )
    if n_eval_jp:
        for m_ in ("jp_direct", "jp_via_graph_open", "jp_hybrid_max"):
            print(
                f"  {m_:22s} mean r@5={agg[f'mean_r5_{m_}']:.3f}  "
                f"r@10={agg[f'mean_r10_{m_}']:.3f}  "
                f"pass_dur={agg[f'n_pass_dur_{m_}']}/{n_eval_jp}  "
                f"pass_easy={agg[f'n_pass_easy_{m_}']}/{n_eval_jp}"
            )
    print(
        f"\n  n_pass_hard (r@10_art_open + r@5_jp_via_graph)  : "
        f"{agg['n_pass_hard']} / {agg['n_questions']} "
        f"({100*agg['n_pass_hard']/agg['n_questions']:.1f} %)"
    )
    print(
        f"  n_pass_easy (r@20_art_open + r@10_jp_via_graph) : "
        f"{agg['n_pass_easy']} / {agg['n_questions']} "
        f"({100*agg['n_pass_easy']/agg['n_questions']:.1f} %)"
    )
    print(f"\n✓ {OUT_DIR}/recall_articles.csv")
    print(f"✓ {OUT_DIR}/recall_jp.csv")
    print(f"✓ {OUT_DIR}/summary.json")
    print(f"  t total : {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
