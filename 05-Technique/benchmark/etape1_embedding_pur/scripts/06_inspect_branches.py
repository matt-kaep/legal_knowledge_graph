"""Analyse détaillée par question pour chaque branche.

Pour chaque question des 11 rubrics CNB 2025, produit :
  - 4 sets de gold : obligatoires, core (linked_article ∈ rubric.core), expected (cumul), expert (cumul)
  - K* (recall ≥ 0.5) sur chaque niveau (open + filtered)
  - Top-10 articles retournés avec texte tronqué
  - Rang dans le classement de chaque gold (oblig, core, expected, expert)
  - Articles gold non résolus dans le pool

Sortie : data/branch_diagnostics.json (utilisé par les subagents pour le diagnostic qualitatif)
"""
from __future__ import annotations
import glob
import json
import re
import sys
from collections import defaultdict
import numpy as np
import pandas as pd
import scipy.sparse as sp
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from etape1 import config
from etape1.eval_recall import recall_at_k, kstar, extract_pourvoi_numbers

KS = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
_POURVOI_RE = re.compile(r"\d{2}-\d{2}\.\d{3}")


def _gold_from_rubric(q: dict) -> dict[str, list[str]]:
    """Extrait 4 niveaux de gold pour une question."""
    aa = q.get("articles_attendus", {})
    oblig = aa.get("obligatoires", [])
    rubric = q.get("rubric", {})

    def _articles(items):
        return [it.get("linked_article") for it in items
                if it.get("linked_article")]

    core = _articles(rubric.get("core", []))
    expected = core + _articles(rubric.get("expected", []))
    expert = expected + _articles(rubric.get("expert", []))
    return {
        "obligatoires": list(dict.fromkeys(oblig)),
        "core":         list(dict.fromkeys(core)),
        "expected":     list(dict.fromkeys(expected)),
        "expert":       list(dict.fromkeys(expert)),
    }


def _load_all_questions() -> list[dict]:
    files = sorted(glob.glob(str(config.RUBRICS_AFFAIRES.parent /
                                   "cnb-*-2025-consolidated.json")))
    out: list[dict] = []
    seen: set[str] = set()
    for f in files:
        d = json.loads(open(f).read())
        for q in d.get("questions", d if isinstance(d, list) else []):
            if q["id"] not in seen:
                seen.add(q["id"])
                out.append(q)
    return out


def _build_pourvoi_map() -> dict[str, list[str]]:
    jp = pq.read_table(config.JP_INDEX, columns=["id", "number", "juris"]).to_pandas()
    jp = jp[jp["juris"] == "CC"]
    out: dict[str, list[str]] = {}
    for r in jp.itertuples():
        n = (r.number or "").strip()
        if _POURVOI_RE.fullmatch(n):
            out.setdefault(n, []).append(r.id)
    return out


def main() -> int:
    print("Loading embeddings + graph…", file=sys.stderr)
    emb = np.load(config.EMB_ARTICLES_ALL)
    order = np.load(config.ARTICLES_ORDER_ALL, allow_pickle=True)
    p2col = np.load(config.PAIRKEY_TO_GRAPHCOL_ALL)
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    full_codes = z["article_codes"]
    art_codes = full_codes[p2col]
    G = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    jp_ids = z["jp_ids"]
    arts_df = pd.read_parquet(config.ARTICLES_PARQUET_ALL).set_index("pair_key")
    resolved = set(arts_df.index)

    questions = _load_all_questions()
    print(f"{len(questions)} questions chargées", file=sys.stderr)
    pourvoi_map = _build_pourvoi_map()

    # Branches dynamiques
    branches: dict[str, set[str]] = defaultdict(set)
    for q in questions:
        br = q.get("branche", "?")
        for level in ["obligatoires", "optionnels"]:
            for pk in q.get("articles_attendus", {}).get(level, []):
                branches[br].add(pk.split(":", 1)[0])

    # Encode questions (CPU)
    print("Encoding questions…", file=sys.stderr)
    m = SentenceTransformer(config.MODEL_ID, device="cpu")
    m.max_seq_length = config.BATCH_MAX_LEN
    Q = m.encode([q["question"] for q in questions],
                  normalize_embeddings=True, convert_to_numpy=True,
                  show_progress_bar=True).astype(np.float32)

    out: list[dict] = []
    for qi, q in enumerate(questions):
        qid = q["id"]
        branche = q.get("branche", "?")
        specialisation = q.get("specialisation", "")

        gold = _gold_from_rubric(q)
        # Pourvois JP gold (CC)
        gold_pourvois = set()
        for jp in q.get("jp_attendues", []):
            gold_pourvois.update(extract_pourvoi_numbers(jp.get("short_ref") or ""))
        gold_jp_ids = {jpid for p in gold_pourvois for jpid in pourvoi_map.get(p, [])}

        # Couverture
        gold_resolved = {lvl: [pk for pk in pks if pk in resolved]
                          for lvl, pks in gold.items()}
        gold_missing = {lvl: [pk for pk in pks if pk not in resolved]
                          for lvl, pks in gold.items()}

        sim = Q[qi] @ emb.T  # (n_emb,)
        # — Open (tous les codes)
        order_open = np.argsort(-sim)
        ranked_open = list(order[order_open])
        cols_open = p2col[order_open]

        # — Filtered (codes de la branche)
        br_codes = branches.get(branche, set())
        mask = np.array([c in br_codes for c in art_codes])
        order_filt = np.argsort(-sim[mask])
        ranked_filt = list(order[mask][order_filt])
        cols_filt = p2col[mask][order_filt]

        # Recall@K et K* par niveau de gold, pour les 2 stratégies
        def _metrics(ranked, cols, golds_dict):
            res = {}
            for lvl, pks in golds_dict.items():
                gold_set = set(pks)
                res[lvl] = {
                    "n_gold":         len(gold_set),
                    "n_gold_resolved": len([p for p in gold_set if p in resolved]),
                    "recall_at_k":    {k: recall_at_k(ranked, gold_set, k) for k in KS},
                    "k_star":         kstar(ranked, gold_set, KS, 0.5),
                    "ranks":          {pk: (int(ranked.index(pk)) + 1 if pk in ranked else None)
                                          for pk in pks},
                }
            # JP via graphe
            if gold_jp_ids:
                jp_recalls = {}
                for k in KS:
                    top = cols[:k]
                    jp_mask = (G[:, top].sum(axis=1) > 0).A1
                    hit = len(set(jp_ids[jp_mask].tolist()) & gold_jp_ids)
                    jp_recalls[k] = hit / len(gold_jp_ids)
                res["jp_via_graph"] = {
                    "n_gold_jp":     len(gold_jp_ids),
                    "n_pourvois":    len(gold_pourvois),
                    "recall_at_k":   jp_recalls,
                    "k_star":        next((k for k in KS if jp_recalls[k] >= 0.5), None),
                }
            return res

        metrics_open = _metrics(ranked_open, cols_open, gold)
        metrics_filt = _metrics(ranked_filt, cols_filt, gold)

        # Top-10 retourné (open + filtered)
        def _top_k_payload(ranked, k=10):
            out = []
            for j in range(min(k, len(ranked))):
                pk = ranked[j]
                try:
                    text = arts_df.loc[pk, "texte"][:200].replace("\n", " ")
                except (KeyError, TypeError):
                    text = "(no text)"
                tag = ""
                if pk in gold["obligatoires"]: tag = "OBLIG"
                elif pk in gold["core"]:        tag = "CORE"
                elif pk in gold["expected"]:    tag = "EXPECTED"
                elif pk in gold["expert"]:      tag = "EXPERT"
                out.append({"rank": j+1, "pair_key": pk, "text": text, "tag": tag})
            return out

        out.append({
            "id":              qid,
            "branche":         branche,
            "specialisation":  specialisation,
            "question":        q["question"],
            "pieges":          q.get("pieges"),
            "gold":            gold,
            "gold_resolved":   gold_resolved,
            "gold_missing":    gold_missing,
            "jp_gold_pourvois": sorted(gold_pourvois),
            "jp_gold_ids":     sorted(gold_jp_ids),
            "metrics_open":    metrics_open,
            "metrics_filtered": metrics_filt,
            "top10_open":      _top_k_payload(ranked_open),
            "top10_filtered":  _top_k_payload(ranked_filt),
            "branch_pool_size": int(mask.sum()),
        })

    out_path = config.DATA / "branch_diagnostics.json"
    out_path.write_text(json.dumps({
        "ks":         KS,
        "threshold":  0.5,
        "pool_size":  len(order),
        "branches":   {br: sorted(codes) for br, codes in branches.items()},
        "questions":  out,
    }, ensure_ascii=False, indent=2))
    print(f"✓ {out_path} ({out_path.stat().st_size / 1024:.1f} KB)", file=sys.stderr)

    # Quick recap stdout
    print("\nRécap K* core par branche (open) :", file=sys.stderr)
    by_br: dict[str, list[int|None]] = defaultdict(list)
    for q in out:
        ks = q["metrics_open"].get("core", {}).get("k_star")
        by_br[q["branche"]].append(ks)
    for br in sorted(by_br):
        vals = by_br[br]
        passed = sum(1 for v in vals if v is not None and v <= 10)
        print(f"  {br:30s} {passed}/{len(vals)} questions avec K*_core ≤ 10", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
