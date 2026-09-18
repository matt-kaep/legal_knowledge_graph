#!/usr/bin/env python3
"""Export structuré des résultats de la campagne 754 vers results_final/.

Produit :
  results_final/scores_<condition>_<modalite>.csv   (par question)
  results_final/summary.json                        (agrégats + métadonnées)
  results_final/manifest.json                       (chemins + sha256)
  results_final/README.md                           (description)
"""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from run_agents import resolve_lkg_repo

HERE = Path(__file__).resolve().parent
REPO = resolve_lkg_repo()
OUT_ROOT = HERE / "results_final"
OUT_ROOT.mkdir(exist_ok=True)

RUNNER_PY = REPO / "05-Technique/benchmark/etape1_embedding_pur/scripts/100_run_b2_direct_llm.py"
CONFIG_PY = REPO / "05-Technique/benchmark/etape1_embedding_pur/etape1/config.py"
spec = importlib.util.spec_from_file_location("b2", RUNNER_PY)
b2 = importlib.util.module_from_spec(spec)
sys.modules["b2"] = b2
spec.loader.exec_module(b2)

GT = json.loads((HERE / "ground_truth_754.json").read_text(encoding="utf-8"))
JP_INDEX = json.loads((HERE / "jp_index_full.json").read_text(encoding="utf-8"))
import re
PV_RE = re.compile(r"n[°o]?\s*(\d{1,2}-\d{2}\.\d{3,4})")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load(condition, modalite, gt_kv):
    out = {}
    d = HERE / "output" / condition
    if not d.exists():
        return out
    for f in d.glob(f"*__{modalite}.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("qid") in gt_kv:
            out[str(rec["qid"])] = rec
    return out


def article_row(rec):
    qid = rec["qid"]
    gold = sorted(set(GT[qid].get("articles_attendus", [])))
    cand = sorted({a for v in GT.values() for a in v.get("articles_attendus", [])})
    resolver = b2.ArticleResolver(cand, b2.load_code_titles(CONFIG_PY))
    ranked = [resolver.resolve(r)[0] for r in rec.get("references", [])]
    return {
        "qid": qid,
        "status": rec.get("status"),
        "n_refs": len(rec.get("references", [])),
        "n_resolved": sum(1 for x in ranked if x),
        "refs_resolved": "|".join(x for x in ranked if x),
        "gold": "|".join(gold),
        "hit_at_10": round(b2.retrieval_metrics.hit_at_k(ranked, set(gold), 10), 6),
        "ndcg_at_10": round(b2.retrieval_metrics.ndcg_at_k(ranked, set(gold), 10), 6),
        "mrr_at_10": round(b2.retrieval_metrics.mrr_at_k(ranked, set(gold), 10), 6),
        "turns": rec.get("turns"),
        "tool_calls": len(rec.get("tool_calls", [])),
        "timed_out": bool(rec.get("timed_out")),
        "elapsed_s": round(rec.get("elapsed_seconds") or 0, 1),
    }


def jp_row(rec):
    qid = rec["qid"]
    gold = set()
    for g in GT[qid].get("gold_jp_ids", []):
        gold.update(JP_INDEX.get(g, {}).get("pourvois", []))
    refs = rec.get("references", [])
    ranked = []
    for r in refs:
        m = PV_RE.search(r)
        if m:
            ranked.append(m.group(1))
    return {
        "qid": qid,
        "status": rec.get("status"),
        "n_refs": len(refs),
        "n_pourvois": len(ranked),
        "pourvois_agent": "|".join(ranked),
        "gold_pourvois": "|".join(sorted(gold)),
        "hit_at_10": round(b2.retrieval_metrics.hit_at_k(ranked, gold, 10), 6),
        "ndcg_at_10": round(b2.retrieval_metrics.ndcg_at_k(ranked, gold, 10), 6),
        "mrr_at_10": round(b2.retrieval_metrics.mrr_at_k(ranked, gold, 10), 6),
        "turns": rec.get("turns"),
        "tool_calls": len(rec.get("tool_calls", [])),
        "timed_out": bool(rec.get("timed_out")),
        "elapsed_s": round(rec.get("elapsed_seconds") or 0, 1),
    }


def main():
    summary = {"campaign": "agent_legal_754", "n_questions": len(GT), "results": {}}
    manifest = {}
    for cond in ("web", "openlegy"):
        for modal in ("articles", "jurisprudence"):
            recs = load(cond, modal, GT)
            rows = [article_row(r) if modal == "articles" else jp_row(r) for r in recs.values()]
            # trie par qid
            rows.sort(key=lambda r: r["qid"])
            csv_path = OUT_ROOT / f"scores_{cond}_{modal}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            m = {
                "questions": len(rows),
                "ok": sum(1 for r in rows if r["status"] == "ok"),
                "timeout": sum(1 for r in rows if r["status"] == "timeout"),
            }
            for metric in ("hit_at_10", "ndcg_at_10", "mrr_at_10"):
                vals = [r[metric] for r in rows if r["status"] == "ok"]
                m[metric] = round(sum(vals) / len(vals), 6) if vals else None
            summary["results"][f"{cond}_{modal}"] = m
            manifest[str(csv_path)] = sha256(csv_path)
    # écrit summary.json ET manifest + README
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest[str(OUT_ROOT / "summary.json")] = sha256(OUT_ROOT / "summary.json")
    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT_ROOT / "README.md").write_text(
        "# Results — campagne agent legal 754\n\n"
        "Campagne 754 questions du benchmark ECIR (eval_rich_retrievable_strict), 2 conditions "
        "(`web` = Brave search, `openlegy` = MCP Légifrance/OpenLegy), 2 modalités par question "
        "(articles, jurisprudence).\n\n"
        "- `scores_<condition>_<modalite>.csv` : score par question\n"
        "- `summary.json` : agrégats NHit@10 / NDCG@10 / MRR@10\n"
        "- `manifest.json` : sha256 des fichiers\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["results"], ensure_ascii=False, indent=2))
    print("exports dans:", OUT_ROOT)


if __name__ == "__main__":
    main()
