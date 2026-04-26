"""Évalue regex V0 (baseline) et V3 (itérée) contre les 10 annotations manuelles.

Produit une table par arrêt + métriques globales + détail des écarts
(FP : regex a produit, pas dans GT; FN : GT mais regex n'a pas produit).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from iterate_regex import (
    extract_pairs_v0, extract_pairs_v1, extract_pairs_v3, load_sample,
)

ANN_FILE = Path(__file__).parent / "manual_annotations.json"


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    recs = load_sample()
    ann = json.loads(ANN_FILE.read_text())["annotations"]

    versions = [("V0 baseline", extract_pairs_v0),
                ("V1 listing+suffixes", extract_pairs_v1),
                ("V3 +anaphore", extract_pairs_v3)]

    all_results = {}
    for vname, fn_extract in versions:
        totals = {"tp": 0, "fp": 0, "fn": 0}
        per_arret = []
        for rid, meta in ann.items():
            text = recs[rid]["text"]
            gt   = set(meta["gt"])
            pred = fn_extract(text)
            tp   = gt & pred
            fp   = pred - gt
            fn   = gt - pred
            totals["tp"] += len(tp); totals["fp"] += len(fp); totals["fn"] += len(fn)
            per_arret.append({
                "id": rid, "jur": meta["jur"],
                "n_gt": len(gt), "n_pred": len(pred),
                "tp": len(tp), "fp": len(fp), "fn": len(fn),
                "missed": sorted(fn), "extra": sorted(fp),
            })
        all_results[vname] = {"totals": totals, "per_arret": per_arret}

    # Print table comparative
    print("=" * 78)
    print("MÉTRIQUES POOLED (10 arrêts, annotation humaine)")
    print("=" * 78)
    print(f"{'Version':<25} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>7} {'R':>7} {'F1':>7}")
    print("-" * 78)
    for vname, res in all_results.items():
        t = res["totals"]
        p, r, f1 = prf(t["tp"], t["fp"], t["fn"])
        print(f"{vname:<25} {t['tp']:>5} {t['fp']:>5} {t['fn']:>5} "
              f"{p:>7.3f} {r:>7.3f} {f1:>7.3f}")

    # Par arrêt (V3 seulement)
    print()
    print("=" * 78)
    print("DÉTAIL PAR ARRÊT (V3)")
    print("=" * 78)
    print(f"{'ID':<26} {'Jur':<4} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4} {'F1':>6}")
    print("-" * 78)
    for rec in all_results["V3 +anaphore"]["per_arret"]:
        p, r, f1 = prf(rec["tp"], rec["fp"], rec["fn"])
        print(f"{rec['id']:<26} {rec['jur']:<4} {rec['n_gt']:>4} "
              f"{rec['n_pred']:>5} {rec['tp']:>4} {rec['fp']:>4} {rec['fn']:>4} {f1:>6.3f}")

    # Détail des écarts pour V3
    print()
    print("=" * 78)
    print("ÉCARTS V3 (par arrêt) — à analyser pour améliorer")
    print("=" * 78)
    for rec in all_results["V3 +anaphore"]["per_arret"]:
        if not rec["missed"] and not rec["extra"]:
            continue
        print(f"\n─── [{rec['jur']}] {rec['id']} ───")
        if rec["missed"]:
            print(f"  FN (GT mais pas détecté, {len(rec['missed'])}):")
            for pk in rec["missed"]:
                print(f"     ⊘ {pk}")
        if rec["extra"]:
            print(f"  FP (détecté mais pas dans GT, {len(rec['extra'])}):")
            for pk in rec["extra"]:
                print(f"     + {pk}")

    # Sauvegarde pour analyse
    out = Path(__file__).parent / "eval_vs_manual_results.json"
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nRésultats sauvés : {out}")


if __name__ == "__main__":
    main()
