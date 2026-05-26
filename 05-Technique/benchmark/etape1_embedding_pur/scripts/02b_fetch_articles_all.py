"""CLI : résout TOUS les pair_keys du graphe (87 821, tous codes) → articles_all.parquet.

Variante full-corpus de 02_fetch_articles.py : utilise config.ALL_CODES au lieu
de config.PENAL_CODES. Sortie : data/articles_all.parquet + data/articles_all_coverage.json.
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
from etape1 import config
from etape1.resolve import resolve_pair_keys, coverage_report


def main() -> int:
    if not config.LEGI_SQLITE.exists():
        print(f"✗ {config.LEGI_SQLITE} absent — lancer ./scripts/_setup_legi.sh d'abord")
        return 1

    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    article_ids = z["article_ids"]
    article_codes = z["article_codes"]
    # Garde tous les articles dont le code_slug est dans ALL_CODES (incl. ceux mappés à None,
    # qui apparaîtront en non-résolus dans le rapport — utile pour diagnostic).
    all_pks = article_ids.tolist()
    print(f"Articles du graphe (tous codes) : {len(all_pks)}")

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    gold = set()
    for q in rubrics:
        aa = q["articles_attendus"]
        gold |= set(aa.get("obligatoires", [])) | set(aa.get("optionnels", []))

    # resolve_pair_keys ignore les slugs absents du code_map ou mappés à None
    # (titre is None → entry stays unresolved).
    code_map = {k: v for k, v in config.ALL_CODES.items() if v is not None}

    print(f"Résolution via {config.LEGI_SQLITE} (mapping {len(code_map)} codes)…")
    resolved = resolve_pair_keys(config.LEGI_SQLITE, all_pks, code_map)

    rep = coverage_report(resolved, gold_pair_keys=gold)
    # Ventilation par code_slug
    from collections import Counter, defaultdict
    by_code_total: Counter = Counter()
    by_code_resolved: Counter = Counter()
    for pk, entry in resolved.items():
        slug = entry["code_slug"]
        by_code_total[slug] += 1
        if entry["texte"] is not None:
            by_code_resolved[slug] += 1
    per_code = []
    for slug in sorted(by_code_total):
        tot = by_code_total[slug]
        res = by_code_resolved[slug]
        per_code.append({
            "code_slug": slug,
            "legi_titre": config.ALL_CODES.get(slug),
            "n_total": tot,
            "n_resolved": res,
            "rate": res / max(tot, 1),
        })
    rep["per_code"] = per_code

    config.ARTICLES_COVERAGE_ALL.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"  Couverture globale : {rep['n_resolved']}/{rep['n_total']} "
          f"({100*rep['resolution_rate']:.1f}%)")
    print(f"  Couverture gold    : {rep['n_gold_resolved']}/{rep['n_gold_total']} "
          f"({100*rep['gold_resolution_rate']:.1f}%)")
    print(f"  Ventilation par code (top + bottom) :")
    sorted_by_rate = sorted(per_code, key=lambda r: r["rate"])
    for r in sorted_by_rate[:5]:
        print(f"    {r['rate']*100:5.1f}%  {r['n_resolved']:>5}/{r['n_total']:<5}  {r['code_slug']}")
    print(f"    ...")
    for r in sorted_by_rate[-5:]:
        print(f"    {r['rate']*100:5.1f}%  {r['n_resolved']:>5}/{r['n_total']:<5}  {r['code_slug']}")

    rows = [
        {"pair_key": pk, **resolved[pk]}
        for pk in all_pks if resolved[pk]["texte"] is not None
    ]
    pd.DataFrame(rows).to_parquet(config.ARTICLES_PARQUET_ALL, index=False)
    print(f"✓ {len(rows)} articles écrits dans {config.ARTICLES_PARQUET_ALL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
