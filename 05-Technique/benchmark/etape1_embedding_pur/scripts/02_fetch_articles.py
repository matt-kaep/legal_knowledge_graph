"""CLI : résout les 8085 pair_keys pénaux du graphe → articles_penal.parquet."""
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
    penal_mask = np.array([c in config.PENAL_CODES for c in article_codes])
    penal_pks = article_ids[penal_mask].tolist()
    print(f"Articles pénaux du graphe : {len(penal_pks)}")

    rubrics = json.loads(config.RUBRICS.read_text())["questions"]
    gold = set()
    for q in rubrics:
        aa = q["articles_attendus"]
        gold |= set(aa.get("obligatoires", [])) | set(aa.get("optionnels", []))

    print(f"Résolution via {config.LEGI_SQLITE}…")
    resolved = resolve_pair_keys(config.LEGI_SQLITE, penal_pks, config.PENAL_CODES)

    rep = coverage_report(resolved, gold_pair_keys=gold)
    config.ARTICLES_COVERAGE.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    print(f"  Couverture globale : {rep['n_resolved']}/{rep['n_total']} "
          f"({100*rep['resolution_rate']:.1f}%)")
    print(f"  Couverture gold    : {rep['n_gold_resolved']}/{rep['n_gold_total']} "
          f"({100*rep['gold_resolution_rate']:.1f}%)")

    rows = [
        {"pair_key": pk, **resolved[pk]}
        for pk in penal_pks if resolved[pk]["texte"] is not None
    ]
    pd.DataFrame(rows).to_parquet(config.ARTICLES_PARQUET, index=False)
    print(f"✓ {len(rows)} articles écrits dans {config.ARTICLES_PARQUET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
