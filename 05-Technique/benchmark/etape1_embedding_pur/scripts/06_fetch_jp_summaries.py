"""CLI : fetch synthese_pour_avocat depuis OVH PostgreSQL pour les 118k JP du graphe pénal.

Récupère, pour chaque jp_id du graphe (source='judilibre'), le champ
`synthese_pour_avocat` de `jp_decisions`. Aligne sur l'ordre des `jp_ids`
du graphe. Sortie :
  data/jp_summaries_penal.parquet  (jp_id, juris, synthese, len_chars)
  data/jp_summary_order.npy        (jp_ids avec synthese, dans l'ordre du graphe)
  data/jp_summary_to_graphrow.npy  (indices dans jp_ids du graphe)
  data/jp_summaries_coverage.json  (statistiques de couverture)

Stratégie : la table fait 1.8M lignes judilibre, on cible 118k jp_ids du graphe.
Plus rapide de filtrer côté serveur en passant la liste via temp table.
"""
from __future__ import annotations
import json
import sys
import time
import numpy as np
import pandas as pd
from etape1 import config, db


def main() -> int:
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    jp_ids = z["jp_ids"].astype(str)
    jp_juris = z["jp_juris"].astype(str)
    print(f"Cible : {len(jp_ids):,} JP du graphe pénal")
    print(f"  juris breakdown : "
          + ", ".join(f"{j}={(jp_juris==j).sum()}" for j in sorted(set(jp_juris))))

    print("\nConnexion OVH PostgreSQL…")
    t0 = time.time()
    with db.connect() as conn:
        with conn.cursor() as cur:
            # Temp table → JOIN est beaucoup plus rapide que ANY(118k array)
            cur.execute("CREATE TEMP TABLE tmp_ids(source_id text PRIMARY KEY)")
            from psycopg2.extras import execute_batch
            execute_batch(
                cur,
                "INSERT INTO tmp_ids(source_id) VALUES (%s) ON CONFLICT DO NOTHING",
                [(jid,) for jid in jp_ids],
                page_size=5000,
            )
            cur.execute("SELECT COUNT(*) FROM tmp_ids")
            n_in_table = cur.fetchone()[0]
            print(f"  Temp table : {n_in_table:,} jp_ids insérés ({time.time()-t0:.1f}s)")

            print("  Query SELECT (filter judilibre) …")
            t1 = time.time()
            cur.execute("""
                SELECT d.source_id, d.synthese_pour_avocat
                FROM jp_decisions d
                JOIN tmp_ids t ON t.source_id = d.source_id
                WHERE d.source = 'judilibre'
                  AND d.synthese_pour_avocat IS NOT NULL
            """)
            rows = cur.fetchall()
            print(f"  Récupéré {len(rows):,} synthèses ({time.time()-t1:.1f}s)")

    # Indexer par source_id pour aligner sur jp_ids du graphe
    by_id = {sid: synth for sid, synth in rows}
    juris_by_id = dict(zip(jp_ids, jp_juris))

    rows_out = []
    keepers_idx = []
    for i, jid in enumerate(jp_ids):
        s = by_id.get(jid)
        if s and s.strip():
            rows_out.append({
                "jp_id":     jid,
                "juris":     juris_by_id[jid],
                "synthese":  s,
                "len_chars": len(s),
            })
            keepers_idx.append(i)
    df = pd.DataFrame(rows_out)
    df.to_parquet(config.JP_SUMMARIES_PARQUET, index=False)

    summary_order = jp_ids[keepers_idx].astype(object)
    summary_to_row = np.array(keepers_idx, dtype=np.int32)
    np.save(config.JP_SUMMARY_ORDER, summary_order)
    np.save(config.JP_SUMMARY_TO_GRAPHROW, summary_to_row)

    # Couverture par juris
    cov = {
        "n_graph_jp":       int(len(jp_ids)),
        "n_resolved":       int(len(df)),
        "resolution_rate":  float(len(df) / max(len(jp_ids), 1)),
        "by_juris":         {},
        "len_p50":          int(df["len_chars"].median()) if len(df) else 0,
        "len_p90":          int(df["len_chars"].quantile(0.9)) if len(df) else 0,
        "len_p99":          int(df["len_chars"].quantile(0.99)) if len(df) else 0,
        "len_max":          int(df["len_chars"].max()) if len(df) else 0,
    }
    for j in sorted(set(jp_juris)):
        n_tot = int((jp_juris == j).sum())
        n_res = int((df["juris"] == j).sum()) if len(df) else 0
        cov["by_juris"][j] = {"total": n_tot, "resolved": n_res,
                               "rate": n_res / max(n_tot, 1)}
    config.JP_SUMMARIES_COVERAGE.write_text(json.dumps(cov, ensure_ascii=False, indent=2))

    print(f"\n=== Récap ===")
    print(f"  Couverture globale : {cov['n_resolved']:,}/{cov['n_graph_jp']:,} "
          f"({100*cov['resolution_rate']:.1f} %)")
    for j, st in cov["by_juris"].items():
        print(f"  juris={j:4s} : {st['resolved']:>6,}/{st['total']:<6,} "
              f"({100*st['rate']:.1f} %)")
    print(f"  Longueur synthese : p50={cov['len_p50']} p90={cov['len_p90']} "
          f"p99={cov['len_p99']} max={cov['len_max']} chars")
    print(f"✓ {config.JP_SUMMARIES_PARQUET}")
    print(f"✓ {config.JP_SUMMARY_ORDER}")
    print(f"✓ {config.JP_SUMMARY_TO_GRAPHROW}")
    print(f"✓ {config.JP_SUMMARIES_COVERAGE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
