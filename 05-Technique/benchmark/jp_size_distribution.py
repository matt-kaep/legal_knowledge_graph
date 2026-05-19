"""Mappe la distribution de taille du corpus Judilibre complet (jp_index.parquet).

Streaming row-group par row-group pour ne pas charger 5 GB en RAM.
Sortie : stats globales + par juridiction + comptes par paliers de contexte LLM.
"""
import numpy as np
import pyarrow.parquet as pq

PARQUET = "05-Technique/benchmark/baseline_b2/jp_index.parquet"
# Facteur chars->tokens. L'analyse pénale du 05-05 utilisait /3.0 (3878 ch = 1293 tok).
# On reporte les deux : /3.0 (cohérent historique) et /4.0 (plus conservateur multilingue).
CHARS_PER_TOK = 3.0

pf = pq.ParquetFile(PARQUET)
lengths = []          # longueur en chars de chaque doc
juris_arr = []        # juridiction de chaque doc

for rg in range(pf.num_row_groups):
    tbl = pf.read_row_group(rg, columns=["text", "juris"])
    texts = tbl.column("text")
    juris = tbl.column("juris").to_pylist()
    # pc.utf8_length serait plus rapide mais on reste simple/robuste
    for t, j in zip(texts.to_pylist(), juris):
        lengths.append(len(t) if t is not None else 0)
        juris_arr.append(j if j is not None else "?")

lengths = np.asarray(lengths, dtype=np.int64)
juris_arr = np.asarray(juris_arr)

def pct_block(arr, label):
    if len(arr) == 0:
        print(f"  [{label}] vide")
        return
    qs = [50, 75, 90, 95, 99, 99.9]
    pcts = np.percentile(arr, qs)
    print(f"  [{label}] n={len(arr):,}")
    print(f"    mean={arr.mean():,.0f} ch (~{arr.mean()/CHARS_PER_TOK:,.0f} tok)")
    for q, p in zip(qs, pcts):
        print(f"    p{q:<5}={p:,.0f} ch (~{p/CHARS_PER_TOK:,.0f} tok)")
    print(f"    max  ={arr.max():,.0f} ch (~{arr.max()/CHARS_PER_TOK:,.0f} tok)")

print("=" * 60)
print(f"CORPUS COMPLET — n = {len(lengths):,} JP")
print("=" * 60)
pct_block(lengths, "GLOBAL")

print("\n--- Par juridiction ---")
for j in sorted(set(juris_arr.tolist())):
    pct_block(lengths[juris_arr == j], j)

print("\n--- Paliers de contexte LLM (en tokens, /3.0) ---")
tok = lengths / CHARS_PER_TOK
total = len(tok)
for thr in [4000, 8000, 16000, 32000, 64000, 128000, 256000]:
    n = int((tok > thr).sum())
    print(f"  > {thr:>7,} tok : {n:>9,} JP ({100*n/total:5.2f} %)")

print("\n--- Idem mais /4.0 (conservateur) ---")
tok4 = lengths / 4.0
for thr in [4000, 8000, 16000, 32000, 64000, 128000, 256000]:
    n = int((tok4 > thr).sum())
    print(f"  > {thr:>7,} tok : {n:>9,} JP ({100*n/total:5.2f} %)")
