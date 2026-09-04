#!/usr/bin/env python3
"""Construit l'index jp_id → {number, juris, text} depuis les JSONL judilibre-v5.

Pour CC : préfère le summary (95% dispo, 471 chars médiane) car dense et propre.
Pour CA/TJ : text intégral seulement (summary absent à 100%).

Écriture par batch de BATCH_SIZE lignes via pyarrow pour éviter l'OOM.
Produit baseline_b2/jp_index.parquet avec colonnes : id, number, juris, text.
"""
from __future__ import annotations
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).parent.resolve()
DB_DIR = HERE.parent / "database-judilibre-v5"
OUT = HERE / "jp_index.parquet"

BATCH_SIZE = 50_000

SCHEMA = pa.schema([
    pa.field("id",     pa.string()),
    pa.field("number", pa.string()),
    pa.field("juris",  pa.string()),
    pa.field("text",   pa.string()),
])


def best_text(d: dict, juris: str) -> str:
    if juris == "CC":
        s = (d.get("summary") or "").strip()
        if len(s) > 100:
            return s
    return (d.get("text") or "").strip()


def main() -> None:
    writer = pq.ParquetWriter(OUT, SCHEMA, compression="snappy")
    total = 0

    for fname, juris in [("cc.jsonl", "CC"), ("ca.jsonl", "CA"), ("tj.jsonl", "TJ")]:
        path = DB_DIR / fname
        print(f"  lecture {fname}…", flush=True)
        ids, numbers, jurises, texts = [], [], [], []

        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                uid = d.get("id")
                if not uid:
                    continue
                ids.append(uid)
                numbers.append(d.get("number") or "")
                jurises.append(juris)
                texts.append(best_text(d, juris))

                if len(ids) >= BATCH_SIZE:
                    batch = pa.table(
                        {"id": ids, "number": numbers, "juris": jurises, "text": texts},
                        schema=SCHEMA,
                    )
                    writer.write_table(batch)
                    total += len(ids)
                    print(f"    {total} lignes écrites…", flush=True)
                    ids, numbers, jurises, texts = [], [], [], []

        if ids:
            batch = pa.table(
                {"id": ids, "number": numbers, "juris": jurises, "text": texts},
                schema=SCHEMA,
            )
            writer.write_table(batch)
            total += len(ids)
            print(f"    {total} lignes écrites (fin {fname})", flush=True)

    writer.close()
    size_mb = OUT.stat().st_size / 1e6
    print(f"✓ {total} JP → {OUT} ({size_mb:.0f} MB)", flush=True)

    # Stats rapides par juridiction
    import pyarrow.dataset as ds
    dataset = pq.read_table(OUT, columns=["juris", "text"])
    import collections
    counts: dict[str, list[int]] = collections.defaultdict(list)
    for juris, text in zip(dataset["juris"].to_pylist(), dataset["text"].to_pylist()):
        counts[juris].append(len(text or ""))
    for j in sorted(counts):
        lens = sorted(counts[j])
        n = len(lens)
        print(f"  {j}: {n} JP, text médiane={lens[n//2]} chars")


if __name__ == "__main__":
    main()
