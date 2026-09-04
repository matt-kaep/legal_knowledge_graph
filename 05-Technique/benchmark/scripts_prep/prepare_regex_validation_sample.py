"""Prépare l'échantillon de 100 arrêts pour la validation du regex d'articles.

À exécuter **en local**, là où les données enrichies sont disponibles.
Produit un petit dossier uploadable sur le cluster (~2-5 Mo) contenant :

    cluster_data/regex_validation/
    ├── sample_100.jsonl      (100 records, 1 par ligne)
    └── manifest.json         (stats + seed pour reproductibilité)

Usage
-----
$ python prepare_regex_validation_sample.py

Puis upload du dossier `cluster_data/regex_validation/` vers le cluster,
et faire pointer le notebook dessus (cf. `PRESAMPLED_PATH` dans le
notebook de validation).
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import Counter
from pathlib import Path

ROOT              = Path(__file__).parent
DATA_ENRICHED_DIR = ROOT / "database-judilibre-enrichie"
OUTPUT_DIR        = ROOT / "cluster_data" / "regex_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Paramètres (identiques au notebook) ────────────────────────────────
N_CC         = 50
N_CA         = 25
N_TJ         = 25
SEED         = 42
MIN_TEXT_LEN = 500

DATASET_PATHS = {
    "CC": DATA_ENRICHED_DIR / "Cour de cassation",
    "CA": DATA_ENRICHED_DIR / "Cours d'appel",
    "TJ": DATA_ENRICHED_DIR / "Tribunal judiciaire",
}


def reservoir_sample(path: Path, k: int, min_text_len: int, seed: int) -> list[dict]:
    """Reservoir sampling sur JSONL — même algo que le notebook."""
    rng = random.Random(seed)
    pool: list[dict] = []
    seen_valid = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("text") or len(rec["text"]) < min_text_len:
                continue
            seen_valid += 1
            if len(pool) < k:
                pool.append(rec)
            else:
                j = rng.randint(0, seen_valid - 1)
                if j < k:
                    pool[j] = rec
    return pool


def main():
    for k, p in DATASET_PATHS.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} introuvable — ajuster DATA_ENRICHED_DIR")

    print(f"Input  : {DATA_ENRICHED_DIR}")
    print(f"Output : {OUTPUT_DIR}")
    print(f"Sampling : {N_CC} CC + {N_CA} CA + {N_TJ} TJ  (seed={SEED})")
    print()

    t0 = time.time()
    sample_cc = reservoir_sample(DATASET_PATHS["CC"], N_CC, MIN_TEXT_LEN, SEED + 1)
    print(f"  CC : {len(sample_cc):>3}  ({time.time()-t0:.1f}s)")
    sample_ca = reservoir_sample(DATASET_PATHS["CA"], N_CA, MIN_TEXT_LEN, SEED + 2)
    print(f"  CA : {len(sample_ca):>3}  ({time.time()-t0:.1f}s cumul.)")
    sample_tj = reservoir_sample(DATASET_PATHS["TJ"], N_TJ, MIN_TEXT_LEN, SEED + 3)
    print(f"  TJ : {len(sample_tj):>3}  ({time.time()-t0:.1f}s cumul.)")

    # Dump JSONL avec juridiction en colonne additionnelle
    out_jsonl = OUTPUT_DIR / "sample_100.jsonl"
    stats_per_jur = {}
    total_bytes = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for jur, recs in [("CC", sample_cc), ("CA", sample_ca), ("TJ", sample_tj)]:
            text_lens = []
            n_with_pairs = 0
            n_pairs_total = 0
            for rec in recs:
                rec_out = dict(rec)
                rec_out["_jurisdiction"] = jur  # tag ajouté pour le notebook
                line = json.dumps(rec_out, ensure_ascii=False)
                f.write(line + "\n")
                total_bytes += len(line) + 1
                text_lens.append(len(rec.get("text") or ""))
                if rec.get("code_article_pairs"):
                    n_with_pairs += 1
                n_pairs_total += len(rec.get("code_article_pairs") or [])
            stats_per_jur[jur] = {
                "n": len(recs),
                "text_len_min":    min(text_lens) if text_lens else 0,
                "text_len_median": int(statistics.median(text_lens)) if text_lens else 0,
                "text_len_max":    max(text_lens) if text_lens else 0,
                "with_code_article_pairs": n_with_pairs,
                "code_article_pairs_total": n_pairs_total,
            }

    # Manifest
    manifest = {
        "version": "1",
        "seed": SEED,
        "min_text_len": MIN_TEXT_LEN,
        "n_per_jurisdiction": {"CC": N_CC, "CA": N_CA, "TJ": N_TJ},
        "n_total": N_CC + N_CA + N_TJ,
        "source_dir": str(DATA_ENRICHED_DIR.resolve()),
        "stats": stats_per_jur,
        "file_size_mb": round(total_bytes / 1e6, 2),
    }
    with open(OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print()
    print("─" * 60)
    print(f"  sample_100.jsonl : {total_bytes/1e6:.2f} Mo")
    print(f"  manifest.json    : stats par juridiction")
    print("─" * 60)
    print()
    for jur, st in stats_per_jur.items():
        print(f"  {jur} — {st['n']} arrêts  ·  text_len med={st['text_len_median']:,}  "
              f"·  {st['with_code_article_pairs']}/{st['n']} avec pairs "
              f"({st['code_article_pairs_total']} paires)")
    print()
    print(f"✅ Prêt à uploader : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
