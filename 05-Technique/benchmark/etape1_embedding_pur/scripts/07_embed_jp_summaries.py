"""CLI : embed les synthèses des JP pénales avec BGE-M3.

Input  : data/jp_summaries_penal.parquet (116k synthèses, p99=801 chars)
Output : data/emb_jp_synthese.npy (N, 1024) fp32 L2-normalisé, aligné sur
         data/jp_summary_order.npy

Aucun chunking attendu (p99 ≈ 160 tokens, max ~240). Donc cap explicite à
BATCH_MAX_LEN reste dispatché à embed_corpus.
"""
from __future__ import annotations
import argparse
import sys
import time
import numpy as np
import pandas as pd
from etape1 import config
from etape1.embed import embed_corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=32,
                    help="Synthèses courtes → batch plus grand OK (32 vs 8 articles).")
    ap.add_argument("--smoke", action="store_true",
                    help="Run sur 100 synthèses pour valider.")
    args = ap.parse_args()

    df = pd.read_parquet(config.JP_SUMMARIES_PARQUET)
    print(f"Synthèses à embedder : {len(df):,}")
    print(f"  longueur chars : p50={df['len_chars'].median():.0f} "
          f"p90={df['len_chars'].quantile(0.9):.0f} "
          f"p99={df['len_chars'].quantile(0.99):.0f} "
          f"max={df['len_chars'].max()}")

    texts = df["synthese"].tolist()
    if args.smoke:
        texts = texts[:100]
        print("→ SMOKE 100")

    t0 = time.time()
    emb = embed_corpus(texts, device=args.device, batch=args.batch)
    assert emb.shape == (len(texts), config.EMB_DIM), emb.shape
    assert not np.isnan(emb).any(), "NaN détectés"

    out = (config.DATA / "emb_jp_synthese.smoke.npy") if args.smoke else config.EMB_JP_SYNTHESE
    np.save(out, emb)
    elapsed = time.time() - t0

    print(f"\n✓ {out} ({len(texts):,} vecteurs, {elapsed/60:.1f} min)")
    print("Alignement :")
    print("  emb_jp_synthese.npy  ⟂  jp_summary_order.npy  ⟂  jp_ids[jp_summary_to_graphrow]")
    print("  Compatible avec G[jp_summary_to_graphrow] pour back-edge.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
