"""CLI : embedde TOUS les articles du graphe (tous codes) + JP.

Variante full-corpus de 03_embed.py. Produit :
  - data/emb_articles_all.npy        (N_all × 1024)
  - data/articles_order_all.npy      (pair_keys ordonnés ⟂ emb)
  - data/pairkey_to_graphcol_all.npy (idx emb → colonne graphe)

Réutilise l'embedding JP existant (config.EMB_JP, identique entre pénal et full).
Paramètres validés : --device mps --batch 8 (BGE-M3, MAX_SEQ via config.BATCH_MAX_LEN).
"""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
from etape1 import config
from etape1.linkage import build_articles_linkage
from etape1.embed import embed_corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true",
                    help="Run sur 20 articles pour valider le pipeline")
    args = ap.parse_args()

    # 1. Linkage articles (full corpus : on autorise tous les codes mappés)
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    arts_df = pd.read_parquet(config.ARTICLES_PARQUET_ALL)
    resolved_pks = set(arts_df["pair_key"])
    all_codes_set = {k for k, v in config.ALL_CODES.items() if v is not None}
    art_order, p2col = build_articles_linkage(
        z["article_ids"], z["article_codes"], resolved_pks, all_codes_set)
    np.save(config.ARTICLES_ORDER_ALL, art_order)
    np.save(config.PAIRKEY_TO_GRAPHCOL_ALL, p2col)
    print(f"Articles à embedder (full corpus) : {len(art_order)}")

    text_by_pk = dict(zip(arts_df["pair_key"], arts_df["texte"]))
    art_texts = [text_by_pk[pk] for pk in art_order]

    if args.smoke:
        art_texts = art_texts[:20]
        art_order = art_order[:20]
        p2col = p2col[:20]
        print("→ SMOKE MODE 20")

    # 2. Embedding articles
    print(f"Embedding articles ({len(art_texts)}) sur {args.device}, batch={args.batch}…")
    emb_arts = embed_corpus(art_texts, device=args.device, batch=args.batch)
    assert emb_arts.shape == (len(art_texts), config.EMB_DIM), emb_arts.shape
    assert not np.isnan(emb_arts).any(), "NaN dans emb_articles_all"
    out = (config.DATA / "emb_articles_all.smoke.npy") if args.smoke else config.EMB_ARTICLES_ALL
    np.save(out, emb_arts)
    print(f"✓ {out}")
    print("Aligned : emb_articles_all.npy ⟂ articles_order_all.npy ⟂ article_ids[pairkey_to_graphcol_all]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
