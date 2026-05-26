"""CLI : produit emb_articles.npy + emb_jp.npy + artefacts de linkage."""
from __future__ import annotations
import argparse
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from etape1 import config
from etape1.linkage import build_articles_linkage, build_jp_linkage
from etape1.embed import embed_corpus


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--smoke", action="store_true",
                    help="Run sur 10 articles + 10 JP pour valider le pipeline")
    args = ap.parse_args()

    # 1. Linkage articles
    z = np.load(config.GRAPH_NPZ, allow_pickle=True)
    arts_df = pd.read_parquet(config.ARTICLES_PARQUET)
    resolved_pks = set(arts_df["pair_key"])
    art_order, p2col = build_articles_linkage(
        z["article_ids"], z["article_codes"], resolved_pks, set(config.PENAL_CODES.keys()))
    np.save(config.ARTICLES_ORDER, art_order)
    np.save(config.PAIRKEY_TO_GRAPHCOL, p2col)
    print(f"Articles à embedder : {len(art_order)}")

    # Texte des articles dans l'ordre de art_order
    text_by_pk = dict(zip(arts_df["pair_key"], arts_df["texte"]))
    art_texts = [text_by_pk[pk] for pk in art_order]

    # 2. Linkage JP (bundle pénal n'a pas de summary → on prend text, juris=CC)
    jp_df = pq.read_table(config.JP_INDEX, columns=["id", "juris", "text"]).to_pandas()
    jp_order, j2row = build_jp_linkage(z["jp_ids"], jp_df, text_col="text", juris_filter="CC")
    np.save(config.JP_ORDER, jp_order)
    np.save(config.JP_TO_GRAPHROW, j2row)
    print(f"JP à embedder (CC, text) : {len(jp_order)}")

    text_by_id = dict(zip(jp_df["id"], jp_df["text"]))
    jp_texts = [text_by_id[jpid] for jpid in jp_order]

    if args.smoke:
        art_texts = art_texts[:10]
        jp_texts = jp_texts[:10]
        print("→ SMOKE MODE 10+10")

    # 3. Embedding
    print(f"Embedding articles ({len(art_texts)})…")
    emb_arts = embed_corpus(art_texts, device=args.device, batch=args.batch)
    assert emb_arts.shape == (len(art_texts), config.EMB_DIM), emb_arts.shape
    assert not np.isnan(emb_arts).any(), "NaN dans emb_articles"
    out_arts = (config.DATA / "emb_articles.smoke.npy") if args.smoke else config.EMB_ARTICLES
    np.save(out_arts, emb_arts)

    print(f"Embedding JP ({len(jp_texts)})…")
    emb_jp = embed_corpus(jp_texts, device=args.device, batch=args.batch)
    assert emb_jp.shape == (len(jp_texts), config.EMB_DIM), emb_jp.shape
    assert not np.isnan(emb_jp).any(), "NaN dans emb_jp"
    out_jp = (config.DATA / "emb_jp.smoke.npy") if args.smoke else config.EMB_JP
    np.save(out_jp, emb_jp)

    print("✓ Embedding terminé. Aligned :")
    print("   emb_articles.npy   ⟂   articles_order.npy   ⟂   article_ids[pairkey_to_graphcol]")
    print("   emb_jp.npy         ⟂   jp_order.npy         ⟂   jp_ids[jp_to_graphrow]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
