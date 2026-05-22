"""CLI : distribution de longueur articles + JP summaries (sous-corpus pénal)."""
from __future__ import annotations
import json
import sys
import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer
from etape1 import config
from etape1.tokenize_stats import compute_token_stats


def main() -> int:
    if not config.ARTICLES_PARQUET.exists():
        print(f"✗ {config.ARTICLES_PARQUET} absent — lancer 02_fetch_articles.py d'abord")
        return 1

    print(f"Chargement tokenizer {config.MODEL_ID}…")
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID)

    arts = pd.read_parquet(config.ARTICLES_PARQUET)
    print(f"Articles : {len(arts)}")
    s_arts = compute_token_stats(arts["texte"].tolist(), tok, max_ctx=config.MAX_CTX)
    print(f"  articles  p50={s_arts['p50']} p90={s_arts['p90']} "
          f"p99={s_arts['p99']} p100={s_arts['p100']} "
          f"over={s_arts['n_over_ctx']} ({100*s_arts['pct_over_ctx']:.2f}%)")

    jp = pq.read_table(config.JP_INDEX, columns=["id", "juris", "summary"]).to_pandas()
    jp = jp.dropna(subset=["summary"])
    jp = jp[jp["summary"].str.len() > 0]
    print(f"JP avec summary : {len(jp)}")
    s_jp = compute_token_stats(jp["summary"].tolist(), tok, max_ctx=config.MAX_CTX)
    print(f"  jp_summary p50={s_jp['p50']} p90={s_jp['p90']} "
          f"p99={s_jp['p99']} p100={s_jp['p100']} "
          f"over={s_jp['n_over_ctx']} ({100*s_jp['pct_over_ctx']:.2f}%)")

    payload = {"articles": s_arts, "jp_summary": s_jp}
    if max(s_arts["n_over_ctx"], s_jp["n_over_ctx"]) == 0:
        payload["truncation_policy"] = "none"
        payload["note"] = "p100 < max_ctx pour les deux corpus → embedding direct sans troncature."
    else:
        payload["truncation_policy"] = "chunk_meanpool_overflow_only"
        payload["note"] = ("Au moins un corpus dépasse max_ctx — chunk+mean-pool "
                            "appliqué aux seuls dépassements (cf. embed.py).")
    config.TOKEN_STATS.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"✓ {config.TOKEN_STATS} écrit (policy: {payload['truncation_policy']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
