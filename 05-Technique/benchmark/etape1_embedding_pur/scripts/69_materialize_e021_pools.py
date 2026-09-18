"""Materialize one real K_in=20 JP candidate pool for E021.

The E017 decision cards cover only previously judged decisions. E021 may
need text for additional retrieved decisions, so ``--summaries`` provides
the complete JP summary source used as an explicit fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_text(card: dict[str, Any]) -> str:
    fields = [
        ("solution", card.get("solution_resume")),
        ("grounds", card.get("fondements_retenus")),
        ("lawyer_summary", card.get("synthese_pour_avocat")),
    ]
    parts = [f"{label}: {value}" for label, value in fields if isinstance(value, str) and value.strip()]
    if not parts:
        raise ValueError("decision card has no allowed text field")
    return "\n".join(parts)


def load_summary_texts(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    summaries = pd.read_parquet(path)
    required = {"jp_id", "synthese"}
    missing = required - set(summaries.columns)
    if missing:
        raise ValueError(f"summary file missing columns: {sorted(missing)}")
    summaries = summaries[["jp_id", "synthese"]].copy()
    summaries["jp_id"] = summaries["jp_id"].astype(str)
    summaries["synthese"] = summaries["synthese"].fillna("").astype(str)
    if (summaries["synthese"].str.strip() == "").any():
        raise ValueError("summary file contains empty synthese values")
    conflicting = summaries.groupby("jp_id")["synthese"].nunique()
    if (conflicting > 1).any():
        raise ValueError("summary file contains conflicting synthese values for one jp_id")
    summaries = summaries.drop_duplicates(subset=["jp_id"], keep="first")
    return dict(zip(summaries["jp_id"], summaries["synthese"], strict=True))


def materialize_pool(
    ranking_path: Path,
    decision_cards_path: Path,
    output_path: Path,
    family: str,
    method: str | None = None,
    k_in: int = 20,
    summaries_path: Path | None = None,
) -> int:
    rankings = pd.read_parquet(ranking_path)
    required = {"qid", "method", "modality", "rank", "item_id"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"ranking file missing columns: {sorted(missing)}")
    rankings = rankings[rankings["modality"].astype(str).str.lower() == "jp"].copy()
    methods = sorted(str(value) for value in rankings["method"].dropna().unique())
    if method is None:
        if len(methods) != 1:
            raise ValueError(f"method must be explicit; found {methods}")
        method = methods[0]
    rankings = rankings[rankings["method"].astype(str) == method].copy()
    rankings["qid"] = rankings["qid"].astype(str)
    rankings["item_id"] = rankings["item_id"].astype(str)
    rankings["rank"] = rankings["rank"].astype(int)
    cards = json.loads(decision_cards_path.read_text(encoding="utf-8"))
    summaries = load_summary_texts(summaries_path)
    decision_cards_sha256 = sha256_file(decision_cards_path)
    summaries_sha256 = sha256_file(summaries_path) if summaries_path else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for qid, group in rankings.sort_values(["qid", "rank"], kind="stable").groupby("qid", sort=True):
            unique_rows = []
            seen_ids = set()
            for row in group.itertuples(index=False):
                if row.item_id in seen_ids:
                    continue
                seen_ids.add(row.item_id)
                unique_rows.append(row)
                if len(unique_rows) == k_in:
                    break
            top = pd.DataFrame(unique_rows)
            ranks = top["rank"].tolist()
            ids = top["item_id"].tolist()
            if len(top) != k_in:
                raise ValueError(f"{family}/{qid}: fewer than {k_in} unique candidates, got source ranks {ranks}")
            candidates = []
            for rank, item_id in zip(ranks, ids):
                card = cards.get(item_id)
                if card is not None:
                    text = candidate_text(card)
                else:
                    text = summaries.get(item_id)
                    if text is None:
                        raise ValueError(
                            f"missing candidate text for {item_id}; provide --summaries for the complete source"
                        )
                candidates.append(
                    {
                        "item_id": item_id,
                        "text": text,
                        "source_rank": rank,
                    }
                )
            output.write(
                json.dumps(
                    {
                        "qid": qid,
                        "modality": "jp",
                        "family": family,
                        "method": method,
                        "source_ranking": str(ranking_path),
                        "source_ranking_sha256": sha256_file(ranking_path),
                        "source_decision_cards_sha256": decision_cards_sha256,
                        "source_summaries_sha256": summaries_sha256,
                        "source_texts_sha256": summaries_sha256 or decision_cards_sha256,
                        "source_rows_considered": int(ranks[-1]),
                        "duplicate_candidates_skipped": int(ranks[-1] - k_in),
                        "candidates": candidates,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--decision-cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--method")
    parser.add_argument("--k-in", type=int, default=20)
    parser.add_argument("--summaries", type=Path, help="Complete JP summary parquet used for missing decision cards")
    args = parser.parse_args(argv)
    print(json.dumps({"questions": materialize_pool(
        args.ranking,
        args.decision_cards,
        args.output,
        args.family,
        args.method,
        args.k_in,
        args.summaries,
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
