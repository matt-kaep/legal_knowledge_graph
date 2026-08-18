"""Materialize one real K_in=20 JP candidate pool for E021."""
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


def materialize_pool(
    ranking_path: Path,
    decision_cards_path: Path,
    output_path: Path,
    family: str,
    method: str | None = None,
    k_in: int = 20,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for qid, group in rankings.sort_values(["qid", "rank"], kind="stable").groupby("qid", sort=True):
            top = group.head(k_in)
            ranks = top["rank"].tolist()
            ids = top["item_id"].tolist()
            if len(top) != k_in or ranks != list(range(1, k_in + 1)):
                raise ValueError(f"{family}/{qid}: expected ranks 1..{k_in}, got {ranks}")
            if len(set(ids)) != k_in:
                raise ValueError(f"{family}/{qid}: duplicate candidate identifiers")
            candidates = []
            for rank, item_id in zip(ranks, ids):
                card = cards.get(item_id)
                if card is None:
                    raise ValueError(f"missing decision card for candidate {item_id}")
                candidates.append(
                    {
                        "item_id": item_id,
                        "text": candidate_text(card),
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
    args = parser.parse_args(argv)
    print(json.dumps({"questions": materialize_pool(
        args.ranking,
        args.decision_cards,
        args.output,
        args.family,
        args.method,
        args.k_in,
    )}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
