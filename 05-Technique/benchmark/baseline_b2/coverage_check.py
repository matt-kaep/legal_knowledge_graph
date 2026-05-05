#!/usr/bin/env python3
"""A.1 — Couverture GT : les pourvois attendus par les rubrics existent-ils
dans nos 118k JP pénales ?

Diagnostic critique avant toute analyse de scoring : si la GT n'est pas dans
le corpus, S̄_jp est plafonné par construction quel que soit le retrieval.

Pour chaque question pénale :
  1. Extraire les pourvois GT via la regex du scorer (format CC XX-XX.XXX)
  2. Vérifier leur présence dans jp_index_penal.parquet (champ `number`)
  3. Rapporter par strate (core / expected / expert)

Sortie :
  Results/coverage_gt.csv         tableau global
  Results/coverage_gt_detail.txt  détail par question avec FN identifiés
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pyarrow.parquet as pq

HERE     = Path(__file__).parent.resolve()
PARQUET  = HERE / "jp_index_penal.parquet"
RUBRICS  = HERE / "rubrics_penal.json"
RESULTS  = HERE / "Results"
RESULTS.mkdir(exist_ok=True)

CSV_OUT      = RESULTS / "coverage_gt.csv"
DETAIL_OUT   = RESULTS / "coverage_gt_detail.txt"

POURVOI_RE = re.compile(r"\b(\d{2})[\s\-]*(\d{2})[.\-]?(\d{3})\b")


def extract_pourvoi(text: str) -> str | None:
    if not text:
        return None
    m = POURVOI_RE.search(text)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}.{m.group(3)}"


def main() -> None:
    print(f"Lecture corpus pénal…", flush=True)
    df = pq.read_table(PARQUET, columns=["id", "number", "juris"]).to_pandas()
    numbers = set(df["number"].dropna())
    n_corpus = len(df)
    print(f"  → {n_corpus} JP, {len(numbers)} numéros uniques\n", flush=True)

    questions = json.loads(RUBRICS.read_text(encoding="utf-8"))["questions"]
    print(f"Questions : {len(questions)}\n", flush=True)

    rows = []
    detail_lines = []
    detail_lines.append(f"# Couverture GT — {n_corpus} JP pénales\n\n")

    totals = {"strate": {"core": [0, 0], "expected": [0, 0], "expert": [0, 0]},
              "raw_jp": [0, 0]}

    for q in questions:
        qid = q["id"]
        rubric = q.get("rubric") or {}
        detail_lines.append(f"\n{'='*78}")
        detail_lines.append(f"  {qid}  [{q.get('branche')}]")
        detail_lines.append(f"  {q.get('specialisation','')[:90]}")
        detail_lines.append(f"{'='*78}")

        per_q_summary = {}

        for strate in ("core", "expected", "expert"):
            items = rubric.get(strate) or []
            jp_refs_raw = [item.get("linked_jp", "") for item in items if item.get("linked_jp")]
            extracted = [(ref, extract_pourvoi(ref)) for ref in jp_refs_raw]

            n_gt = len([_ for _, p in extracted if p])
            n_unparsable = len([_ for _, p in extracted if not p and _])
            n_in_corpus = sum(1 for _, p in extracted if p and p in numbers)
            n_missing   = n_gt - n_in_corpus

            per_q_summary[strate] = (n_gt, n_in_corpus, n_unparsable)
            totals["strate"][strate][0] += n_gt
            totals["strate"][strate][1] += n_in_corpus

            if n_gt + n_unparsable > 0:
                detail_lines.append(f"\n[strate {strate}]  "
                                    f"GT extraits={n_gt}  trouvés={n_in_corpus}  "
                                    f"non parsables={n_unparsable}")
                for ref, pourvoi in extracted:
                    if pourvoi is None:
                        detail_lines.append(f"   ⚠ NON-PARSABLE   « {ref[:80]} »")
                    elif pourvoi in numbers:
                        # Récupère la juris
                        juris = df[df["number"] == pourvoi]["juris"].iloc[0]
                        detail_lines.append(f"   ✓ trouvé  [{juris}]   {pourvoi}   ← « {ref[:80]} »")
                    else:
                        detail_lines.append(f"   ✗ ABSENT          {pourvoi}   ← « {ref[:80]} »")

        rows.append({
            "qid": qid,
            "branche": q.get("branche"),
            "core_gt":          per_q_summary["core"][0],
            "core_in_corpus":   per_q_summary["core"][1],
            "core_coverage":    round(per_q_summary["core"][1] / max(per_q_summary["core"][0], 1), 2),
            "expected_gt":      per_q_summary["expected"][0],
            "expected_in":      per_q_summary["expected"][1],
            "expected_coverage":round(per_q_summary["expected"][1] / max(per_q_summary["expected"][0], 1), 2),
            "expert_gt":        per_q_summary["expert"][0],
            "expert_in":        per_q_summary["expert"][1],
            "expert_coverage":  round(per_q_summary["expert"][1] / max(per_q_summary["expert"][0], 1), 2),
        })

    # CSV global
    with open(CSV_OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Récap totaux
    detail_lines.append(f"\n\n{'═'*78}")
    detail_lines.append("  TOTAUX")
    detail_lines.append(f"{'═'*78}")
    for strate in ("core", "expected", "expert"):
        gt, found = totals["strate"][strate]
        cov = found / max(gt, 1)
        detail_lines.append(f"  {strate:<10s}  GT={gt:3d}  trouvés={found:3d}  "
                            f"couverture={cov*100:5.1f}%")

    DETAIL_OUT.write_text("\n".join(detail_lines), encoding="utf-8")

    # Affichage console
    print(f"{'═'*78}")
    print(f"COUVERTURE GLOBALE par strate")
    print(f"{'═'*78}")
    for strate in ("core", "expected", "expert"):
        gt, found = totals["strate"][strate]
        cov = found / max(gt, 1)
        print(f"  {strate:<10s}  {found:3d} / {gt:3d}  ({cov*100:5.1f}%)")

    print(f"\nDétail par question → {DETAIL_OUT}")
    print(f"CSV → {CSV_OUT}")

    try:
        import pandas as pd
        df_out = pd.read_csv(CSV_OUT)
        print("\n" + df_out.to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
