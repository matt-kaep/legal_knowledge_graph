from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
BENCH_ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"


def resolve_graph_bench_dir(graph_version: str, split: str) -> Path:
    return BENCH_ROOT / graph_version / split


def load_bench_questions(bench_dir: Path) -> list[dict]:
    payload = json.loads((bench_dir / "bench_global.json").read_text())
    return list(payload["questions"])


def metric_rank_tuple(row: dict, modality: str) -> tuple[float, float, float, float, float]:
    is_jp = modality.lower() == "jp"
    suffix = "" if is_jp else "_strict"
    if f"hit{suffix}" not in row:
        suffix = ""
    return (
        float(row[f"hit{suffix}"]),
        float(row[f"ndcg{suffix}"]),
        float(row[f"mrr{suffix}"]),
        float(row[f"m1{suffix}"]),
        float(row[f"m2{suffix}"]),
    )
