from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph")
BENCH_ROOT = REPO / "05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench"
OFFICIAL_TRAIN_SPLIT = "train_augmented_retrievable_strict"
OFFICIAL_N_FOLDS = 5
SHARED_PROTOCOL_DIRNAME = "_protocol"


def resolve_graph_bench_dir(graph_version: str, split: str) -> Path:
    graph_dir = BENCH_ROOT / graph_version / split
    if graph_dir.exists():
        return graph_dir
    legacy_dir = BENCH_ROOT / split
    if legacy_dir.exists():
        return legacy_dir
    return graph_dir


def resolve_official_train_bench_dir() -> Path:
    return BENCH_ROOT / OFFICIAL_TRAIN_SPLIT


def resolve_shared_protocol_dir(split: str = OFFICIAL_TRAIN_SPLIT) -> Path:
    return BENCH_ROOT / SHARED_PROTOCOL_DIRNAME / split


def resolve_shared_fold_paths(split: str = OFFICIAL_TRAIN_SPLIT) -> tuple[Path, Path]:
    protocol_dir = resolve_shared_protocol_dir(split)
    return protocol_dir / "fold_assignments.csv", protocol_dir / "fold_assignments_meta.json"


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
