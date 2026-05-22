"""Étape 1 — chemins, constantes, mapping code_slug → nom LEGI officiel."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

# Inputs read-only (relatifs depuis ROOT)
BENCH = ROOT.parents[0]  # 05-Technique/benchmark/
BUNDLE = BENCH / "baseline_b2" / "penal_bundle"
GRAPH_NPZ = BUNDLE / "graph_penal.npz"
JP_INDEX = BUNDLE / "jp_index_penal.parquet"
RUBRICS = BUNDLE / "rubrics_penal.json"

# LEGI SQLite (produit par Task 3)
LEGI_DIR = DATA / "legi"
LEGI_DIR.mkdir(exist_ok=True)
LEGI_SQLITE = LEGI_DIR / "legi.sqlite"

# Sorties
ARTICLES_PARQUET = DATA / "articles_penal.parquet"
ARTICLES_COVERAGE = DATA / "articles_coverage.json"
ARTICLES_ORDER = DATA / "articles_order.npy"
PAIRKEY_TO_GRAPHCOL = DATA / "pairkey_to_graphcol.npy"
JP_ORDER = DATA / "jp_order.npy"
JP_TO_GRAPHROW = DATA / "jp_to_graphrow.npy"
EMB_ARTICLES = DATA / "emb_articles.npy"
EMB_JP = DATA / "emb_jp.npy"
TOKEN_STATS = DATA / "token_stats.json"
RECALL_CURVES = DATA / "recall_curves.csv"
RECALL_KSTAR = DATA / "recall_kstar.json"

# Modèle
MODEL_ID = "BAAI/bge-m3"
EMB_DIM = 1024
MAX_CTX = 8192

# 4 codes pénaux : code_slug → nom officiel LEGI
PENAL_CODES: dict[str, str] = {
    "code_penal":                              "Code pénal",
    "code_de_procedure_penale":                "Code de procédure pénale",
    "code_de_la_route":                        "Code de la route",
    "code_de_la_justice_penale_des_mineurs":   "Code de la justice pénale des mineurs",
}

# Eval
KS = [1, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
KSTAR_THRESHOLD = 0.5
