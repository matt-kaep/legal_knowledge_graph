#!/usr/bin/env python3
"""[À EXÉCUTER SUR MAC] Prépare un bundle de données pénales pour le cluster.

Filtre le corpus complet (1,07 M JP) aux ~118k JP qui citent au moins un article
d'un code pénal (Code pénal, procédure pénale, justice mineurs, route).

Produit un dossier `penal_bundle/` autonome à transférer sur le cluster :
  penal_bundle/
    ├── jp_index_penal.parquet   # ~118k lignes : id, number, juris, text
    ├── graph_penal.npz          # sous-matrice CSR (penal_jp × all_articles)
    ├── rubrics_penal.json       # 8 questions pénales fusionnées
    ├── eval_rubric.py           # scorer (copié depuis crfpa_benchmark/)
    ├── run_cluster.py           # script all-in-one à lancer sur le cluster
    ├── requirements.txt
    └── README.md
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
from scipy.sparse import csr_matrix, save_npz

HERE      = Path(__file__).parent.resolve()
BENCH_DIR = HERE.parent
GRAPH_NPZ = BENCH_DIR / "graphs_v5" / "graph_bipartite.npz"
PARQUET   = HERE / "jp_index.parquet"
RUBRICS   = BENCH_DIR / "data" / "rubrics"
EVAL_PY   = BENCH_DIR / "crfpa_benchmark" / "eval_rubric.py"

OUT       = HERE / "penal_bundle"

PENAL_CODES = {
    "code_penal",
    "code_de_procedure_penale",
    "code_de_la_route",
    "code_de_la_justice_penale_des_mineurs",
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    print(f"Bundle sortie : {OUT}", flush=True)

    # ─── 1. Charger le graphe et identifier les JP pénales ───────────────
    print("\n[1/5] Identification des JP pénales via citations…", flush=True)
    g = np.load(GRAPH_NPZ, allow_pickle=True)
    mat = csr_matrix(
        (g["data"], g["indices"], g["indptr"]),
        shape=tuple(g["shape"]),
    )
    article_codes = np.array(g["article_codes"])
    jp_ids        = np.array(g["jp_ids"])
    jp_juris      = np.array(g["jp_juris"])

    penal_col_mask = np.isin(article_codes, list(PENAL_CODES))
    penal_cols     = np.where(penal_col_mask)[0]
    print(f"  Articles pénaux : {len(penal_cols)} colonnes")

    penal_subset = mat[:, penal_cols]
    jp_penal_mask = (penal_subset.sum(axis=1) > 0).A1
    penal_jp_idx  = np.where(jp_penal_mask)[0]
    penal_jp_ids  = jp_ids[penal_jp_idx]
    print(f"  JP pénales : {len(penal_jp_ids)}")
    print(f"  Répartition : CC={(jp_juris[penal_jp_idx]=='CC').sum()}, "
          f"CA={(jp_juris[penal_jp_idx]=='CA').sum()}, "
          f"TJ={(jp_juris[penal_jp_idx]=='TJ').sum()}")

    # ─── 2. Extraire le sous-graphe (penal JP × tous articles) ──────────
    print("\n[2/5] Extraction du sous-graphe…", flush=True)
    sub_mat = mat[penal_jp_idx, :].tocsr()
    print(f"  Sous-matrice : {sub_mat.shape}, nnz={sub_mat.nnz}")

    np.savez_compressed(
        OUT / "graph_penal.npz",
        data=sub_mat.data,
        indices=sub_mat.indices,
        indptr=sub_mat.indptr,
        shape=np.array(sub_mat.shape, dtype=np.int64),
        jp_ids=penal_jp_ids,
        jp_juris=jp_juris[penal_jp_idx],
        article_ids=g["article_ids"],
        article_codes=article_codes,
    )
    print(f"  ✓ graph_penal.npz écrit")

    # ─── 3. Filtrer le parquet aux 118k JP pénales ──────────────────────
    print("\n[3/5] Filtrage du parquet…", flush=True)
    full = pq.read_table(PARQUET)  # tient en RAM si on a 8 GB libres
    penal_set = set(penal_jp_ids)
    mask = np.array([uid in penal_set for uid in full["id"].to_pylist()])
    sub_table = full.filter(pa.array(mask))
    pq.write_table(sub_table, OUT / "jp_index_penal.parquet", compression="snappy")
    n_kept = len(sub_table)
    size_mb = (OUT / "jp_index_penal.parquet").stat().st_size / 1e6
    print(f"  ✓ jp_index_penal.parquet : {n_kept} JP, {size_mb:.0f} MB")

    # ─── 4. Fusionner les 8 questions pénales ───────────────────────────
    print("\n[4/5] Fusion des rubrics pénales…", flush=True)
    merged = {"questions": []}
    for fname in ["cnb-penal-2025-consolidated.json",
                  "cnb-procedure-penale-2025-consolidated.json"]:
        data = json.loads((RUBRICS / fname).read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            q["_source_file"] = fname
            merged["questions"].append(q)
    (OUT / "rubrics_penal.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  ✓ {len(merged['questions'])} questions pénales fusionnées")

    # ─── 5. Copier eval_rubric.py + écrire scripts cluster ──────────────
    print("\n[5/5] Copie scripts + requirements…", flush=True)
    shutil.copy(EVAL_PY, OUT / "eval_rubric.py")

    # Copier les scripts cluster (garde les noms explicites côté bundle)
    for src_name, dst_name in [
        ("run_cluster.py",          "run_cluster.py"),
        ("requirements_cluster.txt", "requirements.txt"),
        ("README_CLUSTER.md",        "README.md"),
    ]:
        src = HERE / src_name
        if src.exists():
            shutil.copy(src, OUT / dst_name)

    # Tar.gz
    print("\nCréation de l'archive penal_bundle.tar.gz…", flush=True)
    import tarfile
    archive = HERE / "penal_bundle.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(OUT, arcname="penal_bundle")
    print(f"✓ {archive} ({archive.stat().st_size/1e6:.0f} MB)")

    print("\nÀ transférer sur le cluster :")
    print(f"  scp {archive} cluster:~/")
    print(f"  ssh cluster 'tar xzf penal_bundle.tar.gz && cd penal_bundle && pip install -r requirements.txt && python run_cluster.py'")


if __name__ == "__main__":
    main()
