# Baseline B2 — Embedding naïf sur le benchmark CRFPA

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implémenter la baseline B2 (embedding + graphe, sans entraînement) sur les 41 questions CRFPA et comparer le score S_retrieval au plafond LLM nu (≈ 0,095).

**Architecture:** On embed les 1,07 M JP du corpus avec un encodeur généraliste FR en découpant chaque texte en chunks de 512 tokens et en moyennant les vecteurs (mean pooling), on embed chaque question CRFPA dans le même espace, on récupère les top-K JP voisines au sens cosinus, puis on agrège les articles qu'elles citent via le graphe bipartite — sans aucun entraînement supplémentaire.

**Pourquoi chunking + mean pooling :** CC a un `summary` court (471 chars médiane) mais CA et TJ n'ont aucune metadata — seulement `text` brut de 9-10k chars médiane (~3 000 tokens). Tronquer à 512 tokens couvrirait 15% du document et manquerait les motifs et le dispositif. Le mean pooling sur les chunks donne une représentation complète du document sans perte.

**Tech Stack:** `sentence-transformers` (intfloat/multilingual-e5-base, 384 dims), `numpy`, `scipy.sparse`, scripts Python standalone (pas de notebook pour la reproductibilité).

---

## Structure des fichiers

```
05-Technique/benchmark/
├── baseline_b2/
│   ├── build_jp_index.py        # CRÉER — construit jp_id → {number, text_chunk}
│   ├── embed_corpus.py          # CRÉER — embed 1,07 M JP → jp_embeddings.npy
│   ├── query_naive.py           # CRÉER — question → top-K JP → articles → parsed_canon
│   ├── run_b2.py                # CRÉER — boucle 41 questions, scoring, CSV
│   └── results/                 # Créé par run_b2.py
│       ├── b2_k3.json
│       ├── b2_k5.json
│       ├── b2_k10.json
│       └── comparison_b2_vs_llm.csv
├── graphs_v5/
│   └── graph_bipartite.npz      # EXISTANT — matrice CSR (1 072 646 × 87 821)
├── database-judilibre-v5/
│   ├── cc.jsonl                 # EXISTANT — 553 075 arrêts
│   ├── ca.jsonl                 # EXISTANT — 430 654 arrêts
│   └── tj.jsonl                 # EXISTANT — 142 239 arrêts
└── crfpa_benchmark/
    └── eval_rubric.py           # EXISTANT — scoring parsed_canon vs rubrique
```

---

## Données clés à connaître

- **Graphe** : `graph_bipartite.npz` — CSR rows = JP (`jp_ids` = IDs internes Judilibre), cols = Articles (`article_ids` = pair_key `code:article`).
- **JSONL** : chaque arrêt a `id` (= jp_id du graphe), `number` (pourvoi), `text` (texte intégral). **Le `number` n'est PAS un identifiant unique** : 13k doublons intra-CC, formats hétérogènes en CA/TJ, 16k collisions inter-juridictions sur 50k arrêts. L'identifiant fiable est `id`.
- **Champs riches** : `summary` (95% dispo en CC, 0% en CA/TJ), `titlesAndSummaries` (quasi vide partout). CA/TJ : seulement `text`, médiane 10k chars.
- **Scoring** : `eval_rubric.py::evaluate(parsed_canon, question_obj)` attend `parsed_canon = {"articles": [{"pair_key": "..."}], "jurisprudences": [{"pourvoi": "XX-XX.XXX"}]}`. Le scorer extrait les pourvois par regex format CC (`XX-XX.XXX`) — les JP attendues dans les rubriques CRFPA sont quasi-exclusivement CC.
- **Plafond LLM** : S_retrieval moyen ≈ 0,095 sur les 7 modèles benchmarkés.

---

## Task 1 : Construire l'index JP (jp_id → number + text)

**Pourquoi :** Le graphe utilise `jp_ids` (ObjectId Judilibre). Le scoring CRFPA a besoin du `number` (pourvoi CC format `XX-XX.XXX`). L'embedding a besoin du `text` complet. L'index construit le pont entre les trois, stocké en parquet pour lecture colonne-par-colonne efficace.

**Note sur le `number`** : pas d'identifiant unique (13k doublons intra-CC, formats hétérogènes CA/TJ). On le garde pour le scoring mais on ne s'en sert pas comme clé — seul `id` est la clé.

**Files :**
- Créer : `05-Technique/benchmark/baseline_b2/build_jp_index.py`
- Produit : `05-Technique/benchmark/baseline_b2/jp_index.parquet` (~2 GB — contient le texte intégral)

- [ ] **Étape 1.1 : Écrire le script**

```python
#!/usr/bin/env python3
"""Construit l'index jp_id → {number, juris, text} depuis les JSONL judilibre-v5.

Pour CC : préfère le summary (95% dispo, 471 chars médiane) car dense et propre.
Pour CA/TJ : text intégral seulement (summary absent à 100%).

Produit baseline_b2/jp_index.parquet avec colonnes : id, number, juris, text.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent.resolve()
DB_DIR = HERE.parent / "database-judilibre-v5"
OUT = HERE / "jp_index.parquet"


def best_text(d: dict, juris: str) -> str:
    if juris == "CC":
        s = (d.get("summary") or "").strip()
        if len(s) > 100:
            return s
    return (d.get("text") or "").strip()


def main() -> None:
    rows = []
    for fname, juris in [("cc.jsonl", "CC"), ("ca.jsonl", "CA"), ("tj.jsonl", "TJ")]:
        path = DB_DIR / fname
        print(f"  lecture {fname}…")
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                d = json.loads(line)
                uid = d.get("id")
                if uid:
                    rows.append({
                        "id":     uid,
                        "number": d.get("number") or "",
                        "juris":  juris,
                        "text":   best_text(d, juris),
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT, index=False)
    size_mb = OUT.stat().st_size / 1e6
    print(f"✓ {len(df)} JP → {OUT} ({size_mb:.0f} MB)")
    print(df.groupby("juris")["text"].apply(lambda s: s.str.len().median()).rename("text_len_médiane"))


if __name__ == "__main__":
    main()
```

- [ ] **Étape 1.2 : Lancer le script**

```bash
cd /Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark
mkdir -p baseline_b2
python baseline_b2/build_jp_index.py
```

Sortie attendue :
```
  lecture cc.jsonl…
  lecture ca.jsonl…
  lecture tj.jsonl…
✓ 1125968 JP → baseline_b2/jp_index.parquet (XXXX MB)
juris
CA    10265.0
CC      471.0
TJ     9290.0
Name: text_len_médiane, dtype: float64
```

- [ ] **Étape 1.3 : Vérifier l'alignement avec le graphe**

```bash
python3 -c "
import numpy as np, pandas as pd
from pathlib import Path
base = Path('05-Technique/benchmark')
graph = np.load(base / 'graphs_v5/graph_bipartite.npz', allow_pickle=True)
df = pd.read_parquet(base / 'baseline_b2/jp_index.parquet', columns=['id'])
in_graph = set(graph['jp_ids'])
in_index = set(df['id'])
print('Dans graphe :', len(in_graph))
print('Dans index  :', len(in_index))
print('Intersection:', len(in_graph & in_index))
print('Dans graphe mais pas index (devrait être 0) :', len(in_graph - in_index))
"
```

Sortie attendue : `Intersection: 1072646`, `Dans graphe mais pas index : 0`

- [ ] **Étape 1.4 : Commit**

```bash
git add 05-Technique/benchmark/baseline_b2/build_jp_index.py
git commit -m "feat(b2): build_jp_index — jp_id → number + text (CC:summary, CA/TJ:text)"
```

---

## Task 2 : Embedder le corpus par chunking + mean pooling (embed_corpus.py)

**Pourquoi chunking + mean pooling :** multilingual-e5-base a une fenêtre de 512 tokens. Le texte médian de CA/TJ fait ~3 000 tokens. On découpe chaque document en chunks de 512 tokens avec un overlap de 64 tokens (pour ne pas couper des phrases à mi-chemin), on embède chaque chunk, puis on moyenne les vecteurs. Le vecteur résultant représente l'intégralité du document.

**Overlap 64 tokens** : évite qu'une phrase clé tronquée à la frontière de deux chunks soit mal représentée dans les deux.

**Files :**
- Créer : `05-Technique/benchmark/baseline_b2/embed_corpus.py`
- Produit :
  - `baseline_b2/jp_embeddings.npy` — matrice `(N × 384)` float32, ~1,6 GB
  - `baseline_b2/jp_order.npy` — tableau `(N,)` des jp_ids dans l'ordre des lignes

- [ ] **Étape 2.1 : Installer les dépendances**

```bash
pip install sentence-transformers
```

Vérifier :
```bash
python3 -c "from sentence_transformers import SentenceTransformer; print('OK')"
```

- [ ] **Étape 2.2 : Écrire le script**

```python
#!/usr/bin/env python3
"""Embed le corpus JP par chunking + mean pooling → jp_embeddings.npy.

Stratégie :
  - Tokeniser le texte avec le tokenizer du modèle
  - Découper en chunks de CHUNK_SIZE tokens avec OVERLAP tokens de recouvrement
  - Embedder tous les chunks en batch
  - Pour chaque document : moyenner les vecteurs de ses chunks (mean pooling)

Produit :
  jp_embeddings.npy — (N_jp × 384) float32, L2-normalisé
  jp_order.npy      — (N_jp,) jp_ids dans l'ordre des lignes
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

HERE = Path(__file__).parent.resolve()
PARQUET   = HERE / "jp_index.parquet"
OUT_EMB   = HERE / "jp_embeddings.npy"
OUT_IDS   = HERE / "jp_order.npy"

MODEL_NAME = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 512    # tokens
OVERLAP    = 64     # tokens
BATCH_SIZE = 128    # chunks par batch (réduire à 64 si OOM)
PREFIX     = "passage: "


def chunk_token_ids(token_ids: list[int], size: int, overlap: int) -> list[list[int]]:
    """Découpe une liste de token IDs en chunks avec overlap."""
    if len(token_ids) <= size:
        return [token_ids]
    chunks = []
    step = size - overlap
    for start in range(0, len(token_ids), step):
        chunk = token_ids[start : start + size]
        chunks.append(chunk)
        if start + size >= len(token_ids):
            break
    return chunks


def main() -> None:
    df = pd.read_parquet(PARQUET, columns=["id", "text"])
    ids   = df["id"].to_numpy()
    texts = df["text"].fillna("").tolist()
    n_jp  = len(texts)
    print(f"JP à embedder : {n_jp}")

    model     = SentenceTransformer(MODEL_NAME)
    tokenizer = model.tokenizer

    # Étape 1 : tokeniser tous les textes et construire les chunks
    print("Tokenisation + chunking…")
    doc_chunk_counts: list[int] = []   # combien de chunks par document
    all_chunks: list[str]       = []   # textes des chunks (re-décodés)

    for text in tqdm(texts, desc="tokenise"):
        prefixed = PREFIX + text
        token_ids = tokenizer.encode(prefixed, add_special_tokens=False)
        chunks = chunk_token_ids(token_ids, CHUNK_SIZE - 2, OVERLAP)  # -2 pour [CLS]/[SEP]
        decoded = [tokenizer.decode(c, skip_special_tokens=True) for c in chunks]
        doc_chunk_counts.append(len(decoded))
        all_chunks.extend(decoded)

    print(f"Total chunks : {len(all_chunks)} (moy. {len(all_chunks)/n_jp:.1f} par JP)")

    # Étape 2 : embedder tous les chunks en batch
    print("Embedding des chunks…")
    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=False,  # on normalise après mean pooling
        convert_to_numpy=True,
    ).astype(np.float32)  # (total_chunks × 384)

    # Étape 3 : mean pooling par document
    print("Mean pooling par document…")
    embeddings = np.zeros((n_jp, chunk_embeddings.shape[1]), dtype=np.float32)
    cursor = 0
    for i, n_chunks in enumerate(doc_chunk_counts):
        block = chunk_embeddings[cursor : cursor + n_chunks]  # (n_chunks × 384)
        embeddings[i] = block.mean(axis=0)
        cursor += n_chunks

    # L2-normalisation finale (cosine = dot product après ça)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    embeddings /= norms

    np.save(OUT_EMB, embeddings)
    np.save(OUT_IDS, ids)
    print(f"✓ embeddings : {embeddings.shape} → {OUT_EMB} ({OUT_EMB.stat().st_size/1e9:.2f} GB)")
    print(f"✓ ordre ids  : {ids.shape} → {OUT_IDS}")


if __name__ == "__main__":
    main()
```

- [ ] **Étape 2.3 : Tester sur 500 JP (validation du chunking)**

```bash
python3 -c "
import pandas as pd
df = pd.read_parquet('baseline_b2/jp_index.parquet').head(500)
df.to_parquet('baseline_b2/jp_index_sample.parquet', index=False)
print('Sample 500 JP créé')
"
```

Modifier temporairement dans `embed_corpus.py` :
```python
PARQUET = HERE / "jp_index_sample.parquet"
OUT_EMB = HERE / "jp_embeddings_sample.npy"
OUT_IDS = HERE / "jp_order_sample.npy"
```

Lancer et vérifier la sortie :
```bash
python baseline_b2/embed_corpus.py
```

Sortie attendue (500 JP) :
```
JP à embedder : 500
Tokenisation + chunking…
Total chunks : ~600 (moy. ~1.2 par JP pour CC qui a un summary court)
Embedding des chunks…
Mean pooling par document…
✓ embeddings : (500, 384) → ...
```

Remettre `PARQUET = HERE / "jp_index.parquet"` etc. avant de continuer.

- [ ] **Étape 2.4 : Lancer sur le corpus complet**

> Durée estimée : 4–8h sur CPU Mac. Si Apple Silicon avec MPS, `SentenceTransformer` le détecte automatiquement et sera ~4× plus rapide.

```bash
nohup python baseline_b2/embed_corpus.py > baseline_b2/embed_corpus.log 2>&1 &
echo "PID: $!"
```

Suivre l'avancement :
```bash
tail -f baseline_b2/embed_corpus.log
```

- [ ] **Étape 2.5 : Commit**

```bash
git add 05-Technique/benchmark/baseline_b2/embed_corpus.py
git commit -m "feat(b2): embed_corpus — chunking 512t + mean pooling sur 1M JP"
```

---

## Task 3 : Pipeline de requête naïf (query_naive.py)

**Pourquoi :** Donnée une question CRFPA, on embed la question, on trouve les K JP les plus proches, puis pour chaque JP on récupère les articles cités via la matrice CSR du graphe. On renvoie un `parsed_canon` compatible avec `eval_rubric.evaluate`.

**Files :**
- Créer : `05-Technique/benchmark/baseline_b2/query_naive.py`

- [ ] **Étape 3.1 : Comprendre la correspondance graphe ↔ embeddings**

Le graphe a ses `jp_ids` dans un ordre donné (vecteur numpy de taille 1 072 646).  
L'index embed a ses ids dans l'ordre du parquet (1 125 968, légère différence : certains JP dans le JSONL ne sont pas dans le graphe).  
Il faut construire un mapping : `{jp_id: index_dans_matrice_embeddings}` et `{jp_id: index_dans_matrice_graphe}`.

- [ ] **Étape 3.2 : Écrire query_naive.py**

```python
#!/usr/bin/env python3
"""Pipeline de requête naïf pour la baseline B2.

Pour une question donnée :
  1. Embed "query: <question>" avec multilingual-e5-base
  2. Cosine similarity = dot product (embeddings L2-normalisés)
  3. Top-K JP par similarité
  4. Agrégation des articles : union pondérée par fréquence parmi les K JP
  5. Renvoie parsed_canon compatible eval_rubric.evaluate

Interface :
  from query_naive import NaiveRetriever
  r = NaiveRetriever()
  canon = r.query(question_text, k=5, min_freq=1)
"""
from __future__ import annotations

import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix

HERE = Path(__file__).parent.resolve()
GRAPH_NPZ  = HERE.parent / "graphs_v5" / "graph_bipartite.npz"
EMB_FILE   = HERE / "jp_embeddings.npy"
IDS_FILE   = HERE / "jp_order.npy"

PREFIX_Q = "query: "


class NaiveRetriever:
    """Charge les ressources en mémoire une seule fois, réutilisable pour N requêtes."""

    def __init__(self) -> None:
        print("Chargement des embeddings…")
        self._emb: np.ndarray = np.load(EMB_FILE)       # (N_index, 384)
        self._emb_ids: np.ndarray = np.load(IDS_FILE, allow_pickle=True)  # (N_index,)

        print("Chargement du graphe…")
        data = np.load(GRAPH_NPZ, allow_pickle=True)
        jp_ids_graph   = data["jp_ids"]    # (N_graph,)
        article_ids    = data["article_ids"]  # (87821,)
        mat = csr_matrix(
            (data["data"], data["indices"], data["indptr"]),
            shape=tuple(data["shape"])
        )  # (N_graph × N_article)

        # Index de recherche rapide
        self._emb_id2pos: dict[str, int] = {uid: i for i, uid in enumerate(self._emb_ids)}
        self._graph_id2pos: dict[str, int] = {uid: i for i, uid in enumerate(jp_ids_graph)}

        # IDs dans le graphe qui ont aussi un embedding (intersection)
        common = [uid for uid in jp_ids_graph if uid in self._emb_id2pos]
        print(f"JP avec embedding ET dans le graphe : {len(common)} / {len(jp_ids_graph)}")

        # Sous-matrice : seulement les JP présentes dans les deux
        rows = np.array([self._graph_id2pos[uid] for uid in common], dtype=np.int32)
        self._mat_sub = mat[rows, :]         # (N_common × N_article), CSR
        self._sub_emb = self._emb[[self._emb_id2pos[uid] for uid in common], :]  # (N_common × 384)
        self._sub_ids = np.array(common)     # (N_common,), jp_ids
        self._article_ids = article_ids

        # Mapping jp_id → number (pourvoi) depuis le parquet
        import pandas as pd
        parquet = HERE / "jp_index.parquet"
        df = pd.read_parquet(parquet, columns=["id", "number"])
        self._jp_id2pourvoi: dict[str, str] = dict(zip(df["id"], df["number"]))

        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("intfloat/multilingual-e5-base")
        print("✓ NaiveRetriever prêt")

    def query(self, question: str, k: int = 5, min_freq: int = 1) -> dict:
        """Renvoie parsed_canon compatible eval_rubric.evaluate.

        min_freq : article retenu si cité par au moins min_freq des K JP voisines.
        """
        # 1) Embed question
        q_vec = self._model.encode(
            [PREFIX_Q + question],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]  # (384,)

        # 2) Top-K par dot product (= cosine car L2-normalisé)
        scores = self._sub_emb @ q_vec       # (N_common,)
        top_k_idx = np.argpartition(scores, -k)[-k:]
        top_k_idx = top_k_idx[np.argsort(scores[top_k_idx])[::-1]]

        # 3) JP résultat
        jp_ids_topk = self._sub_ids[top_k_idx]
        jp_scores   = scores[top_k_idx]

        # 4) Articles agrégés par fréquence
        article_counts = np.zeros(self._article_ids.shape[0], dtype=np.int32)
        for idx in top_k_idx:
            row = self._mat_sub.getrow(idx)
            article_counts[row.indices] += 1

        retained_art_idx = np.where(article_counts >= min_freq)[0]
        retained_art_ids = self._article_ids[retained_art_idx]

        # 5) Format parsed_canon
        articles = [{"pair_key": pk} for pk in retained_art_ids]
        jurisprudences = [
            {"pourvoi": self._jp_id2pourvoi.get(uid, ""), "jp_id": uid, "score": float(s)}
            for uid, s in zip(jp_ids_topk, jp_scores)
            if self._jp_id2pourvoi.get(uid, "")
        ]

        return {
            "articles": articles,
            "jurisprudences": jurisprudences,
            "arguments": [],
            "_meta": {
                "k": k,
                "min_freq": min_freq,
                "n_articles": len(articles),
                "n_jp": len(jurisprudences),
            },
        }
```

- [ ] **Étape 3.3 : Test unitaire minimal**

```python
# Lancer depuis 05-Technique/benchmark/
# python3 -c "..."
import sys
sys.path.insert(0, "baseline_b2")
from query_naive import NaiveRetriever
r = NaiveRetriever()
canon = r.query("Quelles sont les règles de responsabilité civile en cas de faute ?", k=5)
print("Articles:", len(canon["articles"]))
print("JP:", len(canon["jurisprudences"]))
print("Exemple article:", canon["articles"][:3])
print("Exemple JP:", canon["jurisprudences"][:2])
# Attendu : plusieurs articles, format pair_key correct (ex: "code_civil:1240")
```

- [ ] **Étape 3.4 : Commit**

```bash
git add 05-Technique/benchmark/baseline_b2/query_naive.py
git commit -m "feat(b2): query_naive — embed question → top-K JP → articles via graphe"
```

---

## Task 4 : Lancer la baseline sur les 41 questions CRFPA (run_b2.py)

**Pourquoi :** Boucle sur les 41 questions, scoring via `eval_rubric.evaluate`, sauvegarde des résultats détaillés + CSV de synthèse comparable au `comparison_crfpa.csv` existant.

**Files :**
- Créer : `05-Technique/benchmark/baseline_b2/run_b2.py`
- Produit : `baseline_b2/results/b2_k{K}.json`, `baseline_b2/results/comparison_b2.csv`

- [ ] **Étape 4.1 : Écrire run_b2.py**

```python
#!/usr/bin/env python3
"""Évalue la baseline B2 (embedding naïf) sur les 41 questions CRFPA.

Usage (depuis 05-Technique/benchmark/) :
    python baseline_b2/run_b2.py
    python baseline_b2/run_b2.py --k 3 5 10  # valeurs de K à tester
    python baseline_b2/run_b2.py --min-freq 2  # articles cités par ≥ 2 JP
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent.resolve()
BENCH_DIR = HERE.parent
RUBRICS_DIR = BENCH_DIR / "data" / "rubrics"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BENCH_DIR / "crfpa_benchmark"))
from eval_rubric import evaluate

CSV_PATH = RESULTS_DIR / "comparison_b2.csv"
CSV_HEADER = [
    "config", "k", "min_freq",
    "n_q_ok", "S_retrieval_mean", "S_e2e_mean",
    "S_bar_art_mean", "S_bar_jp_mean",
    "art_core_mean", "art_expected_mean", "art_expert_mean",
    "jp_core_mean",  "jp_expected_mean",  "jp_expert_mean",
    "n_articles_mean", "n_jp_mean",
]


def load_all_questions() -> list[dict]:
    questions = []
    for f in sorted(RUBRICS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for q in data.get("questions", []):
            questions.append(q)
    return questions


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def run_one_config(retriever, questions: list[dict], k: int, min_freq: int) -> dict:
    per_q = []
    for q in questions:
        t0 = time.time()
        canon = retriever.query(q["question"], k=k, min_freq=min_freq)
        latency = time.time() - t0
        scores = evaluate(canon, q)
        per_q.append({
            "qid":      q["id"],
            "branche":  q.get("branche"),
            "canon":    canon,
            "scores":   scores,
            "latency":  latency,
        })

    ok = per_q
    means = {
        "S_retrieval": _mean([r["scores"]["regime"]["retrieval"] for r in ok]),
        "S_e2e":       _mean([r["scores"]["regime"]["e2e"] for r in ok]),
        "S_bar_art":   _mean([r["scores"]["articles"]["S_bar"] for r in ok]),
        "S_bar_jp":    _mean([r["scores"]["jurisprudences"]["S_bar"] for r in ok]),
        "art_core":    _mean([r["scores"]["articles"]["per_strate"]["core"] for r in ok]),
        "art_expected":_mean([r["scores"]["articles"]["per_strate"]["expected"] for r in ok]),
        "art_expert":  _mean([r["scores"]["articles"]["per_strate"]["expert"] for r in ok]),
        "jp_core":     _mean([r["scores"]["jurisprudences"]["per_strate"]["core"] for r in ok]),
        "jp_expected": _mean([r["scores"]["jurisprudences"]["per_strate"]["expected"] for r in ok]),
        "jp_expert":   _mean([r["scores"]["jurisprudences"]["per_strate"]["expert"] for r in ok]),
        "n_articles":  _mean([r["canon"]["_meta"]["n_articles"] for r in ok]),
        "n_jp":        _mean([r["canon"]["_meta"]["n_jp"] for r in ok]),
    }
    return {"k": k, "min_freq": min_freq, "means": means, "per_question": per_q}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument("--min-freq", type=int, default=1,
                    help="Articles retenus si cités par ≥ min-freq JP voisines")
    args = ap.parse_args()

    questions = load_all_questions()
    print(f"Questions CRFPA : {len(questions)}")

    sys.path.insert(0, str(HERE))
    from query_naive import NaiveRetriever
    retriever = NaiveRetriever()

    init_csv = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="") as cf:
        writer = csv.writer(cf)
        if init_csv:
            writer.writerow(CSV_HEADER)

        for k in args.k:
            config = f"b2_k{k}_mf{args.min_freq}"
            out_json = RESULTS_DIR / f"{config}.json"

            print(f"\n{'='*60}\nConfig : k={k}, min_freq={args.min_freq}\n{'='*60}")
            result = run_one_config(retriever, questions, k=k, min_freq=args.min_freq)
            out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))

            m = result["means"]
            writer.writerow([
                config, k, args.min_freq,
                len(result["per_question"]),
                m["S_retrieval"], m["S_e2e"],
                m["S_bar_art"], m["S_bar_jp"],
                m["art_core"], m["art_expected"], m["art_expert"],
                m["jp_core"], m["jp_expected"], m["jp_expert"],
                m["n_articles"], m["n_jp"],
            ])
            cf.flush()

            print(f"  S_retrieval = {m['S_retrieval']}  (plafond LLM ≈ 0.095)")
            print(f"  S̄_art={m['S_bar_art']}  S̄_jp={m['S_bar_jp']}")
            print(f"  → {out_json}")

    print(f"\n✓ Résultats → {CSV_PATH}")

    try:
        import pandas as pd
        df = pd.read_csv(CSV_PATH)
        cols = ["config", "S_retrieval_mean", "S_bar_art_mean", "S_bar_jp_mean", "n_articles_mean", "n_jp_mean"]
        print("\n" + df[cols].to_string(index=False))
    except ImportError:
        pass


if __name__ == "__main__":
    main()
```

- [ ] **Étape 4.2 : Lancer la baseline**

```bash
cd /Users/matthieu.kaeppelin/Documents/5-Pro/Stages/FE_recherche/legal_knowledge_graph/05-Technique/benchmark
python baseline_b2/run_b2.py --k 3 5 10
```

Sortie attendue (format) :
```
Questions CRFPA : 41
Chargement des embeddings…
...
Config : k=5, min_freq=1
  S_retrieval = X.XXX  (plafond LLM ≈ 0.095)
  S̄_art=X.XXX  S̄_jp=X.XXX
```

- [ ] **Étape 4.3 : Commit**

```bash
git add 05-Technique/benchmark/baseline_b2/run_b2.py baseline_b2/results/
git commit -m "feat(b2): run_b2 — baseline embedding naïf sur 41 questions CRFPA"
```

---

## Task 5 : Analyse et journal

**Pourquoi :** Interpréter les résultats, rédiger l'entrée de journal, décider des suites (expansion 2-hop, variation min_freq, passage à Priorité 2).

**Files :**
- Créer : `01-Projet/journal/2026-05-04.md` (ou date réelle du run)

- [ ] **Étape 5.1 : Lire la table de comparaison**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('05-Technique/benchmark/baseline_b2/results/comparison_b2.csv')
print(df.to_string())
"
```

Comparer avec le plafond LLM (depuis `crfpa_benchmark/results/comparison_crfpa.csv`) :

```bash
python3 -c "
import pandas as pd
llm = pd.read_csv('05-Technique/benchmark/crfpa_benchmark/results/comparison_crfpa.csv')
b2  = pd.read_csv('05-Technique/benchmark/baseline_b2/results/comparison_b2.csv')
print('=== LLM baselines ===')
print(llm[['alias','S_retrieval_mean']].to_string(index=False))
print()
print('=== B2 baselines ===')
print(b2[['config','S_retrieval_mean']].to_string(index=False))
"
```

- [ ] **Étape 5.2 : Rédiger l'entrée de journal**

Créer `01-Projet/journal/YYYY-MM-DD.md` avec :
```markdown
---
date: YYYY-MM-DD
type: journal
tags: [journal, baseline-b2, semaine-4]
---

# YYYY-MM-DD — Baseline B2 : résultats embedding naïf

## Résultats

| Config      | S_retrieval | S̄_art | S̄_jp | N articles | N JP |
|-------------|-------------|--------|-------|------------|------|
| b2_k3_mf1   | X.XXX       | X.XXX  | X.XXX | XX         | 3    |
| b2_k5_mf1   | X.XXX       | X.XXX  | X.XXX | XX         | 5    |
| b2_k10_mf1  | X.XXX       | X.XXX  | X.XXX | XX         | 10   |
| LLM moyen   | ~0.095      | —      | —     | —          | —    |

## Interprétation

[Remplir après le run]

## Prochaine étape

Si S_retrieval > 0.095 → la motivation du KG est démontrée, passer à P2 (fetch articles Légifrance).
Si S_retrieval ≤ 0.095 → itérer sur K, min_freq, expansion 2-hop, ou modèle d'embedding.
```

- [ ] **Étape 5.3 : Commit final**

```bash
git add 01-Projet/journal/ 05-Technique/benchmark/baseline_b2/
git commit -m "feat(b2): résultats baseline B2 — S_retrieval vs plafond LLM"
```

---

## Récapitulatif des risques

| Risque | Probabilité | Mitigation |
|--------|-------------|------------|
| OOM lors du chargement de jp_embeddings.npy (1,7 GB) + graphe en RAM | Faible sur Mac 16 GB+ | Réduire à CC only (553k JP) pour premier test |
| Embedding du corpus trop lent (CPU) | Élevée | Utiliser MPS (Apple Silicon) ou tester sur les 10k premiers JP |
| S_retrieval << 0.095 | Possible | Itérer sur K plus élevé (50, 100), min_freq=1, expansion 2-hop |
| Mauvaise normalisation pourvoi | Faible | L'eval_rubric extrait le pourvoi par regex — format `number` du JSONL correspond |
