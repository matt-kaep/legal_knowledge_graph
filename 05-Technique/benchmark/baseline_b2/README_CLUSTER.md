# Baseline B2 — Droit pénal — Cluster L40S

## Expérience testée

**Hypothèse** : un retrieval naïf basé sur un encodeur texte + agrégation via le KG bat un LLM nu sur le benchmark CRFPA.

**Pipeline** (par question) :
```
Question CRFPA → embed → cosine vs 118k JP pénales → top-K JP voisines
              → leurs articles cités dans le graphe → agrégation → parsed_canon
              → eval_rubric → S_retrieval
```

**Variables** :
- 1 modèle d'embedding par défaut (`e5-base`, 768d), registre extensible
- 4 stratégies d'agrégation (voir ci-dessous)
- 5 valeurs de K (3, 4, 5, 7, 10)

→ **20 configs × 8 questions = 160 évaluations** par modèle.

**Critère de succès** : `S_retrieval_mean > 0.095` (plafond LLM nu).

## Stratégies d'agrégation (le cœur de l'XP)

Donné top-K JP voisines de la question, comment agréger leurs articles cités ?

| Strategy | Logique | Effet attendu |
|----------|---------|---------------|
| `union` | Article retenu s'il est cité par ≥ 1 des K JP | Recall max, précision faible (beaucoup de bruit) |
| `intersection` | Article cité par TOUTES les K JP | Consensus strict, précision élevée mais recall qui s'effondre quand K augmente |
| `majority` | Article cité par ≥ ⌈K/2⌉ JP | Compromis classique |
| `weighted` | Articles classés par Σ similarité des JP citantes, garde top-N (défaut N=5) | Privilégie les articles cités par les JP **les plus proches** |

L'XP compare ces 4 stratégies × 5 K pour identifier la bonne politique de retrieval.

## Contenu du bundle

| Fichier | Description | Taille |
|---------|-------------|--------|
| `jp_index_penal.parquet` | 118 112 JP pénales (id, number, juris, text) | ~346 MB |
| `graph_penal.npz` | Sous-graphe CSR pénal × tous articles | ~2 MB |
| `rubrics_penal.json` | 8 questions pénales fusionnées | ~100 KB |
| `eval_rubric.py` | Module de scoring CRFPA | ~12 KB |
| `run_cluster.py` | **Script all-in-one** | ~21 KB |
| `requirements.txt` | Dépendances Python | <1 KB |

## Setup environnement

```bash
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

export HF_TOKEN=hf_xxx
export HF_HOME=/scratch/hf_cache

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
```

## Lancer la baseline

```bash
# Défaut : e5-base, 4 stratégies, K ∈ {3,4,5,7,10}
python run_cluster.py

# Variantes
python run_cluster.py --only e5-large bge-m3
python run_cluster.py --strategies intersection majority
python run_cluster.py --k 5 10
python run_cluster.py --weighted-top 3
python run_cluster.py --list-questions
```

## Modèles disponibles dans le registre

| Alias | HF id | Dims | Notes |
|-------|-------|------|-------|
| `e5-small` | `intfloat/multilingual-e5-small` | 384 | Rapide, 118M params |
| `e5-base` ⭐ | `intfloat/multilingual-e5-base` | 768 | **Défaut** — équilibré, 278M params |
| `e5-large` | `intfloat/multilingual-e5-large` | 1024 | Qualité, 560M params |
| `bge-m3` | `BAAI/bge-m3` | 1024 | Long context (8k tokens), SOTA multilingue |
| `camembert-base` | `dangvantuan/sentence-camembert-base` | 768 | FR-spécifique |

## Durées estimées sur L40S

| Étape | e5-base | e5-large | bge-m3 |
|-------|---------|----------|--------|
| Embedding 118k JP | ~6 min | ~12 min | ~15 min |
| 20 configs × 8 questions | ~3 min | ~3 min | ~3 min |
| **Total** | **~10 min** | **~15 min** | **~20 min** |

## Sortie

```
results/
├── comparison_b2_penal.csv             # ← clé : 1 ligne par (modèle, strategy, K)
├── <alias>.json                        # détail agrégé par modèle
└── per_question/
    └── <qid>__<alias>__<strategy>__k<K>.json   # détail par évaluation
embeddings/
└── jp_embeddings_<alias>.npy           # cache (resumable)
```

**Métrique principale** : `S_retrieval_mean` dans le CSV — à comparer à 0,095.

## Reprise après interruption

- **Embedding** : memmap + state JSON, relancer reprend automatiquement.
- **Modèle déjà fait** : skip sauf `--force`.
- **Crash sur un modèle** : la boucle continue, statut "error: ..." dans le CSV.

## Récupération

```bash
scp -r cluster:~/penal_bundle/results ./05-Technique/benchmark/baseline_b2/results_cluster/
```

## Métriques (rappel scoring CRFPA)

```
S_retrieval = (0.5·S̄_art + 0.5·S̄_jp) × (1 − τ_hall)
S_e2e       = (0.25·S̄_art + 0.25·S̄_jp + 0.5·S̄_arg) × (1 − τ_hall)
S̄_dim       = (3·S_core + 2·S_expected + S_expert) / 6
S_{d,s}     = |GT_{d,s} ∩ Pred_d| / |GT_{d,s}|     (recall pur)
```
