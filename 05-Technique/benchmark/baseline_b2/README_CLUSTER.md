# Baseline B2 — Droit pénal — Pipeline cluster (L40S)

## Contenu du bundle

| Fichier | Description | Taille |
|---------|-------------|--------|
| `jp_index_penal.parquet` | 118 112 JP pénales (id, number, juris, text) | ~346 MB |
| `graph_penal.npz` | Sous-graphe CSR pénal × tous articles | ~2 MB |
| `rubrics_penal.json` | 8 questions pénales fusionnées | ~100 KB |
| `eval_rubric.py` | Module de scoring CRFPA | ~12 KB |
| `run_cluster.py` | **Script all-in-one** | ~17 KB |
| `requirements.txt` | Dépendances Python | <1 KB |

## Setup environnement

```bash
# 1. venv + torch CUDA
python -m venv venv && source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Cache HF sur disque rapide
export HF_TOKEN=hf_xxx
export HF_HOME=/scratch/hf_cache

# 3. Vérifier le GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Lancer la baseline

```bash
# Tous les modèles du registre (séquentiel : download → embed → eval → purge)
python run_cluster.py

# Subset
python run_cluster.py --only e5-base e5-large

# Variantes K et seuil
python run_cluster.py --k 3 5 10 20 --min-freq 1

# Garder les modèles HF (debug)
python run_cluster.py --no-purge

# Lister les questions
python run_cluster.py --list-questions
```

## Modèles dans le registre

| Alias | HF id | Dims | Notes |
|-------|-------|------|-------|
| `e5-small` | `intfloat/multilingual-e5-small` | 384 | Rapide, 118M params |
| `e5-base` | `intfloat/multilingual-e5-base` | 768 | Équilibré, 278M params |
| `e5-large` | `intfloat/multilingual-e5-large` | 1024 | Qualité, 560M params |
| `bge-m3` | `BAAI/bge-m3` | 1024 | Long context (8k tokens), SOTA multilingue |
| `camembert-base` | `dangvantuan/sentence-camembert-base` | 768 | FR-spécifique |

## Durées attendues sur L40S (48 GB VRAM, ~118k JP pénales)

| Modèle | Embedding | Query × 3K + score |
|--------|-----------|--------------------|
| `e5-small` | ~3 min | <1 min |
| `e5-base` | ~6 min | <1 min |
| `e5-large` | ~12 min | <1 min |
| `bge-m3` | ~15 min | ~1 min |

## Sortie

```
results/
├── comparison_b2_penal.csv           # ← tableau de synthèse (1 ligne par modèle×K)
├── <alias>.json                      # détail agrégé par modèle
└── per_question/
    └── <qid>__<alias>__k<K>.json     # détail par question
embeddings/
└── jp_embeddings_<alias>.npy         # cache embeddings (resumable)
logs/                                  # logs futurs
```

**Métrique clé** : `S_retrieval_mean` dans le CSV — plafond LLM nu ≈ 0,095.

## Comportement de reprise

- **Embedding interrompu** → relancer = reprend depuis le dernier batch (state JSON).
- **Modèle déjà fait** → skip automatique sauf `--force`.
- **Crash sur un modèle** → la boucle continue avec les suivants, statut "error: ..." dans le CSV.

## Récupération des résultats

```bash
# Depuis le Mac
scp -r cluster:~/penal_bundle/results ./05-Technique/benchmark/baseline_b2/results_cluster/
```

## Métriques (rappel scoring CRFPA)

```
S_retrieval = (0.5·S̄_art + 0.5·S̄_jp) × (1 − τ_hall)
S_e2e       = (0.25·S̄_art + 0.25·S̄_jp + 0.5·S̄_arg) × (1 − τ_hall)
S̄_dim       = (3·S_core + 2·S_expected + S_expert) / 6
S_{d,s}     = |GT_{d,s} ∩ Pred_d| / |GT_{d,s}|     (recall pur)
```
