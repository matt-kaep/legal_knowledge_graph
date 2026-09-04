---
title: "Expérience B2 — Embedding naïf sur benchmark CRFPA pénal"
date: 2026-05-05
type: fiche
tags: [experience, baseline, embedding, knowledge-graph, penal, crfpa]
statut: en-cours
pertinence: haute
---

# Expérience B2 — Embedding naïf + KG sur 8 questions CRFPA pénales

## Hypothèse

> Un retrieval naïf (sans entraînement) basé sur un encodeur texte généraliste FR + agrégation via le graphe KG construit (1,07 M JP × 88k articles) **bat le plafond LLM nu** sur le benchmark CRFPA.

Cette expérience est la **Priorité 1** de la roadmap décidée le 2026-04-29. Si elle valide l'hypothèse, elle motive l'investissement dans le contrastive learning (Priorité 4). Sinon, elle indique que le retrieval naïf seul ne suffit pas.

## Pipeline testé (par question)

```
Question CRFPA            "Quelles sont les conditions de la garde à vue ?"
       ↓
   embed("query: ...")    vecteur 768d
       ↓
   cosine vs 118k JP      scores de similarité
       ↓
   top-K JP voisines      ex. K=5 → 5 jp_id Judilibre
       ↓
   articles cités via KG  union des arêtes graph
       ↓
   AGRÉGATION             ← variable d'XP (4 stratégies)
       ↓
   parsed_canon           {articles: [...], jurisprudences: [pourvois]}
       ↓
   eval_rubric            S_retrieval, S̄_art, S̄_jp
```

## Variables expérimentales

### Indépendantes (ce qu'on fait varier)

**1. Stratégie d'agrégation** (cœur de l'XP)

Donné top-K JP voisines de la question, comment combiner leurs articles cités ?

| Strategy | Logique | Hypothèse testée |
|----------|---------|------------------|
| `union` | Article retenu si cité par ≥ 1 des K JP | "Tout signal est utile" |
| `intersection` | Article cité par TOUTES les K JP | "Seul le consensus strict compte" |
| `majority` | Article cité par ≥ ⌈K/2⌉ JP | "Consensus modéré, compromis classique" |
| `weighted` | Articles classés par Σ similarité des JP citantes, garde top-N | "La proximité de similarité importe plus que le compte" |

**2. Nombre de voisins K** : {3, 4, 5, 7, 10}

→ **20 configurations** par modèle.

### De contrôle (fixées)

- **Modèle d'embedding** : `intfloat/multilingual-e5-base` (768d, 278M params), équilibre qualité/vitesse
- **Chunking** : 510 tokens par chunk, overlap 64 tokens, max 8 chunks/doc, mean pooling, L2-normalisation
- **Préfixes E5** : `passage: ` pour les JP, `query: ` pour la question
- **Corpus** : 118 112 JP pénales (CC: 97 060, CA: 14 315, TJ: 6 737), filtrées via citations d'articles pénaux dans le graphe
- **Benchmark** : 8 questions CRFPA pénales (3 droit pénal + 5 procédure pénale)
- **Scoring** : `eval_rubric.evaluate()` — identique aux LLMs, équations finales de la slide Week-3

### Dépendante (ce qu'on mesure)

**Métrique principale** : `S_retrieval = (0.5·S̄_art + 0.5·S̄_jp) × (1 − τ_hall)`

Avec :
- `S̄_d = (3·S_core + 2·S_expected + S_expert) / 6` pour d ∈ {art, jp}
- `S_{d,s} = |GT_{d,s} ∩ Pred_d| / |GT_{d,s}|` (recall pur par strate)
- `τ_hall = 0` en v1 (vérification Légifrance/Judilibre non branchée)

**Métriques secondaires** : N articles retournés, N JP retournées, recall par strate (core / expected / expert).

## Données

### Filtrage du corpus pénal

Stratégie : on garde toute JP qui cite au moins un article d'un code pénal :
- `code_penal` (2 999 articles)
- `code_de_procedure_penale` (3 924 articles)
- `code_de_la_route` (1 125 articles)
- `code_de_la_justice_penale_des_mineurs` (37 articles)

Total : **8 085 colonnes pénales sur 87 821**.
JP qui en citent au moins une : **118 112 sur 1 072 646** (~11%).

Cette stratégie évite les filtres hétérogènes par juridiction (chamber `Chambre criminelle` pour CC, NAC pour TJ, rien d'utile pour CA).

### Représentation textuelle pour l'embedding

| Juridiction | Champ utilisé | Médiane |
|-------------|---------------|---------|
| CC | `summary` (95% dispo) sinon `text` | ~3 100 chars |
| CA | `text` brut | ~10 200 chars |
| TJ | `text` brut | ~9 500 chars |

Le chunking + mean pooling permet de représenter chaque document complet sans troncature.

### Identifiants

- **Clé primaire** : `id` (ObjectId Judilibre, fiable)
- **Pourvoi** (`number`) : **non unique** (13k doublons intra-CC, formats hétérogènes CA/TJ, 16k collisions inter-juridictions). Utilisé seulement pour le scoring CRFPA, dont la regex `\b(\d{2})[\s\-]*(\d{2})[.\-]?(\d{3})\b` ne matche que le format CC `XX-XX.XXX` — donc en pratique le scoring JP ne compare que les arrêts CC.

## Critère de succès

| Niveau | S_retrieval | Interprétation |
|--------|-------------|----------------|
| Plafond LLM nu | ≈ **0,095** | Référence à dépasser |
| **Succès** | **> 0,095** | Le KG apporte de la valeur, motivation pour suite |
| Forte motivation | > 0,2 | Permet d'accélérer Priorité 4 (contrastive) |
| Échec | ≤ 0,095 | Retrieval naïf insuffisant — investiguer pourquoi |

## Infrastructure

### Bundle autonome côté Mac

`05-Technique/benchmark/baseline_b2/penal_bundle.tar.gz` (293 MB) :

| Fichier | Description | Taille |
|---------|-------------|--------|
| `jp_index_penal.parquet` | 118k JP pénales (id, number, juris, text) | 346 MB |
| `graph_penal.npz` | Sous-graphe CSR (118k × 88k, nnz=642k) | 2 MB |
| `rubrics_penal.json` | 8 questions pénales fusionnées | 100 KB |
| `eval_rubric.py` | Scorer CRFPA | 12 KB |
| `run_cluster.py` | Script all-in-one | 26 KB |
| `requirements.txt` | torch, sentence-transformers, etc. | <1 KB |

### Exécution sur cluster L40S

```bash
scp penal_bundle.tar.gz cluster:~/
ssh cluster
tar xzf penal_bundle.tar.gz && cd penal_bundle
pip install -r requirements.txt
export HF_TOKEN=hf_xxx HF_HOME=/scratch/hf_cache
python run_cluster.py        # défaut : e5-base, 4 strategies × 5 K
```

Durée totale estimée : **~10 min** (6 min embedding + 3 min query × 20 configs).

### Robustesse

- **Embedding resumable** via `np.memmap` + state JSON `{offset, n_total, model}` → reprise transparente après Ctrl+C ou crash.
- **Boucle modèle** avec try/except/finally : un modèle qui crash n'arrête pas la suite, le statut est tracé dans le CSV.
- **Cache HF purgé** après chaque modèle (sauf `--no-purge`).

## Sortie attendue

```
results/
├── comparison_b2_penal.csv             # ← clé : 1 ligne par (modèle, strategy, K)
├── e5-base.json                        # détail agrégé
└── per_question/
    └── <qid>__e5-base__<strategy>__k<K>.json
embeddings/
└── jp_embeddings_e5-base.npy           # cache
```

Le CSV contient :
- `S_retrieval_mean`, `S_e2e_mean`
- `S_bar_art_mean`, `S_bar_jp_mean`
- Détail par strate `core / expected / expert` pour articles et JP
- `n_articles_mean`, `n_jp_mean` (sanity check du bruit)
- `embed_time_s`, `query_time_s`

## Lectures attendues des résultats

### Si `weighted` > `majority` > `union`
Le signal de **proximité de similarité** est informatif — le rang dans le top-K importe, pas juste le compte. Bonne nouvelle pour le contrastive learning ensuite.

### Si `intersection` > tout
Le KG est très **structuré thématiquement** : les JP voisines au sens cosinus partagent réellement leurs articles. Indique que le clustering légal est cohérent.

### Si `union` ≈ `weighted`
Aucun gain à filtrer, on peut tout inclure. Indique que le retrieval naïf a peu de discrimination.

### Si N_articles `intersection` ≈ 0 pour K≥4
La diversité des JP voisines est trop forte au-delà de K=3 → suggère d'utiliser des stratégies de consensus modéré (majority).

## Suite décisionnelle

| Résultat | Décision |
|----------|----------|
| `S_retrieval > 0.095` avec une stratégie | **Succès B2** → publier dans le journal, étendre à `e5-large` puis `bge-m3` (long context), passer à P2 (fetch articles Légifrance) |
| `S_retrieval ∈ [0.05, 0.095]` | **Partiel** → tester expansion 2-hop dans le graphe avant d'abandonner le retrieval naïf |
| `S_retrieval < 0.05` | **Échec naïf** → passer directement à P4 (contrastive learning) avec ces résultats comme baseline |

## Limites connues

- **8 questions seulement** : signal statistique limité, mais c'est tout ce qu'il y a en pénal dans CRFPA 2025.
- **τ_hall = 0** : pas de pénalité pour articles/JP hallucinés → optimiste vs LLMs qui hallucinent.
- **Scoring JP biaisé CC** : la regex pourvoi du scorer ne matche que le format CC, donc les JP CA/TJ retournées ne sont pas évaluées même si pertinentes.
- **Pas de S̄_arg** : la dimension "arguments" du scoring est non-implémentée en v1, donc `S_e2e` est None.
- **Embedding généraliste** : multilingual-e5-base n'est pas spécialisé juridique FR. À rapprocher d'un éventuel re-test avec CamemBERT-juridique fine-tuné.

## Liens

- **Bundle** : `05-Technique/benchmark/baseline_b2/penal_bundle.tar.gz`
- **README cluster** : `05-Technique/benchmark/baseline_b2/README_CLUSTER.md`
- **Plan d'origine** : `05-Technique/plans/2026-05-04-baseline-b2-embedding-naif.md`
- **Décisions superviseur** : `01-Projet/journal/2026-04-29.md`
- **Scorer CRFPA** : `05-Technique/benchmark/crfpa_benchmark/eval_rubric.py`
