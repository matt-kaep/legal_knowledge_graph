---
tags: [benchmark, llm, regex, extraction-articles, shortlist]
type: fiche
status: en-cours
created: 2026-04-24
modified: 2026-04-24
---

# Shortlist LLMs — Extraction d'articles de loi depuis arrêts

> Sélection de modèles open-weights à faire tourner sur L40S (40 Go VRAM) pour
> comparer à [[2026-04-24#V3 regex — gelé|regex V3]] (F1=0.920) sur les 20 arrêts
> annotés manuellement (`manual_annotations.json`).

## Objectif

Identifier **à partir de quelle taille/archi un LLM bat le regex** sur la tâche
d'extraction normalisée `code_slug:article`. En creux : montrer qu'un LLM rapide
de petite taille **peut être moins bon qu'un regex bien calibré**.

## Modèles retenus pour la passe 1 (7 modèles)

| # | Modèle | Params | Actifs | Contexte | Licence | Q | VRAM | Note |
|---|---|---|---|---|---|---|---|---|
| 1 | Gemma 4 E2B | ~2B (PLE) | 2B | 128k | Apache 2.0 | Q8 | ~5 Go | Petit, FR solide, multimodal |
| 2 | Qwen3.5-2B | 2B | 2B | 256k | Apache 2.0 | Q8 | ~5 Go | Thinking mode |
| 3 | Gemma 4 E4B | ~4B (PLE) | 4B | 128k | Apache 2.0 | Q8 | ~15 Go | Meilleur FR petit modèle |
| 4 | Qwen3.5-9B | 9B | 9B | 256k→1M | Apache 2.0 | Q8 | ~11 Go | Long contexte réel |
| 5 | Ministral 8B | 8B | 8B | 128k | Mistral Research | Q8 | ~9 Go | FR natif |
| 6 | Gemma 4 26B A4B (MoE) | ~26B | ~4B | 128k | Apache 2.0 | Q6 | ~18 Go | MoE — vitesse d'un 4B ⚠ référence incertaine, voir note |
| 7 | Gemma 4 31B | 31B | 31B | 128k | Gemma license | Q6 | ~22 Go | Baseline du notebook existant `benchmark_m1_m6_sample.ipynb` |

> ⚠ **Note sur Gemma 4 26B A4B** : je n'ai pas retrouvé cette référence dans le
> tableau initial (la famille Gemma 4 annoncée inclut E2B, E4B, 31B mais pas de
> MoE 26B A4B documenté ici). Si tu voulais dire **Qwen3-30B-A3B** (MoE 30B / 3B
> actifs) je le substitue — à confirmer avant de lancer le run cluster.

## Décisions de design — inputs normés pour toutes les comparaisons

Pour qu'une comparaison regex vs LLM tienne debout, on impose à chaque LLM :

1. **Liste des codes autorisés** donnée dans le prompt (52 codes officiels + 10
   variantes historiques/acronymes — même dict que regex V3).
2. **Schéma de sortie JSON strict** validé via `response_format.json_schema` de
   vLLM : liste d'objets `{"code_slug": "...", "article": "..."}`.
3. **Format de l'article imposé** : chaîne compacte `^(L|R|D|A|E)?\d[\d\-]*$`
   + suffixes latins optionnels (ex. `L742-1`, `1649quinquiesB`, `1014`).
4. **Few-shot** : 3 exemples couvrant listing, anaphore, suffixe latin.
5. **Post-traitement identique** : `_normalize_pair_article` + slug du code par
   le même dict que regex → on compare des sets de pair_keys.

## Métriques

- Pooled P/R/F1 sur 20 arrêts (même harness que `eval_vs_manual.py`)
- Détail par arrêt + catégorisation des échecs (hallucination code, omission,
  mauvais slug)
- Latence moyenne par arrêt + tokens émis (coût unitaire)

## Protocole d'exécution

- Serveur vLLM sur cluster (voir `setup_cluster.sh`)
- Un run = 20 arrêts × 1 modèle
- Temperature = 0 (reproductibilité)
- Résultats sauvés dans `results/extraction_articles/<model_id>_<timestamp>.json`
- Script : `eval_llm_vs_regex.py`

## Liens

- [[2026-04-24]] — journal du jour
- [[Design-Benchmark-Avocat-Unifie]]
