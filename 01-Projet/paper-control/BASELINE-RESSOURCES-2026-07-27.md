---
date: 2026-07-27
type: preuve-ressources
status: mesuree
owner: assainissement
tags: [benchmark, ressources, smoke-test, grouped-v2]
---

# Baseline de ressources — campagne `grouped_v2`

## Portée

Cette preuve qualifie uniquement la capacité d'exécution locale. Les scores produits par les smoke tests sont exploratoires, incomplets et interdits d'usage scientifique.

Hôte cible : Mac local, 8 CPU logiques, 16 Gio de RAM installée, GPU non requis. Threads BLAS limités à deux. Graphe testé : `G6-citation-JJ-knn5`, plus gros fichier matriciel de la matrice confirmatoire. Les deux mesures ont parcouru les cinq folds officiels et les 5 603 questions, dans des répertoires temporaires hors `_cv_grouped_v2` et `_final_grouped_v2`.

## Mesures

| Moteur | Configuration limitée | Durée réelle | Pic RSS | Empreinte maximale macOS | Swap pendant le run |
|---|---|---:|---:|---:|---:|
| PPR, une configuration | `k=5`, `art_only`, `alpha=0.5`, cinq folds | 396,46 s | 3 137 568 768 octets, soit 2,92 Gio | 4 389 464 064 octets, soit 4,09 Gio | 0 |
| PPR, grille complète | 48 configurations, un fold de 1 120 questions | 1 310,60 s | 3 167 551 488 octets, soit 2,95 Gio ; observation orchestrée ponctuelle ~3,45 Gio | 4 199 622 144 octets, soit 3,91 Gio | 0 |
| LightGCN | cible Articles, `K=2`, seed 42, LR 0,001, ancrage 1, un epoch | 184,15 s | 9 715 187 712 octets, soit 9,05 Gio | 12 076 573 312 octets, soit 11,25 Gio | 0 |

Commandes mesurées avec `/usr/bin/time -l` :

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -l \
  .venv/bin/python scripts/43_run_cv_ppr.py \
  --graph-version G6-citation-JJ-knn5 \
  --split train_augmented_retrievable_strict \
  --out-dir <repertoire-temporaire> \
  --config 5:art_only:0.5

OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /usr/bin/time -l \
  .venv/bin/python scripts/44_run_cv_lightgcn.py \
  --graph-version G6-citation-JJ-knn5 \
  --split train_augmented_retrievable_strict \
  --out-dir <repertoire-temporaire> \
  --train-k 2 --seed 42 --lr 0.001 --epochs 1 \
  --lambda-anchor 1.0 --negative-sampling-strategy random \
  --selection-target art
```

## Décision d'exploitation

- `ram_minimum_gb_per_ppr_job = 3.5`, plafond arrondi au dixième supérieur de l'observation RSS la plus haute ;
- `cpu_per_ppr_job = 4` : le full-grid a consommé 1 259,57 s user et 22,68 s système en 1 310,60 s réelles, soit ~0,98 cœur moyen ; quatre CPU réservés conservent une marge et permettent au plus deux jobs sur huit CPU ;
- `ram_minimum_gb_per_graph_job = 9.1`, arrondi au dixième de Gio supérieur du pic RSS LightGCN mesuré ;
- le préflight évalue le profil PPR de 3,5 Gio et quatre CPU pour `cosine-control-cv` et `ppr-cv`, et le profil maximal de 9,1 Gio et cinq CPU pour les autres stages ;
- deux jobs PPR peuvent être autorisés si au moins 7,0 Gio sont disponibles sur les huit CPU locaux ;
- un seul job LightGCN simultané sur cet hôte ;
- le plafond historique de 45 Gio reste un upper bound prudent, pas un minimum ;
- toute OOM ou pression mémoire anormale impose un arrêt, un diagnostic et une nouvelle baseline, jamais un abaissement du seuil.

## Empreintes des sorties de smoke

- PPR `summary.csv` : `9b9845d814d41fa13a32ffae1a7411a74df1bb64be6b2d6da5f2c9702a12bd6f` ;
- PPR `fold_metrics.csv` : `30eda2955af255622bf04d31eb3f6608478821918cc00cf589798a69f54eb632` ;
- LightGCN `summary.csv` : `550dd24be89bb34cb7a27729633fadd3d1fbaafcd118bca13e26a80729e4fd5b` ;
- LightGCN `fold_metrics.csv` : `b569ae53bde1c35c557c5667edf162e86062984ee582042b180abb393f804c79` ;
- LightGCN `lightgcn_history_all.csv` : `95f9cba2c4ae8eeb2df1607b1b560c885348e68159dd984770b2e4479ed8937d`.

Ces empreintes servent uniquement à attester les sorties observées pendant la mesure. Elles ne sont pas inscrites dans `REGISTRE-RESULTATS.csv`.
