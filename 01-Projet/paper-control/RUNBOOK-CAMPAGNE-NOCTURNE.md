---
date: 2026-07-26
type: runbook
status: campaign-running
owner: assainissement
tags: [benchmark, campagne, grouped-v2, operations]
---

# Runbook — campagne confirmatoire interne `grouped_v2`

> **Baseline levée le 2026-07-27** : le pic RSS mesuré est de 2,95 Gio par `/usr/bin/time` pour une grille PPR complète sur un fold, avec une observation orchestrée ponctuelle à ~3,45 Gio, et de 9,05 Gio pour LightGCN sur `G6-citation-JJ-knn5`. Le manifeste exige donc 3,5 Gio et quatre CPU pour cosinus/PPR, puis 9,1 Gio et cinq CPU pour le profil maximal. Deux jobs PPR sont permis uniquement si au moins 7 Gio sont disponibles ; LightGCN reste strictement séquentiel. Voir `BASELINE-RESSOURCES-2026-07-27.md`.

## Contrat d'exploitation

Depuis `05-Technique/benchmark/etape1_embedding_pur`, définir :

```bash
PY=.venv/bin/python
RUNNER=scripts/64_run_confirmatory_campaign.py
MANIFEST=configs/confirmatory_campaign_grouped_v2.json
AUTH=$(jq -r '.campaign_id' "$MANIFEST")
```

Les stages possédant des sorties par graphe exigent `--graph-id`; le mode global est refusé afin d'empêcher qu'un job `all` et un job G1 écrivent au même endroit. Les jobs distincts peuvent être parallélisés uniquement jusqu'au `max_safe_parallel_jobs` calculé par le préflight, plafonné à deux. Variables BLAS fixées automatiquement à deux threads par sous-processus. GPU non requis.

Chaque stage écrit ses logs dans `_protocol/grouped_v2/logs/<stage>/...` et son statut atomique dans `_protocol/grouped_v2/status/`. `--resume` ne saute un stage que si manifeste et hashes des artefacts concordent. Après interruption, `--resume` déplace d'abord tout artefact partiel appartenant au stage vers `_protocol/grouped_v2/status/quarantine/<stage>/<graphe>/<timestamp>/`, avec manifeste de chemins et hashes, puis recommence le stage. Rien n'est supprimé silencieusement ; il n'existe pas de checkpoint intra-fold/intra-configuration.

Après crash machine ou `SIGKILL`, inspecter les PID/hostnames des `.lock`. La récupération est uniquement explicite : `$PY $RUNNER --manifest $MANIFEST --stage preflight --recover-stale-locks`. Elle archive dans `status/locks/quarantine/` les seuls verrous locaux dont le PID est démontré mort ; tout verrou vivant, distant ou malformé est conservé.

Liste figée, dans l'ordre du manifeste — G1 doit terminer avant les autres afin de produire le contrôle apparié :

```bash
GRAPHS=(G1 G6-citation-AA-knn5 G6-citation-JJ-knn5 G7-citation-AA-cit1-sem025-knn5 G7-citation-AA-cit1-sem050-knn5 G7-citation-AA-cit1-sem100-knn5 G7-citation-AA-cit025-sem1-knn5 G7-citation-JJ-cit1-sem025-knn5 G7-citation-JJ-cit1-sem050-knn5 G7-citation-JJ-cit1-sem100-knn5 G7-citation-JJ-cit025-sem1-knn5)
```

## Gate 0 — contrôles légers autorisés

```bash
$PY $RUNNER --manifest $MANIFEST --stage preflight
$PY $RUNNER --manifest $MANIFEST --stage cosine-control-cv --dry-run
$PY $RUNNER --manifest $MANIFEST --stage ppr-cv --graph-id G1 --dry-run
$PY $RUNNER --manifest $MANIFEST --stage lightgcn-screen --graph-id G1 --dry-run
```

Succès scientifique : onze graphes, 5 603 questions train, 754 eval, cinq folds, 9 entrées maîtres, 84 copies réellement consommées et 16 fichiers de code hashés. Succès opérationnel : minimum RAM mesuré renseigné et `max_safe_parallel_jobs >= 1`, calculé sur la mémoire réellement disponible (`psutil.available`), pas seulement la RAM installée. Échec : fichier absent, hash différent, metadata de folds incompatible, runtime manquant ou minimum RAM non mesuré. Ces commandes ne lancent aucun calcul lourd ; un dry-run ne crée ni résultat ni statut.

## Gate utilisateur A — autoriser les calculs lourds train/CV

Ne poursuivre qu'après autorisation explicite. L'ordre est strict.

### 1. PPR CV puis sélection

```bash
$PY $RUNNER --manifest $MANIFEST --stage cosine-control-cv --resume
for GRAPH in "${GRAPHS[@]}"; do $PY $RUNNER --manifest $MANIFEST --stage ppr-cv --graph-id "$GRAPH" --resume || break; done
```

Le contrôle cosinus direct est calculé une fois sur G1 (`B2-a` Articles, `B3-a` JP), puis traité comme référence partagée. La sélection PPR est effectuée dans le même stage déterministe que la CV : il n'existe plus de faux stage `ppr-select`. Matrice PPR : 11 graphes × 48 configurations × 5 folds. Mesure historique : environ 1 h 34 par graphe ; extrapolation séquentielle ~17 h 15, ou ~9 h 30 avec deux jobs graphe hors contention. Budget par job : jusqu'à 5 CPU / 45 Go RAM, GPU 0. Succès : `summary.csv` et `champions.json` pour chaque graphe, couverture complète et champions Articles/JP éligibles.

### 2. Screening LightGCN et shortlist

```bash
for GRAPH in "${GRAPHS[@]}"; do $PY $RUNNER --manifest $MANIFEST --stage lightgcn-screen --graph-id "$GRAPH" --resume || break; done
$PY $RUNNER --manifest $MANIFEST --stage lightgcn-shortlist --resume
```

Matrice : 11 graphes × une configuration × deux cibles × 5 folds. Une ancienne CV mono-cible prenait ~23–24 min par graphe ; l'extrapolation prudente du runner bi-cible est ~46–48 min par graphe, soit ~8 h 45 séquentielles. Succès : onze `lightgcn_screen/summary.csv`, puis shortlist hashée de 3 à 5 graphes.

### 3. Tuning, robustesse et gel des epochs

```bash
SHORTLIST=data/doctrine_v3plus_bench/_protocol/grouped_v2/lightgcn_shortlist.json
for GRAPH in $(jq -r '.graph_ids[]' "$SHORTLIST"); do $PY $RUNNER --manifest $MANIFEST --stage lightgcn-tune --graph-id "$GRAPH" --resume || break; done
for GRAPH in $(jq -r '.graph_ids[]' "$SHORTLIST"); do $PY $RUNNER --manifest $MANIFEST --stage lightgcn-seeds --graph-id "$GRAPH" --resume || break; done
for GRAPH in $(jq -r '.graph_ids[]' "$SHORTLIST"); do $PY $RUNNER --manifest $MANIFEST --stage freeze-epochs --graph-id "$GRAPH" --resume || break; done
```

Tuning : shortlist × 12 configurations × deux cibles × 5 folds. À partir de la mesure mono-configuration, compter environ 9–10 h par graphe shortlisté en séquentiel ; la phase complète peut dépasser 45 h et doit être répartie sur plusieurs nuits ou par graphes. Robustesse : chaque champion entraîné est rejoué avec seeds 42/43/44, sans choisir le meilleur seed ; moyenne et écart-type sont agrégés. Le gel produit `frozen_champions.json`, lié par hash à cet agrégat. Succès : champions complets avec `selected_epoch_index`, `replay_epochs`, métrique de sélection, robustesse et hashes.

## Gate utilisateur B — autoriser l'accès à l'évaluation interne

Ce gate est séparé : les champions, graphes, hyperparamètres et epochs doivent être gelés avant tout scoring sur les 754 questions. Le préflight peut lire les octets pour vérifier les hashes scellés ; il ne calcule aucune métrique et n'influence aucune sélection.

### 4. Replay interne unique

```bash
for GRAPH in "${GRAPHS[@]}"; do $PY $RUNNER --manifest $MANIFEST --stage internal-replay --graph-id "$GRAPH" --authorize-internal-eval "$AUTH" --resume || break; done
```

Mesure historique : ~6 min par replay/configuration, non garantie. Le runner impose `fixed_final_epoch`. Succès : `selected_champions.json`, `final_champions_summary.csv` et `rankings.parquet` dans `_final_grouped_v2/<graphe>/`. Aucun choix ne peut être modifié après lecture des résultats sans créer une nouvelle campagne explicitement exploratoire.

Le replay revalide lui-même le manifeste canonique et tous les hashes critiques avant la première lecture puis avant publication ; l'invocation directe de `45_run_final_champions.py` ne contourne donc pas le préflight de provenance.

### 5. Diagnostics et exports papier

```bash
$PY $RUNNER --manifest $MANIFEST --stage diagnostics --authorize-internal-eval "$AUTH" --resume
$PY $RUNNER --manifest $MANIFEST --stage paper-exports --authorize-internal-eval "$AUTH" --resume
```

Après les diagnostics, la tâche A lie E002/E003/E014 à leurs artefacts dans `REGISTRE-EXPERIENCES.csv`, puis classe séparément chaque quadruplet expérience × graphe × famille × cible dans `REGISTRE-RESULTATS.csv`. Tant qu'une preuve ou une classification manque, `paper-exports` échoue volontairement. Succès : tableaux et figure primaire avec protocole, hashes dataset/eval/matrice/manifeste, source, `experiment_id`, couverture et verdict par résultat. La tâche B ne copie aucun chiffre avant cette transmission.

## Contrôles du matin

1. Lire chaque statut `failed` ou incomplet et le dernier log associé.
2. Vérifier le stage sans mutation : `$PY $RUNNER --manifest $MANIFEST --stage <stage> --graph-id <graphe> --check-only`. Ce contrôle exige statut, manifeste, liste exacte et hashes des artefacts.
3. Comparer nombre de graphes et artefacts attendus au manifeste ; rechercher les fichiers `.tmp` résiduels.
4. Ne relancer avec `--resume` qu'après avoir distingué interruption propre, manque de ressource et erreur scientifique.
5. En cas de hash différent, ne pas écraser : archiver le diagnostic, corriger le manifeste ou les entrées, puis démarrer une nouvelle campagne identifiée.

## Critères d'arrêt immédiat

- lecture de l'eval avant gel des champions ; hash de dataset/folds/matrice différent ; fuite de groupes ; couverture < 100 % ; sélection sur une métrique non officielle ; écriture vers `_cv`, `_final` ou `_final_champions` legacy ; OOM répété ; deux processus sur le même graphe/stage.
