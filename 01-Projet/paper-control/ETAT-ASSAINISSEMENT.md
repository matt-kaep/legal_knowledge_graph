---
date: 2026-07-26
type: etat-projet
status: campaign-running
owner: assainissement
tags: [benchmark, assainissement, k-fold]
---

# État A — Assainissement scientifique

## Objectif

Transformer les explorations existantes en un benchmark reproductible, comparable et auditable, puis produire les preuves que le papier peut utiliser.

## État courant

`CAMPAIGN_RUNNING` — la méthode et les garde-fous sont préparés. Le 27 juillet, les baselines ont mesuré 2,95 Gio de pic RSS PPR sur une grille complète d'un fold, avec une observation orchestrée ponctuelle à ~3,45 Gio, et 9,05 Gio pour LightGCN sur le plus gros graphe confirmatoire. Le manifeste réserve 3,5 Gio et quatre CPU par PPR, puis 9,1 Gio et cinq CPU pour le profil maximal. Deux PPR peuvent tourner uniquement lorsque 7 Gio sont réellement disponibles ; LightGCN reste séquentiel.

- Folds vérifiés : 5 603 QID uniques, cinq folds équilibrés, zéro fuite de provenance/texte normalisé.
- Sélection : Recall Articles et Hit JP, fold-first, couverture complète obligatoire.
- Replay LightGCN : `selected_epoch_index` distingué de `replay_epochs`, mode final sans évaluation intermédiaire.
- Onze graphes et toutes les grilles sont figés dans un manifeste hashé.
- Historique G0–G7, negative mining, LLM+RAG et M3 classifié dans le registre.
- `eval_rich_retrievable_strict` reste une évaluation interne déjà consultée, jamais une lockbox.
- Diagnostic G8 E015 : replay G7 JP reproductible et audit juridique sur textes intégraux terminé pour 114 cas. Sur les 34 rattrapages bruts, 30 sont `meme_regle_valide` et 4 relèvent de la même procédure/noyau factuel ; le rattrapage exploratoire audité vaut `0,032714` contre `0,038019` brut. Aucune métrique officielle n'est modifiée et la matérialisation G8 reste bloquée par le filtre anti-même-procédure.
- E016 : chaîne d'évaluation graduée G7 implémentée et préparation locale terminée, sans score à ce stade. Le périmètre contient 754 questions × top-10 JP de `LightGCN-trained_K2`, soit 7 540 positions, 7 487 couples uniques et zéro fiche Step1 manquante. Cinquante-trois positions répètent une JP déjà classée ; elles restent dans K mais leur gain effectif vaut zéro après la première occurrence. L'audit avocat aveugle cible 100 cas stratifiés repondérés.
- Pilote E016 cluster : aucun jugement produit après trois échecs de démarrage documentés. Le snapshot modèle est désormais pin au commit complet et se charge, mais la capture CUDA manque de mémoire sur la partition générique `mm`; le choix d'une partition GPU dédiée reste à valider avant resoumission.

## Gates restants

1. Exécuter la chaîne train/CV autorisée et vérifier chaque statut avant la dépendance suivante.
2. Geler les champions avant l'accès aux 754 questions de l'évaluation interne ; l'autorisation persistante du goal couvre le gate séparé, mais le jeton exact reste obligatoire.
3. Après campagne : inspection des statuts/logs, classification par résultat et transmission des seuls résultats prouvés au papier.

## Artefacts de référence

- `AUDIT-SCIENTIFIQUE-2026-07-26.md`
- `PROTOCOLE-CONFIRMATOIRE.md`
- `RUNBOOK-CAMPAGNE-NOCTURNE.md`
- `REGISTRE-EXPERIENCES.csv`
- `REGISTRE-RESULTATS.csv`
- `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2.json`
- `BASELINE-RESSOURCES-2026-07-27.md`
- `06-Analyses/comparatifs/g8-llm-verified-jp-jp-2026-08-05/Audit-Juridique-G7-G8-2026-08-10.md`
- `06-Analyses/comparatifs/e016-g7-graded-jp-2026-08-11/README.md`

## Preuves légères fraîches

- `pytest` ciblé protocole/runners/replay/orchestration/exports : **141 tests passés** après l'itération 5 (`18.54 s`), un avertissement de dépréciation tiers `pyparsing`.
- `py_compile` : douze scripts critiques de campagne compilés sans erreur.
- Validation structurelle : manifeste JSON et deux registres CSV valides ; `git diff --check` sans erreur sur le périmètre.
- Préflight : manifeste `30a9ab6e923b29e0997e9a183b811c47023f002ab9d79988add4aa8b479bbc29`, `scientific_inputs_ok=true`, 11 graphes, 5 603 train, 754 eval interne, cinq folds, 9 entrées immuables, 84 copies consommées et 16 fichiers de code hashés ; `ok=false`, blocage explicite `ram_minimum_unmeasured`, `max_safe_parallel_jobs=0`.
- Dry-runs : `cosine-control-cv`, `ppr-cv --graph-id G1` et `lightgcn-screen --graph-id G1`, tous trois exit 0 sans sous-processus de calcul.
- Inventaire après dry-run : `_cv_grouped_v2` absent ; `_final_grouped_v2` absent.
- Commandes reproductibles : voir le Gate 0 de `RUNBOOK-CAMPAGNE-NOCTURNE.md`.
- Review-loop : cinq itérations, quatre axes indépendants ; dernière passe sans constat critique ou important après correction du gate direct de replay.
- E015 : 114/114 cas qualifiés sur textes intégraux Judilibre, 880/880 JP matérialisées, 34 rattrapages recalculés après audit ; `Hit@10 exact = 0,250000` inchangé, rattrapage exploratoire `0,032714`, aucune arête G8 finale matérialisée.

## Besoins reçus du papier

Voir `SYNC-PAPIER-VERS-ASSAINISSEMENT.md`.

## Dernière mise à jour

2026-08-11 — E016 enregistré, préflight passé et bundles préparés. Le pilote train-only ne produit encore aucune réponse : cache et snapshot corrigés, puis OOM CUDA sur `nodemm02`; resoumission suspendue avant un quatrième changement de lancement.
