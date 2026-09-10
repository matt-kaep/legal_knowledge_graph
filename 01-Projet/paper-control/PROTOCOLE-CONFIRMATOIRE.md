---
date: 2026-07-26
type: protocole
status: fige-pour-campagne-interne
owner: assainissement
tags: [benchmark, protocole, k-fold, reproductibilite]
---

# Protocole confirmatoire

Ce document est le contrat figé de la campagne confirmatoire interne `grouped_v2`. Toute modification impose un nouveau manifeste et invalide la reprise des stages issus de l'ancien hash.

## Données

- Apprentissage et tuning : `train_augmented_retrievable_strict`, 5 603 questions.
- Évaluation interne : `eval_rich_retrievable_strict`, 754 questions.
- L'évaluation interne a déjà été consultée : elle ne constitue pas une lockbox inédite.
- Les affirmations finales nécessiteront une nouvelle lockbox jamais consultée.

## Validation

- Cinq folds partagés par toutes les méthodes et tous les graphes.
- Le seed officiel est 42. Le générateur refuse tout autre seed et toute réécriture du namespace `grouped_v2`; la paire CSV/métadonnées est publiée comme un répertoire atomique sous verrou exclusif.
- Groupement transitif par provenance `(source, doc_id, section_id)` et énoncé normalisé.
- Toute reformulation ou duplication normalisée reste dans le même fold.
- Agrégation : moyenne dans chaque fold, puis moyenne non pondérée des cinq folds.
- Comparaisons de graphes appariées sur les mêmes folds et les mêmes questions.

## Sélection

- Articles : `Recall@10`, puis `NDCG@10`, puis `MRR@10`.
- JP : `Hit@10`, puis `NDCG@10`, puis `MRR@10`.
- Une configuration incomplète sur un fold ou sur des questions attendues n'est pas éligible.
- Les deux modalités sont toujours reportées.
- Aucun hyperparamètre n'est choisi sur l'évaluation interne.

## LightGCN

- L'epoch est choisi dans les folds de validation.
- Le replay utilise un nombre d'epochs figé avant l'évaluation interne.
- `selected_epoch_index` est l'index zéro-based de l'historique CV ; `replay_epochs = selected_epoch_index + 1` est le nombre transmis au runner.
- Le replay utilise `fixed_final_epoch` et n'évalue aucun epoch intermédiaire sur l'évaluation interne.
- Le nombre de couches, le learning rate, l'ancrage, le seed et le negative mining sont traités comme des hyperparamètres déclarés.
- Le negative mining principal reste `random` tant qu'une ablation propre n'a pas validé une autre stratégie.

## Comparabilité et traçabilité

- Mêmes espaces candidats, datasets, folds, métriques, seeds et budgets de tuning.
- Une ablation ne modifie qu'un facteur causal à la fois.
- Chaque run confirmatoire produit un manifeste contenant versions, hashes, configuration et couverture.
- Tout run ne satisfaisant pas ces règles porte explicitement le statut `exploratoire`, `non_comparable` ou `invalide`.
- Le contrôle cosinus direct est calculé une seule fois sur G1 et réutilisé comme référence partagée ; il n'est pas présenté comme onze expériences dépendantes du graphe.
- Les caches de questions, embeddings, ordres d'identifiants, matrice et base LEGI qui matérialise G1 sont scellés par hash dans le manifeste.
- Les fichiers d'identifiants historiques au dtype objet restent chargés uniquement après vérification de leur hash scellé ; aucune entrée `.npy` non fiable n'est acceptée dans cette campagne.
- Le statut d'expérience et le verdict par résultat sont enregistrés séparément dans `REGISTRE-EXPERIENCES.csv` et `REGISTRE-RESULTATS.csv`.

## Implémentation figée

- Manifeste : `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2.json`.
- Orchestrateur : `05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py`.
- Audit : `01-Projet/paper-control/AUDIT-SCIENTIFIQUE-2026-07-26.md`.
- Runbook et gates d'autorisation : `01-Projet/paper-control/RUNBOOK-CAMPAGNE-NOCTURNE.md`.
