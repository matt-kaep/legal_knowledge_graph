---
date: 2026-07-26
type: audit
status: scientifically-prepared-blocked-resource-baseline
owner: assainissement
tags: [benchmark, audit, grouped-v2, reproductibilite]
---

# Audit scientifique — préparation `grouped_v2`

## Conclusion

Le chemin confirmatoire interne est préparé et isolé des artefacts historiques. Les contrôles scientifiques légers passent ; aucun entraînement, full sweep, replay sur les 754 questions, embedding ou appel LLM n'a été exécuté. Le statut est `SCIENTIFICALLY_PREPARED / BLOCKED_RESOURCE_BASELINE` : la RAM minimale d'un job n'a pas été mesurée et le préflight interdit donc le lancement sans seuil inventé. Les 45 Gio historiques sont conservés comme plafond prudent uniquement.

## Preuves d'entrée vérifiées

- Train strict : 5 603 QID uniques ; hash `7e6cf685dcf2c6d1794fbbd4cd46fbcf26ec9f6107a38c7ebf478b1450e9b9ac`.
- Folds : cinq folds de 1 120, 1 120, 1 121, 1 121 et 1 121 QID ; aucun QID manquant, supplémentaire ou dupliqué.
- Hash folds : `70b3d1fba45b3ce45386e97277319d5ca7e3802c60d361eb5a9d90461577c1ad`.
- Groupes : 505 groupes ; taille maximale 44 ; zéro groupe de provenance et zéro groupe de texte normalisé traversant plusieurs folds.
- Évaluation interne : 754 questions ; hash `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59` ; déjà consultée historiquement, donc non assimilable à une lockbox.
- Les onze benches train référencés par le manifeste ont le même hash que le train officiel ; les matrices et tableaux d'identifiants partagés sont vérifiés par le préflight.

## Écarts trouvés et résolution

| Priorité | Écart constaté | Risque | Résolution vérifiable |
|---|---|---|---|
| P0 | Une métrique primaire globale `Hit@10` classait aussi les Articles. | Sélection contraire au protocole. | Mapping Articles `Recall@10`, JP `Hit@10`, tests de priorité et départage. |
| P0 | Le mode historique LightGCN pouvait sélectionner un checkpoint en évaluant chaque epoch sur le jeu passé au runner final. | Fuite de l'évaluation interne dans la sélection. | Mode `fixed_final_epoch` sans évaluation intermédiaire ; replay groupé forcé sur ce mode. |
| P0 | L'index d'epoch zéro-based pouvait être transmis comme nombre d'epochs. | Replay raccourci d'un epoch. | Champs distincts `selected_epoch_index` et `replay_epochs`; conversion testée. |
| P0 | Un champion incomplet pouvait atteindre un fallback legacy. | Sélection avec folds/QID manquants. | En présence du schéma groupé, seuls les candidats `eligible_champion` sont acceptés ; absence de métriques officielles = erreur. |
| P1 | Aucun manifeste unique ni namespace de campagne n'existait. | Mélange de graphes, hashes ou sorties historiques. | Manifeste exact de onze graphes ; sorties `_cv_grouped_v2`, `_final_grouped_v2` et `_protocol/grouped_v2`. |
| P1 | Screening, tuning et robustesse LightGCN pouvaient partager une sortie. | Écrasement de la preuve de sélection. | Namespaces séparés et shortlist liée au hash du manifeste. |
| P1 | La reprise reposait implicitement sur l'existence de fichiers. | Stage incomplet pris pour terminé. | Statut JSON atomique, hashes d'artefacts, logs et manifeste identique requis pour `--resume`. |
| P2 | Les diagnostics de profondeur confondaient parfois fraction d'attendus retrouvés et présence d'au moins un attendu. | Interprétation ambiguë. | Deux colonnes séparées : `expected_coverage_at_k` et `any_expected_answer_at_k`. |

## Corrections issues de la revue indépendante

- Replay LightGCN cible-spécifique : identité complète propagée aux métriques et rankings ; aucun mélange Articles/JP ou d'epochs.
- Comparaisons G6/G7 contre G1 appariées sur configuration et fold, sans inclure l'identité du graphe dans la clé de jointure.
- Shortlist reverifiée contre hashes, provenance et sources à chaque consommation.
- Gel `frozen_champions.json` dépendant des trois seeds et de leur agrégat de robustesse.
- Aucune régénération de surface legacy en `grouped_v2`; dry-run sans écriture, logs non écrasés et code shell non nul si une sortie attendue manque.
- Export papier bloqué sans classification explicite et artefact de preuve existant dans le registre.

## Invariants désormais exécutables

- Agrégation fold-first non pondérée et couverture exacte des cinq folds/QID.
- Sélection Articles `Recall@10 > NDCG@10 > MRR@10`; JP `Hit@10 > NDCG@10 > MRR@10`.
- Deux sélections LightGCN distinctes, puis epoch de replay dérivé exclusivement des historiques CV.
- Replay groupé refusé si protocole, hashes, couverture ou `replay_epochs` sont absents/incompatibles.
- Shortlist : G1, deux contrôles G6 et meilleur G7 par modalité, dédupliqués, maximum cinq graphes.
- Exports papier refusés sans provenance, `experiment_id`, hashes et statut scientifique autorisé.

## Limites scientifiques restantes

- Les résultats historiques restent exploratoires, non comparables ou invalides selon le registre ; ils ne sont pas convertis rétroactivement en preuves confirmatoires.
- `eval_rich_retrievable_strict` permet seulement des conclusions `confirmee_interne` ou `refutee`, jamais `confirmee_lockbox`.
- La durée du tuning LightGCN dépasse vraisemblablement une nuit en séquentiel. Les estimations du runbook sont des extrapolations de mesures historiques, pas des garanties.
- Les sorties confirmatoires n'existent pas encore : le présent audit valide la préparation et les garde-fous, pas les résultats futurs.

## Gates légers

Les commandes de vérification, leur résultat frais et l'inventaire d'absence de calcul lourd sont consignés dans `ETAT-ASSAINISSEMENT.md`. Gate final après cinq itérations : 141 tests passés (`18.54 s`), douze scripts compilés, JSON/CSV et diff-check valides, manifeste `30a9ab6e923b29e0997e9a183b811c47023f002ab9d79988add4aa8b479bbc29`, `scientific_inputs_ok=true`, dry-runs valides, namespaces confirmatoires absents et blocage opérationnel explicite `ram_minimum_unmeasured`. Le lancement suit exclusivement `RUNBOOK-CAMPAGNE-NOCTURNE.md` après mesure de ressources.
