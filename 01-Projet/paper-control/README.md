---
date: 2026-07-26
type: coordination
status: active
tags: [projet, coordination, benchmark, papier]
---

# Contrôle partagé — assainissement scientifique et papier

Ce dossier est la source de vérité de la coordination entre les deux tâches du projet.

## Tâches

- **A — Assainissement scientifique** : rend le protocole, le code et les preuves publiables.
- **B — Papier** : construit l'argument scientifique et rédige le manuscrit.

## Lecture obligatoire au démarrage

### Tâche A

1. `AGENTS.md`
2. `PROTOCOLE-CONFIRMATOIRE.md`
3. `ETAT-ASSAINISSEMENT.md`
4. `ETAT-PAPIER.md`
5. `SYNC-PAPIER-VERS-ASSAINISSEMENT.md`
6. `REGISTRE-AFFIRMATIONS.csv`
7. `REGISTRE-RESULTATS.csv`

### Tâche B

1. `AGENTS.md`
2. `PROTOCOLE-CONFIRMATOIRE.md`
3. `ETAT-PAPIER.md`
4. `ETAT-ASSAINISSEMENT.md`
5. `SYNC-ASSAINISSEMENT-VERS-PAPIER.md`
6. `REGISTRE-EXPERIENCES.csv`
7. `REGISTRE-RESULTATS.csv`

## Mise à jour obligatoire

Après une décision, un run important, une réfutation ou une modification du plan du papier :

- A met à jour `ETAT-ASSAINISSEMENT.md`, `REGISTRE-EXPERIENCES.csv` et `SYNC-ASSAINISSEMENT-VERS-PAPIER.md` ;
- B met à jour `ETAT-PAPIER.md`, `REGISTRE-AFFIRMATIONS.csv` et `SYNC-PAPIER-VERS-ASSAINISSEMENT.md`.

Chaque canal de synchronisation doit commencer par un résumé courant, puis conserver un journal daté des transmissions.

## Statuts scientifiques

- `proposee` : affirmation à tester ;
- `exploratoire` : observation utile, mais protocole insuffisant pour conclure ;
- `confirmatoire_en_cours` : protocole figé, preuve pas encore complète ;
- `confirmee_interne` : confirmée sur l'évaluation interne déjà consultée ;
- `confirmee_lockbox` : confirmée sur une lockbox jamais consultée ;
- `refutee` : expérience compatible avec le protocole et résultat contraire ;
- `non_comparable` : protocoles, candidats ou données incompatibles ;
- `invalide` : fuite, sélection sur eval, couverture incomplète ou artefact insuffisant.

## Règles anti-conflit

- Une tâche ne modifie pas les fichiers possédés par l'autre.
- Une demande passe par le canal sortant du demandeur.
- Une correction urgente est signalée dans le canal ; le propriétaire applique la modification.
- Les chiffres du papier sont référencés par `experiment_id`, jamais recopiés depuis une slide.
- `REGISTRE-EXPERIENCES.csv` qualifie la validité et la complétude d'une expérience ; `REGISTRE-RESULTATS.csv` porte le verdict scientifique par graphe, famille et cible.
- Les slides et le papier sont des sorties ; les registres et artefacts sont les sources.

## Format d'une transmission

```markdown
### YYYY-MM-DD — sujet

- Décision ou résultat :
- Pourquoi l'autre tâche est concernée :
- Artefacts ou sections affectés :
- Action demandée :
- Statut / échéance :
```
