---
title: Évaluation graduée des JP retournées par G7 et audit avocat
date: 2026-08-11
status: review-required
experiment_id: E016
tags:
  - benchmark
  - jurisprudence
  - g7
  - llm-judge
  - audit-avocat
---

# Évaluation graduée des JP retournées par G7 et audit avocat

## 1. Décision

Exécuter les trois chantiers suivants dans cet ordre :

1. construire et geler un instrument LLM de pertinence juridique graduée ;
2. l'appliquer puis le faire contrôler par un avocat uniquement sur les top-10 JP de G7 ;
3. utiliser cette mesure validée pour diagnostiquer G7, puis concevoir des expériences d'amélioration séparées.

Le premier run ne compare pas plusieurs méthodes. Il répond à la question : « quelle est la qualité juridique des dix JP retournées par G7 pour chacune des 754 questions ? »

## 2. Motivation

Le `Hit@10 exact` reste reproductible, mais il considère comme échec toute JP absente de la Ground Truth historique. L'audit E015 a montré que certaines JP retournées et non exactes appliquent néanmoins une règle juridiquement compatible. Il faut donc conserver deux lectures complémentaires :

- récupération exacte de la Ground Truth historique ;
- pertinence juridique graduée de chaque JP effectivement retournée.

La seconde lecture ne remplace pas la Ground Truth et ne devient pas automatiquement confirmatoire. Elle reste une évaluation interne jusqu'à validation du juge et, pour une affirmation finale, jusqu'à utilisation d'un jeu d'évaluation final indépendant.

## 3. Périmètre et exclusions

### Inclus dans E016

- méthode : `LightGCN-trained_K2`, replay G7 gelé, top-10 JP ;
- population : 754 questions de `eval_rich_retrievable_strict` ;
- profondeur : `K=10`, soit 7 540 positions attendues ;
- entrée JP : carte Step1 existante utilisée pour G8 ;
- jugement LLM A--E et `non_jugeable` ;
- score gradué normalisé, distributions et justifications ;
- échantillon avocat de 100 couples issus uniquement des sorties G7.

### Exclus de ce premier run

- cosine brut, PPR, autres graphes ou autres méthodes ;
- création d'une nouvelle fiche juridique ou nouvelle analyse Step1 ;
- ajout du texte intégral dans l'entrée du LLM ;
- utilisation d'une relation G8, de la Ground Truth, du rang ou du nom de méthode par le juge ;
- matérialisation d'arêtes G8 finales ;
- modification ou tuning de G7 ;
- sélection de prompt, modèle, poids ou seuil sur les 754 questions internes.

## 4. Contrat d'entrée

Chaque travail unique est identifié par `(qid, jp_id)`. Le payload visible par le juge contient uniquement :

1. le texte de la question ;
2. la carte Step1 de la JP :
   - `synthese_pour_avocat` ;
   - `fondements_retenus` ;
   - `cited_articles` ;
   - `solution_resume` ;
   - `arguments_parties[{argument, reponse_juge}]`.

Les champs suivants restent dans les artefacts de reconstruction, jamais dans le prompt : méthode, rang, statut exact contre la Ground Truth, relations G8 et distances de graphe.

La carte est réutilisée telle qu'elle existe. Aucun champ vide n'est régénéré par un autre LLM.

## 5. Taxonomie et arbre de décision

- **A — Règle directement applicable à la question** : la JP applique une règle ou un raisonnement qui permet de répondre directement à la question.
- **B — Apport par distinction, exception ou limite** : la JP traite la même question juridique et apporte une nuance substantielle utile pour répondre.
- **C — Même question juridique, sans solution exploitable** : la JP rencontre le problème, mais ne fournit pas de réponse utilisable sur le fond.
- **D — Proximité factuelle ou procédurale seulement** : même contexte, faits ou procédure, sans règle utile pour répondre.
- **E — Aucun rapport juridique substantiel** : hors sujet ou proximité lexicale superficielle.
- **`non_jugeable`** : la carte Step1 est objectivement insuffisante pour qualifier le couple. Cette sortie n'est pas une classe d'hésitation et ne doit jamais absorber un cas frontière A--E.

Arbre appliqué dans cet ordre :

1. carte insuffisante pour comprendre la JP : `non_jugeable` ;
2. aucune question juridique précise commune : E ;
3. proximité seulement factuelle ou procédurale : D ;
4. même question, sans solution de fond exploitable : C ;
5. règle ou raisonnement répondant directement : A ;
6. apport substantiel uniquement par distinction, exception, limite ou qualification complémentaire : B.

## 6. Sortie minimale

Le juge renvoie un JSON strict :

```json
{
  "classe": "A | B | C | D | E | non_jugeable",
  "justification": "Une phrase juridiquement concrète."
}
```

La justification doit nommer la règle, la solution ou l'absence de solution qui détermine la classe, puis relier ce point à la question. Elle ne doit pas paraphraser la définition de la classe. Les formulations génériques comme « la décision applique directement la règle permettant de répondre » sont interdites.

Le LLM ne reproduit pas de chaîne de pensée, de question juridique intermédiaire, de liste de règles ni de seconde fiche JP.

## 7. Score et exports

Gains :

- A : `1` ;
- B : `0,5` ;
- C, D, E et `non_jugeable` : `0`.

Pour chaque question `q` :

```text
score_gradue@10(q) = somme(gain_i × premiere_occurrence_i, i=1..10) / 10
```

Le dénominateur reste toujours `K=10`. Une position manquante ou `non_jugeable` ne réduit pas le dénominateur. Une JP répétée dans le top-10 est jugée une seule fois : sa première occurrence peut rapporter son gain A/B, chaque occurrence suivante consomme une position mais vaut zéro. Le score global G7 est la moyenne macro sur les 754 questions.

Exports obligatoires :

- score global `score_gradue@10` dans `[0, 1]` ;
- score par question ;
- comptes et proportions A, B, C, D, E et `non_jugeable` ;
- taux `non_jugeable@10` ;
- nombre et taux de positions JP répétées ;
- `Hit@10 exact` historique reporté séparément, jamais fusionné avec le score gradué ;
- détail des 7 540 positions : `qid`, `jp_id`, rang, classe, gain, justification et statut technique ;
- provenance de la classe : `llm` ou `preflight_missing_card` ;
- manifest contenant hashes des questions, rankings, cartes, prompt, schéma, script et modèle.

Le score n'est pas pondéré par le rang. Une A en position 1 et une A en position 10 ont le même gain dans E016.

## 8. Modèle et calibration

Le premier instrument réutilise l'infrastructure locale/cluster éprouvée par M3 et G8 :

- modèle : `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` ;
- température : `0` ;
- sortie : JSON Schema strict ;
- un couple question--JP par appel ;
- cache write-through par `(qid, jp_id, model_id, prompt_hash, card_hash)`.

Le prompt est calibré sur un pilote séparé issu du train, avec des cas attendus directs, complémentaires, connexes, procéduraux et hors sujet. Aucun exemple, seuil ou formulation n'est choisi en lisant les sorties des 754 questions internes.

Après calibration, le modèle, le prompt, le schéma, `K`, les gains et les paramètres de génération sont hashés et gelés avant la préparation du run E016.

## 9. Pipeline des chantiers 1 et 2

### Gate 1 — Préflight G7 et cartes

- retrouver le ranking G7 gelé utilisé par E015 ;
- prouver 754 questions et dix positions par question, ou matérialiser explicitement les positions manquantes ;
- récupérer les cartes Step1 existantes sans relancer Step1 ;
- calculer la couverture des cartes et les champs vides ;
- produire un manifest de sources et hashes.

### Gate 2 — Pilote de calibration hors eval

- construire un pilote train fixe et stratifié ;
- exécuter le juge ;
- inspecter les classes et justifications ;
- corriger le prompt uniquement à ce stade ;
- geler le contrat complet.

### Gate 3 — Run G7 interne complet

- conserver les 7 540 positions et générer un travail aveugle par couple unique ;
- dédupliquer les éventuels couples répétés ;
- juger chaque couple unique une fois ;
- reconstruire les dix positions de chaque question ;
- exiger zéro travail technique manquant avant agrégation ;
- produire scores, distributions, détail et manifest.

### Gate 4 — Échantillon avocat G7

Sélectionner 100 couples uniques avec seed fixe et allocation cible : 25 A, 20 B, 15 C, 15 D, 15 E et 10 `non_jugeable`. Si une strate ne contient pas assez de couples, prendre toute la strate et redistribuer son quota aux classes de score identique les moins représentées, puis à la population restante. À l'intérieur de chaque classe, équilibrer autant que possible les rangs `1--3` / `4--10` et la présence / absence dans la Ground Truth.

Conserver pour chaque ligne la taille de sa strate et sa probabilité d'inclusion. Reporter les résultats bruts de l'échantillon stratifié et les estimations repondérées selon les proportions A--E / `non_jugeable` du run complet ; ne jamais présenter l'accord brut d'un échantillon surreprésentant A/B comme une précision populationnelle.

L'avocat reçoit la question et la décision, avec la carte Step1 et le texte intégral accessibles. Il ne voit pas la classe LLM, la méthode, le rang, la Ground Truth ni les relations G8. Il utilise la même taxonomie A--E / `non_jugeable` et fournit une justification juridique courte.

Mesures de validation :

- matrice de confusion complète ;
- accord A/B contre C/D/E ;
- précision et rappel de A et B ;
- accord pondéré sur les gains `1 / 0,5 / 0` ;
- désaccord moyen absolu sur les gains ;
- intervalles d'incertitude reportés avec la taille d'échantillon.

Gate minimal avant usage substantiel dans le papier : accord pondéré sur les gains d'au moins `0,70` et précision du groupe positif A/B d'au moins `0,85`. Si le gate échoue ou si le prompt est modifié après lecture de l'échantillon, ces 100 cas deviennent une calibration et une nouvelle validation indépendante est obligatoire.

## 10. Gestion des erreurs

- timeout, erreur vLLM, réponse vide ou JSON invalide : erreur technique, retry borné, jamais `non_jugeable` ;
- erreur technique persistante : run incomplet, aucune agrégation finale ;
- carte absente : position conservée, classe `non_jugeable`, justification indiquant l'absence de carte et `label_source=preflight_missing_card` ;
- carte présente mais champs partiels : le juge applique l'arbre et n'utilise `non_jugeable` que si la qualification juridique est impossible ;
- doublon `(qid, jp_id)` : un jugement en cache ; première occurrence scorée, occurrences suivantes marquées `duplicate_position` et de gain effectif nul ;
- sortie hors taxonomie ou justification générique : invalide, retry ;
- moins de dix JP retournées : positions manquantes explicites de gain zéro, dénominateur inchangé.

## 11. Tests et critères de clôture

Tests unitaires :

- calcul à dénominateur K fixe ;
- gains A/B/C/D/E/`non_jugeable` ;
- reconstruction des positions et doublons ;
- payload aveugle sans méthode, rang, Ground Truth ni G8 ;
- validation du JSON et rejet des justifications génériques ;
- séparation erreur technique / `non_jugeable` ;
- sélection avocat déterministe et stratifiée.

Tests d'intégration :

- petit run mock local de bout en bout ;
- pilote réel train avec cache/reprise ;
- agrégation reproduite depuis les réponses sans nouvel appel LLM ;
- vérification des hashes et des 7 540 positions finales.

E016 est techniquement complet lorsque le manifest est fermé, les 754 questions et 7 540 positions sont présentes, aucune erreur technique ne reste ouverte et tous les exports se recalculent depuis le détail brut.

E016 reste `exploratoire` tant que le contrôle avocat n'est pas terminé. Même après validation, il décrit une évaluation interne déjà consultée et non un jeu d'évaluation final indépendant.

## 12. Chantier 3 — Diagnostic puis amélioration

Une fois l'instrument gelé et le run G7 contrôlé :

1. croiser exactitude historique et classes A--E pour séparer alternative valable, erreur réelle et Ground Truth douteuse ;
2. rattacher les relations G8 uniquement comme variables explicatives : lien LLM brut `same_rule_application`, `same_legal_issue`, filtre procédure/noyau factuel, absence de paire candidate et distances structurelles ;
3. ajouter ultérieurement le cosine brut avec le même juge gelé pour mesurer les passages A/B gagnés ou perdus entre cosine et G7 ;
4. formuler des hypothèses d'amélioration de G7 par type d'erreur ;
5. tester chaque modification dans une expérience distincte, avec tuning train/CV et sans sélection sur les 754 questions internes.

Les distances G8 de deux ou trois sauts restent des diagnostics de connectivité, jamais une équivalence juridique ni une note de pertinence.

## 13. Artefacts attendus

Sous un dossier E016 dédié :

- `manifest.json` ;
- `g7_rankings_top10.parquet` ;
- `judge_jobs.jsonl` ;
- `judge_responses.jsonl` ou shards équivalents ;
- `graded_jp_detail.csv` ;
- `graded_jp_per_question.csv` ;
- `graded_jp_summary.json` ;
- `lawyer_audit_sample.csv` ;
- `lawyer_annotations.csv` ;
- `lawyer_agreement.json` ;
- rapport Markdown distinguant résultats exacts, gradués, audit avocat et limites.

Le registre Task A reçoit E016 avant le run, puis son statut et ses artefacts sont mis à jour à chaque gate. Aucun chiffre n'est transmis au papier comme résultat prouvé avant clôture des contrôles correspondants.
