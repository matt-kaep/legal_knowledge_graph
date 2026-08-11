---
date: 2026-07-26
type: synchronisation
owner: assainissement
recipient: papier
tags: [coordination, benchmark, papier]
---

# Canal A vers B — Assainissement vers papier

## Résumé courant

- Aucun résultat historique ne doit être présenté comme confirmatoire avant son enregistrement dans `REGISTRE-EXPERIENCES.csv`.
- `eval_rich_retrievable_strict` est une évaluation interne déjà consultée, pas une lockbox finale.
- Les folds groupés et l'intégration des runners/replay sont audités. La baseline mémoire est maintenant mesurée et la campagne est en cours avec un seul job graphe ; aucun résultat confirmatoire n'est transmissible avant la fin des gates.
- Les sections méthodes et protocole peuvent suivre le contrat figé ; les sections résultats doivent attendre les exports post-campagne.
- E015 dispose désormais d'un audit humain sur textes intégraux : 30/34 rattrapages bruts sont juridiquement valides après exclusion de quatre cas de même procédure/noyau factuel. Ce résultat reste exploratoire et ne doit pas entrer comme gain LightGCN dans le tableau principal.
- E016 est enregistré pour mesurer, sur G7 seulement, la pertinence graduée A–E des 7 540 positions top-10 puis la contrôler sur 100 cas avocat. La préparation trouve 7 487 couples uniques, zéro fiche manquante et 53 positions JP répétées qui seront pénalisées comme places perdues. Aucun score E016 n'est encore disponible.

## Action demandée à B

- Utiliser les statuts scientifiques du dossier de contrôle.
- Référencer chaque chiffre par `experiment_id`.
- Laisser les conclusions G7 et negative mining conditionnelles.

## Journal des transmissions

### 2026-07-26 — Initialisation

- Décision ou résultat : séparation entre explorations historiques et future campagne confirmatoire.
- Pourquoi B est concernée : le manuscrit actuel contient des chiffres issus d'explorations sur l'eval interne.
- Artefacts affectés : `07-Redaction/papier-v0/`.
- Action demandée : inventorier les affirmations quantitatives et retirer tout vocabulaire confirmatoire non prouvé.
- Statut : à traiter.

### 2026-07-26 — Préparation de campagne terminée

- Décision ou résultat : protocole `grouped_v2` figé, manifeste de onze graphes et runbook reproductible disponibles.
- Preuve : `AUDIT-SCIENTIFIQUE-2026-07-26.md`; aucun entraînement ni replay interne exécuté pendant l'assainissement.
- Artefacts affectés : méthodes/protocole du papier uniquement à ce stade.
- Action demandée : conserver C002–C005 comme proposées/exploratoires ; attendre les exports portant `experiment_id` et statut avant d'insérer des chiffres confirmatoires.
- Statut : transmis.

### 2026-07-26 — Correction après review-loop

- Décision : ajout du contrôle cosinus partagé, scellement données/code/runtime, jeton d'autorisation eval, verrous, refus d'écrasement et verdicts par résultat.
- Preuve : 123 tests ciblés verts ; préflight scientifique valide, mais gate ressources en échec explicite `ram_minimum_unmeasured`.
- Action demandée : conserver toutes les affirmations quantitatives en attente ; ne pas interpréter « méthode préparée » comme « résultats produits ».
- Statut : transmis.

### 2026-07-26 — Gate antérieur supersédé

- Preuve : 103 tests ciblés verts, préflight hashé et aucun namespace de résultat confirmatoire créé.
- Décision supersédée : ce gate précédait l'ajout de E014 et de `REGISTRE-RESULTATS.csv` ; il ne décrit plus le contrat courant.
- Action demandée : ne pas anticiper la classification ; attendre la transmission post-replay de la tâche A.
- Statut : supersédé par les entrées review-loop suivantes.

### 2026-07-26 — Gate final post-review-loop

- Preuve : 141 tests ciblés verts (`18.54 s`), douze scripts compilés, JSON/CSV et diff-check valides, manifeste `30a9ab6e923b29e0997e9a183b811c47023f002ab9d79988add4aa8b479bbc29`.
- Préflight : `scientific_inputs_ok=true`, 84 copies consommées et 16 fichiers de code vérifiés ; `ok=false` uniquement parce que `ram_minimum_unmeasured`, donc aucun lancement autorisé.
- Décision : méthode scientifiquement préparée, mais aucune preuve confirmatoire produite ; E002/E003/E014 et `REGISTRE-RESULTATS.csv` restent en attente.
- Inventaire : `_cv_grouped_v2` et `_final_grouped_v2` absents après les trois dry-runs.
- Statut : transmis.

### 2026-07-27 — Baseline mémoire et lancement autorisé

- Décision : profils mesurés distincts — PPR 3,5 Gio/quatre CPU, LightGCN 9,1 Gio/cinq CPU. Deux PPR sont permis seulement avec 7 Gio disponibles ; LightGCN reste limité à un job.
- Preuve : `BASELINE-RESSOURCES-2026-07-27.md`; les scores des smoke tests restent exploratoires et exclus du registre scientifique.
- Artefacts affectés : manifeste `confirmatory-g1-g6-g7-grouped-v2-2026-07-27`, état A et runbook.
- Action demandée : aucune insertion de chiffre tant que les exports confirmatoires et leurs verdicts ne sont pas transmis.
- Statut : campagne en cours.

### 2026-08-05 — Diagnostic G8 brut sur replay G7

- Décision ou résultat : le replay G7 `JJ/cit1-sem025/knn5` à l'epoch 7 fixé reproduit `JP Hit@10 = 0,250`. Le diagnostic E015 observe un rattrapage `same_rule_application` G8 brut de `0,0380` par question, soit `0,2880` exact-ou-compatible.
- Limite : ce n'est pas une amélioration officielle de LightGCN. G8 n'est pas matérialisé, ses liens LLM bruts n'ont pas encore subi le filtre même procédure/noyau factuel, et l'eval interne a déjà été consultée.
- Preuve : `06-Analyses/comparatifs/g8-llm-verified-jp-jp-2026-08-05/README.md` ; sorties détaillées conservées sur le cluster dans le dossier de diagnostic E015.
- Action demandée : ne pas intégrer ces chiffres au tableau principal ni les qualifier de confirmatoires. Ils peuvent être évoqués uniquement comme analyse exploratoire de pertinence juridique alternative, après audit humain.
- Statut : transmis avec réserve.

### 2026-08-10 — Audit juridique complet du diagnostic E015

- Décision ou résultat : les 34 rattrapages par règle ont été audités exhaustivement sur les textes intégraux Judilibre : 30 `meme_regle_valide`, 4 `meme_procedure_ou_noyau_factuel`. Le `Hit@10 exact` reste `0,250000`; le rattrapage exploratoire passe de `0,038019` brut à `0,032714` audité et l'indicateur exact-ou-compatible de `0,288019` à `0,282714`.
- Preuve : `06-Analyses/comparatifs/g8-llm-verified-jp-jp-2026-08-05/Audit-Juridique-G7-G8-2026-08-10.md`; `audit-g7-g8-2026-08-10/audit_summary.json`; 114 verdicts et 880 textes Judilibre conservés.
- Diagnostic complémentaire : 11/30 erreurs couvertes mais non rattrapées appliquent une même règle, principalement sous-typée `same_legal_issue`; 7/20 erreurs hors couverture exacte de paire appliquent une même règle; l'échantillon stratifié de 30 liens LLM bruts contient 25 mêmes règles valides, 3 mêmes procédures/noyaux factuels et 2 faux positifs.
- Limites : les trois proportions d'échantillon ne sont pas des estimations de population; l'eval interne a déjà été consultée; aucune double annotation ni lockbox inédite; aucune arête G8 finale matérialisée.
- Action demandée : si E015 est mentionnée, employer uniquement la formulation exploratoire du rapport; ne pas présenter `0,282714` comme gain LightGCN ni `25/30` comme précision globale de G8. Attendre un filtre anti-même-procédure figé, une matérialisation versionnée et une future lockbox pour toute affirmation finale.
- Statut : transmis avec réserve; E015 reste `exploratoire`.

### 2026-08-11 — Lancement du chantier E016, évaluation graduée G7

- Décision ou résultat : le protocole et la chaîne reproductible E016 sont implémentés pour juger les 7 540 positions G7 avec question + fiche Step1, classes A–E/`non_jugeable`, score fixed-K et audit avocat aveugle repondéré.
- Pourquoi B est concernée : E016 pourra compléter le `Hit@10` exact par une mesure de pertinence juridique, mais seulement après le run complet et le gate avocat.
- Artefacts affectés : `06-Analyses/comparatifs/e016-g7-graded-jp-2026-08-11/README.md`; scripts 74–80; entrée E016 du registre.
- Action demandée : ne publier aucun score avant la transmission des artefacts agrégés ; conserver explicitement le statut interne/exploratoire même si le gate avocat est franchi.
- Statut : implémentation transmise, exécution LLM et validation avocat en attente.

### 2026-08-11 — Préflight et préparation E016

- Décision ou résultat : le préflight confirme 754 questions et 7 540 rangs. Les fiches Step1 couvrent les 4 865 JP distinctes ; 7 487 couples uniques seront jugés. Le ranking contient 53 répétitions sur 30 questions, issues de doublons du pool JP LightGCN.
- Traitement : chaque couple unique est jugé une fois ; la première occurrence conserve son gain A/B éventuel, toute répétition ultérieure consomme une place du top-10 et vaut zéro.
- Preuve : manifest E016 local hashé et bundle train-only de 30 questions/298 couples, sans sortie LLM réelle à ce stade.
- Action demandée : ne pas interpréter la couverture des fiches ni le nombre de doublons comme un score de pertinence ; attendre le run LLM et le gate avocat.
- Statut : Gate 1 terminé ; Gate 2 réel GPU en attente.

### 2026-08-11 — Pilote E016 suspendu au gate ressource

- Décision ou résultat : les jobs Slurm `935280`, `935290` et `935297` ont échoué avant toute réponse LLM, respectivement sur le cache, la résolution de révision puis une OOM de capture CUDA sur `nodemm02`.
- Preuve : le troisième job charge bien le snapshot complet du juge et 16,47 Gio de poids ; `judge_responses.jsonl` reste absent.
- Conséquence scientifique : aucune classe ni métrique E016 n'existe encore ; le prompt n'a pas été calibré sur l'eval interne.
- Action demandée : aucune pour B. A doit choisir une ressource dédiée, avec préférence pour `L40S` déjà éprouvée par G8, avant de reprendre Gate 2.
- Statut : blocage technique de lancement, pas blocage scientifique du protocole.

### 2026-08-11 — Pilote E016 validé et campagne complète lancée

- Décision ou résultat : le pilote train-only `935516` sur L40S produit 298/298 réponses `ok`, sans erreur ni JSON invalide. La distribution A=83, B=58, C=4, D=25, E=130, NJ=0 est un diagnostic de calibration et non un score G7 à publier.
- Preuve : `judge_responses.jsonl`, `run_summaries/judge-judge_responses.json` et agrégats de calibration sous `data/doctrine_v3plus_bench/calibration/E016-g7-graded-jp-v1/`.
- Exécution complète : `935563` a échoué avant toute réponse sur une erreur ECC matérielle de `node51`; le même run, sans changement de prompt, modèle ou données, est resoumis sous `935568` en excluant ce nœud.
- Action demandée : ne publier aucun score E016 avant agrégation des 754 questions et contrôle avocat. La frontière B reste le point de vigilance prioritaire de l'audit.
- Statut : Gate technique du pilote passé ; jugement complet en cours de lancement, audit avocat en attente.

### 2026-08-11 — Jugement complet E016 terminé, audit avocat préparé

- Décision ou résultat : le job `935568` termine les 7 487 couples avec 7 487 réponses `ok`, zéro erreur et zéro sortie invalide. L'agrégation donne un score gradué@10 brut de `0,427122` et la distribution A=2 442, B=1 592, C=102, D=428, E=2 976, NJ=0.
- Distinction obligatoire : le champ technique `exact_hit_at_10=0,263926` est un indicateur binaire « au moins une gold dans le top-10 », pas le `Hit@10` officiel du benchmark. Ne pas les comparer comme s'ils partageaient la même définition.
- Audit : 100 cas aveugles sont préparés, stratifiés A=27, B=22, C=17, D=17, E=17, avec poids d'inclusion dans la clé privée. La frontière B reste prioritaire.
- Action demandée : ne pas présenter `0,427122` comme score validé avant accord pondéré avocat et précision pondérée A/B ; conserver le statut exploratoire interne.
- Statut : jugement et agrégation terminés ; gate avocat en attente.
