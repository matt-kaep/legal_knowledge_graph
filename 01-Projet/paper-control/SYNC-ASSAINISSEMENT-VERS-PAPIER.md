---
date: 2026-07-26
type: synchronisation
owner: assainissement
recipient: papier
tags: [coordination, benchmark, papier]
---

# Canal A vers B — Assainissement vers papier

## Transmission prioritaire — Checkpoint A3 du 2026-09-02

- **Aucun résultat de modèle ni chiffre de performance nouveau n’est transmissible.** A3 fige le contrat des candidats et vérifie les runners ; les trois tests de fumée ne sont pas des expériences reportables. E024--E030 restent non lancées.
- L’univers structurel de chaque graphe est **23 859 Articles** et **115 304 décisions uniques**. L’univers officiel de candidats retournables est désormais **13 236 Articles** et **114 851 décisions uniques**, avec un ordre commun hashé : Articles `c312dfaaa91a61fca49def5b4489b5b1443894f522c20b06b482276af4e0844c`, JP `065c42517513d7cbf7f050d2b310d7d274b24067bd47db771395816590718b1a`. Les nœuds structurels restants sont auxiliaires de propagation et ne peuvent pas apparaître dans un ranking.
- Le nouveau train `train_augmented_retrievable_strict_no_eval_overlap_effective_retrieval_v3` contient 5 578 questions. L’évaluation reste inchangée à 754 questions, SHA-256 `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59`. Les cinq folds groupés seed 42 font 1 116 / 1 115 / 1 116 / 1 115 / 1 116 questions, sans fuite de provenance ni de texte normalisé.
- Les labels stricts Article et JP sont tous présents dans l’univers officiel, pour le train comme pour l’évaluation. Pour LightGCN, la projection hashée conserve 46 606 des 51 137 occurrences de labels Article étendus, exclut explicitement 4 531 occurrences non récupérables et laisse zéro question d’entraînement sans positif récupérable.
- Preuve : `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json`, SHA-256 `c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3`; manifeste local A3 SHA-256 `92af5d04ef2cfea473bf37d187570ebb890b7d4537c49ecae72c800e13456a6b`. Les 295 tests du benchmark passent ; ce contrôle de code ne crée aucun nouveau résultat de modèle.
- Formulation autorisée, uniquement pour la méthode : « Les graphes conservent des nœuds auxiliaires pour la propagation, tandis que les trois méthodes sont évaluées sur un même univers ordonné de candidats retournables. » Ne pas annoncer de résultat de performance ou de campagne terminée avant la transmission post-E024/E025.

## Transmission prioritaire — Checkpoint A2 du 2026-09-02

- **Aucun nouveau résultat de modèle n’est transmissible.** L’évaluation interne reste inchangée (754 questions, SHA-256 `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59`) et aucun PPR, LightGCN, cosine, reranking, LLM direct ou juge automatique n’a été lancé sur le nouveau checkpoint.
- L’option de retrait validée a été appliquée au train uniquement : 22 QID strictement non récupérables ont été retirés ; le nouveau train compte 5 578 questions. Cinq folds groupés seed 42 ont été régénérés (1 116 / 1 115 / 1 116 / 1 115 / 1 116), sans fuite de provenance ni de texte normalisé. Preuve : `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_candidate_coverage_v2.json`, SHA-256 `784928dd9a88670bf09ae3cc4cfc061629bd9ca5190d1d564d93d61d6bd56555`.
- **Ne pas présenter ce snapshot comme prêt pour des résultats.** Les fichiers de graphe contiennent 23 859 Articles et 115 304 décisions uniques, mais les runners actuels ne peuvent scorer que 13 236 Articles et 114 851 décisions ayant une représentation. L’écart est explicite dans le manifeste ; il doit être résolu avant tout replay. La projection des positifs étendus LightGCN n’est pas interchangeable entre les deux univers et le code la bloque volontairement.
- Formulation autorisée si nécessaire dans la méthode : « Les données d’entraînement ont été re-gelées après suppression des questions dont une référence stricte était absente de l’espace de candidats déclaré ; l’évaluation est restée inchangée. » Ne pas écrire qu’une campagne sur ce snapshot est terminée ou que les 23 859 / 115 304 identifiants ont tous été scorés.

## Transmission prioritaire — Checkpoint A du 2026-09-01

- État : le checkpoint de données est terminé, mais les calculs sont volontairement non lancés. Ne modifier aucun tableau de résultats à partir de cette transmission.
- Split figé : `train_augmented_retrievable_strict_no_eval_overlap_v1` contient 5 600 questions ; trois questions identiques à l'évaluation ont été retirées du train. `eval_rich_retrievable_strict` reste exactement à 754 questions, sans modification. La règle, les QID retirés, les effectifs et tous les SHA-256 sont dans `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_v1.json`.
- Folds : `grouped_v3_no_eval_overlap_v1`, cinq folds de 1 120 questions, seed 42, zéro groupe de provenance ou de texte normalisé réparti sur plusieurs folds. Les artefacts lourds locaux sont hashés dans ce manifeste.
- Point bloquant scientifique : 22 questions du train gelé ont une référence absente de l'espace canonique de candidats ; 0 des 754 questions d'évaluation sont concernées. A ne choisira pas silencieusement de les retirer ou de les ignorer. La décision documentée est nécessaire avant PPR/LightGCN et toute sélection de paramètres.
- Statut des anciens résultats : E022, E017 et E021 sont archivés et vérifiables, mais ne sont pas des résultats du nouveau snapshot. Ne pas les combiner avec E024–E030 ni les présenter comme les résultats du checkpoint.
- Expériences prévues après ce gate : PPR (E024), LightGCN (E025), cosine/BGE-M3 (E026), LLM direct (E027), courbes par rang (E028), reranking comparable (E029) et LLM-as-a-Judge séparé (E030, exploratoire jusqu'à l'accord avocat).

## Résumé courant

- Aucun résultat historique ne doit être présenté comme confirmatoire avant son enregistrement dans `REGISTRE-EXPERIENCES.csv`.
- `eval_rich_retrievable_strict` est une évaluation interne déjà consultée, pas une lockbox finale.
- Les folds groupés et l'intégration des runners/replay sont audités. La baseline mémoire est maintenant mesurée et la campagne est en cours avec un seul job graphe ; aucun résultat confirmatoire n'est transmissible avant la fin des gates.
- Les sections méthodes et protocole peuvent suivre le contrat figé ; les sections résultats doivent attendre les exports post-campagne.
- E015 dispose désormais d'un audit humain sur textes intégraux : 30/34 rattrapages bruts sont juridiquement valides après exclusion de quatre cas de même procédure/noyau factuel. Ce résultat reste exploratoire et ne doit pas entrer comme gain LightGCN dans le tableau principal.
- E016 a jugé les 7 540 positions top-10 de G7 : score gradué brut `0,427122`, encore non validé. L'analyse descriptive trouve au moins une A/B pour 498 des 555 questions sans JP exacte ; ce signal peut refléter une Ground Truth incomplète ou un juge trop permissif. Le contrôle avocat de 100 cas reste obligatoire.
- E017 a terminé ses 33 replays LightGCN et les 14 309 jugements gradués : 7 348 réponses E016 sont réutilisées et 6 961 sont calculées sur GPU. Les scores inter-graphes sont désormais agrégés, mais restent exploratoires et en attente de l'audit avocat E016.
- E018 relie les faux négatifs exacts des 33 replays aux liens G8-Large bruts. Il sert à préparer un audit humain des alternatives juridiques, pas à créer une métrique de retrieval ni à présenter G8 comme une amélioration.

## Action demandée à B

- Utiliser les statuts scientifiques du dossier de contrôle.
- Référencer chaque chiffre par `experiment_id`.
- Laisser les conclusions G7 et negative mining conditionnelles.

## Journal des transmissions

### 2026-08-12 — E019-A : les listes expliquent la proximité des scores

- Décision ou résultat : lecture seule des 33 rankings E017, pour 11 graphes, 55 paires et les seuils top-1/top-3/top-5/top-10. G6-AA et G7-JJ c1/s0,50 partagent 7,04 JP au top-10 (Jaccard 0,565) ; leurs différences représentent environ trois JP exclusives par liste, A/B dans 46,34 % des cas pour G6 et 47,54 % pour G7.
- Interprétation autorisée : les scores gradués proches correspondent à des listes largement communes, complétées par quelques alternatives. Le résultat décrit le comportement des rankings ; il ne démontre pas la supériorité de G7, une causalité des liens JP--JP, ni la validité du juge LLM.
- Artefacts ou sections affectés : `e019_jp_ranking_overlap_pair_metrics.csv`, `e019_jp_ranking_overlap_per_question.parquet`, README E017 et deck G7--G8.
- Action demandée : conserver E019-A comme diagnostic exploratoire. E019-B est prêt pour une comparaison réduite G0--G7, mais son lancement exige une sélection train-only homogène des familles historiques G0--G5.
- Statut : `exploratoire`.

### 2026-08-12 — E017 : analyse approfondie intégrée au deck

- Décision ou résultat : le deck E016--E017 contient désormais cinq slides. Le tableau complet confirme deux vainqueurs distincts : G6-AA pour le `Hit@10` exact (`0,2685`) et G7-JJ c1/s0,50 pour le score gradué (`0,4317`). G6 est meilleur au rang 1 ; G7 passe devant à partir du rang 5. Les deux top-10 partagent 7,04 JP en moyenne et environ 46 % des positions restent sans gain.
- Interprétation autorisée : les sous-groupes et deux décisions lues suggèrent que les liens JP--JP récupèrent mieux les familles jurisprudentielles transversales, tandis que les liens article--article préservent mieux l'ancrage des questions précises et des exceptions procédurales.
- Limites : la comparaison ne constitue pas une ablation causale, le benchmark est une évaluation interne déjà consultée et le score gradué reste en attente de l'audit avocat E016.
- Artefacts affectés : `01-Projet/presentations/E016-Evaluation-JP-Graduee-2026-08-11.tex`, son PDF et la section « Diagnostic approfondi pour la présentation » du README E017.
- Statut : `exploratoire_agrege_en_attente_audit_avocat`.

### 2026-08-12 — E018 : diagnostic transversal G8-Large × E017

- Décision ou résultat : les 24 882 couples replay--question E017 sont comparés à G8-Large brut. Les cas sans hit exact mais raccordés par « même règle » (1 061) ou seulement par « même question » (1 490) ont un score gradué moyen supérieur aux 15 766 cas sans lien G8 ; cette observation sélectionne des cas à lire, elle ne mesure pas un gain de retrieval.
- Pourquoi B est concernée : la dissociation exact / pertinence alternative peut alimenter la discussion des limites du benchmark, à condition de ne pas l'interpréter comme une validation de G8 ou du juge LLM.
- Artefacts ou sections affectés : `06-Analyses/comparatifs/g8-llm-verified-jp-jp-2026-08-05/G8-Large-Analyse-Descriptive-2026-08-12.md`, script `scripts/85_analyze_e017_g8_large_raw_diagnostics.py` et exports E018 sous `data/doctrine_v3plus_bench/E017-intergraph-graded-jp-v1/`.
- Action demandée : conserver `E018` comme diagnostic post-hoc exploratoire ; ne pas ajouter de score composite ni de résultat G8 au tableau principal avant filtre, audit humain et ablation G7 contre G7+G8.
- Statut : `exploratoire`.

### 2026-08-12 — E017 agrégé, réserve avocat maintenue

- Décision ou résultat : les 33 replays et les 14 309 jugements JP gradués sont complets. G6 citation AA donne le meilleur `Hit@10` exact moyen (`0,2685`, ET `0,0051`) ; G7 JJ citation 1 / sémantique 0,50 donne le meilleur score gradué@10 (`0,4317`, ET `0,0020`).
- Pourquoi B est concernée : le résultat explique une dissociation entre Ground Truth exacte et pertinence graduée, utile pour la discussion des limites et non pour désigner une méthode finale.
- Artefacts ou sections affectés : `06-Analyses/comparatifs/e017-intergraph-graded-jp-2026-08-11/README.md` et les exports `e017_graph_seed_metrics.csv`, `e017_graph_metrics.csv`, `e017_per_question_metrics.csv`, `e017_summary.json` sous `data/doctrine_v3plus_bench/E017-intergraph-graded-jp-v1/`.
- Action demandée : ne pas insérer ces chiffres dans un tableau confirmatoire et ne pas conclure à la supériorité d'un graphe avant le paquet avocat E016 ; la différence entre score gradué et `Hit@10` doit rester une observation exploratoire.
- Statut : `exploratoire_agrege_en_attente_audit_avocat`.

### 2026-08-12 — E017 passé au jugement GPU

- Décision ou résultat : 33/33 CV et 33/33 replays complets ; pool exhaustif de 248 820 positions, 14 309 couples uniques, 7 348 réponses E016 réutilisées et 6 961 jugements nouveaux.
- Preuve : pilote A100 `937671` terminé avec 30/30 réponses `ok`, puis run complet A100 `937682` observé `RUNNING` sur `node05`.
- Pourquoi B est concernée : le tableau inter-graphes pourra recevoir un score JP gradué par graphe et seed seulement après résolution complète du journal et agrégation fixed-K.
- Action demandée : ne reprendre encore aucun score E017 ; conserver le statut `exploratoire_internal_evaluation` et la réserve sur l'audit avocat E016.
- Statut : calcul GPU en cours.

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

### 2026-08-18 — Export versionné pour la session Papier

- Branche et manifeste : `paper/ecir-2027-reproducibility`, manifeste portable `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json`, campagne `confirmatory-g1-g6-g7-grouped-v2-repro-v1-2026-08-18`, SHA-256 fichier `18abc26eda35f121cf10cc9eddbce690cc8cbf367020b8080d284b90ef0413ed`, SHA-256 canonique de préflight `b852b0dbf9460ac8541c5447e61b37ea5987c99f0126a0bf74f0947a36a69152`. Le manifeste historique est conservé immuable.
- Audit versionné : `results/benchmark-repro-v1/audit.json`, SHA-256 `a25a00ecdcc4e3827b22d14fb416444c618ef267495b0788650182c78136ceec`; il vérifie les hashes des entrées/code et classe `_final_grouped_v2` comme manquant.
- Reprise autorisée pour B : `results/benchmark-repro-v1/internal_eval_articles.csv` (SHA `e0650daf35e2ea4d799ccaf73840dde75f94f231141ae6b914baaaa8dd925423`), `internal_eval_jp_exact.csv` (SHA `755c4025cbfc93936294a4544c40c4e678b8f18e6a7633b80916088ab54294a6`), `internal_eval_jp_llm_as_a_judge.csv` (SHA `0199bcf105d05ddd8045ef990151c0fa335b054fada4d8b3de4d3bd5e4200a60`), `train_cv_retrieval.csv` (SHA `bb62a2ef41d8413ad21c5cb6bcc1d7571825238aa19525218f6ec09d02c8645e`), et `e016_jp_llm_and_exact_context.csv` (SHA `fd256704442245d7444e478f6d8cb00a9874a2f5ff6f8259e8acf37ec2867558`). Chaque ligne contient méthode, graphe, split, folds/seeds, moyenne, dispersion, configuration, statut, source et SHA de l'artefact source.
- Formulation autorisée Articles : « E017 fournit un export interne exploratoire de Recall@10 sur les 11 graphes, 3 seeds et 754 questions ; la sélection d'epoch est issue des folds train-only et le replay est gelé. » Ne pas transformer ce score d'évaluation interne en preuve confirmatoire ni en lockbox finale.
- Formulation autorisée JP exacte : « L'export sépare le `Hit@10` officiel, NDCG@10 et MRR@10 du diagnostic binaire `exact_any_gold_at_10`. » Pour E016, les valeurs récupérées sont LLM-as-a-Judge `0,4271220159`, exact_any `0,2639257294`, Hit@10 officiel `0,25`, NDCG `0,1648240979`, MRR `0,1404630331`; l'audit avocat reste manquant.
- Formulation autorisée juge : « Les scores LLM-as-a-Judge E016/E017 sont exploratoires et en attente du contrôle humain avocat ; ils ne soutiennent pas une supériorité de graphe. » La comparaison G6/G7 ne doit pas être appelée ablation causale.
- État historique au 2026-08-18 : l'évaluation finale PPR était manquante et E021 était alors seulement préparé. Ce point est supersédé par la transmission E021 v5 ci-dessous ; l'évaluation finale PPR reste à produire.
- Paquet humain : `results/audit/e016-lawyer-audit/README.md` et `manifest.json`. Les CSV sample/key restent dans le checkout de données local ; `lawyer_agreement.json` doit être produit par l'annotation aveugle avant tout changement de statut.
- Statut de transmission : exploitable pour les tableaux internes, méthodes de protocole et limites ; non autorisé pour une formulation confirmatoire, un classement final ou une affirmation de supériorité.

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

### 2026-08-11 — Analyse descriptive E016, exact contre gradué

- Décision ou résultat : le `Hit@10` officiel reste `0,250000`. Sur 754 questions, 199 ont au moins une JP exacte dans le top-10, alors que 692 ont au moins une JP classée A/B. Parmi les 555 sans hit exact, 498 ont une A/B et 57 n'en ont aucune.
- Contrôle de cohérence : 191/204 couples exacts uniques sont classés A/B ; 3 821/7 283 couples non exacts uniques le sont aussi. Le gain moyen décroît du rang 1 (`0,5623`) au rang 10 (`0,3289`).
- Interprétation autorisée : la Ground Truth peut être fortement incomplète, mais cette hypothèse reste indissociable d'un éventuel surclassement par le juge tant que l'audit avocat n'est pas terminé.
- Limite nouvelle : l'échantillon avocat actuel contient seulement 2 couples exacts et aucune JP attendue non retournée. Il valide le juge sur les sorties G7, pas la pertinence générale des 978 couples de Ground Truth.
- Action demandée : ne pas intégrer ces chiffres comme résultat validé dans le papier. Après le gate avocat, prévoir un audit séparé des couples de Ground Truth avant le diagnostic G8 et les propositions d'amélioration G7.
- Statut : analyse exploratoire transmise ; verdict scientifique et diagnostic G8 toujours en attente.

### 2026-08-11 — Campagne E017 inter-graphes lancée

- Décision ou résultat : les onze graphes G1/G6/G7 sont en calcul LightGCN avec trois seeds et cinq folds groupés. Les CV actifs sont `936154` et `936188`; les replays `936155` et `936189` sont déjà soumis avec dépendance par graphe.
- Preuve : 91 entrées vérifiées par SHA-256 sur le cluster; 33 tâches scientifiques isolées; artefact `06-Analyses/comparatifs/e017-intergraph-graded-jp-2026-08-11/README.md`.
- Limite : aucun score E017 n'est encore produit. L'évaluation reste interne/exploratoire et le futur score gradué restera conditionné par l'audit avocat E016.
- Action demandée : ne pas ajouter de classement inter-graphes au papier ou aux slides avant transmission des replays complets et de l'agrégation contrôlée par la tâche A.
- Statut historique au 2026-08-11 : calcul cluster en cours ; supersédé par la transmission E021 v5 datée du 2026-08-18 ci-dessous.

### 2026-08-18 — Transmission E021 reranking comparable v5

À reprendre par Papier, avec statut strictement exploratoire et incomplet :

- Manifeste : `experiments/reranking-comparable/manifest_cluster_gpu_runtime_v5.json`, SHA-256 `d6f8d45602218248f54a32b84160f4f1276e441efcf50c9638a7338d1a5f8cd4`.
- Jobs : 2 262 unités (`K_in=20`, `K_out=10`, 754 questions × 3 familles), SHA `36f03198d39ec764095d3340ea1f8dc006b941e585245e30bd8e4c14a0a5afdf`.
- Réponses : SHA `780e53c1d69481660869d4c0f9e68b377be7d4ab2f0b5c869eae1522b9a3a9fb`; 2 249 clés `(famille,qid)` valides, 13 manquantes : cosine 7, PPR 1, LightGCN 5.
- Export JP : `results/reranking-comparable/E021-cluster-gpu-runtime-v5/metrics.json`, SHA `077bea64f34b4382ca00e251359c67d782d0c3a5c807fb571567fa357c7e5954`.
- Audit : `results/reranking-comparable/E021-cluster-gpu-runtime-v5/audit.json`, SHA `8bc00d9c5850c985343c036bb589703c4e4d302bf2c95fb86a86cc0aedd67e4d`.
- Code : runner SHA `6495fc37727a531f5e00555713c76d810faceb4ee1be146fb20ddc4a62aa97db`; agrégateur SHA `6d6b76d6b56b4b88aa16961bd46bf9935f347ced514b60cc78f8100ad8b4bbfc`.

Valeurs exportées sur les réponses valides uniquement, avec écart-type d’échantillon par question :

| Famille | Couverture | Hit@10 officiel | NDCG@10 | MRR@10 |
|---|---:|---:|---:|---:|
| cosine/BGE-M3 | 747/754 | 0,2707496653 ± 0,4383721651 | 0,2195999266 ± 0,3809056369 | 0,2065229171 ± 0,3769360233 |
| PPR | 753/754 | 0,2768924303 ± 0,4405456056 | 0,2283575587 ± 0,3876235637 | 0,2167414996 ± 0,3848983077 |
| LightGCN | 749/754 | 0,2930574099 ± 0,4480084150 | 0,2437210300 ± 0,3969861945 | 0,2323081569 ± 0,3949802308 |

Formulation autorisée : « E021 fournit un export interne exploratoire d’un reranker commun appliqué à trois viviers JP réels, avec métriques exactes séparées et couverture explicitée ; 13 unités restent manquantes. » Ne pas écrire qu’une famille surclasse une autre, ne pas appeler cette évaluation confirmatoire, et ne pas produire de résultat Articles à partir de cet E021 JP-only. `exact_any_gold_at_10` reste un diagnostic séparé du `Hit@10` officiel.

Le LLM-as-a-Judge n’est pas le reranker E021 et doit rester dans les exports E016/E017 séparés ; l’audit avocat `lawyer_agreement.json` reste en attente dans le chantier A.

### 2026-08-31 — Décision de présentation E021

- E021 sera livré à Papier comme tableau JP annexe, distinct du tableau principal de benchmark. Il comparera le même reranker appliqué aux trois viviers réels : similarité, navigation dans le graphe et modèle d'apprentissage sur graphe.
- Condition avant insertion : reprendre les 13 unités manquantes et exporter les métriques exactes sur couverture complète. Le tableau indiquera la méthode source, le nombre de questions, Hit@10 officiel, NDCG@10 et MRR@10.
- Le LLM-as-a-Judge reste hors de ce tableau : c'est une évaluation distincte, pas le reranker.

### 2026-08-31 — Préparation Télécom, sans nouveau résultat à reprendre

- Décision ou résultat : les sorties PPR finales des 11 graphes sont présentes sur Télécom et leur manifeste historique est identifié. A prépare E022, un audit non destructif qui revalidera les champions issus des cinq folds, les 754 questions, les rangs et les métriques exactes. Aucun PPR n'est relancé.
- E021 : un job GPU reprenable est prêt pour les 13 réponses de reranking manquantes. Il conserve le JSONL historique, n'ajoute que les clés famille/question absentes et produit un nouveau reçu de couverture.
- Artefacts : `experiments/confirmatory-recovery/manifest_ppr_final_audit_v1.json`, `experiments/reranking-comparable/manifest_cluster_gpu_runtime_v5_resume_v1.json` et `05-Technique/benchmark/etape1_embedding_pur/scripts/run_telecom_reproducibility.sh`.
- Action demandée : aucune modification des tableaux du papier avant réception des deux audits. E021 reste un tableau JP annexe, sans score LLM-as-a-Judge.
- Statut : préparation vérifiée localement et poussée dans le commit `ddda94e` de `paper/ecir-2027-reproducibility`; jobs Télécom non encore soumis (passerelle SSH momentanément indisponible).

### 2026-09-01 — Calculs Télécom soumis, aucun chiffre nouveau à reprendre

- Les jobs d'audit et de récupération sont maintenant en cours : E022 / Slurm `969381` (CPU, audit PPR non destructif) et E021 / Slurm `969382` (L40S, reprise de 13 unités de reranking seulement). Le commit de soumission portable est `fe20588` sur `paper/ecir-2027-reproducibility`.
- Aucun résultat nouveau n'est transmis au papier à ce stade. Les métriques PPR finales et le tableau JP de reranking restent en attente des rapports hashés, de la couverture complète et du contrôle par A.
- Le LLM-as-a-Judge et l'audit avocat restent hors de ces jobs et de tout tableau E021.

### 2026-09-01 — Tentatives v1 arrêtées ; aucune donnée nouvelle à reprendre

- Les deux jobs v1 ont échoué avant de produire un résultat nouveau : E022 `969381` a exposé un décalage de profondeur entre le résumé historique PPR (@20) et la métrique requise (@10) ; E021 `969382` a rencontré une révision distante de modèle devenue indisponible. Le JSONL E021 n'a reçu aucune nouvelle réponse.
- A prépare les reprises v2 sans relancer PPR : l'audit v2 vérifiera que les onze résumés @20 concordent avec les classements, puis exportera les métriques exactes @10 depuis ces mêmes classements. Le reranking v2 utilise le snapshot local figé déjà présent sur Télécom pour ne dépendre d'aucun service externe.
- Statut Papier : ne reprendre aucun chiffre, tableau ou classement de ces tentatives. La prochaine transmission contiendra uniquement des sorties v2 hashées et validées par A.

### 2026-09-01 — Reprises v2 en cours, toujours aucun chiffre à reprendre

- E022 v2 / Slurm `969521` audite les onze sorties PPR existantes sans les recalculer. E021 v2 / Slurm `969522` reprend uniquement les 13 unités manquantes depuis le snapshot local figé du reranker.
- Les entrées E021 ont été revalidées avant soumission et aucune réponse historique n'a été modifiée. Attendre les rapports v2, leurs hashes et le contrôle de couverture par A avant toute modification du papier.

### 2026-09-01 — E022 terminé ; E021 reste en reprise, rien à intégrer

- E022 / Slurm `969521` a audité avec succès les onze sorties PPR existantes. Son rapport hashé vérifie les champions sélectionnés sur les folds train-only, la couverture et le recalcul exact des métriques à 10 depuis les classements archivés. A matérialise maintenant les tables légères ; ne pas reprendre les chiffres avant cette exportation versionnée.
- E021 / Slurm `969522` a exécuté les 13 unités manquantes, mais seulement deux réponses sont valides : onze sorties du modèle répètent un identifiant, malgré un vivier de 20 décisions distinctes. Le reçu est explicitement incomplet ; aucune métrique partielle ne doit être reprise.
- A prépare une reprise v3 limitée aux onze unités, avec même modèle, prompt, température, questions et viviers. Elle documente une normalisation déterministe des doublons de sortie, nécessaire car vLLM ne sait pas imposer l'unicité dans ce schéma JSON. Le reranking reste un tableau JP annexe exploratoire, séparé du LLM-as-a-Judge.
- Statut Papier : attendre les tables PPR versionnées et le reçu E021 complet. Aucun changement de manuscrit demandé à ce stade.

### 2026-09-01 — E022 PPR : tables exactes disponibles pour le papier

- E022 est maintenant `confirmee_interne` : l'audit Slurm `969521` a contrôlé, sans rejouer PPR, les onze graphes, les 5 folds de sélection train-only, les 754 questions et les classements archivés. Rapport source : SHA-256 `d0217fb5ae304d5c640101044bad75475d33cca67d94b6458a7e72f9d01d06d4`.
- À reprendre : `results/benchmark-repro-v1/ppr_final_table_articles.csv` (SHA `d3b05f49d9c6d957dac29dc3b3107fb4e65e701394eb35f19a054ffa8f4e45a3`) pour Recall@10, NDCG@10 et MRR@10 ; `results/benchmark-repro-v1/ppr_final_table_jp.csv` (SHA `ebace97fc605254049ed268ce6c1ae61e577b83178d1c709ce91fe58195fc83c`) pour Hit@10 officiel, NDCG@10 et MRR@10. Le détail complet de 66 valeurs est `ppr_final_internal_eval_exact.csv`, SHA `2f311319e25b9a876a1904534f56ef53ae6729310f34aaa0ef27e60816918bb0`.
- Formulation autorisée : « Les résultats PPR ont été sélectionnés sur cinq folds groupés d'entraînement, puis évalués à 10 résultats sur les 754 questions ; les tableaux distinguent les tâches Articles et Jurisprudence, avec Hit@10 officiel séparé des diagnostics binaires. » Reprendre les configurations et écarts-types tels quels dans les CSV ; ne pas reprendre les anciens résumés historiques à 20 comme métriques à 10.
- E021 n'est pas encore transmissible : la reprise GPU v3 ne concerne que les onze unités de reranking restantes. Le LLM-as-a-Judge reste séparé de ce reranker et sans nouvelle valeur dans cette transmission.

### 2026-09-01 — E021 reranking : tableau JP annexe complet disponible

- E021 est désormais complet : Slurm `969635` a produit un reçu vérifié de 2 262/2 262 unités, soit 754/754 questions pour chacun des trois viviers. Reçu : `results/reranking-comparable/E021-cluster-gpu-runtime-v5-resume-v3/completion_receipt.json`, SHA `456b810a773ae1cafe7ec8d5ec909b19986c797379deec7079e463018484d8e5`.
- Tableau à reprendre : `results/reranking-comparable/E021-cluster-gpu-runtime-v5-resume-v3/table_jp_reranking_exact.csv`, SHA `cbf3785e2afa394e755372dca2a81012f82991493317cec45f6ad50a8dd05b4b`. Détail des neuf valeurs : `internal_eval_jp_reranking_exact.csv`, SHA `8020ee093ce4822425731602bcb24ff6222c9cdf0c42fd76a3e3a4b90dac16f5`.

| Vivier reranké | Hit@10 officiel | NDCG@10 | MRR@10 |
|---|---:|---:|---:|
| Cosine / BGE-M3 | 0,2708885942 ± 0,4385007964 | 0,2202137204 ± 0,3816775665 | 0,2072581154 ± 0,3777762464 |
| PPR | 0,2765251989 ± 0,4403684495 | 0,2280546973 ± 0,3874553522 | 0,2164540440 ± 0,3847236266 |
| LightGCN | 0,2928824050 ± 0,4476458218 | 0,2436718744 ± 0,3969293745 | 0,2323591638 ± 0,3949335717 |

- Formulation autorisée : « Un même reranker, figé à `K_in=20` et `K_out=10`, a été appliqué aux trois viviers JP réels sur les mêmes 754 questions ; les métriques exactes sont rapportées séparément. » Statut : `exploratoire` et tableau annexe, sans LLM-as-a-Judge. Ne pas le présenter comme une évaluation Articles ni comme une conclusion de supériorité.
- Note de reproduction : 11 réponses du modèle contenaient 19 répétitions d'identifiants. La reprise v3, documentée dans `manifest_cluster_gpu_runtime_v5_resume_v3.json`, conserve la première occurrence et complète 19 positions selon l'ordre gelé du vivier ; les 2 251 autres réponses ne sont pas modifiées.

### 2026-09-01 — État de vérification de la branche

- La branche publiable passe 322 tests dans son worktree isolé. Huit tests supplémentaires nécessitent les jeux de données et graphes lourds volontairement absents de Git ; ils échouent explicitement en préflight sur fichier absent, sans produire ni modifier de résultat.
- Les exports et tableaux transmis ci-dessus sont eux vérifiés dans ce worktree, et les calculs ayant besoin des données ont été exécutés puis contrôlés sur Télécom. Aucun changement de manuscrit n'est demandé par cette note opérationnelle.

### 2026-09-01 — Branche de publication propre, résultats à reprendre

- La branche à utiliser pour le dépôt public est `paper/ecir-2027-reproducibility-clean`, dérivée de `main` et limitée au code de reproduction, configurations, prompts, schémas, tests, manifests, exports légers et documents A. Elle n'ajoute ni fichier du manuscrit, ni PDF, ni donnée lourde ou brute. L'ancienne branche `paper/ecir-2027-reproducibility` est conservée comme archive de récupération mais ne doit pas être la PR de publication.
- E022 PPR : `results/benchmark-repro-v1/ppr_final_internal_eval_exact.csv`, SHA Git `2f311319e25b9a876a1904534f56ef53ae6729310f34aaa0ef27e60816918bb0`, contient 66 valeurs exactes, 11 graphes et 754 questions. Tables : `ppr_final_table_articles.csv` SHA `d3b05f49d9c6d957dac29dc3b3107fb4e65e701394eb35f19a054ffa8f4e45a3`, `ppr_final_table_jp.csv` SHA `ebace97fc605254049ed268ce6c1ae61e577b83178d1c709ce91fe58195fc83c`. Le manifeste miroir explique le hash CRLF d'origine et le hash LF Git. `audite` dans le CSV décrit la matérialisation ; les 66 entrées du registre portent `confirmee_interne`.
- E017 LightGCN : les trois exports internes existants couvrent 11 graphes, trois seeds (42, 43, 44) et 754 questions. Les métriques exactes Articles/JP sont séparées ; le LLM-as-a-Judge reste distinct, exploratoire et en attente de l'audit avocat.
- E021 reranking : reçu complet SHA `456b810a773ae1cafe7ec8d5ec909b19986c797379deec7079e463018484d8e5`, métriques source SHA `08b96023f25a9a36d8041c4f8ef5341e4927466dfae581a92b9e4156e12e2d1c`, tableau SHA `cbf3785e2afa394e755372dca2a81012f82991493317cec45f6ad50a8dd05b4b`. À présenter seulement comme tableau JP annexe exploratoire, sans juge LLM ni conclusion de supériorité.
- Vérification de la branche propre sous Python 3.12 : 266 tests passent ; les huit échecs restants demandent explicitement des données exclues du dépôt et échouent par `FileNotFoundError`. `psutil` et `psycopg2-binary` sont désormais déclarés, et le lanceur Télécom déduit la branche du checkout local.
- Point d'entrée GitHub : PR brouillon https://github.com/matt-kaep/legal_knowledge_graph/pull/2 (`paper/ecir-2027-reproducibility-clean` vers `main`). Elle est à relire avant fusion ; son état GitHub ne change pas les statuts scientifiques ci-dessus.

### 2026-09-01 — Brief de reprise pour la session Papier

Utilisez la branche `paper/ecir-2027-reproducibility-clean` (PR brouillon : https://github.com/matt-kaep/legal_knowledge_graph/pull/2) comme unique point d'entrée. Les fichiers du manuscrit ne sont pas dans cette branche ; elle fournit les tableaux et preuves à citer.

1. **Tableaux principaux PPR.** Prenez les valeurs, écarts-types et configurations directement dans `results/benchmark-repro-v1/ppr_final_table_articles.csv` (Articles : Recall@10, NDCG@10, MRR@10) et `results/benchmark-repro-v1/ppr_final_table_jp.csv` (Jurisprudence : Hit@10 officiel, NDCG@10, MRR@10). Les deux tableaux couvrent les 11 graphes et 754 questions ; leur détail complet est `ppr_final_internal_eval_exact.csv`. Ils sont documentés comme résultats internes confirmés dans le registre A.
2. **Résultats exacts LightGCN.** Les exports à consulter sont `results/benchmark-repro-v1/internal_eval_articles.csv` et `results/benchmark-repro-v1/internal_eval_jp_exact.csv` : 11 graphes, 754 questions et trois graines d'exécution (42, 43, 44). Articles et Jurisprudence doivent rester deux tableaux distincts, avec leurs métriques propres.
3. **Tableau complémentaire de reranking.** Si un tableau annexe JP est retenu, utiliser exclusivement `results/reranking-comparable/E021-cluster-gpu-runtime-v5-resume-v3/table_jp_reranking_exact.csv`. Il compare le même reranker appliqué aux trois viviers (similarité, PPR et LightGCN), sur les mêmes 754 questions, avec 20 candidats en entrée et 10 en sortie. Le présenter comme complément exploratoire, sans conclusion de supériorité.
4. **À ne pas utiliser pour soutenir une conclusion.** `internal_eval_jp_llm_as_a_judge.csv` est une métrique d'évaluation séparée, non le reranker ; elle reste exploratoire tant que l'audit avocat n'a pas produit `lawyer_agreement.json`. Ne pas l'employer pour une revendication de supériorité.

Règles de rédaction : ne pas publier les identifiants internes d'expérience ; ne pas appeler `exact_any_gold_at_10` le Hit@10 officiel ; ne pas présenter une comparaison G6/G7 comme une ablation causale. Pour chaque chiffre, reprendre la valeur exacte depuis le CSV et conserver séparés Articles, Jurisprudence, métriques exactes et métrique par juge automatique.
