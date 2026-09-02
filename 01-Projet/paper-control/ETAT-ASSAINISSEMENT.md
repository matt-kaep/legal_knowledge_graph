---
date: 2026-07-26
type: etat-projet
status: checkpoint-a3-complete-no-full-campaign
owner: assainissement
tags: [benchmark, assainissement, k-fold]
---

# État A — Assainissement scientifique

## Campagne B1 — PPR en cours ; LightGCN B1-r1 arrêté techniquement et archivé

La première soumission B1 du 2 septembre 2026 reste archivée par le manifeste `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3.json` (SHA-256 fichier `f1107b126e11b0457dc28a4fbe3db621b1c061a932e2acda1eb1368bac0be649`). Son préflight distant a validé 50 entrées hashées et le job cosine `977100` a terminé ; les onze PPR (`977101`) et huit LightGCN lancés (`977102`) ont échoué **avant tout calcul CV**. La cause vérifiée est un défaut de propagation du split A3 vers le chargeur de folds : les fichiers A3 étaient bien présents sur Télécom, mais les deux lanceurs demandaient le chemin historique par défaut. Les tâches LightGCN restantes ont été annulées après confirmation de la même erreur. Ces sorties ne fournissent aucun champion, aucune métrique CV ni résultat reportable.

- B1-r1 est un nouveau manifeste immuable : `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r1.json` (SHA-256 fichier `1b612a182742244dad59006e6d01b826a0285f01123aeeae67321b48c9de5e9a`). Il référence explicitement A3 (SHA-256 `c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3`), la tentative interrompue, les scripts corrigés, les données, les graphes, les grilles de CV, les ressources Slurm et un espace de sorties entièrement neuf.
- Préflight local B1-r1 réussi : 50 entrées immuables vérifiées ; train de 5 578 questions, évaluation inchangée de 754 questions ; univers retournable commun de 13 236 Articles et 114 851 JP. Les 23 tests ciblés PPR/LightGCN/B1-r1 passent (`1,86 s`), les nouveaux wrappers Slurm valident et `git diff --check` est vierge. La suite complète est requise avant le commit et la relance.
- Le contrat métrique B1-r1 utilise le Hit@K normalisé, après dédoublonnage du ranking. Pour les Articles, Hit@10 et Recall@10 sont identiques dans ce benchmark : le maximum vérifié est de dix labels stricts par question d’évaluation. NDCG@10 et MRR@10 restent exportés séparément.
- E026 (cosine/BGE-M3, Slurm `977124`) a terminé sous B1-r1 ; ses rankings et métriques attendent les contrôles de couverture/hash avant inscription au registre des résultats. Le smoke PPR isolé (`977125`) a couvert les cinq folds, les 5 578 QID et les deux modalités sur A3 ; E024 est **en cours** sur les onze graphes (Slurm `977157`). À l'observation de 11:27, les onze tâches sont au fold 2/5, entre 400 et 800 questions sur 1 115 pour le fold courant ; aucun champion n'est encore disponible.
- Le smoke LightGCN isolé (`977158`, G1, cinq folds, une époque) a validé la même couverture et le gate mémoire. La CV complète E025 B1-r1 (`977184`, sous-job exécuté `977185`) a été **arrêtée volontairement après 4 h 52 min 23 s** : quatre entraînements de 30 époques ont coûté environ 58 minutes chacun. La grille actuelle représente 24 entraînements par fold, donc environ 116 heures par graphe pour cinq folds, alors que l'allocation est limitée à 24 heures. Les dix autres graphes ont été annulés avant exécution. Quatre historiques partiels du seul fold 0 de G1 sont conservés sur Télécom (SHA-256 `83231b0665026d6d9f1124f838f8738abfc9724e836e19434517935855c149cb`, `27ce0c28e4c2d739c25071cd4393c68d4b55ccb020160f52a73c609d45794c9c`, `d84c4ecc71791735d82e73c4e59a358bf9c1ff22569e9676568a8f733c3306d1`, `2a58e7ca79417e80b0bec90bb5be68fc5b34e1b4faf9469be3dcb990f8ae6b98`). Elles ne constituent ni une CV complète, ni un champion, ni un résultat ; elles sont non reportables. Une reprise B1-r2 devra être scellée séparément, avec sorties atomiques et reprise par configuration avant toute nouvelle soumission.
- E028 (courbes K=1–100) dépend exclusivement des rankings B1-r1 gelés ; E027, E029 et E030 ne sont pas autorisées dans B1.

## Checkpoint A3 — Univers de candidats figé, sans campagne complète

Le 2 septembre 2026, la décision scientifique « option 2 » a été matérialisée sans modifier l’évaluation. L’univers structurel du graphe reste **23 859 Articles** et **115 304 décisions uniques**. L’univers officiel de candidats retournables, commun à cosine, PPR et LightGCN, est maintenant l’intersection stable entre les représentations et les nœuds du graphe : **13 236 Articles** (ordre SHA-256 `c312dfaaa91a61fca49def5b4489b5b1443894f522c20b06b482276af4e0844c`) et **114 851 décisions uniques** (ordre SHA-256 `065c42517513d7cbf7f050d2b310d7d274b24067bd47db771395816590718b1a`).

- Preuve versionnée : `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json` (SHA-256 `c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3`) et manifeste local de snapshot A3 (SHA-256 `92af5d04ef2cfea473bf37d187570ebb890b7d4537c49ecae72c800e13456a6b`). A2 et les versions antérieures restent immuables.
- Le nouveau snapshot train est `train_augmented_retrievable_strict_no_eval_overlap_effective_retrieval_v3`, avec **5 578** questions. L’évaluation `eval_rich_retrievable_strict` est byte-for-byte identique, avec **754** questions et SHA-256 `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59`. Les folds `grouped_v5_no_eval_overlap_effective_retrieval_v3`, seed 42, contiennent 1 116 / 1 115 / 1 116 / 1 115 / 1 116 questions ; aucune provenance ni aucun texte normalisé ne traverse plusieurs folds.
- Contrôles stricts : zéro chevauchement train--évaluation ; zéro label strict Article ou JP absent dans le train ou l’évaluation. Pour les labels Article étendus utilisés par LightGCN : 51 137 occurrences, 46 606 conservées, 4 531 exclues explicitement, zéro question sans positif récupérable.
- Les nœuds restants sont explicitement `auxiliary_non_returnable_nodes` : 10 623 Articles et 453 décisions. Ils peuvent propager de l’information dans les graphes, mais ne peuvent ni être classés ni être retournés. Les onze graphes ont été vérifiés directement : mêmes effectifs et même ordre de candidats.
- Des garde-fous de code refusent maintenant tout candidat hors univers officiel, tout doublon de ranking et toute projection LightGCN non explicitement documentée. La baseline cosine sait aussi extraire correctement l’incidence Article--JP d’un graphe hybride carré.
- Validation reproductible : **295 tests passent** (`16,00 s`) avec les données A3, les onze scripts modifiés compilent, `git diff --check` est vierge et les CSV/manifeste/hashes sont cohérents. Les deux tests historiques de replay ont été réparés sans copier la version modifiée du checkout utilisateur : `24_build_global_table.py` provient d’un blob Git compatible et les manifests historiques restent scellés.
- Seuls trois tests de fumée techniques ont été exécutés : cosine (2 questions), PPR sur un graphe (2 questions) et LightGCN sur un graphe, un fold et une époque (2 questions). Ils ont produit exclusivement des rankings dans l’univers officiel. **Aucun run complet E024--E030, aucune sélection de champion et aucune métrique de modèle reportable n’a été lancée.**
- Une première tentative A3 ayant oublié l’intersection avec les nœuds du graphe a été conservée, sans suppression, dans `data/doctrine_v3plus_bench/_invalid_attempts/`; elle n’a exécuté aucun modèle et est explicitement invalide.

## Checkpoint A2 — Données figées, calculs bloqués avant exécution

Le 2 septembre 2026, l’option 1 validée a été appliquée sans toucher à l’évaluation : 22 questions train/CV dont un label strict Article ou JP n’était pas récupérable dans l’espace de nœuds du graphe ont été retirées. Le nouveau snapshot `train_augmented_retrievable_strict_no_eval_overlap_candidate_covered_v2` contient **5 578** questions ; `eval_rich_retrievable_strict` reste byte-for-byte identique à **754** questions (SHA-256 `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59`). Les cinq folds seed 42 ont été régénérés : 1 116 / 1 115 / 1 116 / 1 115 / 1 116 questions, avec zéro fuite de provenance ou de texte normalisé.

- Preuve versionnée : `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_candidate_coverage_v2.json` (SHA-256 `784928dd9a88670bf09ae3cc4cfc061629bd9ca5190d1d564d93d61d6bd56555`). Les artefacts locaux, leurs chemins relatifs et leurs hashes y sont listés ; `benchmark_freeze_no_eval_overlap_v1` reste immuable.
- Les contrôles sur le nouvel instantané sont tous nuls : chevauchement train--évaluation par QID, chevauchement par texte normalisé, labels stricts Article absents et labels stricts JP absents. Ils passent dans l’univers de nœuds demandé (23 859 Articles ; 115 304 décisions uniques) comme dans l’univers réellement consommé aujourd’hui par les runners.
- **Blocage matériel avant tout modèle :** `load_retrieval_view()` ne score actuellement que les identifiants disposant d’une représentation : 13 236 Articles et 114 851 décisions uniques, plutôt que les 23 859 / 115 304 nœuds du graphe. La projection LightGCN produite pour l’univers brut exclut 1 951 labels étendus ; le même audit dans l’univers réellement scoré en exclurait 4 531. Aucun entraînement ne peut donc démarrer sans choisir et documenter un seul univers de candidats. Le runner refuse désormais cette projection incompatible, plutôt que de filtrer des labels silencieusement.
- Aucun calcul PPR, cosine/BGE-M3, LightGCN, reranking, LLM direct ou LLM-as-a-Judge n’a été lancé dans ce checkpoint. E024--E030 sont uniquement replanifiés et restent bloqués par E023/E031.
- Validation logicielle : les 46 tests ciblant le freeze, les folds, PPR et LightGCN passent ; la suite entière reste bloquée à la collecte de deux tests historiques, car `45_run_final_champions.py` importe `24_build_global_table.py`, absent de cette branche propre. Le fichier existe seulement dans le checkout utilisateur sous une modification non suivie par cette branche ; il n’a pas été copié.

## Objectif

Transformer les explorations existantes en un benchmark reproductible, comparable et auditable, puis produire les preuves que le papier peut utiliser.

## État courant

`CHECKPOINT_A_TERMINE_BLOQUE_SUR_COUVERTURE_TRAIN` — le 1er septembre, le train et l'évaluation ont été re-gelés sans chevauchement, en préservant strictement les 754 questions d'évaluation. Les trois QID/textes strictement identiques ont été retirés du train : 5 603 → 5 600 questions. Le nouveau protocole `grouped_v3_no_eval_overlap_v1` contient cinq folds de 1 120 questions, sans fuite de provenance ni de texte normalisé. La preuve versionnée est `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_v1.json`.

- Aucun calcul de modèle n'est autorisé dans ce checkpoint : PPR, LightGCN, cosine/BGE-M3, LLM direct, reranking et juge automatique sont seulement listés comme replays à venir dans E024–E030.
- L'évaluation est saine vis-à-vis de l'espace de candidats : 0/754 question avec référence Article ou jurisprudence absente. Les candidats sont dédoublonnés de façon stable avant score/métrique (23 859 Articles ; 115 304 décisions uniques, depuis 117 374 lignes).
- Une anomalie historique distincte bloque volontairement la sélection : 22/5 600 questions train comportent une référence absente de ces espaces de candidats. Elle ne vient ni des trois retraits ni d'une différence entre graphes. Une décision de politique train/CV doit être enregistrée avant la moindre sélection de paramètres.
- Les résultats E022 (PPR), E017 (LightGCN) et E021 (reranking) restent archivés et traçables, mais ne sont pas les sorties du snapshot `grouped_v3_no_eval_overlap_v1` ; ils ne doivent pas être mélangés aux futurs résultats E024–E030.

`CAMPAIGN_RUNNING` — la méthode et les garde-fous sont préparés. Le 27 juillet, les baselines ont mesuré 2,95 Gio de pic RSS PPR sur une grille complète d'un fold, avec une observation orchestrée ponctuelle à ~3,45 Gio, et 9,05 Gio pour LightGCN sur le plus gros graphe confirmatoire. Le manifeste réserve 3,5 Gio et quatre CPU par PPR, puis 9,1 Gio et cinq CPU pour le profil maximal. Deux PPR peuvent tourner uniquement lorsque 7 Gio sont réellement disponibles ; LightGCN reste séquentiel.

- Folds vérifiés : 5 603 QID uniques, cinq folds équilibrés, zéro fuite de provenance/texte normalisé.
- Sélection : Recall Articles et Hit JP, fold-first, couverture complète obligatoire.
- Replay LightGCN : `selected_epoch_index` distingué de `replay_epochs`, mode final sans évaluation intermédiaire.
- Onze graphes et toutes les grilles sont figés dans un manifeste hashé.
- Historique G0–G7, negative mining, LLM+RAG et M3 classifié dans le registre.
- `eval_rich_retrievable_strict` reste une évaluation interne déjà consultée, jamais une lockbox.
- Diagnostic G8 E015 : replay G7 JP reproductible et audit juridique sur textes intégraux terminé pour 114 cas. Sur les 34 rattrapages bruts, 30 sont `meme_regle_valide` et 4 relèvent de la même procédure/noyau factuel ; le rattrapage exploratoire audité vaut `0,032714` contre `0,038019` brut. Aucune métrique officielle n'est modifiée et la matérialisation G8 reste bloquée par le filtre anti-même-procédure.
- E016 : jugement complet et agrégation terminés sur 754 questions × top-10 JP de `LightGCN-trained_K2`, soit 7 540 positions et 7 487 couples uniques. Les 7 487 réponses sont valides. Le score gradué brut vaut `0,427122`; distribution A=2 442, B=1 592, C=102, D=428, E=2 976, NJ=0. Cinquante-trois répétitions restent dans K mais leur gain effectif vaut zéro après la première occurrence.
- Exécution E016 cluster : pilote `935516` validé, puis run complet `935568` terminé sur `node52` en 13 min 15 s avec zéro erreur après exclusion de `node51`, affecté par une erreur ECC sur la tentative `935563`. L'échantillon avocat aveugle de 100 cas est préparé et stratifié A=27, B=22, C=17, D=17, E=17 ; l'interprétation du score reste bloquée sur cet audit.
- Analyse descriptive E016 : le `Hit@10` officiel reste `0,250000`, mais 692/754 questions ont au moins une A/B. Parmi les 555 questions sans JP exacte dans le top-10, 498 ont une A/B et 57 n'en ont aucune. Sur les couples uniques, 191/204 exacts et 3 821/7 283 non exacts sont classés A/B. Cette dissociation suggère une Ground Truth possiblement incomplète, sans le prouver avant audit avocat.
- Limite d'audit identifiée : le paquet avocat actuel contrôle le juge sur les sorties G7 mais ne contient que 2 couples exacts et aucune JP attendue absente du top-10. Une évaluation séparée des 978 couples de Ground Truth sera nécessaire pour juger directement la qualité du benchmark.
- E017 : les 33 CV et 33 replays LightGCN sont complets pour 11 graphes × seeds 42/43/44. Les 248 820 positions JP donnent 14 309 couples uniques ; 7 348 réponses E016 sont réutilisées par identité exacte et 6 961 jugements sont nouveaux. Le pilote A100 `937671` et le run complet A100 `937682` sont terminés avec zéro réponse invalide ou erreur ; les 14 309 jugements sont agrégés. G6 citation AA maximise le `Hit@10` exact moyen (`0,2685`, écart-type `0,0051`), tandis que G7 JJ citation 1 / sémantique 0,50 maximise le score gradué (`0,4317`, écart-type `0,0020`). Cette dissociation reste exploratoire et en attente de l'audit avocat E016.
- Diagnostic E017 pour présentation : G6-AA reste meilleur au rang 1, tandis que G7-JJ c1/s0,50 dépasse son score gradué cumulé à partir du rang 5. Les top-10 partagent 7,04 JP en moyenne ; 46,51 % des positions G6-AA et 46,27 % des positions G7-JJ sont sans gain, doublons inclus. Les analyses par sous-groupe et deux décisions lues suggèrent une meilleure couverture transversale par les liens JP--JP et un meilleur ancrage des questions précises par les liens article--article. Cette lecture est mécanistique, non causale, et reste conditionnée par l'audit avocat.
- E018 : diagnostic transversal des 33 replays E017 contre les 596 392 paires candidates G8-Large. Les faux négatifs exacts raccordés par le lien LLM brut « même règle » ou « même question » ont des scores gradués moyens plus élevés que les erreurs sans lien G8. Les corrélations inter-replay servent à définir un audit humain, pas à valider G8 : les arêtes restent brutes et le filtre même procédure/noyau factuel manque encore.
- E019-A : analyse de recouvrement achevée, en lecture seule sur les 33 replays E017. Pour G6-AA contre G7-JJ c1/s0,50, les top-10 partagent 7,04 JP (Jaccard 0,565) ; les environ trois substitutions de chaque côté sont A/B dans 46,34 % et 47,54 % des cas. La matrice couvre 55 paires de graphes et les seuils 1/3/5/10. C'est un diagnostic exploratoire des listes, sans nouvelle métrique ni effet causal inféré. E019-B prépare ensuite une comparaison réduite G0--G7, mais n'est pas lancée : les sélections historiques G0--G5 doivent être harmonisées train-only avant tout classement global.

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
- `06-Analyses/comparatifs/e017-intergraph-graded-jp-2026-08-11/README.md`

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

2026-09-02 — B1 initiale a échoué techniquement avant CV pour PPR et LightGCN. B1-r1 est corrigée et préflightée à distance : cosine `977124`, smoke PPR `977125` et smoke LightGCN `977158` terminés et vérifiés ; PPR complet `977157` est en cours. LightGCN CV complet `977184` a été arrêté volontairement après quatre historiques partiels de G1/fold 0 : la grille non reprenable dépasse largement la limite Slurm de 24 h et ne produit aucun résultat reportable. Aucun score B1-r1 n’est encore reportable.

### 2026-08-18 — Audit/export de reproductibilité et branche dédiée

- Branche isolée : `paper/ecir-2027-reproducibility`, sans modification de `07-Redaction/`. Le manifeste historique `confirmatory_campaign_grouped_v2.json` est conservé ; la version portable actuelle est `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json`, campagne `confirmatory-g1-g6-g7-grouped-v2-repro-v1-2026-08-18`, SHA-256 fichier `18abc26eda35f121cf10cc9eddbce690cc8cbf367020b8080d284b90ef0413ed`, SHA-256 canonique de préflight `b852b0dbf9460ac8541c5447e61b37ea5987c99f0126a0bf74f0947a36a69152`.
- Préflight frais : hashes scientifiques et de code valides (`9` entrées immuables, `84` copies de matrices/identifiants, `16` fichiers de code), 11 graphes, 5 603 questions train, 754 questions eval interne, 5 folds. Le gate ressources refuse l'exécution : 4 520 706 048 octets disponibles contre 9 771 050 598 requis pour le job graphe maximal ; aucun calcul n'a été relancé.
- PPR : `_cv_grouped_v2` est complet pour 11/11 graphes, avec 5/5 folds et 5 603/5 603 questions par champion train-only. L'export `results/benchmark-repro-v1/train_cv_retrieval.csv` contient 22 lignes Articles/JP ; SHA-256 `bb62a2ef41d8413ad21c5cb6bcc1d7571825238aa19525218f6ec09d02c8645e`. Le replay PPR interne `_final_grouped_v2` reste manquant.
- E017 : les 33/33 replays LightGCN récupérés sont couverts par 11 graphes × seeds 42/43/44, 754 questions et rangs 1–10 par replay. Les exports Articles, JP exact et JP LLM-as-a-Judge sont séparés : respectivement `internal_eval_articles.csv` SHA `e0650daf35e2ea4d799ccaf73840dde75f94f231141ae6b914baaaa8dd925423`, `internal_eval_jp_exact.csv` SHA `755c4025cbfc93936294a4544c40c4e678b8f18e6a7633b80916088ab54294a6`, et `internal_eval_jp_llm_as_a_judge.csv` SHA `0199bcf105d05ddd8045ef990151c0fa335b054fada4d8b3de4d3bd5e4200a60`; source agrégé `e017_graph_metrics.csv` SHA `eaa5f13ad47e0182380346ac4b56a7e0ca5445bb35e9359f73b4b975c0a39896`.
- E017 reste `exploratoire_agrege_en_attente_audit_avocat` : le meilleur `Hit@10` officiel récupéré est G6-AA `0,2685307987` (ET `0,0050615221`) ; le meilleur score LLM-as-a-Judge est G7-JJ citation 1 / sémantique 0,50 `0,4316976127` (ET `0,0020124656`). Ces chiffres sont traçables mais non utilisables comme preuve de supériorité dans le papier avant le contrôle humain.
- E016 : hashes et agrégats vérifiés sur 754 questions, 7 540 positions et 7 487 couples uniques. Le score LLM brut est `0,4271220159`; l'indicateur binaire `exact_any_gold_at_10` vaut `0,2639257294`, tandis que le `Hit@10` officiel recalculé vaut `0,25`, avec NDCG `0,1648240979` et MRR `0,1404630331`. `lawyer_audit_sample.csv` et `lawyer_audit_key.csv` existent ; `lawyer_agreement.json` manque. Le statut reste exploratoire.
- Résultats matérialisés dans `REGISTRE-RESULTATS.csv` : 131 lignes hashées, avec split, folds/seeds, moyenne, dispersion, configuration, statut, source et SHA-256. `results/benchmark-repro-v1/audit.json` porte le SHA `a25a00ecdcc4e3827b22d14fb416444c618ef267495b0788650182c78136ceec`; `data-manifest.json` porte le SHA `2042725dc262ddb2c045c00660cbeee42362609007c1d148be297a012ee614b3` et interdit toute redistribution avant contrôle de licence.
- Reranking comparable : E021 avait été enregistré comme `proposee` avec un protocole préparé (`K_in=20`, `K_out=10`, mêmes 754 questions et même reranker pour cosine/BGE-M3, PPR et LightGCN). Ce protocole a depuis été exécuté sous le contrat v5 ; voir la section datée ci-dessous pour le statut et les résultats partiels. Coût planifié minimal : 2 262 appels (3 × 754).
- Action demandée à B : reprendre uniquement les exports `results/benchmark-repro-v1/`, avec le statut exploratoire des evals internes ; ne pas présenter les scores E016/E017 comme confirmatoires ni le binaire `exact_any_gold_at_10` comme `Hit@10`.

### 2026-08-18 — E021 reranking comparable exécuté sur cluster

- Contrat v5 : `experiments/reranking-comparable/manifest_cluster_gpu_runtime_v5.json`, SHA-256 `d6f8d45602218248f54a32b84160f4f1276e441efcf50c9638a7338d1a5f8cd4`. Runtime Python 3.12.3, vLLM 0.19.1, snapshot `4033b16200f4152e55e100ea12dc388c537df622`, L40S, fenêtre 16 384 tokens.
- 2 262 unités exécutées, 754 questions par famille, `K_in=20`, `K_out=10`, mêmes questions et mêmes pools cosine/BGE-M3, PPR et LightGCN. Jobs SHA `36f03198d39ec764095d3340ea1f8dc006b941e585245e30bd8e4c14a0a5afdf`; réponses SHA `780e53c1d69481660869d4c0f9e68b377be7d4ab2f0b5c869eae1522b9a3a9fb`.
- Le schéma utilise `items.enum` sur les IDs du pool ; l'unicité reste contrôlée par le parser local, car vLLM 0.19.1 refuse `uniqueItems`. Runner SHA `6495fc37727a531f5e00555713c76d810faceb4ee1be146fb20ddc4a62aa97db`; agrégateur SHA `6d6b76d6b56b4b88aa16961bd46bf9935f347ced514b60cc78f8100ad8b4bbfc`.
- Couverture valide : cosine 747/754, PPR 753/754, LightGCN 749/754 ; 13 unités restent invalides (cosine 7, PPR 1, LightGCN 5). Les 153 enregistrements invalides de l'historique ne sont pas métrés.
- Export JP : `results/reranking-comparable/E021-cluster-gpu-runtime-v5/metrics.json`, SHA `077bea64f34b4382ca00e251359c67d782d0c3a5c807fb571567fa357c7e5954`. Audit : `results/reranking-comparable/E021-cluster-gpu-runtime-v5/audit.json`, SHA `8bc00d9c5850c985343c036bb589703c4e4d302bf2c95fb86a86cc0aedd67e4d`.
- Moyennes JP sur réponses valides, écart-type d'échantillon par question : cosine `Hit@10=0,2707496653 ± 0,4383721651`, `NDCG@10=0,2195999266 ± 0,3809056369`, `MRR@10=0,2065229171 ± 0,3769360233` ; PPR `Hit@10=0,2768924303 ± 0,4405456056`, `NDCG@10=0,2283575587 ± 0,3876235637`, `MRR@10=0,2167414996 ± 0,3848983077` ; LightGCN `Hit@10=0,2930574099 ± 0,4480084150`, `NDCG@10=0,2437210300 ± 0,3969861945`, `MRR@10=0,2323081569 ± 0,3949802308`.
- E021 reste `exploratoire_incomplet_non_paper_ready` : `eval_rich_retrievable_strict` a déjà été consultée, aucune lockbox inédite n'est démontrée, aucune conclusion de supériorité n'est autorisée, et cette campagne ne contient aucune métrique Articles. Le LLM-as-a-Judge est séparé.
- Décision de périmètre du 2026-08-31 : le reranking E021 est une expérience annexe. Le livrable final attendu est un tableau JP distinct du tableau principal, comparant les trois viviers au même reranker. Il pourra être publié après récupération des 13 unités manquantes ; le juge automatique n'en sera pas une colonne ni une condition de calcul.

### 2026-08-31 — Préparation contrôlée Télécom : audit PPR et reprise E021

- PPR final : les 11 dossiers existants sous `_final_grouped_v2` ont été retrouvés sur Télécom, avec 33 fichiers requis (`selected_champions.json`, `final_champions_summary.csv`, `rankings.parquet`). Leur hash de manifeste `3787811ba0877278522db54bcccd6efdd6aa1b2d66132541f64a362cc5681c18` est identifié : `confirmatory_campaign_grouped_v2_repro_v1_cluster_node_runtime.json`. Aucun replay PPR ne sera relancé. L'audit E022 vérifiera les champions train-only, les 754 questions, les rangs et le recalcul des métriques top-10 sans modifier les sorties historiques.
- E021 : les 13 unités manquantes restent exactement 7 cosine/BGE-M3, 1 PPR et 5 LightGCN. Le manifeste de reprise `manifest_cluster_gpu_runtime_v5_resume_v1.json` fige le hash initial du JSONL de réponses, les 2 262 jobs et le modèle. Le job L40S ne traite que les unités absentes ou invalides, puis écrit des métriques et un reçu séparés ; il échoue explicitement si la couverture reste incomplète.
- Scripts Télécom préparés : `71_audit_ppr_final_recovery.py`, `72_finalize_e021_resume.py`, `sbatch_ppr_final_audit.sh`, `sbatch_e021_reranking_resume.sh` et `run_telecom_reproducibility.sh`. Le reçu E021 recoupe désormais chaque réponse avec le hash du job correspondant. Les tests ciblés de ces contrôles et des contrats E021 existants passent à 17/17 ; la soumission cluster n'a pas encore été effectuée.
- Publication : commit `ddda94e` poussé sur `paper/ecir-2027-reproducibility` (PR brouillon existante). Deux contrôles SSH en lecture seule vers Télécom ont expiré ; aucun job, fichier de données ou artefact distant n'a été modifié par cette étape.
- Statut scientifique : aucune nouvelle valeur n'est promue par cette préparation. E021 reste exploratoire et incomplet jusqu'au reçu de couverture complète ; PPR final reste en attente de l'audit E022 avant transcription dans les exports versionnés.

### 2026-09-01 — Soumission Télécom et surveillance

- Le lanceur portable a été corrigé et poussé dans le commit `fe20588` : les valeurs par défaut de données et Python sont désormais transmises à SSH par un marqueur explicite, au lieu d'arguments vides perdus par le shell distant. Le test de transport SSH est passé avec ce comportement réel, avec cinq autres tests ciblés de l'audit et du reçu E021.
- E022 est soumis sous Slurm `969381` (partition CPU, `nodecpu10`). Il effectue uniquement l'audit de 33 fichiers PPR existants pour les onze graphes et écrit un rapport séparé sous `_protocol/ppr_final_audit_v1/` ; aucun PPR n'est relancé et `_final_grouped_v2` reste immuable.
- E021 est soumis sous Slurm `969382` (partition L40S, `node39`). Il vérifie les hashes initiaux, traite seulement les 13 unités manquantes, puis écrira les métriques et le reçu sous `resume_v1/`. L'historique JSONL v5 est append-only.
- Aucun résultat nouveau n'est enregistré dans `REGISTRE-RESULTATS.csv` tant que les jobs ne sont pas terminés et que les couvertures, métriques et hashes de sortie n'ont pas été contrôlés.

### 2026-09-01 — Tentatives v1 auditées, reprises v2 préparées

- Les jobs E022 `969381` et E021 `969382` ont chacun quitté le cluster avec le code `1:0`, avant toute production de nouvelle métrique ou de nouvelle réponse. Les journaux et les sorties v1 sont conservés comme preuves d'exécution échouée.
- E022 v1 a révélé une hypothèse erronée de l'auditeur, pas une modification des artefacts PPR : les champions sont sélectionnés sur 5 603 questions train-only, les fichiers `final_champions_summary.csv` historiques résument les listes à 20 résultats, et les classements stockés permettent de recalculer exactement les métriques requises à 10. Les onze résumés @20 ont été comparés aux classements stockés et concordent. Le manifeste v2 déclare explicitement les deux manifests historiques (`G1` et `G6/G7`), conserve les doublons bruts comme diagnostic et écrit l'audit dans `_protocol/ppr_final_audit_v2/`.
- E021 v1 a échoué lors de l'initialisation vLLM car la révision distante figée renvoie désormais 404. Le snapshot local exact `4033b16200f4152e55e100ea12dc388c537df622` est présent sur Télécom ; la reprise v2 démarre vLLM sur ce snapshot, avec résolution distante désactivée et le même nom de modèle côté API. Le JSONL de réponses n'a pas changé lors de l'échec v1.
- Les manifests v2 (`manifest_ppr_final_audit_v2.json` et `manifest_cluster_gpu_runtime_v5_resume_v2.json`) sont préparés séparément des manifests v1. Aucun chiffre nouveau n'est encore promu ; E022 reste en audit et E021 reste exploratoire/incomplet jusqu'aux reçus v2.

### 2026-09-01 — Reprises v2 soumises sur Télécom

- Après synchronisation du commit `2181c9a`, E022 v2 est soumis sous Slurm `969521` (CPU, `nodecpu10`) et E021 v2 sous Slurm `969522` (L40S, `node39`).
- Avant soumission E021, les SHA-256 des jobs (`36f03198d39ec764095d3340ea1f8dc006b941e585245e30bd8e4c14a0a5afdf`) et de l'historique de réponses (`780e53c1d69481660869d4c0f9e68b377be7d4ab2f0b5c869eae1522b9a3a9fb`) ont été recontrôlés, ainsi que la présence du snapshot local exact. Aucun artefact historique n'a été modifié avant cette reprise.

### 2026-09-01 — E022 audité ; reprise E021 v3 préparée

- E022 v2 / Slurm `969521` est terminé avec le code `0:0`. Le rapport `ppr_final_audit_v2/audit.json`, SHA-256 `d0217fb5ae304d5c640101044bad75475d33cca67d94b6458a7e72f9d01d06d4`, couvre les onze graphes, les 5 603 questions de sélection, les 754 questions d'évaluation et les classements conservés. Il confirme que les résumés historiques sont à 20, puis recalcule les métriques officielles à 10 sans rejouer PPR. La matérialisation légère et les tables versionnées restent à produire à partir de ce rapport hashé.
- E021 v2 / Slurm `969522` a initialisé le snapshot local figé et exécuté les 13 unités restantes. Deux réponses sont valides ; onze réponses répètent au moins un identifiant de décision et sont rejetées par le parser v2. Le reçu `resume_v2/completion_receipt.json` est donc explicitement `incomplete`, SHA-256 `1b23ccb1771da40f84a0620823ba94ade8beddce3a2ec7d4da7ab4893de0d279`; l'historique append-only contient désormais 2 251 clés valides et a le SHA-256 `fe318254dc20a91b532e599ea4316f3495a43891b11862e2258b7e7d19766410`.
- La reprise E021 v3 est préparée dans `experiments/reranking-comparable/manifest_cluster_gpu_runtime_v5_resume_v3.json`. Elle garde les mêmes jobs, viviers, modèle, snapshot, prompt et température. Comme vLLM 0.19.1 ne prend pas en charge `uniqueItems`, elle conserve la première occurrence de chaque identifiant modèle, puis complète seulement les positions vacantes selon l'ordre figé du vivier réel ; chaque ajout est enregistré par réponse. Les réponses v1/v2 et leurs manifests ne sont pas modifiés.
- Tests ciblés de la reprise, de l'agrégation, des viviers, du reçu, de l'audit PPR, de l'export PPR et du lanceur : 24 passés. Aucun résultat E021 partiel n'est inscrit dans `REGISTRE-RESULTATS.csv` ni transmis au papier avant une couverture de 754/754 par famille et un reçu complet.

### 2026-09-01 — Résultats PPR E022 matérialisés et enregistrés

- La matérialisation `73_materialize_e022_ppr_results.py` a relu le rapport E022 complet et les onze classements PPR archivés, sans relancer PPR. Elle vérifie les hashes de chaque `rankings.parquet` et `selected_champions.json`, puis recalcule les métriques officielles à 10 et leur écart-type d'échantillon sur les 754 questions.
- Exports Git légers : `results/benchmark-repro-v1/ppr_final_internal_eval_exact.csv` (66 lignes, SHA-256 LF `2f311319e25b9a876a1904534f56ef53ae6729310f34aaa0ef27e60816918bb0`), `ppr_final_table_articles.csv` (`d3b05f49d9c6d957dac29dc3b3107fb4e65e701394eb35f19a054ffa8f4e45a3`) et `ppr_final_table_jp.csv` (`ebace97fc605254049ed268ce6c1ae61e577b83178d1c709ce91fe58195fc83c`). Le rapport amont E022 reste SHA-256 `d0217fb5ae304d5c640101044bad75475d33cca67d94b6458a7e72f9d01d06d4`.
- Le manifeste Télécom brut et son miroir Git sont séparés : `ppr_final_materialization_manifest.json` (SHA `e50da8b65804415624da94bb8a1841bc4069bc6c698a520549c02f4ba9ecf3c3`) et `ppr_final_repository_mirror_manifest.json` (SHA `640092c81df6da9425b5a81465bb4e9c73ef699454c75a8a83a09de3d62252b7`). Le second explique l'unique transformation de sérialisation CRLF-vers-LF ; les contenus et l'ordre des lignes sont inchangés.
- `REGISTRE-RESULTATS.csv` comporte 66 entrées E022 nouvelles (33 Articles : Recall@10, NDCG@10, MRR@10 ; 33 JP : Hit@10 officiel, NDCG@10, MRR@10), toutes `confirmee_interne`. Les doublons JP bruts restent indiqués par question comme diagnostic, tandis que le calcul officiel applique la déduplication canonique des identifiants avant le score.

### 2026-09-01 — E021 reranking comparable complet et matérialisé

- E021 v3 / Slurm `969635` est terminé avec le code `0:0` sur L40S. Le reçu `results/reranking-comparable/E021-cluster-gpu-runtime-v5-resume-v3/completion_receipt.json`, SHA-256 `456b810a773ae1cafe7ec8d5ec909b19986c797379deec7079e463018484d8e5`, vérifie les 2 262 unités et les trois couvertures complètes de 754/754 questions.
- Sorties exactes : métriques source SHA `08b96023f25a9a36d8041c4f8ef5341e4927466dfae581a92b9e4156e12e2d1c`; détail à neuf lignes `internal_eval_jp_reranking_exact.csv`, SHA `8020ee093ce4822425731602bcb24ff6222c9cdf0c42fd76a3e3a4b90dac16f5`; tableau JP à trois lignes `table_jp_reranking_exact.csv`, SHA `cbf3785e2afa394e755372dca2a81012f82991493317cec45f6ad50a8dd05b4b`; manifeste de matérialisation SHA `4e5e9b02daeaaab3d514f6cd04f54f4b5f3d15dc0e91a960f35d381e1d5b58ba`.
- Même reranker figé (Gemma 4 26B, révision `4033b16200f4152e55e100ea12dc388c537df622`, température 0, `K_in=20`, `K_out=10`) : cosine/BGE-M3 `Hit@10=0,2708885942`, `NDCG@10=0,2202137204`, `MRR@10=0,2072581154` ; PPR `0,2765251989`, `0,2280546973`, `0,2164540440` ; LightGCN `0,2928824050`, `0,2436718744`, `0,2323591638`. Les écarts-types par question figurent dans le tableau.
- Les onze unités reprises ont produit 19 occurrences d'identifiants dupliqués. La v3 a supprimé chaque répétition après la première et complété exactement 19 positions par l'ordre original du vivier de 20 décisions ; les 2 251 réponses précédemment valides sont inchangées. Cette règle est figée dans le manifeste v3 et exposée dans chaque configuration.
- E021 est désormais complet mais reste `exploratoire` en tant que reranking annexe. Il n'a aucune métrique Articles, ne contient pas le LLM-as-a-Judge et ne soutient pas une conclusion de supériorité.

### 2026-09-01 — Suite de tests et isolation de la branche publiable

- La suite depuis le worktree isolé donne 322 tests passés. Les trois tests d'import de `jp_analysis` qui empêchaient la collecte ont été réparés par des imports de paquet non ambigus ; le sous-module passe aussi 53/53 tests lorsqu'il est lancé seul, tout en conservant son mode d'exécution direct.
- Huit tests restent en échec dans ce worktree, tous parce qu'ils vérifient volontairement des fichiers de benchmark/graphes absents : jeux de questions train/eval, folds, graphes G0/G6/G7 et matrices. La branche ne les inclut pas car les données lourdes restent hors Git. Les erreurs sont des `FileNotFoundError` de préflight et non des échecs de métrique ou de pipeline.
- Vérifications de code et d'artefacts : exports PPR/E021 ciblés, reçu E021, audit PPR, lanceur Télécom et agrégations passent ; `py_compile` et `git diff --check` passent. La reproduction complète exige le data manifest et le préflight sur un checkout disposant des données, comme Télécom.

### 2026-09-01 — Surface GitHub de reproduction assainie

- La première branche `paper/ecir-2027-reproducibility` conserve les preuves de récupération mais son diff vers `main` contient 165 commits historiques et des éléments hors périmètre de publication. Elle est conservée sans réécriture comme archive technique et n'est pas la branche à intégrer au papier.
- La branche de publication `paper/ecir-2027-reproducibility-clean` repart de `origin/main`. Elle contient seulement le pipeline benchmark courant, les prompts, schémas, manifests E021/E022, tests, exports agrégés légers, manifeste de données et suivi A. Aucun ajout de `07-Redaction/`, fichier `.tex`, PDF, corpus, matrice, embedding ou classement brut n'y est autorisé.
- Le lanceur Télécom dérive maintenant la branche à synchroniser du checkout local plutôt que d'un nom codé en dur. Les dépendances déclarées incluent `psutil` et `psycopg2-binary`, réellement importées par l'orchestrateur et le préparateur de cartes JP ; Python 3.10+ est requis.
- Vérification isolée sous Python 3.12 : 266 tests passent. Les 8 échecs restants sont exclusivement des `FileNotFoundError` voulus en absence des entrées du manifeste (jeux train/eval, folds, graphes et matrices). Les checks de syntaxe Python/Bash, les hashes d'exports et `git diff --check` passent.
- Les hashes de tous les exports légers sont recoupés : E022 = 66 métriques / 11 graphes / 754 questions ; E021 = 9 métriques / 3 familles avec 754/754 chacune ; E017 = 11 graphes, trois seeds et 754 questions. Dans l'export E022, `audite` qualifie la matérialisation hashée ; la qualification scientifique officielle de chaque résultat est `confirmee_interne` dans le registre.
- Publication : PR brouillon GitHub `#2` (`paper/ecir-2027-reproducibility-clean` vers `main`) : https://github.com/matt-kaep/legal_knowledge_graph/pull/2. Son contrôle GitGuardian est vert ; elle n'est ni fusionnée ni une autorisation de promouvoir les scores exploratoires.
