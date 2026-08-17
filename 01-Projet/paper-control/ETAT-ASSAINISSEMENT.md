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

2026-08-12 — E019-A terminé : recouvrement top-K des 11 graphes E017, sur 55 paires, puis contrat E019-B préparé pour la comparaison réduite G0--G7 × méthodes JP. Résultats exploratoires ; audit avocat E016 toujours en attente.

### 2026-08-18 — Audit/export de reproductibilité et branche dédiée

- Branche isolée : `paper/ecir-2027-reproducibility`, sans modification de `07-Redaction/`. Le manifeste historique `confirmatory_campaign_grouped_v2.json` est conservé ; la version portable actuelle est `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json`, campagne `confirmatory-g1-g6-g7-grouped-v2-repro-v1-2026-08-18`, SHA-256 fichier `18abc26eda35f121cf10cc9eddbce690cc8cbf367020b8080d284b90ef0413ed`, SHA-256 canonique de préflight `b852b0dbf9460ac8541c5447e61b37ea5987c99f0126a0bf74f0947a36a69152`.
- Préflight frais : hashes scientifiques et de code valides (`9` entrées immuables, `84` copies de matrices/identifiants, `16` fichiers de code), 11 graphes, 5 603 questions train, 754 questions eval interne, 5 folds. Le gate ressources refuse l'exécution : 4 520 706 048 octets disponibles contre 9 771 050 598 requis pour le job graphe maximal ; aucun calcul n'a été relancé.
- PPR : `_cv_grouped_v2` est complet pour 11/11 graphes, avec 5/5 folds et 5 603/5 603 questions par champion train-only. L'export `results/benchmark-repro-v1/train_cv_retrieval.csv` contient 22 lignes Articles/JP ; SHA-256 `bb62a2ef41d8413ad21c5cb6bcc1d7571825238aa19525218f6ec09d02c8645e`. Le replay PPR interne `_final_grouped_v2` reste manquant.
- E017 : les 33/33 replays LightGCN récupérés sont couverts par 11 graphes × seeds 42/43/44, 754 questions et rangs 1–10 par replay. Les exports Articles, JP exact et JP LLM-as-a-Judge sont séparés : respectivement `internal_eval_articles.csv` SHA `e0650daf35e2ea4d799ccaf73840dde75f94f231141ae6b914baaaa8dd925423`, `internal_eval_jp_exact.csv` SHA `755c4025cbfc93936294a4544c40c4e678b8f18e6a7633b80916088ab54294a6`, et `internal_eval_jp_llm_as_a_judge.csv` SHA `0199bcf105d05ddd8045ef990151c0fa335b054fada4d8b3de4d3bd5e4200a60`; source agrégé `e017_graph_metrics.csv` SHA `eaa5f13ad47e0182380346ac4b56a7e0ca5445bb35e9359f73b4b975c0a39896`.
- E017 reste `exploratoire_agrege_en_attente_audit_avocat` : le meilleur `Hit@10` officiel récupéré est G6-AA `0,2685307987` (ET `0,0050615221`) ; le meilleur score LLM-as-a-Judge est G7-JJ citation 1 / sémantique 0,50 `0,4316976127` (ET `0,0020124656`). Ces chiffres sont traçables mais non utilisables comme preuve de supériorité dans le papier avant le contrôle humain.
- E016 : hashes et agrégats vérifiés sur 754 questions, 7 540 positions et 7 487 couples uniques. Le score LLM brut est `0,4271220159`; l'indicateur binaire `exact_any_gold_at_10` vaut `0,2639257294`, tandis que le `Hit@10` officiel recalculé vaut `0,25`, avec NDCG `0,1648240979` et MRR `0,1404630331`. `lawyer_audit_sample.csv` et `lawyer_audit_key.csv` existent ; `lawyer_agreement.json` manque. Le statut reste exploratoire.
- Résultats matérialisés dans `REGISTRE-RESULTATS.csv` : 56 lignes hashées, avec split, folds/seeds, moyenne, dispersion, configuration, statut, source et SHA-256. `results/benchmark-repro-v1/audit.json` porte le SHA `a25a00ecdcc4e3827b22d14fb416444c618ef267495b0788650182c78136ceec`; `data-manifest.json` porte le SHA `2042725dc262ddb2c045c00660cbeee42362609007c1d148be297a012ee614b3` et interdit toute redistribution avant contrôle de licence.
- Reranking comparable : E021 est enregistré comme `proposee`, protocole préparé avec `K_in=20`, `K_out=10`, mêmes 754 questions et même reranker pour cosine/BGE-M3, PPR et LightGCN. Aucun résultat n'est produit : le pool PPR interne, le modèle fournisseur et les droits de redistribution ne sont pas encore gelés. Coût planifié minimal : 2 262 appels (3 × 754), coût monétaire non estimé avant gel du modèle.
- Action demandée à B : reprendre uniquement les exports `results/benchmark-repro-v1/`, avec le statut exploratoire des evals internes ; ne pas présenter les scores E016/E017 comme confirmatoires ni le binaire `exact_any_gold_at_10` comme `Hit@10`.
