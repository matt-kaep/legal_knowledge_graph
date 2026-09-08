---
date: 2026-09-08
type: synchronisation
owner: assainissement
recipient: papier
tags: [coordination, benchmark, papier]
---

# Canal A vers B — Assainissement vers papier

## Mise à jour — E029 autorisé sur GPU, aucun score encore disponible (2026-09-09)

- Le préflight CPU output-512 est terminé sur Télécom (`986427`, `0:0`, 18 min 56 s), sans appel de modèle. Son reçu hashé `.../_campaign_b2_e029_a3_k70_article192_output512_preflight_v1_20260909/preflight_receipt.json` vaut `da5033e7be5a0e59f0ffdf5172a5a3e976cdc58227f11baefc83f9deea5a9d8d` : 70 conditions compatibles, 52 780 appels prévus, zéro dépassement de contexte et zéro incompatibilité de budget de réponse. Les audits associés valent `86a6ee8d83467b96ac373921d574dfd6efdb1a471595d5591203d1c4c3005811` (contexte) et `3049cb4d8776bb2416e0a9bb97a9ecd23f4bb73aade3ac2733f43d6e4f6a1120` (sortie ; maximum conservatif 425 tokens).
- Le manifeste GPU distinct est `configs/b2_reranking_comparable_a3_k70_article192_output512_execution_v2.json`, SHA `c33f4474aa47224a884eabc7e9f616f7b7b7cd1d6db49113619449dce1522586`. Il conserve A3, les viviers, K=10--70, les prompts, Gemma et température zéro; seule la réserve de sortie passe de 256 à 512 tokens. Il interdit explicitement de réutiliser l'archive tronquée output-256.
- Les cinq lots L40S ont été soumis : cosine `986445`, PPR `986446`, LightGCN graines `42/43/44` = `986447/986448/986449`; l'agrégation CPU `986450` dépend de leur succès intégral. Ils sont initialement `PENDING (Resources)` et ne produisent encore ni appel modèle, ni ranking, ni score.
- Statut pour Papier : **ne pas intégrer de résultat E029 ou E030 à ce stade**. Après ces cinq lots et leur agrégation contrôlée, les rankings gelés permettront le lancement distinct de E030. E030 demeure une mesure exploratoire centrale de pertinence juridique, non une validation humaine du juge.

## Mise à jour — paquet publiable A3/B1 et tableaux exacts versionnés (2026-09-08)

- Les tableaux à reprendre sont maintenant versionnés dans la branche : `results/benchmark-a3-b1/main_table_articles.csv` (SHA `44cb8a4bd81ccc5368cc8f802a242c352076f378100184613a0068159e0b1c9d`) et `main_table_jurisprudence.csv` (SHA `0ad76a437f903c06acd3d4e8cf824939c0aa2813d158e7d4f41369fbe4c875b0`). Le reçu `main_table_manifest.json` (SHA `f7d2b7399a6a81eb4442020da6a27e858be12ab3d8aa95b5b4295c9c7aeb021b`) les relie au CSV source B1-A3 G6, SHA `9569ea74bb3d7fa50ef5fd80cd86cbc8879ec95fb41686ca8a34bdd79f20f7ad`.
- Ces fichiers reprennent exactement les six cellules canoniques cosine / PPR / LightGCN G6, avec méthode, graphe, tâche, graines, configuration gelée et statut. Ils ne remplacent pas les ablations scoped G1/G7 et ne mélangent ni E017, ni E021, ni E022.
- Le README racine est désormais l’entrée de reproduction A3/B1 ; `data-manifest-a3.json` (SHA `18b486737b6c25abde6a11db5e0df008a61fcabf02c99fa7927e99c13dec5e73`) inventorie 30 entrées B1 hashées et non redistribuées. La suite complète passe (`393 passed`).

## Décision — E030 sera un résultat exploratoire central, après E029 (2026-09-08)

- Dès que les rankings E029 sont complets, contrôlés et gelés, A lancera E030. Le juge recevra seulement la question et une fiche anonymisée par candidat ; aucune information sur la méthode, le graphe, la graine, le rang, le score ou la vérité terrain ne sera visible.
- Le tableau E030 restera **distinct** du tableau des métriques exactes. Il est scientifiquement utile comme mesure complémentaire de pertinence juridique, même avant l’audit avocat ; son statut sera explicitement `exploratoire`.
- L’absence actuelle de `lawyer_agreement.json` n’empêche pas le calcul ni la communication descriptive du score. Elle interdit de présenter le juge comme validé par des avocats, ou d’en déduire une conclusion de supériorité entre méthodes ou graphes.
- Contrôle technique déjà effectué : les cinq tests du runner préparé passent. Ils attestent notamment que la fiche JP ne contient que `synthese`, que les métadonnées de retrieval sont absentes de la partie visible, et que le score conserve dix positions avec gains A=1, B=0,5. Ce n’est pas encore une autorisation d’exécution : A doit d’abord disposer des rankings E029 hashés, puis B doit figer le manifeste d’exécution (modèle/révision, snapshot/tokenizer et budget inclus).
- L'exécuteur versionné de A est prêt : `scripts/110_run_b2_e030_llm_judge.py`, SHA `ca137cae20048d93f244bbd97870eb8637238f664775bb4f3a44b61e685fe731`. Il ne reçoit que des jobs déjà gelés, écrit les réponses append-only puis refuse toute matérialisation incomplète. Une sortie directe sans référence A3 résolue devient un zéro terminal, sans appel et sans fiche visible, mais reste dans les dix positions. Cela prépare le lancement sans créer de liste, appel modèle ou score E030.
- Le gel versionné des jobs est également prêt : `scripts/111_freeze_b2_e030_judge_jobs.py`, SHA `279b76e85cffea701e0887478ed59a0d445a75d8569533e337c7c799d58cd30e`. À partir d’un ranking final explicitement gelé, il impose 10 positions par question, A3 et les cartes candidate comme sources hashées, puis produit une liste immuable ; pour les fichiers multi-conditions, le graphe, la graine ou la profondeur est filtré explicitement et ce filtre devient partie du reçu. Il ne rend toujours visible au juge que question + `texte`/`synthese`. Les 10 tests E030 ciblés et la suite complète (`406 passed`) sont verts. Il reste à lier les sources E029 achevées, le snapshot/tokenizer et le budget dans le manifeste d’exécution, puis à soumettre les appels.

## Réparation E029 — campagne 256 archivée, préflight 512 en cours (2026-09-09)

- Ne reprendre aucun nombre du contrat E029 `c9296cea…` : le shard cosine a été arrêté après 24 min dès l’identification de JSON tronqués par le plafond de sortie 256. Son ledger partiel SHA `bbccfcfcca8438201e7d3b001127ecf98426310de7b53e3761208e289448815d` contient 3 634 lignes, dont 2 220 erreurs dues à l’annulation ; il est archivé et non reportable. Les autres shards et l’agrégateur n’ont pas démarré.
- Le successeur n’altère aucune entrée scientifique. Son manifeste CPU sans modèle est `configs/b2_reranking_comparable_a3_k70_article192_output512_preflight_v1.json`, SHA `4965bf9c776ff7102819011af58c4859f34f384b4fe6b3d7a2c69c806c10f825` : mêmes listes A3, prompts, texte Article 192 tokens, synthèse JP, modèle Gemma et température zéro ; seule la réserve de sortie passe à 512. Aucun score E029/E030 n’est encore transmis au papier.

## Mise à jour — tableau structurel canonique G1/G6/G7 disponible (2026-09-08)

- Source unique : `.../_campaign_b1_a3_structural_graphs_20260908/a3_structural_graphs.csv`, SHA `80ba400a88ebef618df5a9ce1430e66fe637607bf1051f891297bc93774f5b0f`; reçu SHA `2307ac04cfffd4a337211c56037045954d8ee9678212c0ba9c6843260a144bb7`.
- À reprendre : G1 = 593 172 arêtes citation Article--JP, 45 composantes, composante principale `99,92494671925116 %`, degré moyen/médian `8,399906537423973 / 3`; G6-AA = 639 034 arêtes (593 172 citation + 45 862 AA), 37 composantes, `99,93910771561887 %`, `9,049358152839634 / 4`, normalisation symétrique, poids `AA=1`, `citation=1`; G7-AA = mêmes nœuds et arêtes, aucune normalisation, poids `AA=0,25`, `citation=1`.
- Toujours distinguer 117 374 lignes JP brutes de 115 304 décisions uniques, 23 859 nœuds Article structurels de 13 236 candidats Article retournables et 114 851 candidats JP retournables. Le CSV comporte les percentiles, maximum, isolés et hashes des graphes sources.
- Le fichier ne décrit que les graphes pénaux G1/G6/G7. **Ne pas écrire de chiffre pour un « graphe complet » distinct** : A n'a pas encore identifié de source structurelle non ambiguë pour cette notion.

## Mise à jour — E028 r5 rétablit le manifeste de code courant, sans changer les résultats (2026-09-08)

- R4 reste une archive intègre de son exécution. R5 est le nouveau manifeste courant : `configs/paper_ready_existing_retrieval_a3_r5.json`, SHA `1d987e9479b30b33d212ba3d9ef79131592f826b8e43310c3322a10d6302d224`, reçu SHA `bcf4aa500cfe9cd6ee80ef5f00dd0634b98260482e448ad65d6445eb17eb4f84`.
- R5 épingle le script qui sait aussi relire les sources PPR scoped (SHA `abcb08369012b12ca06aacf1362d975851c6b0727dba41d3131846f04d1e8ffd`) et a été dérivé des mêmes rankings cosine/PPR globaux A3. Les métriques restent byte-for-byte identiques à r4 (SHA `fa7abb380c984b652d890576ca2e238f9a7110b9c368b8c0541f152f1d47d5a2`).
- **Action Papier :** aucune valeur ne change. Employer r5 seulement comme preuve de reproductibilité courante ; il ne crée ni nouveau retrieval, ni nouveau classement, ni nouveau résultat.

## Mise à jour — cellules PPR/LightGCN G1, G6, G7 contrôlées depuis les rankings (2026-09-08)

- A a terminé une dérivation déterministe, sans modèle ni retrieval, des replays scoped PPR et LightGCN G1. Manifeste : `configs/paper_ready_retrieval_a3_scoped_ppr_g1_lightgcn_g1_r1.json`, SHA `f4b0cbb5f459b940c61df22ecfd521228d74dd34060ca96d45d228842d789775`. Reçu : `.../_campaign_b1_a3_scoped_exact_metrics_r1_20260908/depth_curves/depth_curves_manifest.json`, SHA `11539fa232438f0fbde5d0a2b6728894ffcdd65b73dda2c5a3698aa8427df4e5`.
- Le reçu atteste les 754 questions, les rangs 1--100 sans doublon, les deux univers A3 et les rankings source hashés. Le CSV de métriques exactes est `.../ranking_metrics_at_10.csv`, SHA `69b0243be7d7ea17cfa88745918480b92f4430d01fe39ab9303cc27e8c983e6a`.
- À reprendre seulement comme cellules scoped / ablations, jamais comme sélection post-évaluation : PPR G1 Articles Hit/Recall@10 `0,5459570123363227`, NDCG@10 `0,311061139772892`, MRR@10 `0,2639015199360027`; PPR G1 JP Hit@10 `0,22855879752431477`, NDCG@10 `0,14533424111226456`, MRR@10 `0,12104385078523008`; PPR G6 JP `0,22822723253757737 / 0,14574720359336762 / 0,12102858826996758`; PPR G7 Articles `0,5531756557618627 / 0,31544924625056936 / 0,26798766367731885`.
- LightGCN G1, moyenne des graines 42/43/44 : Articles Hit/Recall@10 `0,561023746368574`, NDCG@10 `0,40954248455077263`, MRR@10 `0,40138116991565265`; JP Hit@10 `0,26492042440318303`, NDCG@10 `0,1782094594240493`, MRR@10 `0,15336652491824906`. Configuration gelée avant évaluation : Articles K2/lr 0,0005/lambda 0,5/7 époques ; JP K3/lr 0,001/lambda 1/4 époques.
- Formulation autorisée : « Les cellules G1/G6/G7 ont été sélectionnées par validation croisée sur l’entraînement, puis leurs rankings top-100 ont été contrôlés contre le contrat A3 avant calcul des métriques. » Ne pas appeler ces cellules un classement causal ou sélectionner G1/G6/G7 après lecture de ces résultats.

## Mise à jour — E028 r4 répare la reproductibilité, sans changer aucun chiffre (2026-09-08)

- A a conservé le manifeste et les exports r3 comme archive immuable, puis a créé r4 (SHA `0855a35878ff33abdc147637c7d151908fbd5b459c4a0b59d2b0cee2d7a52e89`). R4 fixe le script qui l'a produit (SHA `6ed1c895c29d262d2bea9a2a518e1b03ce4a1ca3960d4627d93bac7b77dedd20`) au lieu de modifier la preuve historique r3; r5 est désormais le successeur de code courant.
- R4 a relu uniquement les rankings top-100 cosine et PPR B1-A3 déjà gelés, a validé leurs hashes puis a produit courbes K=1–100, CSV de métriques et figures. Reçu : `.../_campaign_b1_a3_paper_ready_existing_retrieval_r4_20260908/depth_curves/depth_curves_manifest.json`, SHA `5229c033d218a858c82ca79956d0a2b791875052b667b1d47d56be2d0e6635a6`.
- Les métriques et courbes sont byte-for-byte identiques à r3 (métriques SHA `fa7abb380c984b652d890576ca2e238f9a7110b9c368b8c0541f152f1d47d5a2`). **Action Papier :** aucune valeur de tableau ne change et r4 ne remplace pas l’export distinct à trois méthodes cosine/PPR/LightGCN G6 ; il rend seulement la provenance de code de la dérivation cosine/PPR à nouveau rejouable.

## Mise à jour — E029 successeur : préflight complet, exécution GPU gelée, aucun score (2026-09-08)

- Le reranking E029 successeur est prêt à être soumis, mais **ne fournit encore aucun résultat**. Son contrat exploratoire est : même reranker Gemma figé, `K_in=10,20,30,40,50,60,70`, `K_out=10`, mêmes 754 questions, viviers cosine/PPR/LightGCN G6 et trois seeds LightGCN séparées. Articles : préfixe de 192 tokens Gemma du champ `texte`; JP : champ `synthese` complet. Le retrieveur, le graphe, la seed, le rang et les scores ne sont pas montrés au modèle.
- Preuve CPU : reçu `05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_campaign_b2_e029_a3_k70_article192_preflight_v1_20260908/preflight_receipt.json`, SHA `a3878e63faf467bf1c8311440076af595f0e8544c881b6dc791d45fec9537740`, et audit de contexte SHA `8e82219ba9cb2ca22c484a709669135ba472627cb9499406f5445e0ee9bc1779`. Ils attestent cinq lots hashés, 52 780 requêtes, 70/70 conditions compatibles, zéro dépassement et zéro appel modèle.
- Manifeste d’exécution GPU : `05-Technique/benchmark/etape1_embedding_pur/configs/b2_reranking_comparable_a3_k70_article192_execution_v1.json`, SHA `c9296ceaed851a01ace69e2e2ade42ad35b01dc70107fda21813840b554467c1`. Il reste distinct des archives E029 plein texte et de E030.
- Exécution : la soumission H100 `986162`--`986166` a été annulée avant démarrage pour utiliser la capacité L40S immédiatement disponible, explicitement autorisée par le manifeste. Les remplaçants strictement équivalents sont `986412` cosine (**RUNNING**, node39, depuis 23:45 CEST), `986413` PPR, `986414`/`986415`/`986416` LightGCN seeds 42/43/44 (en attente de ressources L40S). Il n’existe encore ni classement reranké complet, ni métrique.
- L'agrégateur hashé est planifié sous `986417` sur CPU et ne se déclenchera qu'en `afterok` des cinq shards. Il refuse une couverture incomplète et conserve les trois graines LightGCN séparées : aucune moyenne ni aucun tableau ne pourra être créé sur des sorties partielles.
- Action Papier : ne rien intégrer ni chiffrer pour ce contrat tant que les cinq reçus GPU, les rankings top-10, les métriques exactes et l’agrégation hashée n’ont pas été contrôlés par A. E030 n’est pas lancé.

## Mise à jour — nouveau préflight E029 autorisé, aucun résultat encore (2026-09-08)

La décision de représentation est désormais figée pour un **nouveau** contrat E029 : profondeurs `10,20,30,40,50,60,70`, sortie top-10; Articles réduits au préfixe de 192 tokens du tokenizer Gemma exactement versionné; JP conservées sous forme de `synthese` complète. Cette règle est identique pour cosine, PPR et les trois graines LightGCN. Le modèle ne verra jamais le retrieveur, le graphe, la graine, le rang ou le score source.

Preuve de protocole : `05-Technique/benchmark/etape1_embedding_pur/configs/b2_reranking_comparable_a3_k70_article192_preflight_v1.json`, SHA-256 `d40676004a1601f3b0bb5523409993506973f6db5af4884d4a175fe49a88f347`. Il prévoit 70 conditions × 754 questions = 52 780 jobs, mais interdit encore tout appel LLM. Les 33 tests E029 ciblés passent. Les cinq lots neufs et l’audit de contexte doivent être produits et hashés avant qu’A puisse créer le manifeste GPU.

**Action Papier :** ne rien intégrer à ce stade. La formulation pourra devenir « analyse annexe exploratoire de reranking sur viviers gelés » seulement après le préflight complet puis les sorties GPU contrôlées. Les archives plein texte E029 antérieures restent séparées et ne doivent ni être agrégées ni comparées à cette nouvelle représentation.

## Mise à jour — E029 bloquée par le contexte, zéro appel modèle (2026-09-08)

Le reranking comparable n'a aucun score ni sortie LLM intégrable. Le manifeste v4 est `05-Technique/benchmark/etape1_embedding_pur/configs/b2_reranking_comparable_a3_fullmatrix_cpu_preflight_v4.json` (SHA-256 `a6026645ff0ddaac4e5375231189b974e37873369368c9e99c2b82145905b31a`). Les 100 viviers restent validés byte à byte par `.../_campaign_b2_e029_fullmatrix_cpu_preflight_v3_20260908/input_pool_validation.json` (SHA-256 `0a95bc75c0750c1b2de7acae19843ce000e50a2abffd7599f469047a1985ce06`). Le reçu final v4 est `.../_campaign_b2_e029_fullmatrix_cpu_preflight_v4_20260908/cpu_preflight_receipt.json` (SHA-256 `d3da604a0225c97156bcbdc5bd9caf94d75de32bb4ab2901e28ace1de9b6db0f`).

Statut à reprendre dans le papier : **aucun résultat E029 ou E030 n'est reportable**. V2 est archivé (seeds LightGCN fusionnées), v3 est archivé (échec d'import avant comptage), tous deux sans appel modèle. L'audit exact v4 couvre les 100 prompts avec Articles `texte` intégral, JP `synthese` intégrale, contexte 16 384 et réserve de sortie 256 : 42 conditions passent, 58 échouent, 24 509 prompts dépassent le budget d'entrée de 16 128 tokens et `model_calls=0`. Aucune troncature, profondeur supprimée ou matrice GPU partielle n'est autorisée. E029 attend une règle de représentation compacte commune, explicitement gelée ; E030 reste séparée, sans score et exploratoire jusqu'à l'audit avocat.

Une simulation CPU d'aide à la décision, non intégrée au protocole, a ensuite mesuré des plafonds uniformes de 64, 96, 128 et 160 tokens par candidat sur les dix viviers réels à `K_in=100`. Reçu : `.../_campaign_b2_e029_fullmatrix_cpu_preflight_v4_20260908/representation_cap_feasibility_v2.json`, SHA-256 `c516d3bc9137043059bdfdea21ace898da686fd887188fca6f20d5093350da6b`, zéro appel modèle. Les plafonds 64 et 96 tiennent toutes les sources et questions ; 128 laisse 42 prompts hors budget et 160 en laisse 5 740. Ceci fournit une option mesurée pour un futur contrat, mais n'est pas une règle appliquée ni un résultat à citer : aucun GPU E029/E030 ne peut être lancé avant la décision explicite de la représentation.

Un second préflight répond au besoin d'une troncature limitée des seuls Articles pour conserver une profondeur supérieure à 50. Sur cosine, PPR et les trois graines LightGCN, avec les 754 questions et le template de chat exact : à `K_in=50`, 288 tokens par Article passent toutes les conditions (maximum 15 171 tokens) tandis que 320 laisse 3 prompts hors budget ; à `K_in=70`, 192 passent toutes les conditions (maximum 15 434) tandis que 224 laisse 111 prompts hors budget. Reçu SHA-256 `7784ca394d7ee784c349862fe9ca2827dcfe493684fc8f3a642b447dc89b0044`, zéro appel modèle. C'est une aide à la décision, non une règle appliquée : le papier ne doit pas décrire cette représentation avant le gel d'un manifeste successeur.

## Résumé courant — E029 matrice complète en préflight CPU, aucun résultat LLM nouveau

- Checkpoint initial : `01-Projet/paper-control/INVENTAIRE-E029-E030-A3-2026-09-08.md` ; suivi : `PROGRESS-E029-E030.md`. L'inventaire était sans calcul. Son successeur CPU `b2_reranking_comparable_a3_fullmatrix_cpu_preflight_v2.json` est désormais lancé pour reconstruire 100 viviers K=10..100, mais n'effectue aucun appel modèle.
- Les seuls chiffres réutilisables pour le tableau principal restent les résultats retrieval A3 déjà transmis. Rankings top-100 complets et hashés : cosine `7de0504d...`, PPR `8f15c4d3...`, LightGCN G6 `b1448dd5...`.
- L'archive E029 possède 42 conditions sur les 100 demandées (31 668/75 400 unités) ; elle reste immuable et non agrégée. Le successeur emploie PPR G6 explicitement pour les deux tâches : G6 Articles final, G6 JP rejoué depuis son champion CV train-only distinct. Aucun score E029 ne doit être repris dans le papier à ce stade.
- E030 n'a aucun résultat. Les scores Judge historiques E016/E017 restent exploratoires et `lawyer_agreement.json` manque.
- La structure pénale G1/G6/G7 est exportée ; la structure du graphe complet distinct reste à produire. Ne pas présenter une approximation structurelle.

### 2026-09-08 — Gate de non-calcul E029/E030

- A mesure désormais le budget de contexte K=10..100, Articles et JP séparés, prompt et cartes inclus. Aucune troncature, profondeur supprimée ou profondeur substituée ne sera introduite silencieusement.
- Action B : ne pas intégrer E029/E030 ; le futur tableau reranking affichera toutes les profondeurs ou des `--` explicitement justifiés.

## Handoff historique — résultats A3 et archive E029 partielle (2026-09-08)

Ce bloc conserve les éléments archivés avant l'inventaire Phase 0. Il ne remplace pas le résumé courant : E029 est actuellement en préflight CPU pour une matrice homogène de 100 conditions, et ses sorties historiques de 42 conditions ne constituent pas un tableau complet.

### Ce que le papier peut utiliser maintenant

- **Contrat commun A3 :** entraînement/CV sur 5 578 questions ; évaluation interne inchangée de 754 questions (SHA-256 `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59`) ; 13 236 Articles et 114 851 décisions retournables. Manifeste A3 `05-Technique/benchmark/etape1_embedding_pur/configs/benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json`, SHA-256 `c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3`.
- **Tableau principal Articles (métriques exactes, A3) :** cosine Hit/Recall@10 `0,4315102521999074`, NDCG@10 `0,32102115314737484`, MRR@10 `0,32082228116710876` ; PPR G6-AA Hit/Recall@10 `0,5768809734326976`, NDCG@10 `0,39573594882260704`, MRR@10 `0,3735463769946528` ; LightGCN G6-AA, moyenne des graines 42/43/44, Hit/Recall@10 `0,5510694286556356`, NDCG@10 `0,40073652473423804`, MRR@10 `0,39287538770297387`.
- **Tableau principal Jurisprudence (métriques exactes, A3) :** cosine Hit@10 `0,21794871794871795`, NDCG@10 `0,15397741432193576`, MRR@10 `0,13577691465622502` ; PPR G7-AA (citation 1, sémantique 0,25) Hit@10 `0,23065870910698497`, NDCG@10 `0,142981907459661`, MRR@10 `0,11715506715506716` ; LightGCN G6-AA, moyenne des graines 42/43/44, Hit@10 `0,2663572060123784`, NDCG@10 `0,1777719765610496`, MRR@10 `0,15267848371296647`.
- **Baseline LLM directe E027 :** le modèle ne voit que la question, sans vivier ni corpus. Articles : Hit/Recall@10 `0,09497442212959453`, NDCG@10 `0,08860546818088738`, MRR@10 `0,11098006399730537`. Jurisprudence : `0` pour les trois métriques. Le manifeste est `configs/b2_direct_llm_a3_v1.json`, SHA-256 `a6da87fca6658ae33b73da22b34bd66f061a8a20d9c492e7ae9e891fd35905d7`.
- **Formulation autorisée :** « Sur l’évaluation interne A3, après sélection par validation croisée sur l’entraînement puis gel des configurations, les méthodes sont évaluées sur le même univers ordonné de candidats retournables. » G6 est la configuration principale présélectionnée du papier ; il ne doit pas être décrit comme un champion causal parmi les onze graphes.

### Reranking LLM E029 — archive de 42 conditions, annexe exploratoire incomplète

- Les cinq shards ont produit les **31 668** réponses prévues. Les appels étaient déjà terminés quand un défaut du routeur CLI a empêché la matérialisation des résultats. Le correctif ne modifie ni prompt, ni vivier, ni réponse ; la reprise a fait **zéro** appel modèle. Le manifeste initial reste immuable : `configs/b2_reranking_comparable_a3_execution_v1.json`, SHA-256 `259cff89780dfb4bfa39159cc0f608fdd295c559391858d376149170e8ff8515`. Le manifeste de réparation chaîné est `configs/b2_reranking_comparable_a3_materialization_repair_v1.json`, SHA-256 `4e841c3b683590fb9382e665cd73ada8f1a5f854818049d483df0314068a17d1`.
- Les rankings top-10 et métriques sont complets et hash-validés : cosine JP (5 278 réponses, reçu de reprise SHA `4cdc245baf4145f03b3dbac3ed6abe46b6f43369b6953b62268eac92c8476089`) ; PPR (6 032, `47376ddee2b42a1e02b1a2cddde09b54a2579b20018afccf7dda827fa1313b62`) ; LightGCN G6 graines 42/43/44 (6 786 chacune, reçus `8c369e654f36ff883cecdfa8da4af964589491a9a976857304fd6a0481ccc871`, `054ca5625471e2abf7551b27b9a5bb75f1805c7e529657afdcdd250505fb93f0`, `a5329e9527885018fd10620dfecb825431962cffb1de046f6c3d05618b6fb0f1`). Chaque liste contient 754 questions par condition, dix rangs, aucun doublon parmi les identifiants résolus ; les réponses invalides sont des positions nulles explicites, jamais remplacées.
- Contrat contenu : Articles = champ `texte` intégral non tronqué ; JP = champ `synthese` non tronqué par décision, ni décision complète, ni `resume_avocat`, ni arguments. Aucun appel à la base OVH n’a lieu à l’inférence. Le modèle est Gemma `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` à la révision `4033b16200f4152e55e100ea12dc388c537df622`, température 0, fenêtre 16 384 tokens et réserve de sortie 256 tokens.
- Quelques repères exacts **à réserver à une annexe exploratoire**, jamais au tableau principal : PPR Articles, vivier 10 puis reranking : Hit@10 `0,5768809734326976`, NDCG@10 `0,5067788547230162`, MRR@10 `0,5381399730537662` ; PPR JP, vivier 10 : Hit@10 `0,2306587091069849`, NDCG@10 `0,1933259213191533`, MRR@10 `0,1841075112626836` ; LightGCN G6 JP, vivier 10, moyenne des trois graines : Hit@10 `0,2663572060123784`, NDCG@10 `0,2239652668557736`, MRR@10 `0,2171062972787109`. Les profondeurs non compatibles restent `--` et ne doivent pas être interpolées.
- **Formulation autorisée uniquement :** « Dans une analyse annexe exploratoire, un LLM réordonne des viviers gelés produits par les retrieveurs, sans modifier ceux-ci. » Ne pas dire que le reranking est une métrique LLM-as-a-Judge, une sélection de graphe, ou une conclusion de supériorité.

### LLM-as-a-Judge E030 — autorisation GPU reçue, gel incomplet

La session Assainissement a reçu l’autorisation de préparer puis lancer les nouveaux runs GPU. Aucun job E030 n’est cependant encore soumis : le manifeste existant est explicitement un préflight (`b2_llm_as_judge_a3_preflight_v1.json`) et n’a pas de modèle, de révision, de listes source ni de budget scellés. Les scores E030 resteront exploratoires jusqu’à l’audit avocat.

**Décisions demandées à la session Papier / responsable scientifique avant gel :**

1. **Modèle du juge :** valider ou refuser la proposition de reprendre exactement Gemma `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` à la révision `4033b16200f4152e55e100ea12dc388c537df622`, température 0. Ce choix est cohérent avec E027/E029, mais il ne doit pas être supposé sans validation explicite.
2. **Périmètre des listes rerankées :** faut-il juger les 42 conditions E029 compatibles (toutes profondeurs et trois graines LightGCN), ou figer seulement une profondeur définie avant lecture des scores ? La seconde option est plus légère, mais doit indiquer à l’avance quelle profondeur est retenue et pourquoi. Dans les deux cas, les listes directes, cosine, PPR et LightGCN G6 devront être hashées et jugées séparément.
3. **Contenu JP vu par le juge :** le préflight prévoit une carte de décision anonymisée et aveugle (sans rang, retrieveur, graphe, graine ni vérité terrain). Confirmer que cette carte doit être fondée sur `synthese`, ou transmettre le schéma de carte souhaité si les champs `arguments` et réponse du juge sont requis. Ce choix change directement le coût et la validité de l’évaluation.

**Rappel d’interprétation :** un juge reçoit une question et dix candidats anonymisés ; il attribue A, B, C, D, E ou `non_jugeable`. Les gains sont A=1, B=0,5, les autres=0 ; le dénominateur reste dix et une répétition après la première occurrence vaut zéro. Ce n’est ni un reranker ni une métrique exacte. Aucun score ne peut étayer une conclusion de supériorité avant `lawyer_agreement.json`.

### État de branche et prochain geste

- Branche : `paper/ecir-2027-reproducibility-clean`, PR publique : <https://github.com/matt-kaep/legal_knowledge_graph/pull/2>. Le papier LaTeX n’a pas été modifié.
- Assainissement finalise maintenant les exports versionnés E029, les registres et le commit de réparation. Dès réponse aux trois points E030 ci-dessus, la campagne du juge est gelée, soumise et surveillée sans modifier les listes déjà hashées.

## Mise à jour A3 ciblée — 2026-09-08

- Un inventaire sans E017/E021/E022 confirme : CV PPR complète pour G1, G6-AA et G7-AA-citation1/sémantique0,25 ; les replays présents ne couvrent que G6--Articles et G7--JP. La tentative complémentaire v1 a échoué avant évaluation et sans résultat, faute d’un champ redondant dans les champions locaux ; son successeur v2 vérifie le chemin source et le SHA-256 et tourne sous `985673`. Les quatre replays restent sans score à ce stade.
- G1 LightGCN a 120/120 reçus CV A3 valides. Son replay est scellé sur trois graines, avec Articles K2/lr 0,0005/lambda 0,5/7 époques et JP K3/lr 0,001/lambda 1/4 époques ; aucun chiffre G1 n’est encore transmissible. G7 LightGCN n’a aucune CV et ne sera pas lancé dans cette priorité.
- Le replay G1 a commencé sur une A30 libre (`985671`, après attente sans disponibilité H100). L’export structurel G1/G6/G7 est également en cours (`985678`) ; aucun des deux ne produit encore un chiffre à reprendre.
- **Action Papier :** ne rien ajouter au tableau à partir de ces nouvelles cellules avant la transmission post-replay et le contrôle des hashes/rankings. Les résultats G6 déjà transmis restent les seuls chiffres LightGCN reportables.

## Handoff consolidé — Checkpoints A3, B1 et B2 (2026-09-08)

- **A3 validé :** train/CV 5 578 questions ; évaluation interne inchangée de 754 questions (SHA `850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59`) ; cinq folds groupés sans fuite ; 13 236 candidats Articles et 114 851 JP. Manifeste `benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json`, SHA `c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3`.
- **B1-paper-ready utilisable :** les six valeurs exactes cosine/PPR/LightGCN G6, les preuves de CV/gel, les rankings top-100 et les courbes sont dans la transmission prioritaire ci-dessous. G6 est l’instanciation LightGCN principale présélectionnée, pas un champion démontré des onze graphes ; PPR sélectionne séparément G6 Articles et G7 JP sur train/CV.
- **E027 utilisable comme baseline LLM directe séparée :** le modèle ne voit que la question ; ses métriques exactes et hashes sont ci-dessous. **E029 est en cours mais non reportable :** cinq lots L40S et 31 668 entrées attendues. Le reranker voit le texte intégral `texte` pour les Articles, mais la `synthese` non tronquée pour chaque JP — pas la décision complète, ni `resume_avocat` ou des arguments. Les viviers sont des exports hashés ; il n'y a aucun appel direct à OVH pendant l'inférence. Aucun score n'est transmis. **E030 n’est pas lancé** et tout score de juge resterait exploratoire avant audit avocat.
- **Action Papier :** remplir maintenant les tableaux principaux Articles et JP avec les résultats B1, conserver les deux tâches distinctes et préparer un emplacement annexe pour E029. Ne pas utiliser E016/E017/LLM-as-a-Judge comme preuve de supériorité et ne pas transformer les sorties partielles E029 en résultats.
- **Formulation autorisée :** « Sur l’évaluation interne A3, après sélection par validation croisée sur l’entraînement et gel des configurations, nous observons les métriques exactes rapportées ci-dessous. »

## Transmission prioritaire — résultats exacts G6 et baseline LLM directe disponibles

- État E029 : le reranking exploratoire est en cours, sans score exporté. Les cinq shards L40S `985512` (cosine JP), `985513` (PPR), `985514`/`985515`/`985516` (LightGCN graines 42/43/44) ont démarré ; à 3 min 24, les réponses HTTP 200 partielles étaient 39, 58, 65, 73 et 85. Rien de ce run n'est encore réutilisable dans le papier. E030 n'est pas lancé.

- La configuration `G6-citation-AA-knn5` est le graphe principal présélectionné pour le papier, **pas** un champion validé contre les onze graphes. La CV G6 est complète (120/120 reçus) et les configurations ont été sélectionnées exclusivement par les cinq folds de train/CV, puis gelées : Articles K=2, lr=0,0005, lambda=0,5, 9 époques ; JP K=3, lr=0,0005, lambda=0,5, 5 époques. Preuves de CV/gel : `37fce90ff05643b54502ae5b40b22154792d94bb691880401de6ec0e89de5c4d`, `2d525254b10e263ff3f25ac4bc8e2e4c9c56b974d71f0653bfc45276a5d13fab`, `afa4aef329ff833c2199af7086e359505575b89dbe0a26f6c39863bccd3884b8`.
- Le replay LightGCN final `984841` a terminé `0:0` (trois graines 42/43/44) et a été contrôlé par le job de dérivation `984925` (`0:0`). Source unique à reprendre : `05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_campaign_b1_a3_paper_ready_retrieval_g6_lightgcn_v2_20260907/depth_curves/ranking_metrics_at_10.csv`, SHA-256 `9569ea74bb3d7fa50ef5fd80cd86cbc8879ec95fb41686ca8a34bdd79f20f7ad`. Reçu racine : `depth_curves_manifest.json`, SHA-256 `0913a9b1b83dae34f2cd55f84ae772ab5c73202e225b5f98c982c309035914de`; manifeste : `configs/paper_ready_retrieval_a3_g6_lightgcn_v2.json`, SHA-256 `33b8c2b9004396fe5fa2b59db9a05cbb7015d3aa9ece714fbe852cce8e2fad54`.
- Tableau Articles utilisable : cosine Hit/Recall@10 `0,4315102521999074`, NDCG@10 `0,32102115314737484`, MRR@10 `0,32082228116710876` ; PPR Hit/Recall@10 `0,5768809734326976`, NDCG@10 `0,39573594882260704`, MRR@10 `0,3735463769946528` ; LightGCN G6 (moyenne des trois graines) Hit/Recall@10 `0,5510694286556356`, NDCG@10 `0,40073652473423804`, MRR@10 `0,39287538770297387`.
- Tableau jurisprudence utilisable : cosine Hit@10 `0,21794871794871795`, NDCG@10 `0,15397741432193576`, MRR@10 `0,13577691465622502` ; PPR Hit@10 `0,23065870910698497`, NDCG@10 `0,142981907459661`, MRR@10 `0,11715506715506716` ; LightGCN G6 (moyenne des trois graines) Hit@10 `0,2663572060123784`, NDCG@10 `0,1777719765610496`, MRR@10 `0,15267848371296647`.
- Les rankings des trois méthodes ont été hashés avant lecture ; le contrôle couvre 754 questions, les 100 positions retournées, zéro doublon, zéro candidat hors des univers A3 (13 236 Articles, 114 851 décisions) et l'ordre candidat attendu. Courbes K=1--100 et figures utilisables : CSV `98f7db2e44a4ab8ce82f334e23d21e09dcaa366b727f7e6b62fb13d2a45e614d`, PNG `388f1078dde1f1f3b7a9eedbdbf9e38aaf63e65ca9ffc80e93a85d3fd6da8a54`, PDF `e6b2eac3a8bca55350268c212a895e452a8e8d174c07580d437694dcb346b041`.
- Formulation autorisée : « Sur l’évaluation interne A3, après sélection par validation croisée sur l’entraînement et gel des configurations, nous observons les métriques exactes ci-dessus. » Ne pas affirmer une supériorité inter-graphes, ne pas mélanger E017/E021/E022 et ne pas qualifier E030 de validé.
- E027 est terminé une seule fois : `984939` Articles (`0:0`) et `984940` jurisprudence (`0:0`). Le modèle direct ne voit que la question et son prompt fixe ; il ne reçoit ni vivier ni corpus. À reprendre comme baseline exacte sur l'évaluation interne A3 : Articles Hit/Recall@10 `0,09497442212959453`, NDCG@10 `0,08860546818088738`, MRR@10 `0,11098006399730537` ; jurisprudence Hit@10, NDCG@10 et MRR@10 `0`. Sources hashées : métriques/rankings Articles `e813da7413370f8406e6f186e8d95a38122a8634b3a1d42fc263f120a65cd16e` / `c44158f14a8501f9c10bc836a99f593729564f3711e60a3f662fe0d002acd1a4`, JP `86d55401175e959e90886ba52b395f95eeb3836728c4610d5f6911012c13d0ff` / `8b5551200e06e57c8b12d0e2349ee6eaf1a52d2df1fc7a0e8686cf36abdf696d`. Le contrôle indépendant confirme 754 questions × 10 positions, zéro doublon parmi les références résolues et zéro référence résolue hors A3 ; les positions vides sont des zéros, pas des candidats. E029 reste distincte : 15 conditions cosine/PPR sont compatibles et 25 doivent apparaître `--`; le préflight G6 v1 CPU `984986` a terminé `0:0`, sans appel LLM, reçu `b5b32784691016a8ba3fb927a0ea64320a8f9d06ae37ba2297087b1ce18188ce`. Le préflight G6 seedé v2 `985046` a terminé `0:0` (trois graines, 27 viviers, `model_calls=0`) ; son reçu vaut `e1635a0a014b2e66ac0ecfa509bda14c57f2d131f107775160371c7a0f1b1d89`. Les jobs CPU v1 `985481` (`1:0`) puis v2 `985493` (`127:0`) ont échoué avant tout lot ou appel modèle : v1 supposait un répertoire unique pour les viviers, puis v2 utilisait `dirname($0)` alors que Slurm copie son wrapper hors du dépôt. Le successeur v3 `985497` a ensuite terminé `0:0` en 48 secondes : son reçu SHA-256 `ca10235c8a9570b58ec8527581c404f26882834c31af7d235b52ce13710d80a3` certifie cinq lots et 31 668 entrées de modèle sans aucun appel modèle. Le manifeste GPU `configs/b2_reranking_comparable_a3_execution_v1.json`, SHA-256 `259cff89780dfb4bfa39159cc0f608fdd295c559391858d376149170e8ff8515`, a été soumis sur L40S sous `985512` (cosine JP), `985513` (PPR), `985514`/`985515`/`985516` (LightGCN graines 42/43/44) ; le contrôle de soumission les montre tous en attente. Aucun résultat de reranking n'est encore à reprendre. E030 a maintenant un contrat préparatoire distinct, mais aucune liste top-10, modèle ou score n’est gelé ; il n’y a donc aucun résultat de juge à reprendre. Tout score futur demeure exploratoire jusqu'à l'audit avocat.

## Transmission prioritaire — B1-r2 LightGCN en cours, aucun nouveau score

- E025 est effectivement en cours, mais **aucun résultat LightGCN B1-r2 n’existe encore** : aucun champion de validation croisée, aucun replay final, aucun classement des 754 questions et aucune comparaison PPR--LightGCN à reprendre dans le papier.
- La preuve de préparation est `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r2.json`, SHA-256 `9427430f436ca5fc1e2d2bdc9858880c7ed45daaaf48d6b88ca170df614e440c`. Il hérite de B1-r1 par hash et conserve A3 : 5 578 questions train/CV, 754 questions d’évaluation, 13 236 candidats Articles, 114 851 candidats JP, onze graphes et cinq folds groupés.
- La seule modification opérationnelle est l’exécution : 1 320 tâches CUDA atomiques, une par combinaison graphe/fold/configuration/cible, avec reçu hashé. L’agrégateur refuse toute sélection de champion si la couverture n’est pas totale. Le préflight durable Télécom a vérifié 52 entrées immuables : `.../_campaign_b1_a3_effective_retrieval_r2_cuda_atomic_20260903/lightgcn_cv/preflight.json`, SHA-256 `9d816a7ec6c9af83d7e66a2c7a403257e990e388aecd692181b57a72240d3f0a`; le plan des tâches vaut `37fce90ff05643b54502ae5b40b22154792d94bb691880401de6ec0e89de5c4d`. La sonde A40 Slurm `978803` est terminée `0:0` en 51 min 56 s, avec CUDA/NVIDIA A40 et couverture A3 attestées ; son reçu technique vaut `5635f077b24eff847e0d7d5503d237a61cf498b4cb6c7c32d0502775999de7cb`. Au contrôle intégral du 7 septembre, **177 reçus** sont validés : G1 compte ses 120 tâches CV, G6 avec liens Article--Article en compte 57. Chaque reçu a été contrôlé de bout en bout (appartenance au plan, hashes des fichiers, CUDA/A40, couverture A3 et zéro question sans positif récupérable). Huit GPU sont actifs et 22 tâches attendent le quota, sans échec Slurm observé. Aucun score de sonde ou de lot partiel n’est transmissible. Télécom autorise 30 sous-tâches soumises mais seulement huit GPU actifs pour ce compte (`QOSMaxGRESPerUser`) ; l’attente résiduelle est donc un quota, non un défaut de pipeline.
- Les résultats exacts B1-r1 cosine et PPR restent inchangés et transmissibles selon les sections ci-dessous. Ne pas utiliser E017, E021, E022, les quatre historiques LightGCN B1-r1, ni la préparation B1-r2 comme résultats.

## Transmission prioritaire — CV PPR B1-r1 terminée et gelée ; replay final en attente

- La soumission B1 initiale reste archivée par `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3.json`, SHA-256 `f1107b126e11b0457dc28a4fbe3db621b1c061a932e2acda1eb1368bac0be649`. PPR (`977101`) et LightGCN (`977102`) ont échoué avant toute sélection CV car les lanceurs cherchaient le chemin de folds historique au lieu du split A3 ; les tâches LightGCN restantes ont été annulées. Cosine (`977100`) a terminé comme sortie technique isolée. Ne pas utiliser ces sorties initiales dans le papier.
- La reprise B1-r1 est gelée par `05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r1.json`, SHA-256 `1b612a182742244dad59006e6d01b826a0285f01123aeeae67321b48c9de5e9a`. Elle référence exclusivement A3, SHA-256 `c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3` : 5 578 questions train/CV, 754 questions d’évaluation, 13 236 candidats Articles et 114 851 candidats JP. Son préflight valide 50 entrées hashées.
- E026 B1-r1 (cosine, Slurm `977124`) est validée pour les métriques exactes : Articles Hit@10 `0,4315102521999074`, NDCG@10 `0,32102115314737484`, MRR@10 `0,32082228116710876` ; JP Hit@10 `0,21794871794871795`, NDCG@10 `0,15397741432193576`, MRR@10 `0,13577691465622502`. Preuve : ranking top-100 SHA-256 `7de0504d5e7a7a7caef303d0beb8019c28faac7dec1f38997e581ca782597334` et export A3 `cosine_exact_metrics_v2/ranking_metrics_at_10.csv`, SHA-256 `30f8b11d4392c5d4cbc3d7cdb90fcd2209504e72aeee83f1cc21a1b0f48ac007`. Le classement couvre 754/754 questions par tâche, sans doublon ni candidat hors univers. Le timing détaillé reste à compléter : ne pas encore le présenter.
- La CV PPR E024 B1-r1 est terminée : les onze sous-jobs de `977157` ont tous quitté le cluster avec `0:0`, chaque sortie couvrant les cinq folds et les 5 578 questions train. Le gel inter-graphes est `.../_campaign_b1_a3_effective_retrieval_r1_20260902/frozen/ppr_champions.json`, SHA-256 `61e80ca1d1f746f4c759c6c118bb79fa605ed6c869bb04498b9b6abeee48db6d`; il est explicitement `train_cv_only` et référence les hashes des onze synthèses CV. Champion Articles : G6 avec liens Article--Article, `k_in=50`, graines `both`, alpha `0,5`. Champion JP : G7 avec liens Article--Article, poids citation `1` et sémantique `0,25`, même configuration PPR. **Ce sont des choix de configuration sur train/CV, pas des résultats évalués à reprendre dans le papier.**
- Les dispatches PPR `978593` et `978599` sont des traces techniques non reportables : ils ont échoué avant toute lecture de l'évaluation, car le staging du manifeste hashé n'était d'abord pas transmis puis utilisait `/tmp`, non partagé avec le nœud Slurm. Aucun fichier `ppr_final`, ranking ou score B1-r1 n'a été produit. Le replay `978600` est maintenant en cours après préflight hashé sur le nœud : même manifeste parent B1-r1 et mêmes champions gelés. Seule formulation autorisée à ce stade : « Les configurations PPR ont été sélectionnées sur les cinq folds groupés de l’entraînement puis gelées avant l’évaluation. » Ne pas écrire un score PPR B1-r1 ni comparer PPR à cosine ou LightGCN.
- Le smoke GPU LightGCN B1-r1 (`977158`, G1, cinq folds, une époque) a aussi terminé avec couverture complète et gate mémoire respecté. La CV complète LightGCN B1-r1 (`977184`, sous-job `977185`) a été arrêtée volontairement : quatre historiques partiels du seul fold 0 de G1 existent, mais aucune CV complète, aucun champion ni replay final. La grille actuelle demanderait environ 116 heures par graphe pour cinq folds, au-delà de la limite Slurm de 24 heures ; les traces sont archivées, non reportables, et une reprise B1-r2 devra être scellée avant nouveau calcul. E028 produira les courbes comparatives uniquement à partir des rankings gelés des campagnes complètes. E017, E021 et E022 sont explicitement exclus ; E027, E029 et E030 ne sont pas lancées.
- **Seule la baseline cosine exacte est transmissible au papier.** PPR B1-r1 est méthodologiquement gelé mais ne devient transmissible qu'après son replay final contrôlé ; LightGCN B1-r1 reste arrêté techniquement.
- Contrat de résultat à retenir : le tableau principal utilisera Hit@10 normalisé, après dédoublonnage. Pour les Articles, Hit@10 = Recall@10 dans ce benchmark (maximum vérifié de dix labels stricts par question). NDCG@10 et MRR@10 seront exportés distinctement. Ne pas confondre ce Hit@10 avec « au moins une réponse exacte ».
- Formulation autorisée uniquement pour la méthode : « La campagne de comparaison a été pré-enregistrée sur le snapshot A3, avec sélection par validation croisée sur l’entraînement puis gel des configurations avant évaluation. » Attendre la transmission post-replay pour tout chiffre ou classement.

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

### 2026-09-03 — PPR B1-r1 final : résultats exacts désormais disponibles

Le replay PPR B1-r1 a terminé sur Télécom (`978600`, `0:0`, 4 min 55 s) après le gel train/CV des champions. Il est indépendant des sorties historiques E017/E021/E022. La preuve racine est `.../_campaign_b1_a3_effective_retrieval_r1_20260902/ppr_final_exact_metrics_v1/depth_curves_manifest.json`, SHA-256 `851b9f771a45cf1c633900dd84a8922cfd0ea67d99e5142180827a89cb65732d`. Elle relie le manifeste B1-r1, A3, le script de dérivation et le ranking top-100 source (SHA-256 `8f15c4d3e1ba465af7030e24f0e78772cae776fe8f12f5020f578d600964a938`). Les 754 questions de chaque tâche ont 100 positions, sans doublon ni candidat hors des univers A3 de 13 236 Articles et 114 851 décisions.

- Articles, champion PPR `G6-citation-AA-knn5` avec configuration gelée `k_in=50`, graines `both`, `alpha=0,5` : Hit/Recall@10 `0,5768809734326976`, NDCG@10 `0,39573594882260704`, MRR@10 `0,3735463769946528`.
- Jurisprudence, champion PPR `G7-citation-AA-cit1-sem025-knn5` avec la même configuration : Hit@10 officiel `0,23065870910698497`, NDCG@10 `0,142981907459661`, MRR@10 `0,11715506715506716`. Ne pas substituer à ce Hit@10 l'indicateur binaire `exact_any_gold_at_10`.
- Les valeurs et le hash sont dans `REGISTRE-RESULTATS.csv` (lignes `R-E024-B1R1-PPR-*`). Les courbes PPR K=1–100 existent également en CSV (SHA-256 `a52373c8f1c027244f4369149f2bbcb50f57f2635c5d988ecb70bf550f0f5258`). La figure comparative doit attendre LightGCN B1-r2, qui n'est pas encore calculé.

Formulation utilisable : « Après sélection des configurations exclusivement sur les cinq folds d'entraînement, nous avons gelé un champion PPR distinct par tâche et l'avons rejoué sur les 754 questions d'évaluation. » Ne pas présenter les valeurs train/CV de sélection comme résultats d'évaluation, ne pas mélanger les expériences historiques et ne pas annoncer de comparaison PPR–LightGCN avant le run LightGCN B1-r2.
