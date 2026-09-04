---
title: E017 - Comparaison LightGCN inter-graphes par score JP gradue
date: 2026-08-11
status: exploratoire-agrege-en-attente-audit-avocat
tags:
  - benchmark
  - lightgcn
  - jurisprudence
  - exploratoire
---

# E017 — Comparaison LightGCN inter-graphes par score JP gradué

## Statut scientifique

`exploratoire_internal_evaluation` — les 33 replays LightGCN et les 14 309 jugements gradués sont complets sur `eval_rich_retrievable_strict`, déjà consulté. Les chiffres ci-dessous expliquent les écarts entre graphes, mais ils ne sont pas transmissibles comme résultats confirmatoires : le score gradué reste en attente de validation par l'audit avocat E016.

## Périmètre

- 11 graphes du manifeste `confirmatory_campaign_grouped_v2.json` : G1, deux G6 et huit G7.
- LightGCN figé : K=2, learning rate 0,001, ancrage 1, négatifs aléatoires, 30 epochs maximum.
- Seeds : 42, 43 et 44.
- Sélection d'epoch : cinq folds groupés, train uniquement, séparément pour Articles et JP.
- Replay : top-10 sur les 754 questions d'évaluation interne, après gel des epochs.
- Jugement gradué : réutilisation du cache E016 et jugement L40S des seuls couples question-JP nouveaux.

## Contrat et données

- Manifeste matérialisé : `data/doctrine_v3plus_bench/_protocol/e017_intergraph_graded_jp/campaign_manifest.json`.
- 33 tâches scientifiques `(graphe, seed)` regroupées en 11 éléments Slurm.
- 91 fichiers d'entrée vérifiés par taille et SHA-256 sur le cluster.
- Aucun secret PostgreSQL ni corpus Judilibre brut transféré.
- Runtime CPU observé : Python 3.12.3, NumPy 1.26.4, Pandas 3.0.2, SciPy 1.13.1, PyTorch 2.10.0+cu128, device CPU.

## Exécution cluster

Les deux premières tentatives sont conservées comme incidents techniques sans résultat scientifique :

- `936124` / `936125` : arrêt avant entraînement, `fold_metadata.json` absent du premier manifeste de transfert ; replay annulé.
- `936154` / `936155` : trois graphes poursuivis ; huit graphes arrêtés avant entraînement car leurs copies locales des identifiants partagés n'avaient pas été transférées.

Exécution finale après correction et vérification des 91 entrées :

- CV initial pour les indices 0, 1 et 7 : `936154` ; statuts finaux complets.
- Replays correspondants : `936155` ; statuts finaux complets.
- CV de reprise pour les indices 2–6 et 8–10 : `936188` ; statuts finaux complets.
- Replays correspondants : `936189` ; statuts finaux complets.

Au contrôle du 12 août 2026, les 33 CV et 33 replays sont `complete`. Les 33 rankings JP contiennent chacun 754 questions et 7 540 positions à K=10, avec une séquence de rangs complète.

## Pool gradué et lancement GPU

- Positions conservées pour l'agrégation : 248 820, soit 33 rankings × 754 questions × K=10.
- Couples uniques question–JP : 14 309 ; jurisprudences distinctes : 8 124.
- Doublons internes aux top-10 : 1 564 positions, conservées avec gain nul après la première occurrence.
- Cache E016 réutilisé par identité exacte : 7 348 réponses `ok`.
- Jugements nouveaux à calculer : 6 961.
- Fiches juridiques : 8 124/8 124 disponibles ; 3 332 ont été récupérées localement sans transférer les accès PostgreSQL.
- Pilote A100 `937671` : `COMPLETED`, 30/30 réponses `ok`, 0 invalide, 0 erreur, 10,902 s d'inférence hors démarrage vLLM.
- Le pilote H100 `937667` a été annulé avant démarrage faute de ressource disponible ; aucun calcul ni artefact n'en provient.
- Run complet A100 : `937682`, terminé avec code retour `0` en 15 min 01 s le 12 août 2026 ; 6 931 nouveaux jugements `ok`, zéro invalide et zéro erreur. Les 30 réponses du pilote sont présentes dans le journal final, qui contient 14 309 réponses `ok` au total.
- Modèle et snapshot inchangés par rapport à E016 : `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, révision `519bdca117c8f10a9a578d1b70b5c0d54c59b7ba`.

## Agrégation terminée (2026-08-12)

L'agrégateur `scripts/84_summarize_e017_intergraph_graded_jp.py` conserve les classements par couple `(graphe, seed)`, calcule les métriques exactes officielles avec `metrics.py`, puis agrège la moyenne et l'écart-type sur les trois seeds. Le score gradué est calculé séparément avec le contrat E016 ; il ne remplace pas `Hit@10`.

| Lecture | Graphe | Valeur moyenne sur 3 seeds | Dispersion inter-seed |
|---|---|---:|---:|
| Meilleur `Hit@10` exact | G6 citation AA | 0,2685 | 0,0051 |
| Meilleur NDCG@10 exact | G6 citation AA | 0,1816 | 0,0015 |
| Meilleur score gradué@10 | G7 JJ, citation 1 / sémantique 0,50 | 0,4317 | 0,0020 |
| G7 JJ, citation 1 / sémantique 0,25 |  | Hit@10 0,2437 ; score gradué@10 0,4278 | 0,0047 ; 0,0018 |

Ces lectures montrent une dissociation : la variante la mieux classée par Ground Truth exacte n'est pas celle qui maximise le score gradué du juge LLM. Cela peut refléter une pertinence juridique alternative, une Ground Truth incomplète, ou une permissivité du juge ; E017 seul ne permet pas de trancher.

## Diagnostic approfondi pour la présentation (2026-08-12)

La comparaison détaillée porte sur les deux variantes qui gagnent selon l'une des deux lectures : G6 citation AA pour les métriques exactes et G7 JJ citation 1 / sémantique 0,50 pour le score gradué. Elle ne constitue pas une ablation causale : la famille de liens, leur pondération et la normalisation ne sont pas identiques.

## E020 - JP qui bougent et Gold observées

Le diagnostic E020 sépare les JP communes des JP exclusives à chaque paire de graphes, puis joint les classes graduées. Pour G6-AA contre G7-JJ c1/s0,50, les JP communes sont A/B dans 57,29 % des cas, contre 45,54 % pour les exclusives G6 et 46,17 % pour les exclusives G7. Les substitutions restent donc partiellement utiles, mais moins que le noyau commun.

Parmi les 978 couples Ground Truth, 263 ont été retournés au moins une fois et possèdent déjà un jugement E017 : 226 sont A, 23 B, 2 C, 2 D et 10 E. Ce sous-ensemble observé est cohérent avec la Ground Truth, mais il ne permet pas de conclure pour les 715 Gold jamais retournées. Voir `../e019-g0-g7-reduced-method-comparison-2026-08-12/E020-Mouvements-JP-Et-Gold-2026-08-12.md`.

### Rang, classes négatives et recouvrement

- G6-AA est meilleur au rang 1 : 68,92 % de A/B contre 66,84 % pour G7-JJ. G7 passe devant sur le score cumulé fixed-K au rang 5 (`0,2408` contre `0,2392`) et termine à `0,4317` contre `0,4289` au rang 10.
- La proportion A/B sur les 22 620 positions des trois seeds est proche : 53,80 % pour G6-AA et 53,99 % pour G7-JJ. Les classes C/D/E représentent respectivement 46,20 % et 46,01 % ; après ajout des doublons à gain nul, les positions sans gain atteignent 46,51 % et 46,27 %.
- Le rang 10 reste moins pertinent que le rang 1, mais G7 y conserve davantage de A/B : 47,83 % contre 45,62 %.
- Les top-10 partagent en moyenne 7,04 JP par question et seed, avec un Jaccard moyen de 0,565. Parmi les positions exclusives, G7 contient 46,17 % de A/B contre 45,54 % pour G6 : le faible gain gradué vient d'environ trois substitutions, pas d'un reclassement massif des JP partagées.

### E019-A -- matrice de recouvrement des listes JP

E019-A relit les 33 rankings E017 sans entraînement, reranking ni nouveau jugement. Il compare les top-1, top-3, top-5 et top-10 pour les 55 paires de graphes, sur les trois seeds et les 754 questions. Son rôle est descriptif : il mesure si deux scores proches proviennent des mêmes JP ou d'alternatives différentes.

- Pour G6-AA contre G7-JJ c1/s0,50, l'intersection moyenne vaut 0,64 au top-1, 2,01 au top-3, 3,43 au top-5 et 7,04 au top-10. Le Jaccard top-10 est 0,565.
- Les JP exclusives représentent donc environ 2,89 positions G6 et 2,91 positions G7 par liste top-10. Elles sont A/B dans 46,34 % des cas côté G6 et 47,54 % côté G7 ; l'écart gradué tient à quelques substitutions plutôt qu'à une liste radicalement différente.
- Certaines variantes AA de G7 sont presque identiques : `cit1/sem0,25` contre `cit1/sem0,50` partage 9,63 JP sur 10 en moyenne (Jaccard 0,945). A l'inverse, les comparaisons AA/JJ les plus éloignées ne partagent qu'environ 5,78 JP sur 10 (Jaccard 0,429).

Les exports sont `e019_jp_ranking_overlap_pair_metrics.csv` (55 paires x 4 seuils) et `e019_jp_ranking_overlap_per_question.parquet` (diagnostic top-10 compressé) sous `data/doctrine_v3plus_bench/E017-intergraph-graded-jp-v1/`. E019-A ne démontre ni qu'un graphe est meilleur, ni un effet causal des arêtes : il décrit les listes déjà évaluées.

### Dispersion entre questions

- G7-JJ gagne contre G6-AA sur 301/754 questions après moyenne des trois seeds, fait égalité sur 163 et perd sur 290. L'écart moyen est `+0,002807`, la médiane est nulle ; 193 gains et 184 pertes sont stables sur les trois seeds.
- G7-JJ est plus polarisé : 6,76 % de questions à score nul contre 5,66 % pour G6-AA, mais 2,21 % de top-10 parfaits contre 1,64 %. Son premier quartile est plus bas (`0,15` contre `0,20`) et son troisième quartile plus haut (`0,70` contre `0,65`).
- Les questions sous `0,25` restent nombreuses et proches : 30,24 % pour G7-JJ contre 29,75 % pour G6-AA. Les questions à score au moins `0,50` sont 42,71 % contre 43,55 %.

### Sous-groupes et lecture juridique

- G7-JJ progresse surtout sur les questions `articulation_textes` (`+0,0121`, n=330) et de granularité `transversale` (`+0,0119`, n=278).
- Il régresse sur les `cas_pratique` (`-0,0170`, n=108) et légèrement sur les questions `precise` (`-0,0022`, n=145).
- Cas stable favorable à G7 : pour les limites de l'action civile après une infraction terroriste, G7 ajoute des décisions A/B qui appliquent l'exigence d'un préjudice personnel et direct provenant des éléments constitutifs de l'infraction. Le texte de la décision `5fca758a15bbab62b2809ef7` sur la constitution de partie civile de la commune de Nice confirme directement cette règle. G6 retrouve la JP attendue, puis retourne davantage de décisions seulement thématiques sur le terrorisme.
- Cas stable favorable à G6 : pour le journaliste poursuivi pour recel de pièces de l'instruction produites afin d'établir sa bonne foi en diffamation, G6 retourne les décisions qui articulent recel, secret de l'instruction et nécessité des droits de la défense, dont `6079a8c69ba5988459c4ee07`. G7 dérive vers la bonne foi journalistique et la diffamation en général.

L'hypothèse mécanistique est donc double : les liens JP--JP semblent mieux traverser les fondements textuels pour récupérer des formulations alternatives d'une même question de droit, tandis que les liens article--article semblent mieux préserver la frontière textuelle d'une exception ou d'une condition procédurale précise. Les deux cas lus illustrent cette hypothèse mais n'en estiment pas la fréquence ; l'audit avocat et une ablation à facteur unique restent nécessaires.

Artefacts produits dans `data/doctrine_v3plus_bench/E017-intergraph-graded-jp-v1/` :

- `e017_graph_seed_metrics.csv` — 33 lignes, une par replay ;
- `e017_graph_metrics.csv` — 11 lignes, moyenne et écart-type par graphe ;
- `e017_per_question_metrics.csv` — 24 882 lignes, diagnostic par question ;
- `e017_summary.json` — hashes des sources et des sorties.
- `01-Projet/presentations/E016-Evaluation-JP-Graduee-2026-08-11.pdf` — deck de cinq slides intégrant le tableau des onze graphes et le diagnostic approfondi.

## Étapes suivantes

1. Exploiter le paquet avocat E016 avant toute interprétation de la différence entre score gradué et score exact.
2. Diagnostic E018 : sur les 33 replays, relier chaque faux négatif exact aux liens G8-Large bruts. Les résultats montrent que le rattrapage « même règle » co-varie avec l'exact, tandis que le rattrapage « même question » co-varie avec le score gradué ; ces associations restent exploratoires, sans arête G8 matérialisée. Voir `../g8-llm-verified-jp-jp-2026-08-05/G8-Large-Analyse-Descriptive-2026-08-12.md`.
3. Auditer aveuglément les quatre catégories E018 (exact, même règle, même question seule, sans lien), puis définir le filtre G8 même procédure/noyau factuel, geler une matérialisation et évaluer son effet causal séparément.
