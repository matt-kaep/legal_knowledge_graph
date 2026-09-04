---
date: 2026-08-11
type: fiche
status: en-cours
tags: [benchmark, jurisprudence, llm-judge, evaluation-graduee, avocat, g7]
---

# E016 — Évaluation graduée des JP retournées par G7

## Statut scientifique

`exploratoire_en_attente_audit_avocat` — le jugement complet et l'agrégation sont terminés. Le score reste provisoire jusqu'au contrôle des 100 cas avocat et aucun résultat ne devient confirmatoire sur `eval_rich_retrievable_strict`, déjà consulté.

## Périmètre figé

- graphe : `G7-citation-JJ-cit1-sem025-knn5` ;
- méthode : `LightGCN-trained_K2` ;
- évaluation : 754 questions, dix JP par question, soit 7 540 positions ;
- entrée du juge : question + fiche Step1 G8 existante ;
- sortie : `classe` parmi A, B, C, D, E, `non_jugeable`, plus une justification juridique concrète d'une phrase ;
- gains : A = 1, B = 0,5, autres classes = 0 ;
- score : `(n_A + 0,5 × n_B) / 10` sur les premières occurrences, avec dénominateur fixe et répétitions suivantes à zéro ;
- juge initial : `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit`, snapshot `519bdca117c8f10a9a578d1b70b5c0d54c59b7ba`, température 0, schéma JSON strict ;
- audit : 100 cas stratifiés, avocat aveugle au label LLM, accès possible au texte intégral.

Le juge ne reçoit jamais le rang, la méthode, les JP gold, une relation G8 ni une distance dans le graphe. Les erreurs techniques bloquent l'agrégation ; elles ne sont jamais transformées en `non_jugeable`.

Le préflight a détecté 53 positions répétant une JP déjà présente plus haut dans le top-10, sur 30 questions. Elles proviennent de doublons historiques dans le pool JP utilisé par LightGCN. E016 conserve ces positions pour évaluer la sortie réellement produite, juge chacun des 7 487 couples uniques une fois et attribue un gain effectif nul à chaque répétition après la première. Une même JP ne peut donc pas rapporter deux fois des points.

## Chaîne d'artefacts

```text
rankings.parquet + bench_global.json + jp_decisions.step1_raw
  -> rankings_topk.parquet + decision_cards.json + judge_jobs.jsonl + manifest.json
  -> judge_responses.jsonl
  -> graded_jp_detail.csv + graded_jp_per_question.csv + graded_jp_summary.json
  -> lawyer_audit_sample.csv + lawyer_audit_key.csv + lawyer_evidence.jsonl
  -> lawyer_agreement.json
```

Les données et sorties volumineuses restent dans le checkout principal sous :

`05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/G7-citation-JJ-cit1-sem025-knn5/eval_rich_retrievable_strict/E016-g7-graded-jp-v1/`

## Exécution locale

Depuis la racine du dépôt :

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py preflight
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py prepare
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py status
```

`prepare` lit les fiches Step1 présentes dans PostgreSQL sans les recalculer et sans écrire en base. Le manifeste refuse l'écrasement par défaut ; `prepare --force` ne doit être utilisé que pour reconstruire volontairement un bundle dont le changement sera ensuite visible dans les hashes.

## Calibration autorisée

Le prompt se calibre uniquement sur le train :

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/75_prepare_g7_graded_jp_eval.py \
  --profile calibration \
  --question-limit 30 \
  --seed 42 \
  --out-dir 05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/calibration/E016-g7-graded-jp-v1
```

Le profil `calibration` refuse tout chemin contenant `eval_rich_retrievable_strict`. Le pilote réel doit être intégralement inspecté. Si le prompt change, sa version et son hash changent avant la préparation E016 complète.

## Jugement GPU

Après transfert du dépôt et du bundle de jobs sur le cluster :

```bash
sbatch 05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_g7_graded_jp_judge.sh full
```

Le runner écrit en JSONL append-only et reprend les jobs déjà valides. Pour rejouer les sorties `invalid` ou `error` après correction de l'incident :

```bash
RETRY_NON_OK=1 sbatch 05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_g7_graded_jp_judge.sh full
```

Le statut n'est complet que lorsqu'une réponse `ok` existe pour chaque job dont la fiche est disponible.

## Agrégation et paquet avocat

Après rapatriement de `judge_responses.jsonl` :

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py summarize
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py status
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py \
  select-lawyer-audit --seed 42 --sample-size 100
```

Pour joindre les décisions intégrales, répéter `--corpus CHEMIN_JSONL` pour chaque corpus Judilibre nécessaire. Le CSV avocat omet la classe LLM, sa justification, le rang, l'exactitude gold et les poids. La clé privée conserve ces champs et ne doit pas être transmise avant la fin des annotations.

Une fois les 100 annotations remplies :

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/79_summarize_g7_graded_jp_lawyer_audit.py
```

Le gate exploratoire exige un accord sur les gains d'au moins 0,70 et une précision pondérée A/B d'au moins 0,85. Les intervalles utilisent 2 000 bootstrap stratifiés déterministes, graine 42. Si le prompt est modifié après cet audit, ces 100 cas deviennent une calibration et un nouvel audit indépendant est requis.

## Résultats

Préparation locale terminée :

- 754 questions et 7 540 positions ;
- 4 865 JP distinctes ;
- 7 487 couples question–JP uniques à juger ;
- 4 865 fiches Step1 disponibles, aucune position sans fiche ;
- 53 positions répétées, soit 30 questions affectées ;
- bundle de calibration train-only : 30 questions, 300 positions, 298 couples uniques, aucune fiche manquante.

Le jugement complet des 754 questions est agrégé. Les sorties donnent :

- `score_gradue@10` moyen : `0,427122` ;
- distribution : A=`2 442`, B=`1 592`, C=`102`, D=`428`, E=`2 976`, `non_jugeable`=`0` ;
- taux `non_jugeable@10` : `0` ;
- 53 positions répétées sur 7 540 (`0,7029 %`), pénalisées après leur première occurrence ;
- échantillon avocat : 100 cas, aveugle aux labels LLM, stratifié A=27, B=22, C=17, D=17, E=17 et accompagné des poids d'inclusion dans la clé privée.

Le champ technique `exact_hit_at_10=0,263926` produit par l'agrégateur indique seulement si au moins une JP gold apparaît dans le top-10. Il ne doit pas être confondu avec le `Hit@10` officiel du benchmark, défini par $|A_q\cap R_q[:K]|/\min(|A_q|,K)$ et reporté séparément.

## Analyse descriptive provisoire

Cette section croise les sorties graduées avec la Ground Truth uniquement après le jugement aveugle. Elle décrit le run complet, mais son interprétation juridique reste conditionnée par l'audit avocat.

### Exactitude historique contre pertinence graduée

- le `Hit@10` officiel de G7 est `0,250000` ;
- 199 questions sur 754 (`26,39 %`) contiennent au moins une JP exacte dans le top-10 et 555 n'en contiennent aucune ;
- 692 questions sur 754 (`91,78 %`) contiennent néanmoins au moins une JP classée A ou B ; seules 62 (`8,22 %`) ont un score gradué nul ;
- parmi les 555 questions sans JP exacte, 498 (`89,73 %`) ont au moins une A/B et 232 (`41,80 %`) ont un score gradué d'au moins `0,5` ;
- le score gradué moyen vaut `0,4111` sans JP exacte contre `0,4719` lorsqu'une JP exacte est présente ;
- la corrélation par question entre `Hit@10` officiel et score gradué est faible : Pearson `0,078`, Spearman `0,098`.

| JP exacte dans le top-10 | aucune A/B | au moins une A/B | total |
|---|---:|---:|---:|
| non | 57 | 498 | 555 |
| oui | 5 | 194 | 199 |
| total | 62 | 692 | 754 |

Cette dissociation est compatible avec deux explications qui ne peuvent pas encore être départagées : soit G7 retourne de nombreuses alternatives juridiquement pertinentes absentes de la Ground Truth, soit le juge LLM surclasse une partie des décisions. Le gate avocat a précisément pour rôle de mesurer ce second risque.

### Contrôle interne de cohérence

Après exclusion des 53 positions répétées, les 7 487 couples uniques se répartissent ainsi :

- 204 couples correspondent à une JP exacte ; 191 (`93,63 %`) sont classés A ou B, dont 170 A, 21 B, 2 C, 2 D et 9 E ;
- 7 283 couples ne correspondent pas à une JP exacte ; 3 821 (`52,46 %`) sont classés A ou B, dont 2 259 A et 1 562 B ;
- ces 3 821 couples non exacts A/B touchent 685 questions, dont 498 des 555 questions sans aucun hit exact.

Le signal gradué décroît avec le rang : le gain effectif moyen passe de `0,5623` au rang 1 à `0,3289` au rang 10 ; la part A/B passe de `66,58 %` à `43,10 %`. Cette décroissance est cohérente avec un ranking qui concentre davantage de décisions jugées utiles en tête, mais elle ne constitue pas une comparaison à une baseline aléatoire.

### Premières familles de diagnostic

- **57 questions sans hit exact et sans A/B** : échecs G7 les plus nets dans cette grille ;
- **498 questions sans hit exact mais avec A/B** : candidates principales pour mesurer l'incomplétude de la Ground Truth ;
- **5 questions avec hit exact mais score nul** : désaccords prioritaires entre Ground Truth et juge ;
- **13 couples exacts classés C, D ou E** : cas prioritaires pour vérifier soit la pertinence de la Ground Truth, soit une erreur du juge ou de la fiche.

Le type de question montre aussi une dissociation : `cas_pratique` obtient le meilleur `Hit@10` officiel (`0,4213`) mais un score gradué moyen de `0,3958`, tandis que `articulation_textes` obtient un `Hit@10` de `0,1758` mais un score gradué de `0,4676`. Ce contraste est un axe de diagnostic, pas encore une preuve d'incomplétude différentielle.

### Limite du paquet avocat actuel

L'échantillon de 100 cas valide la qualité du juge LLM sur la population des sorties G7. Il contient 98 couples non exacts et seulement 2 couples exacts, conformément à leur faible fréquence dans les 7 487 couples uniques. Il ne permet donc pas, à lui seul, d'estimer solidement la qualité de toute la Ground Truth :

- il n'audite presque pas les couples question--JP exacts ;
- il n'inclut aucune JP attendue absente du top-10 G7 ;
- il ne transforme pas automatiquement les A/B non exactes en nouvelles annotations de référence.

Après le gate avocat, une évaluation directe du benchmark devra donc échantillonner séparément les 978 couples question--JP de la Ground Truth, y compris ceux que G7 ne retourne pas. Le diagnostic G8 et les distances de graphe restent différés jusque-là, conformément au protocole.

### Pilote train-only et lancement complet

Trois premières soumissions n'ont produit aucun jugement :

- `935280` : échec avant vLLM, cache par défaut `/scratch/kaeppelin-22` non accessible ;
- `935290` : échec avant chargement, révision courte non résolue en ligne après évolution du dépôt Hugging Face ;
- `935297` : le snapshot local complet `519bdca117c8f10a9a578d1b70b5c0d54c59b7ba` charge correctement 16,47 Gio de poids, puis la capture CUDA échoue par manque de mémoire sur `nodemm02` dans la partition générique `mm`.

Le job `935516`, exécuté ensuite sur `L40S`/QOS `normal`, valide le pilote train-only : 298 couples chargés, 298 réponses `ok`, zéro erreur, zéro sortie invalide et 28,439 secondes d'inférence après démarrage du serveur. La distribution de calibration est A=83, B=58, C=4, D=25, E=130 et `non_jugeable`=0. Elle sert au contrôle du contrat et ne constitue pas un résultat E016 sur G7.

La première soumission complète `935563` a échoué avant le chargement du modèle et avant toute réponse sur une erreur matérielle `CUDA uncorrectable ECC error` de `node51`. Le protocole et les artefacts sont restés inchangés. Le même run, resoumis sous `935568` avec `node51` exclu, s'est terminé sur `node52` en 13 min 15 s : 7 487 couples chargés, 7 487 réponses `ok`, zéro erreur et zéro sortie invalide.

## Limites obligatoires

- `eval_rich_retrievable_strict` n'est pas une lockbox inédite ;
- E016 évalue seulement G7 dans ce premier chantier ;
- le contrôle avocat porte sur 100 cas stratifiés et doit être repondéré ;
- G8 et les distances de graphe restent exclus du jugement et seront ajoutés seulement au diagnostic post-E016 ;
- aucune modification ou sélection d'hyperparamètre G7 ne peut être décidée sur les 754 questions à partir de ce score.

## Références d'implémentation

- `docs/superpowers/specs/2026-08-11-g7-graded-jp-evaluation-lawyer-audit-design.md`
- `docs/superpowers/plans/2026-08-11-g7-graded-jp-evaluation.md`
- `05-Technique/benchmark/etape1_embedding_pur/prompts/g7_graded_jp_judge_v1.txt`
- scripts `74` à `80` dans `05-Technique/benchmark/etape1_embedding_pur/scripts/`.
