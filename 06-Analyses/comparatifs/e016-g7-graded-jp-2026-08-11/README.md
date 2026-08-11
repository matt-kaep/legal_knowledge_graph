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
