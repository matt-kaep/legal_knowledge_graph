# Legal Knowledge Graph — reproduction ECIR 2027

Ce dépôt est le paquet de reproduction du benchmark A3/B1 : code, manifests,
prompts, schémas, tests et exports agrégés légers. Il ne contient pas le
corpus juridique, les embeddings, les graphes binaires, les rankings détaillés
ni les réponses des modèles de langage.

La campagne A3 sépare deux tâches : retrouver des Articles et retrouver des
décisions de jurisprudence. Elle fige 5 578 questions d'entraînement, 754
questions d'évaluation interne et les univers retournables communs de 13 236
Articles et 114 851 décisions. Les graphes gardent davantage de nœuds comme
auxiliaires de propagation, mais ces nœuds ne peuvent jamais être renvoyés.

## Résultats A3/B1 à citer

Les exports actuels, exacts et légers sont :

- [Articles](results/benchmark-a3-b1/main_table_articles.csv) — SHA-256
  `44cb8a4bd81ccc5368cc8f802a242c352076f378100184613a0068159e0b1c9d` ;
- [Jurisprudence](results/benchmark-a3-b1/main_table_jurisprudence.csv) —
  SHA-256 `0ad76a437f903c06acd3d4e8cf824939c0aa2813d158e7d4f41369fbe4c875b0` ;
- [reçu de matérialisation](results/benchmark-a3-b1/main_table_manifest.json)
  — SHA-256 `f7d2b7399a6a81eb4442020da6a27e858be12ab3d8aa95b5b4295c9c7aeb021b`.

Chaque ligne porte la méthode, le graphe, la tâche, les graines, la
configuration gelée et le hash de son CSV source. Le contrat A3 est
`benchmark_freeze_no_eval_overlap_effective_retrieval_a3.json`, SHA-256
`c4dda4279fa33fd15970cf78d10dd22a9456afb6f15d2831e5d8e9f73bbc14b3`.

`Hit@10` est la métrique principale pour les deux tâches :

```
|dedup(top-10) ∩ références attendues| / min(|références attendues|, 10)
```

Pour les Articles, il est numériquement égal à Recall@10 dans ce benchmark,
car aucune question n'a plus de dix références strictes. Ce n'est pas le
diagnostic binaire « au moins une réponse exacte ». NDCG@10 et MRR@10 restent
exportés dans les tableaux.

Les scores sont confirmés sur l'évaluation interne A3 après sélection par
validation croisée sur l'entraînement et gel des configurations. Ils ne sont
pas une lockbox jamais consultée.

## Données, provenance et préflight

Les 30 entrées nécessaires à B1, leur taille, leur SHA-256 et leur rôle sont
dans [le manifeste A3](results/benchmark-a3-b1/data-manifest-a3.json),
SHA-256 `18b486737b6c25abde6a11db5e0df008a61fcabf02c99fa7927e99c13dec5e73`.
Ce manifeste a été régénéré depuis A3/B1 et rehashé ; il ne vaut pas licence de
redistribution. Chaque source doit être obtenue ou reconstruite par une
personne autorisée, à son chemin relatif indiqué, puis vérifiée avant calcul.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r 05-Technique/benchmark/etape1_embedding_pur/requirements.txt
export LKG_REPO="$PWD"
export LKG_DATA_ROOT="/chemin/autorise/vers/les-donnees"
export LKG_PYTHON="$PWD/.venv/bin/python"

# Vérifie les splits, folds, candidats, graphes et hashes A3/B1.
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/94_run_b1_a3_campaign.py \
  --manifest 05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r1.json \
  --stage preflight

# Régénère le manifeste local, sans téléchargement ni modification de données.
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/108_build_a3_data_manifest.py \
  --data-root "$LKG_DATA_ROOT" \
  --output results/benchmark-a3-b1/data-manifest-a3-regenerated.json
```

## Rejouer la campagne B1

Les anciens manifests restent des archives immuables. La campagne courante est
composée de successeurs versionnés afin qu'aucun artefact déjà scellé ne soit
écrasé.

```bash
B1=05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_b1_a3_r1.json

# Baseline cosine/BGE-M3 : Articles et jurisprudence, top-100, 754 questions.
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/94_run_b1_a3_campaign.py \
  --manifest "$B1" --stage cosine

# PPR : soumettre les onze graphes sur CPU ; la sélection se fait uniquement
# dans les cinq folds d'entraînement. Voir le wrapper Slurm pour les ressources.
sbatch 05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_b1_a3_r1_ppr_cv.sh

# LightGCN : CV atomique CUDA, avec minimum 9,1 Gio libres, puis agrégation,
# gel et replay seulement après couverture complète des tâches.
sbatch 05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_b1_a3_r2_lightgcn_cv.sh
```

Le runner LightGCN refuse un agrégat partiel. Les replays finaux utilisent les
configurations gelées et trois graines (`42;43;44`) ; ils ne choisissent ni
epoch, ni paramètre, ni graphe sur l'évaluation. Les commandes exactes de gel,
replay, métriques et courbes sont données par les manifests successeurs
`confirmatory_campaign_b1_a3_paper_ready_g6_replay_v2.json` et
`paper_ready_retrieval_a3_scoped_ppr_g1_lightgcn_g1_r1.json`.

Après un replay hashé, matérialiser les tables légères sans changer de nombre :

```bash
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/109_materialize_a3_b1_paper_tables.py \
  --source-metrics /chemin/vers/ranking_metrics_at_10.csv \
  --out-dir results/benchmark-a3-b1-nouvelle-version
```

## LLM : deux expériences distinctes

- **E029 — reranking comparable.** Le modèle Gemma voit la question et les
  candidats de vrais viviers cosine, PPR ou LightGCN. Il renvoie une permutation
  top-10. La campagne actuelle teste `K_in=10…70`, avec préfixe Article de 192
  tokens Gemma et synthèse JP complète. Les cinq jobs H100 sont soumis ; aucun
  score E029 n'est encore reportable.
- **E030 — LLM-as-a-Judge.** Ce n'est pas un reranker. Après gel des top-10
  E029, le juge évalue séparément chaque fiche anonymisée. Les classes sont A,
  B, C, D, E et `non_jugeable`, avec gains A=1, B=0,5 et 0 sinon, dénominateur
  fixe 10 et gain nul après la première répétition. Ce score sera important
  mais présenté dans un tableau exploratoire distinct tant que l'audit avocat
  n'est pas terminé.

Les coûts dépendent de la file Télécom. À titre d'ordre de grandeur, PPR est
CPU et parallélisable par graphe ; LightGCN est séquentiel par GPU et constitue
le coût principal de CV ; E029 contient 52 780 appels Gemma répartis sur cinq
H100. Les manifests de chaque campagne fixent les ressources, le modèle et les
limites de contexte avant soumission.

## Tests

```bash
"$LKG_PYTHON" -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests -q
```

Les tests unitaires et de contrat fonctionnent sans corpus. Les tests qui
relisent les données exigent `LKG_DATA_ROOT`. La coordination scientifique, les
statuts et les hashes complémentaires sont conservés dans
`01-Projet/paper-control/`; aucun fichier du manuscrit `07-Redaction/` n'est
modifié par ce paquet.
