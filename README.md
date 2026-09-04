# Legal Knowledge Graph — package de reproductibilité

Ce dépôt contient le code, les configurations, les prompts, les tests, les
manifests et les exports agrégés permettant de reconstruire et de vérifier le
benchmark de recherche. Il ne contient ni le corpus juridique, ni les
embeddings, ni les classements bruts, ni les réponses de modèles de langage.

Le protocole compare onze graphes sur deux tâches séparées : recherche
d'articles et recherche de jurisprudence. Les paramètres sont choisis avec les
cinq folds groupés de l'ensemble d'entraînement ; l'évaluation interne contient
754 questions. Les métriques des articles sont Recall@10, NDCG@10 et MRR@10.
Celles de la jurisprudence sont Hit@10, NDCG@10 et MRR@10. `Hit@10` est la
métrique officielle de couverture ; ce n'est pas l'indicateur de diagnostic
`exact_any_gold_at_10`.

## Installation et données

Python 3.10 ou plus récent est requis (Python 3.12 est la version recommandée).
Créez un environnement, puis installez les dépendances :

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r 05-Technique/benchmark/etape1_embedding_pur/requirements.txt
export LKG_REPO="$PWD"
export LKG_DATA_ROOT="/chemin/vers/donnees_lkg"
export LKG_PYTHON="$PWD/.venv/bin/python"
```

Les fichiers nécessaires et leurs sommes SHA-256 sont décrits dans
[`results/benchmark-repro-v1/data-manifest.json`](results/benchmark-repro-v1/data-manifest.json).
Placez-les sous `LKG_DATA_ROOT` en conservant leur arborescence relative. Le
préflight vérifie leur présence et leurs hashes avant tout calcul. Les données
ne sont pas versionnées ici ; leur acquisition et leur éventuelle diffusion
doivent être traitées séparément.

## Deux modes d’exécution

### Vérifier les résultats déjà produits

Cette commande n'entraîne rien et ne modifie pas les sorties historiques. Elle
contrôle les manifests, les hashes, la couverture et régénère les exports
agrégés dans un dossier de sortie choisi :

```bash
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/90_audit_and_export_reproducibility.py \
  --manifest 05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json \
  --output-dir reproducibility-artifacts/audit
```

Pour l'audit PPR historique, qui est lui aussi strictement en lecture seule :

```bash
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/71_audit_ppr_final_recovery.py \
  --campaign-manifest 05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1_cluster_node_runtime.json \
  --recovery-manifest experiments/confirmatory-recovery/manifest_ppr_final_audit_v2.json \
  --data-root "$LKG_DATA_ROOT" \
  --output "$LKG_DATA_ROOT/05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/_protocol/ppr_final_audit_v2/audit.json"
```

### Rejouer le benchmark

L'orchestrateur applique successivement : préflight, baseline cosine/BGE-M3,
PPR, sélection LightGCN sur les folds d'entraînement, gel des epochs, replay
final, métriques et exports. Le replay interne ne doit être lancé qu'après le
gel explicite des paramètres :

```bash
MANIFEST=05-Technique/benchmark/etape1_embedding_pur/configs/confirmatory_campaign_grouped_v2_repro_v1.json
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage preflight
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage cosine-control-cv
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage ppr-cv
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage lightgcn-screen
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage lightgcn-shortlist
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage lightgcn-tune
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage lightgcn-seeds
"$LKG_PYTHON" 05-Technique/benchmark/etape1_embedding_pur/scripts/64_run_confirmatory_campaign.py --manifest "$MANIFEST" --stage freeze-epochs
```

Les étapes `internal-replay` et `paper-exports` sont ensuite disponibles avec
le même manifeste. Utilisez `--resume` seulement pour reprendre une étape
interrompue et jamais pour remplacer un artefact scellé.

## Cluster Télécom

Le lanceur ne choisit pas un compte, une machine ni une branche cachés : ils
sont tous des variables explicites. Il synchronise la branche locale courante,
soumet un audit PPR en lecture seule ou la reprise hashée du reranking, puis
permet de consulter la file.

```bash
export REMOTE_HOST="utilisateur@gpu-gw.enst.fr"
export REMOTE_REPO="legal_knowledge_graph_repro"
export REMOTE_BRANCH="$(git branch --show-current)"
05-Technique/benchmark/etape1_embedding_pur/scripts/run_telecom_reproducibility.sh sync-code
05-Technique/benchmark/etape1_embedding_pur/scripts/run_telecom_reproducibility.sh submit-ppr-audit
05-Technique/benchmark/etape1_embedding_pur/scripts/run_telecom_reproducibility.sh submit-e021-resume
05-Technique/benchmark/etape1_embedding_pur/scripts/run_telecom_reproducibility.sh queue
```

L'audit PPR demande 2 CPU, 12 Go RAM et 30 minutes au maximum. La reprise du
reranking utilise une L40S, 8 CPU, 48 Go RAM et jusqu'à 6 heures ; elle ne
calcule que les unités manquantes ou invalides et vérifie le hash de chaque
entrée avant agrégation.

## Résultats versionnés

- PPR : les tableaux Articles et Jurisprudence sont dans
  `results/benchmark-repro-v1/ppr_final_table_articles.csv` et
  `results/benchmark-repro-v1/ppr_final_table_jp.csv`. L'audit E022 est
  complet sur les onze graphes et 754 questions. Dans l'export PPR, le libellé
  `audite` décrit la matérialisation hashée ; le statut scientifique officiel
  `confirmee_interne` est enregistré dans
  `01-Projet/paper-control/REGISTRE-RESULTATS.csv`.
- LightGCN E017 : les résultats exacts Articles et Jurisprudence sont dans
  `results/benchmark-repro-v1/internal_eval_articles.csv` et
  `results/benchmark-repro-v1/internal_eval_jp_exact.csv`. Ils agrègent
  33 replays (onze graphes × trois seeds).
- Évaluation graduée par juge LLM :
  `results/benchmark-repro-v1/internal_eval_jp_llm_as_a_judge.csv`. Ses gains
  sont A=1, B=0,5 et 0 sinon, avec dénominateur fixe 10 et gain nul pour une
  répétition. Elle est distincte des métriques exactes.
- Reranking comparatif E021 :
  `results/reranking-comparable/E021-cluster-gpu-runtime-v5-resume-v3/table_jp_reranking_exact.csv`.
  Il couvre 754 questions pour chacun des trois viviers (cosine/BGE-M3, PPR et
  LightGCN), avec `K_in=20` et `K_out=10`.

Les résultats PPR exacts peuvent être utilisés comme résultats de l'évaluation
interne selon le protocole. L'évaluation par juge LLM reste exploratoire tant
que l'accord de l'audit avocat n'est pas produit. Le reranking est un tableau
supplémentaire exploratoire : il ne fonde pas une conclusion principale.

## Tests

```bash
"$LKG_PYTHON" -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests -q
```

Dans un worktree sans les données, les tests qui exigent explicitement les
matrices, graphes et splits signalent leur absence. Les tests unitaires et les
tests de contrat peuvent toutefois être exécutés sans corpus.

Les registres et la transmission scientifique sont conservés sous
`01-Projet/paper-control/`. Aucun fichier du manuscrit (`07-Redaction/`) ne
fait partie de cette branche de publication.
