---
type: handoff
sujet: M3 LLM-judge — récolte des résultats cluster + insertion tableau global
session_estimee: 1 session (~1-2h) APRÈS la fin du run cluster
date_creation: 2026-06-08
tasks_liees: [#23]
parent: handoff-M3-LLM-judge.md
---

# Handoff — Récolte & agrégation M3 (après run cluster)

## Où on en est (session du 2026-06-08)

La **plomberie M3 est écrite, testée en mock, et prête pour le cluster**. Ce qui
est fait :

1. **Prompts figés** (few-shot 3 classes, JSON strict) :
   - `…/etape1_embedding_pur/prompts/m3_judge_art_v1.txt`
   - `…/etape1_embedding_pur/prompts/m3_judge_jp_v1.txt`
   - schéma : `…/etape1_embedding_pur/schemas/m3_judge_format.json`
2. **Rankings dumpés** : scripts 18 et 20 étendus (additif) écrivent
   `data/global_bench/rankings.parquet` — colonnes `qid, method, k_in, modality,
   rank, item_id`. Contient les 6 méthodes de 18 (B2-a, B3-a, B3-e, B4-d, B4-e,
   B4-f) + PPR-row-a0.85 / PPR-row-a0.95 (art+jp) de 20. **Garantit que M3 juge
   exactement les rankings mesurés par M1/M2.**
3. **Script juge** : `scripts/23_eval_m3_llm_judge.py` — appels mono-item
   concurrents (ThreadPool, continuous batching vLLM), cache write-through
   (`m3_judge_cache.csv`, reprise après crash), `response_format` json_schema
   strict + enum + temperature=0, strip HTML des articles. Testé `--mock` :
   dédup 831/1239 sur 20 q, idempotence OK, sanity wiring OK.
4. **Launcher cluster** : `scripts/run_m3_judge_on_cluster.sh {pilot|full}` —
   démarre vLLM Gemma 26B, attend /health, lance le juge, tue vLLM. **PAS de
   pip install** (env --user fragile).

## Décisions / écarts vs handoff d'origine

- **Pas de batch 10-items/prompt** → mono-item concurrent (steer : évite
  overflow contexte 16k vLLM + mélange d'indices ; débit via concurrence).
- **LLM-seul / LLM+RAG (P3) DIFFÉRÉS** — non implémentés. Restent à faire dans
  une session dédiée (parsing des articles cités dans la génération). Ne PAS
  oublier : ce sont 2 entrées du tableau global côté articles.
- **B4-a / B4-c NON dumpées** (≡ B2-a / B3-a, droppées Week-9).
- **REPO surchargeable** par env `LKG_REPO` dans script 23 (portabilité cluster).
- **Dénominateur M3 — À TRANCHER avec Johnny.** Param `--denom`, défaut
  `k_fixed` = `(2·#n2+#n1)/(2K)`, K=10 fixe (déf gelée Week-9). Alternative
  `k_ranking` = 2·(taille réelle du ranking). N'impacte QUE les méthodes
  renvoyant <10 items : **B3-e (≈9,86 items/q)** et **B4-d (≈9,57)** — toutes
  les autres ont 10 items → identiques. Effet mesuré : B3-e passe de 0,496
  (k_ranking) à 0,487 (k_fixed). Switcher = **re-agrégation seule, ZÉRO
  re-jugement** (l'agrégation relit le cache). Trancher avant d'insérer les
  chiffres dans le tableau.
- **Cache clé sur (prompts+modèle)** : `m3_judge_cache_<hash8>.csv`. Éditer un
  prompt ⇒ nouveau cache ⇒ re-jugement propre (pas de contamination par les
  labels de l'ancien prompt). Conséquence : après ajustement prompt au pilote,
  le `full` re-juge tout proprement.
- **Agrégation robuste aux qids dupliqués** : bench_global a ~2 qids en double
  (ranking dupliqué) ; `aggregate` déduplique par item_id.

## À FAIRE cette session (après run cluster)

### 0. Pré-requis : le run cluster a tourné
Le gate validation (étape 4 du handoff parent) doit avoir été franchi et le
`full` lancé sur le cluster. Récupérer `data/global_bench/` du cluster :
- `m3_judge_cache.csv`  (tous les jugements bruts)
- `eval_m3.csv`         (M3(q) par méthode)
- `eval_m3_summary.json` (agrégats + sanity)

### 1. Sanity-check des labels (15 min)
- Lire la section `sanity_gt_singleton` de `eval_m3_summary.json` :
  les items GT-singleton doivent être **majoritairement n2**. Si <70% n2 →
  prompt mal calibré, ré-itérer (v2) avant d'exploiter les chiffres.
- Inspecter à la main 10 cas où une méthode score n2 sur un item **hors-GT** :
  ce sont les vrais positifs récupérés (la valeur ajoutée de M3 sur GT sparse).

### 2. Insertion dans le grand tableau global (Chantier 2)
- Ajouter une colonne/ligne M3 (moyenne + distribution n2/n1/n0) par méthode,
  à côté de M1/M2/Hit@K/MRR/NDCG.
- Format slides : voir `15_format_slides_tables.py`.

### 3. Analyse fine
- Comparer M3 vs M1 par méthode : l'écart M3−M1 mesure les vrais positifs
  hors-GT. Attendu fort sur B2-a (cosine pur, topiquement large) et PPR.
- Cas-type validé en session : `q_presomption_limite_condamnation` (GT=
  `code_civil:9-1`, M1=0 côté articles car top-cosine = articles pénaux voisins,
  mais top-JP cite littéralement l'art. 9-1) → M3 devrait être >0 là où M1=0.

## Fichiers clés
| Path | Rôle |
|---|---|
| `…/scripts/23_eval_m3_llm_judge.py` | le juge |
| `…/scripts/run_m3_judge_on_cluster.sh` | launcher cluster |
| `…/data/global_bench/rankings.parquet` | rankings champions (entrée du juge) |
| `…/data/global_bench/eval_m3*.{csv,json}` | sorties M3 (à produire au cluster) |
| `…/prompts/m3_judge_{art,jp}_v1.txt` | prompts |

## Reprise rapide d'un run interrompu
Le cache est write-through : relancer `run_m3_judge_on_cluster.sh full` reprend
là où ça s'est arrêté (skip les `(qid,item,modality)` déjà jugés).
