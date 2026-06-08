---
type: handoff
sujet: M3 LLM-judge sur toutes les méthodes du tableau global
session_estimee: 1 session (setup + lancement) + compute background ~8-12h Gemma cluster
date_creation: 2026-06-08
tasks_liees: [#23]
---

# Handoff — Implémenter M3 (LLM-as-judge) sur l'ensemble des baselines

## Objectif de la session

Implémenter la métrique **M3 LLM-as-judge** et la **calculer sur toutes les méthodes** du tableau global de référence (B2-a, B3-a, B3-e, B4-a/c/d/e/f, PPR row α=0,85/0,95, et plus tard LightGCN). M3 capte la pertinence sémantique des top-K **indépendamment de la GT** — crucial parce que notre GT est sparse (|GT| moy ≈ 1,23 articles) et qu'on perd les vrais positifs absents de la GT.

## Définition retenue (figée Week-9)

Pour chaque triplet (question $q$, article $a_i$ dans le top-$K$ d'une méthode) :
- LLM classifie en triplet **{n2: pertinent, n1: proche, n0: non pertinent}**
- Agrégation par question : $M3(q) = (2 \cdot \#n2 + \#n1) / (2K)$ ∈ [0, 1]
- Agrégation finale : moyenne sur les 971 questions
- Détail conservé : distribution (n2, n1, n0) par méthode pour analyse fine

## Contexte projet — où on en est

- **Cohorte d'évaluation** : 971 questions (970 doctrine_qgen + 1 CRFPA, sur 977 nominal, 6 droppées car embeddings manquants ; 7e CRFPA droppée pour incompatibilité PyTorch).
- **Panel métriques actuel** : M1 (Recall@K), M2 (rang moyen normalisé). Hit@K, MRR@K, NDCG@K en cours d'implémentation (Chantier 1, handoff séparé pas nécessaire — devrait être terminé avant cette session).
### Méthodes à évaluer — liste complète et déduppée

M3 s'évalue **par modalité** (judge d'articles vs judge de JP, prompts distincts). Une méthode qui produit deux rankings (articles ET JP, comme PPR) compte pour deux entrées.

**⊕ Côté ARTICLES** (6 entrées maintenant + 1 future) :

| Priorité | Méthode | K_in | Rationale |
|---|---|---|---|
| P1 | **B3-e** | 10 | champion strict (M1 = 0,711) |
| P1 | **PPR row-norm α=0,95** | — | champion étendu (M1 ext = 0,441) |
| P2 | **B2-a** cosine art. open pool | — | baseline pure cosine |
| P2 | **PPR row-norm α=0,85** | — | variante PPR (champion étendu strict 0,421) |
| P3 | **LLM seul (Gemma)** | — | génère réponses → parser numéros d'articles cités (handling spécial) |
| P3 | **LLM + RAG** (cosine top-K → Gemma) | — | RAG retrieval = B2-a, mais articles cités par la génération (handling spécial) |
| Futur | **LightGCN** articles | — | quand handoff LightGCN abouti |

**⊕ Côté JP** (5 entrées maintenant + 1 future) :

| Priorité | Méthode | K_in | Rationale |
|---|---|---|---|
| P1 | **B4-e RRF** | 20 | champion JP (M1 = 0,416) |
| P1 | **B3-a** cosine JP direct | — | baseline pure cosine JP |
| P2 | **B4-d** intersection | 50 | meilleur K_in B4-d (M1 = 0,348) |
| P2 | **B4-f** citation-weighted | 10 | référence négative (effondrement à M1 = 0,154) — utile pour calibrer M3 sur cas où GT-recall est mauvais |
| P2 | **PPR row-norm α=0,95** côté JP | — | confirmer saturation (Jaccard 1,00 vs cosine) avec M3 |
| Futur | **LightGCN** JP | — | quand handoff LightGCN abouti |

### Méthodes EXPLICITEMENT droppées (ne pas faire tourner M3 dessus)

| Méthode | Raison du drop |
|---|---|
| **B4-a** cross-union articles | $\equiv$ B2-a (re-rank par cosine collapse sur le top cosine initial — vérifié Week-9) |
| **B4-c** cross-union JP | $\equiv$ B3-a (même mécanique) |
| **B3-e** K_in=20, K_in=50 | non-champions, M1 inférieur à K_in=10 |
| **B4-d, B4-e, B4-f** sur K_in autres que champion | non-champions documentés |
| **PPR sym-norm** | collapse à cosine (vérifié Week-9 : 0,187 ≈ B2-a 0,185), redondant |
| **PPR α ∈ {0,1; 0,3; 0,5; 0,7}** | non-champions du fine sweep (cf. `21_ppr_curves.py`) |

### Total compute estimé

- **Côté articles** : 4 méthodes graphe/cosine × 971 q × K=10 = **38 840 (q, art) à juger**
- **Côté JP** : 5 méthodes × 971 q × K=10 = **48 550 (q, jp) à juger**
- **Total déduppé inter-méthodes possible** : beaucoup de paires (q, item) reviennent dans plusieurs méthodes — caching par (q, art_id) et (q, jp_id) réduit nettement. **Estimation après dédup : ~30-40k appels uniques** côté articles, ~35-45k côté JP.
- Avec batch 10 items/appel : ~7-8k appels Gemma effectifs
- Wall-clock cluster ~5s/appel : **~10-12h compute background**

### Handling spécial : LLM seul et LLM + RAG

Pour ces deux méthodes, **pas de ranking top-K classique** : la sortie est une génération textuelle. Pipeline d'extraction nécessaire :

1. **LLM seul** : prompter Gemma de répondre à la question avec « cite les articles applicables (numéro + code) ». Parser la sortie pour extraire la liste ordonnée d'articles cités (regex sur patterns `art\. \d+ du Code [a-z ]+`).
2. **LLM + RAG** : prompter Gemma avec top-10 articles cosine en contexte + même consigne de citation. Parser pareil.

Le « top-K » résultant est l'ordre d'apparition dans la réponse. **À discuter avec Johnny** : faut-il limiter à K=10 ou prendre tout ce que le LLM cite (peut être < 10) ? Recommandation : capper à 10 pour cohérence avec les autres méthodes ; si LLM cite < 10, juger seulement ceux-là (M3 reste défini, juste sur moins d'items).

**Ces deux méthodes peuvent être lancées dans une session séparée** si on veut isoler la complexité parsing — le handoff LLM-seul/RAG pourrait alors être autonome.

## Contexte technique

- **LLM cible** : Gemma 4 26B (déjà utilisé pour `doctrine_qgen` génération de questions)
- **Cluster** : env --user fragile (cf. mémoire `cluster-user-env-fragile.md`). **NE PAS faire de `pip install` non-pinné**. Vérifier l'env existant d'abord.
- **Ressources cluster** : matthieu.kaeppelin@hector.legal — accès via le runner Hector pour Gemma vLLM
- **Pipeline réutilisable** : voir le code de génération `doctrine_qgen` (utilisé pour générer les 1707 questions) — même backbone Gemma cluster, on adapte le prompt et le post-processing
- **Volume** : voir section « Total compute estimé » plus bas — ~30-40k jugements uniques côté articles + ~35-45k côté JP après dédup inter-méthodes, ~7-8k appels Gemma effectifs en batch 10, **~10-12h compute wall-clock**

## Plan de la session

### Étape 1 — Prompt design (30-45 min)
Écrire le prompt LLM-judge. Spécifications :
- **Input** : énoncé question + texte intégral d'un article candidat
- **Output structuré** : label dans {n2, n1, n0} + justification 1 phrase
- **Format** : JSON strict pour parsing (`{"label": "n2", "raison": "..."}`)
- **Calibration** : exemples few-shot (3-5 exemples annotés à la main couvrant les 3 classes)
- **Sortir l'article ENTIER**, pas seulement le numéro — Gemma doit voir le texte

Sauvegarder le prompt dans `05-Technique/benchmark/etape1_embedding_pur/prompts/m3_judge_v1.txt`.

### Étape 2 — Script `23_eval_m3_llm_judge.py` (1-1.5h)
Architecture :
```
Pour chaque méthode m du tableau:
    Pour chaque question q (sur 971):
        ranked = methode_topK[m][q]              # déjà calculé par scripts 18/20
        article_texts = [load_article(a) for a in ranked]
        prompt = format_prompt(q.enonce, article_texts)
        response = gemma_call(prompt)            # batch idéalement 10 art / appel
        labels = parse_json(response)            # [n2, n1, n0, n1, ...]
        record(m, q, labels)
sauvegarder eval_m3.csv
```

Réutiliser le client vLLM/Gemma de `doctrine_qgen` (chemin à retrouver — sans doute `04-Donnees/scripts/` ou `05-Technique/judilibre/`). Idempotence : si une (méthode, qid) existe déjà dans le CSV, skip.

### Étape 3 — Test pilote 20 questions × 2 méthodes (30 min)
Avant de lancer le run complet :
- Pilote sur 20 questions stratifiées (mix singleton/multi-GT) × 2 méthodes (B3-e + PPR α=0,95)
- Valider la qualité des labels par inspection humaine (sur 5-10 cas où LLM dit n2 mais GT dit "hors-GT" → ce sont les vrais positifs récupérés)
- Ajuster prompt si dérive (trop strict / trop laxe)

### Étape 4 — Lancement run complet en background (15 min setup)
- Lancer le run sur cluster Gemma en background (sbatch / nohup selon convention Hector)
- Logger progression → fichier
- **Attention** : commande à fournir dans `.sh`, pas en copier-coller terminal (cf. mémoire `feedback_terminal_copy.md`)

### Étape 5 — Pendant que ça tourne (~10-13h)
La session peut se terminer ici. Le compute tourne seul. Au retour :
- Récolter `eval_m3.csv`
- Calculer agrégats par méthode
- Insérer ligne M3 dans le grand tableau global (Chantier 2)
- Analyser la distribution (n2, n1, n0) par méthode — particulièrement les vrais positifs hors-GT

## Fichiers clés

| Path | Rôle |
|---|---|
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/bench_global.json` | 971 questions (enoncé + GT) |
| `05-Technique/benchmark/etape1_embedding_pur/data/global_bench/eval_m1_m2.csv` | Rankings actuels par méthode (à étendre) |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/18_eval_m1_m2.py` | Référence pour structure du loop éval |
| `05-Technique/benchmark/etape1_embedding_pur/scripts/20_ppr_naive.py` | Référence pour PPR rankings |
| `05-Technique/benchmark/etape1_embedding_pur/etape1/config.py` | Config (paths embeddings, articles, JP) |
| `04-Donnees/.../legi_articles.parquet` (à confirmer) | Texte intégral des articles |
| `01-Projet/journal/2026-06-03.md` | Décisions Week-9 |

**IMPORTANT** : les CSV de rankings actuels (`eval_m1_m2.csv`) stockent les **métriques** mais pas forcément les **rankings**. Il faut peut-être modifier les scripts 18/20 pour sauvegarder un `rankings.parquet` (qid, méthode, k_in, rank, article_id) avant cette session, ou recalculer les rankings dans le script 23.

## Décisions déjà prises (ne pas re-discuter)

1. ✅ Triplet n2/n1/n0 — pas binaire, pas note continue.
2. ✅ Agrégation $M3(q) = (2 \cdot \#n2 + \#n1) / (2K)$ — pondération validée.
3. ✅ Gemma 4 26B comme judge (pas plus gros, pas OpenAI).
4. ✅ Reporter le détail (n2, n1, n0) **en plus** de M3 agrégé.
5. ✅ Appliquer à **toutes les méthodes** du tableau, pas seulement les champions.

## Critères de succès de la session

- [ ] Prompt M3 v1 fixé, sauvegardé, testé sur 20 q pilote
- [ ] Script `23_eval_m3_llm_judge.py` écrit, lancé sur cluster Gemma en background
- [ ] Plan de récolte/agrégation documenté pour la session suivante
- [ ] Pas de pip install non-pinné sur l'env cluster

## Pièges connus

- **Format JSON strict** : Gemma a tendance à ajouter du texte autour. Prévoir extraction regex/json5 robuste avec fallback.
- **Coût per-question** : si batch trop large (K=10 articles dans un seul prompt), Gemma peut tronquer ou mélanger les indices. Tester pilote avant scale.
- **Contexte 32k Gemma 4** : 1 question + 10 articles peut atteindre la limite. Cf. mémoire `doctrine-qgen-4-sections-overflow.md`. Si dépassement → batch par 5 articles.
- **Sanity check final** : sur les questions où GT singleton + méthode score Hit@K=1, le LLM-judge devrait classer en n2 l'article ground-truth. Sinon le prompt est mal calibré.

## Pour relancer cette session

Coller en début de session Claude Code :
```
Reprends le handoff `01-Projet/handoffs/handoff-M3-LLM-judge.md`.
Exécute la session telle que planifiée. Demande validation
avant le lancement du run cluster (étape 4).
```
