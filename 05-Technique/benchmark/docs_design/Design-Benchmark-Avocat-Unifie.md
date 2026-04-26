---
tags: [benchmark, design, mode-avocat, unifie, scoring, baselines]
type: design-document
status: brouillon
created: 2026-04-21
modified: 2026-04-21
---

# Benchmark "mode avocat" unifié — design

> [!info] Origine
> Spec issue du point superviseur avec Jhony, soir du 21 avril 2026.
> Consolide et remplace la coexistence de [[Design-Rubrique-Hierarchisee]] (CRFPA)
> et [[Design-Benchmark-Rapprochements]] en un benchmark unique aligné sur le
> besoin métier.

---

## 1. Vision

Reproduire la tâche réelle d'un avocat qui prépare un dossier :

```
Question juridique  ──►  système à évaluer  ──►  { articles,
                                                    JP avec sens favorable/défavorable,
                                                    arguments }
```

Le benchmark mesure la capacité du système à fournir **ces trois sorties conjointement**, en une seule passe.

---

## 2. Format de l'input

**Question juridique en langage avocat** — niveau de complexité variable :

- **Niveau simple** (M1) : "Quelles sont les conditions de la rupture brutale d'une relation commerciale établie ?"
- **Niveau appliqué** (M2) : question + résumé du dossier + position du client

Champs optionnels :
- `case_summary` (le cas)
- `client_position` (demandeur / défendeur)
- `specialisation` (droit commercial, social, pénal, etc.)

---

## 3. Format de la sortie attendue

```json
{
  "articles": [
    {
      "pair_key": "code_civil:1240",
      "code_slug": "code_civil",
      "article_num": "1240",
      "pertinence": "haute|moyenne|basse",
      "role": "fondement|exception|procedure"
    }
  ],
  "jurisprudences": [
    {
      "pourvoi": "15-13263",
      "ecli": "ECLI:FR:CCASS:2016:C100045",
      "chamber": "Civ. 1re",
      "date": "2016-01-14",
      "sens_vs_question": "favorable|defavorable|nuance|neutre",
      "importance": 1|2|3,
      "role": "principe|interpretation|espece"
    }
  ],
  "arguments": [
    {
      "titre": "Le trouble anormal de voisinage suppose une atteinte répétée",
      "fondements": ["code_civil:1240"],
      "jp_appuyees": ["15-13263"],
      "points_clefs": ["caractère répété", "intensité mesurable", "absence de justification"]
    }
  ]
}
```

Conforme à [[Format-Fondement-Juridique]] et [[Format-Jurisprudence]].

---

## 4. Ground truth (sourcing)

Le benchmark "mode avocat" se construit par **fusion** de sources qui existent déjà :

| Source | Ce qu'elle apporte | Statut |
|---|---|---|
| Sujets + grilles CNB (CRFPA) | Gabarit de question + arguments + articles + JP attendus | Disponible (11 matières × 2 ans) |
| Copies IEJ Strasbourg | Validation des articles + JP effectivement utilisés par les top candidats | Disponible |
| Rapprochements Cass (Judilibre) | Ground truth sur les JP *liées* — enrichit la GT JP | Disponible |
| Cases Hector synthétiques | Cas réalistes avec annotation 7 dimensions (favorable/défavorable déjà annoté) | M6 MVP (5 cas) |
| Doctrine (Actu-Juridique, Village-Justice) | Extension thématique | Phase 2 |

### Enrichissement pour le champ `sens_vs_question`

Trois sources complémentaires :
1. **Annotation manuelle** sur les cas CRFPA (copie IEJ = position prise par le candidat)
2. **Annotation automatique par LLM** validée par échantillon humain
3. **Pattern Hector** (7 dimensions dont `is_favorable`) sur les cas M6

---

## 5. Baselines à évaluer

| # | Config | Pipeline |
|---|---|---|
| **B1** | LLM seul | Question → LLM → JSON (zero-shot) |
| **B2** | LLM + RAG | Retrieve top-K articles Légifrance + top-K JP Judilibre (embedding) → LLM augmenté |
| **C1** | LLM + **GraphRAG** | Retrieve + traversée du graphe bipartite (Phase B) → LLM augmenté |
| **C2** | **GNN + BERT** | BERT-encode la question + Graph Neural Network sur le graphe bipartite → ranking |
| **C3** | Retrieval pur (embedding) | Embedding question vs embeddings articles/JP, pas de LLM → liste brute |

### Modèles candidats

- **LLM** : Gemma 3, Mistral-7B, Saul-Instruct, GPT-5, Claude Opus
- **Embedder** : `multilingual-e5-large`, `bge-m3`, éventuellement `text-embedding-3-large`
- **GNN** : GraphSAGE ou GAT sur le graphe bipartite Phase B
- **BERT FR juridique** : Legal-CamemBERT ou JuriBERT

---

## 6. Scoring (brouillon à affiner)

### 6.1 Principe général

Le score combine trois composantes avec pénalité pour hallucination :

```
Score_total = w_art · S_articles + w_jp · S_jurisprudences + w_arg · S_arguments
```

Poids par défaut (à calibrer) : `w_art = 0.35`, `w_jp = 0.40`, `w_arg = 0.25`.

### 6.2 S_articles — F1 pénalisé

```
TP = |articles_pred ∩ articles_GT|
FP = |articles_pred \ articles_GT|       (hallucinations / excédent)
FN = |articles_GT \ articles_pred|       (oublis)

Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
S_articles = F1 = 2·P·R / (P + R)
```

**Pénalité dépassement** : une variante stricte `F_β<1` sous-pondère le rappel pour punir plus les hallucinations que les oublis.

### 6.3 S_jurisprudences — F1 pondéré par le sens

```
Pour chaque JP :
  match =  1.0  si prédite + dans la GT + sens correct
           0.5  si prédite + dans la GT + sens incorrect  (malus inversion)
           0.0  si prédite + pas dans la GT               (hallucination)

S_jp = Σ match(jp) / |GT_jp|  (recall pondéré)
       * precision_ensembliste sur pourvois

Avec pondération par importance (1/2/3) : JP de principe pèsent plus.
```

### 6.4 S_arguments — similarité sémantique sur points-clés

Pour chaque argument prédit, on calcule :

```
sim(arg_pred, arg_GT) = cosine(embed(arg_pred), embed(arg_GT))
```

Matching bipartite par similarité maximale, puis moyenne sur les arguments attendus. Pénalité si plus d'arguments produits que dans la GT (dilution).

### 6.5 Variantes à tester

- **V1 — strict** : F_0.5 sur articles (punit hallucinations), match exact pour JP, similarité ≥ 0.8 pour arguments.
- **V2 — souple** : F1 classique, demi-points sur sens JP, similarité ≥ 0.6.
- **V3 — pondéré par importance** : JP × importance, articles × fréquence-hub.

Calibration à faire sur **5-10 questions annotées manuellement** avant de figer.

---

## 7. Métriques de graphe à maîtriser

Jhony insiste sur la capacité à **expliquer chaque graphe** en 30 secondes à l'oral. Pour chaque graphe produit (Phase A × 3 périmètres + Phase B × 3 périmètres = 6 graphes), tenir une fiche avec :

### 7.1 Métriques structurelles

| Métrique | Définition | Ce que ça dit |
|---|---|---|
| `|V|`, `|E|` | Nœuds, arêtes | Taille |
| Densité | $\frac{2|E|}{|V|(|V|-1)}$ (dirigé : $\frac{|E|}{|V|(|V|-1)}$) | "Sparsité" |
| Composantes connexes (non dirigé) | Nombre de sous-graphes disjoints | Fragmentation |
| Plus grosse CC | `max(|V_c|)` | Cohésion effective |
| Nœuds isolés | degré 0 | "Orphelins" |
| Degré moyen, max | distribution | Super-hubs ou pas |

### 7.2 Centralités

| Métrique | Ce que ça dit |
|---|---|
| Degré entrant | "Le plus cité" |
| Degré sortant | "Le plus citeur" |
| PageRank (non dirigé) | Arrêts-pivots de la lignée |
| Betweenness | Nœuds-ponts entre clusters |

### 7.3 Liens apparents vs effectifs

À documenter pour chaque graphe :
- **Arêtes réelles** (explicites dans la donnée) : `rapproche`, `cite`.
- **Arêtes induites** (projections) : `co_cite` articles (deux articles cités ensemble), `co_rapp` JP (deux JP citées ensemble). À calculer optionnellement.

### 7.4 Fiche-type par graphe

Chaque graphe livre une fiche avec 3 blocs :
1. **Conception** (2 lignes) : nœuds + arêtes + périmètre
2. **Chiffres** (tableau) : métriques ci-dessus
3. **Interprétation** (2 lignes) : ce qu'on voit structurellement

---

## 8. Articulation avec ce qui existe déjà

| Livrable Week 2 | Utilisation dans le benchmark unifié |
|---|---|
| 38 Q CRFPA (M1') | Socle initial de questions "mode avocat" (après reformat) |
| 1 532 Q Rapprochements (M3) | Source d'enrichissement de la GT sur les JP liées |
| Graphes Phase A | Support pour la baseline C1 (GraphRAG) — côté JP |
| Graphes Phase B | Support pour C1 et C2 — ajoute les articles |
| Format canonique articles/JP | Directement réutilisable pour l'output JSON |

---

## 9. Plan d'attaque (brouillon)

### Étape 1 — Spec définitive du benchmark (semaine 3)
- Reformulation des 38 Q CRFPA en format "mode avocat" unifié
- Ajout du champ `sens_vs_question` sur toutes les JP (annotation manuelle + LLM)
- Validation de 10 cas pilotes à la main

### Étape 2 — Scoring (semaine 3)
- Implémenter les 3 variantes V1/V2/V3
- Calibrer sur les 10 cas pilotes
- Fixer les poids `w_art`, `w_jp`, `w_arg`

### Étape 3 — Harness et baselines (semaines 3-4)
- Harness d'évaluation commun (entrée standardisée, sortie JSON, scoring unifié)
- Implémenter B1, B2, C1 d'abord
- C2 (GNN + BERT) ensuite
- C3 (retrieval pur) en dernier

### Étape 4 — Fiches de graphes (semaine 3)
- 6 fiches (A × 3 + B × 3) avec métriques et interprétation
- Exécution d'un script dédié qui calcule en un bloc toutes les métriques manquantes (betweenness, etc.)

### Jalon 16 mai
- Premier scoreboard B1 vs B2 vs C1 (minimum) sur le benchmark unifié

---

## 10. Questions ouvertes

- Annotation `sens_vs_question` : entièrement humaine, LLM + validation, ou les deux ?
- Calibration des poids `w_art`, `w_jp`, `w_arg` : arbitraire ou apprise par régression sur scores humains ?
- Taille cible du benchmark unifié : 38 Q (MVP) → 100 Q (CRFPA étendu 2022-2024) → 300 Q (+ doctrine) ?
- C2 (GNN + BERT) : entraîné comment ? Quel gold pour la supervision ?

---

## Connexions

- [[2026-04-21]] — journal du jour (point Jhony intégré)
- [[Design-Rubrique-Hierarchisee]] — socle pour la partie rubrique / arguments
- [[Design-Benchmark-Rapprochements]] — socle pour l'enrichissement JP
- [[Design-Graphes-Phase-AB]] — infrastructure pour C1, C2
- [[Format-Fondement-Juridique]] — format articles
- [[Format-Jurisprudence]] — format JP avec `sens_vs_question`
- [[Benchmark-KG-Juridique-FR-Design]] — design global (à mettre à jour)
