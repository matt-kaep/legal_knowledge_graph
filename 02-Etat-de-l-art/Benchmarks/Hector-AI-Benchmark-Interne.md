---
tags: [benchmark, hector, interne, reference, production]
categorie: "Benchmarks"
titre_complet: "Hector AI — Benchmark interne de l'agent jurisprudence-search"
auteurs: "Équipe Hector AI (Matthieu Kaeppelin et coll.)"
annee: 2026
type: "Benchmark interne production"
venue: "Hector AI (privé)"
status: "lu"
pertinence: "haute"
confidentialite: "interne"
chemin_local: "/Users/matthieu.kaeppelin/Documents/5-Pro/Hector AI/Tech/hector-monorepo/apps/backend/services/hector-agent-jurisprudence-search/benchmark/"
created: 2026-04-16
modified: 2026-04-16
---

# Hector AI — Benchmark interne

> [!warning] Confidentiel
> Ressource propriétaire Hector AI. À ne pas publier tel quel dans le mémoire. Les **principes et patterns** peuvent être réutilisés sous forme abstraite, mais pas le code ni les données clients.

## Contexte

Hector AI = outil d'IA pour les avocats (ma propre entreprise). Le service `hector-agent-jurisprudence-search` est l'agent qui recherche de la jurisprudence pertinente pour un dossier client. Il inclut déjà **un benchmark de production** que je peux utiliser comme **baseline avancée** ou **comparaison industrielle** pour mon projet de stage.

## Structure du benchmark (9 fichiers)

| Fichier | Taille | Rôle |
|---|---|---|
| `benchmark_quality.ts` | 68 KB | ~90+ test cases stratifiés par spécialisation |
| `benchmark_quality_test_cases.ts` | 40 KB | Dataset des cases de test |
| `benchmark_relevance.ts` | 45 KB | Pipeline retrieval + scoring Mistral Small |
| `benchmark_e2e_prod.ts` | 41 KB | Bench end-to-end sur l'API prod |
| `benchmark_e2e.ts` | 34 KB | Bench end-to-end sur vrais dossiers |
| `benchmark_e2e_30.ts` | 33 KB | Variante 30 cas |
| `benchmark_models.ts` | 36 KB | Comparaison multi-modèles avec double juge |
| `benchmark_second_judge.ts` | 23 KB | Meta-judge (Sonnet 4.6) |
| `verify_benchmark.ts` | 17 KB | Vérification post-hoc |

## Architecture générale

```
Input : dossier client (question + position + résumé)
     │
     ▼
┌──────────────────────────────────┐
│  GrokLegalPlanner                │  ← Mistral Large génère 5 arguments
│  (planification des arguments)   │
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  Pour chaque argument :          │
│  - KeywordGenerator              │  ← génère mots-clés
│  - MassSearchExecutor            │  ← recherche Judilibre parallèle
│  - IntersectionScorer            │  ← scoring
│  - CompositeEnricher             │  ← enrichissement des décisions
│  - BalancedSelector              │  ← sélection des top-K
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  Pour chaque décision :          │
│  3 modèles analysent EN PARALLÈLE│
│  - Mistral Small 4               │
│  - Grok 4.1 Fast NR              │
│  - Grok 4.1 Fast Reasoning       │
│                                  │
│  Output par modèle :             │
│  {                               │
│    client_in_decision,           │
│    sens_arret,                   │
│    is_favorable: boolean,        │
│    dispositif_summary,           │
│    relevant, relevance: 0-1,     │
│    reasoning,                    │
│    principles: [{title, content, │
│                  citation}]      │
│  }                               │
└──────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────┐
│  Judge 1 : Grok 4.1 Reasoning    │  ← verdict: dispositif_ok,
│                                  │    favorable_ok, principles_ok
│  Judge 2 : Claude Sonnet 4.6     │  ← meta-judge, vérifie Judge 1
│                                  │    en relisant la décision
└──────────────────────────────────┘
     │
     ▼
   Rapport JSON
```

## Test cases — 5 spécialisations × ~20 questions = 90+

| Spécialisation | Exemples de questions |
|---|---|
| **Droit Pénal** | Légitime défense, harcèlement moral pénal, abus de confiance, complicité, sursis, escroquerie, contrainte, vol avec effraction, récidive, recel, garde à vue, mise en danger… |
| **Droit Social** | Licenciement faute grave, économique, CDD→CDI, harcèlement, rupture conventionnelle, heures supp, inaptitude, salarié protégé, résiliation judiciaire, non-concurrence, travail dissimulé, grève, co-emploi… |
| **Droit de la Famille** | Prestation compensatoire, résidence enfant, droit de visite, pension alimentaire, partage communauté, adoption simple, régime matrimonial, autorité parentale, faute divorce, obligation alimentaire… |
| **Droit Commercial** | Rupture brutale, déséquilibre significatif, concurrence déloyale, vices cachés, retard paiement, dirigeant, cautionnement, cession fonds, franchiseur, agent commercial… |
| **Droit Civil - Baux** | Congé reprise, décence, clause résolutoire, préemption, éviction commerciale, trouble de jouissance… |

Chaque test case :

```typescript
interface TestCase {
  question: string;              // formulée comme un avocat
  specialisation: string;
  clientPosition: string;        // "défense" ou "demande" avec détails
  caseSummary: string;           // résumé du dossier
  procedureType?: string;        // "Procédure au fond", etc.
}
```

## Métriques utilisées

### Métriques principales (sur l'analyse)
- **dispositif_ok** : le modèle a-t-il bien compris qui gagne/perd ?
- **favorable_ok** : le modèle a-t-il bien identifié si c'est favorable au client ?
- **principles_ok** : les principes juridiques extraits sont-ils corrects ?
- **relevance (0-1)** : score de pertinence de la décision

### Seuils de pertinence testés
0.30, 0.40, 0.50, 0.60 — impact sur precision/recall

### Double évaluation
- **Judge 1 (Grok Reasoning)** : verdict automatique
- **Judge 2 / Meta-judge (Sonnet 4.6)** : vérifie Judge 1 en relisant la décision

> C'est la méthodologie la plus rigoureuse que j'aie vue — **Judge-on-Judge** élimine le biais du juge unique.

## Prompts par juridiction

Le système utilise **4 prompts différents** selon la juridiction :
- `promptCassation.ts` — arrêts de Cassation
- `promptCourAppel.ts` — arrêts de Cour d'appel
- `promptTribunal.ts` — jugements de Tribunal judiciaire
- `promptJuridictionsFond.ts` — autres juridictions du fond

> Pattern intéressant : adapter le prompt au **type de document juridique**, pas un prompt générique.

## Modèles testés

| Modèle | Usage |
|---|---|
| Mistral Small 4 (2603) | analyse rapide |
| Grok 4.1 Fast Non-Reasoning | analyse rapide |
| Grok 4.1 Fast Reasoning | analyse avec raisonnement |
| Grok 4.1 Reasoning | **Judge 1** |
| Claude Sonnet 4.6 | **Meta-judge** |
| Mistral Large | **Planner** (génération arguments) |

## Parallélisation

- 7 dossiers en parallèle pour le planner
- 5 arguments en parallèle pour le search
- 3 modèles en parallèle par décision
- 2 judges en parallèle par analyse

> Gros investissement sur la performance — à noter pour le design de notre benchmark académique.

## Liens avec mon projet stage

> [!important] Utilisation possible dans le stage
> Le benchmark Hector est un **atout majeur**, mais avec des contraintes :
> 1. **Code propriétaire** → ne pas citer le code, seulement les patterns abstraits
> 2. **Données clients confidentielles** → ne pas utiliser les vrais dossiers
> 3. **Questions de benchmark** → réutilisables (elles ne dépendent pas des clients)

### Ce qu'on peut tirer pour le stage

1. **Les 90+ test questions** (synthétiques, pas clients) = réutilisables
   - Stratifiées sur 5 spécialisations → cohérent avec les 9 codes Les-Audits
   - Formulées par des avocats → réalistes
2. **La méthodologie Judge-on-Judge** (Grok → Sonnet meta) = à reprendre dans notre benchmark
3. **L'approche "favorable/défavorable client"** = cas d'usage métier réaliste à benchmarker
4. **Les prompts par juridiction** = pattern à reprendre pour notre pipeline GraphRAG
5. **Les seuils de pertinence testés** (0.30-0.60) = ordre de grandeur pour notre RAG

### Ce qu'on apporte de nouveau dans le stage

1. **Graphe structuré** (Hector utilise du retrieval vectoriel + keyword → notre GraphRAG peut battre cette baseline)
2. **Métriques de traçabilité** explicites (citation precision/recall)
3. **Sensibilité et reproductibilité** (Hector ne les mesure pas)
4. **Temporalité** (Hector ne gère pas le versioning)
5. **Ouverture** (Hector est fermé, nous publions)

### Positionnement dans les 5 configurations
| Baseline | Correspond à |
|---|---|
| B1 (LLM seul) | — |
| B2 (LLM + RAG vectoriel) | — |
| B3 (LLM multi-step) | — |
| **B4 (LLM + RAG + agent)** | **≈ pipeline Hector actuel** |
| 🎯 Cible (LLM + GraphRAG) | **notre contribution** |

> Notre cible doit **battre** le pipeline Hector (qui est déjà à l'état de l'art de l'industrie) pour prouver la valeur du graphe.

## Précautions éthiques et légales

- [ ] Confirmer avec Hector AI que je peux citer le **pattern général** dans mon mémoire
- [ ] **Ne pas utiliser** de code propriétaire dans le rendu
- [ ] **Anonymiser** tout usage (pas de nom "Hector" dans le mémoire sauf accord)
- [ ] Clarifier la **propriété intellectuelle** de mon travail de stage vs Hector
- [ ] **Ne pas utiliser** les vrais dossiers clients (que des cases synthétiques)

## Connexions

- [[Alhajar-2025-Les-Audits-Affaires]] — benchmark public, peut se comparer
- [[Butler-Butler-2026-Legal-RAG-Bench]] — taxonomie d'erreurs à adopter
- [[Louis-2022-BSARD]] — retrieval FR
- [[Benchmark-KG-Juridique-FR-Design]] — note de conception à mettre à jour

## Notes personnelles

- **Trouvaille stratégique** : j'ai déjà **en interne** l'équivalent d'un benchmark académique solide
- Le pipeline Hector est probablement **supérieur** à la plupart des papiers RAG juridiques publics
- Le **Judge-on-Judge** (Grok + Sonnet) est une idée originale à revendiquer
- Notre stage doit **apporter quelque chose de plus** que Hector — d'où l'importance du **graphe structuré**, de la **temporalité**, et de la **traçabilité** explicite
- **Question ouverte** : peut-on envisager un papier co-signé Hector AI + labo de recherche ?
