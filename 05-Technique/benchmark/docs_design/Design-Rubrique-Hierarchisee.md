---
tags: [benchmark, methodologie, design, rubrique, ground-truth]
type: design-document
status: proposition
created: 2026-04-20
modified: 2026-04-20
---

# Design de la rubrique hiérarchisée pour le benchmark

> Formalisation du format d'évaluation décidé le [[2026-04-20]].
> Complète [[Benchmark-KG-Juridique-FR-Design]] (vue d'ensemble) et s'appuie sur [[Format-Fondement-Juridique]] et [[Format-Jurisprudence]] pour les références.

---

## 1. Problème à résoudre

L'évaluation d'un système *génératif* de recherche juridique ne peut pas reposer sur :

- **Similarité sémantique à une "réponse GT texte"** → trop laxiste (deux réponses opposées peuvent être proches)
- **Exact match** → trop strict (reformulations tuent le score)
- **Étiquette qualitative de difficulté** (easy/medium/hard) → subjective, introduit un biais d'annotateur

**Solution retenue** : évaluer la couverture d'une **rubrique hiérarchisée de points attendus**, inspirée de HealthBench (OpenAI 2025) et PLawBench (Shi et al. 2026).

---

## 2. Principe : rubrique en 3 strates

Chaque question du benchmark est associée à une **rubrique** — liste structurée de points de raisonnement attendus dans la réponse. Ces points sont classés en 3 strates avec des poids croissants en qualité discriminante :

| Strate | Poids | Ce qu'elle évalue | Signal discriminant |
|---|---|---|---|
| **`core`** | 3 | Un étudiant L2 doit les citer | "le système répond juste" |
| **`expected`** | 2 | Attendu d'un praticien confirmé | "le système répond bien" |
| **`expert`** | 1 | Points de navigation profonde (sujets liés, exceptions, articulations) | **"le système exploite la structure du KG"** |

### Pourquoi cette structure discrimine RAG vs GraphRAG

- Un **RAG vectoriel** ramène les chunks sémantiquement proches de la question → couvre `core` + quelques `expected`, plafonne vite
- Un **GraphRAG** navigue les relations du KG → peut remonter les points `expert` qui sont à 2-3 sauts (ex : articulation C. civ. / C. santé publique pour le trouble de voisinage)
- La différence de score sur la strate `expert` **est** la mesure de la valeur ajoutée du KG

---

## 3. Schéma Pydantic

```python
from pydantic import BaseModel, Field
from typing import Literal

class RubricPoint(BaseModel):
    """Un point attendu dans la réponse."""
    text: str                      # description textuelle, utilisé par LLM-as-judge
    linked_article: str | None = None    # clé canonique code:article
    linked_jp: str | None = None         # ID de décision (ECLI ou Judilibre)
    # weight hérité de la strate parente

class Rubric(BaseModel):
    """Rubrique hiérarchisée pour une question du benchmark."""
    core: list[RubricPoint] = Field(default_factory=list)
    expected: list[RubricPoint] = Field(default_factory=list)
    expert: list[RubricPoint] = Field(default_factory=list)

    @property
    def total_points(self) -> int:
        return len(self.core) + len(self.expected) + len(self.expert)

    @property
    def total_weight(self) -> float:
        return 3 * len(self.core) + 2 * len(self.expected) + 1 * len(self.expert)
```

---

## 4. Format complet d'un cas de benchmark

```yaml
id: Q_042
branche: civil
specialisation: Droit Civil
question: "Mon voisin fait du bruit à 23h tous les soirs, que puis-je faire ?"
perspective: demande         # demande | defense | neutre
date_posee: 2026-04-20

rubric:
  core:
    - text: "Fondement : trouble anormal de voisinage"
      linked_article: "code_civil:1240"
    - text: "Responsabilité objective : pas besoin de prouver une faute"
      linked_article: "code_civil:1240"

  expected:
    - text: "Critère d'anormalité du trouble (horaire + répétition)"
      linked_jp: "ECLI:FR:CCASS:1998:96.20.991"
    - text: "Procédure : mise en demeure préalable puis tribunal judiciaire"
    - text: "Nécessité de rapporter la preuve (témoignages, constats)"

  expert:
    - text: "Articulation pénale : tapage nocturne art. R.1334-31 CSP"
      linked_article: "code_de_la_sante_publique:R1334-31"
    - text: "Possibilité de cumul action civile + action pénale"
    - text: "Si location : obligation de jouissance paisible du bailleur"
      linked_article: "code_civil:1719"
    - text: "Nuance Cass. civ. 3e 2012 sur les bruits inhérents au voisinage normal"
      linked_jp: "ECLI:FR:CCASS:2012:11.10.861"
    - text: "Exception : antériorité de l'activité (art. L.113-8 C. construction)"

articles_attendus:              # union indexable de rubric.*.linked_article
  obligatoires: ["code_civil:1240"]
  optionnels: ["code_de_la_sante_publique:R1334-31", "code_civil:1719"]

jp_attendues:                   # union indexable de rubric.*.linked_jp
  - ref: "ECLI:FR:CCASS:1998:96.20.991"
    short_ref: "Cass. civ. 2e, 19 mai 1998, n° 96-20.991"
    sens_vs_question: favorable
    importance: 1
  - ref: "ECLI:FR:CCASS:2012:11.10.861"
    short_ref: "Cass. civ. 3e, 14 mars 2012, n° 11-10.861"
    sens_vs_question: nuance
    importance: 2

pieges:                         # optionnel, anti-hallucination
  not_to_cite_articles: ["code_civil:1382"]    # abrogé depuis 2016
  not_to_cite_jp: []
```

> Les listes `articles_attendus` et `jp_attendues` sont **dérivables** des `linked_*` de la rubrique — à générer automatiquement pour éviter les incohérences.

---

## 5. Format de sortie imposé au LLM testé

Le LLM ne répond **pas en texte libre**. Il produit un **JSON structuré** avec le même squelette pour tous les systèmes testés (B1 à C) :

```python
class SystemOutput(BaseModel):
    """Ce que tout système sous test doit produire pour une question."""

    reponse_synthetique: str                # paragraphe de synthèse 2-5 phrases
    points_cles: list[str]                  # bullets des raisonnements structurants
    fondements_textuels: list[ArticleCitation]
    jurisprudence_favorable: list[JPCitation]
    jurisprudence_defavorable: list[JPCitation]
    distinctions_importantes: list[str] = []    # nuances / exceptions
    sujets_connexes: list[str] = []             # ouvertures / branches voisines

class ArticleCitation(BaseModel):
    code: str          # slug canonique (cf. Format-Fondement-Juridique)
    article: str       # forme canonique
    role: str          # "fondement principal" | "fondement alternatif" | ...
    extrait: str | None = None

class JPCitation(BaseModel):
    decision_id: str   # ECLI priorité
    short_ref: str
    apport: str        # en quoi elle sert ici
```

**Avantages** :
- Évaluable point par point
- Comparable entre toutes les architectures
- Directement exploitable en prod (UI avocat)
- Force les systèmes à expliciter ce qu'ils font

---

## 6. Métriques dérivées

### 6.1 Score de rubrique (couverture pondérée)

```
rubric_score = (3·cov_core + 2·cov_expected + 1·cov_expert) / total_weight
```

où `cov_strate = (points couverts) / (points de la strate)`.

### 6.2 Métriques auxiliaires

| Métrique | Formule | Strate clé |
|---|---|---|
| `recall_core` | `|core_couverts| / |core|` | qualité de base |
| `recall_expected` | idem sur `expected` | qualité praticien |
| `recall_expert` | idem sur `expert` | **apport du KG** |
| `precision_articles` | articles cités ∈ articles_attendus / articles cités | anti-bruit |
| `precision_jp` | idem pour JP | anti-bruit |
| `adverse_coverage` | `|JP_défavorables_citées ∩ attendues_defavorables| / |attendues_defavorables|` | test *dossier* |
| `anti_hallucination` | 1 − (sources inventées ou abrogées citées / sources citées) | sûreté |
| `f1_expert_weighted` | F1 sur strate expert, pondéré par importance JP | métrique synthétique |

### 6.3 Évaluation "un point est-il couvert ?"

Méthode : **LLM-as-judge** avec prompt standardisé.

Pour chaque point de la rubrique, le juge répond `COUVERT | PARTIELLEMENT_COUVERT | NON_COUVERT` sur la base de la `SystemOutput` fournie.

Double-judge (cf. pattern Judge-on-Judge de [[Benchmark-KG-Juridique-FR-Design]] §5.5) pour les points de strate `expert` où l'évaluation est la plus subjective.

---

## 7. Implications sur la difficulté (décision 2026-04-20)

**Abandon de l'étiquette qualitative a priori** (`easy/medium/hard` de `schema.py:44`).

À la place, **3 proxies mesurables** extraits de la rubrique :

| Proxy | Formule | Sens |
|---|---|---|
| `|rubric|` | `rubric.total_points` | Complexité globale |
| `nb_sources_attendues` | `|articles_attendus| + |jp_attendues|` | Largeur du retrieval |
| `ratio_expert` | `|expert| / total_points` | Profondeur de navigation KG |

> La difficulté devient **émergente** à l'analyse (bucketing a posteriori) plutôt que subjective à l'annotation.

L'enum `Difficulty` de `schema.py` reste utilisable pour le module M6 (décisions piège) mais n'est plus une dimension obligatoire des questions M1-M5.

---

## 8. Qualité de la rubrique = plafond du benchmark

> [!warning] Risque central
> **La qualité de la rubrique est le plafond absolu du benchmark.**
> - Si un point `expert` pertinent est oublié → système pénalisé en precision pour une bonne réponse non-listée
> - Si un point `core` est en fait marginal → systèmes qui le citent sont surévalués
> - Si la hiérarchie core/expected/expert est mal calibrée → les métriques ne discriminent plus

**Conséquence opérationnelle** : la construction des rubriques est le vrai projet. Elle conditionne toute l'évaluation. → cf. question ouverte [[2026-04-20#5. Questions pour toi avant d'itérer]] sur la constitution de la ground truth.

---

## 9. Connexions

- [[Benchmark-KG-Juridique-FR-Design]] — design global (6 modules, 5 configurations)
- [[Format-Fondement-Juridique]] — format des `linked_article` / `articles_attendus`
- [[Format-Jurisprudence]] — format des `linked_jp` / `jp_attendues`
- [[Format-QCM-benchmark-juridique-FR]] — format alternatif pour Module 4
- [[2026-04-20]] — décision d'introduire la rubrique hiérarchisée
- `05-Technique/benchmark/schema.py` — implémentation Pydantic à étendre
