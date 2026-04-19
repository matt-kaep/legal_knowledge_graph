---
tags: [article, benchmark, legalbench, raisonnement-juridique]
categorie: "Benchmarks"
titre_complet: "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning in Large Language Models"
auteurs: "Guha et al. (40 co-auteurs)"
annee: 2023
type: "Article de conférence"
venue: "arXiv / Stanford"
url: "https://arxiv.org/abs/2308.11462"
doi: ""
pdf_local: ""
status: "lu"
pertinence: "haute"
created: 2026-04-16
modified: 2026-04-16
---

# LegalBench — Guha et al. 2023

> [!info] Metadonnees
> **Auteurs** : Neel Guha + 39 co-auteurs (dont Christopher Ré, Daniel E. Ho, Adam Chilton, Sharad Goel)
> **Annee** : 2023 | **Venue** : arXiv (soumis 20 août 2023)
> **Institution** : Stanford (lead) + collaboration multi-institutions
> **Licence** : CC BY-4.0
> **URL** : [Lien](https://arxiv.org/abs/2308.11462)

## Resume

Premier benchmark collaboratif de grande ampleur visant à mesurer les capacités de **raisonnement juridique** des LLMs. Construit par un consortium interdisciplinaire de **juristes et de data scientists**, il regroupe **162 tâches** structurées selon le cadre **IRAC** (Issue, Rule, Application, Conclusion) + Interpretation + Rhetoric. 20 LLMs open-source et commerciaux évalués. Résultat : 143 pages, 79 tableaux — devient la référence du domaine.

## Contributions principales

1. Premier benchmark à grande échelle **conçu par des juristes** (pas par des informaticiens seuls)
2. Structuration autour du **raisonnement juridique formel** (IRAC) plutôt que tâches NLP génériques
3. Évaluation empirique massive (20 modèles × 162 tâches)
4. Mise à disposition libre (CC BY-4.0) → standard de facto du domaine

## Methodologie

### Donnees
- **162 tâches** juridiques distinctes
- Construites **manuellement par des professionnels du droit**
- Processus collaboratif interdisciplinaire (juristes + data scientists)

### Taxonomie des taches — le cadre IRAC

Le raisonnement juridique est décomposé en 6 catégories :

| Catégorie | Question testée | Exemple de tâche |
|---|---|---|
| **Issue-spotting** | Quel est le problème juridique ? | Identifier le domaine applicable |
| **Rule-recall** | Quelle règle s'applique ? | Restituer la règle pertinente |
| **Rule-application** | Comment la règle s'applique-t-elle aux faits ? | Qualification des faits |
| **Rule-conclusion** | Quelle est la conclusion juridique ? | Solution du cas |
| **Interpretation** | Que signifie tel terme juridique ? | Compréhension sémantique |
| **Rhetoric** | Comment argumenter ? | Structure argumentative |

> IRAC = modèle dominant d'enseignement du raisonnement juridique en common law américain.

### Modeles evalues
- **20 LLMs** open-source + commerciaux
- Non exhaustivement listés dans l'abstract (détail dans le papier complet)

## Resultats cles

*(Détails dans les 79 tableaux du papier — à lire intégralement pour chiffres précis)*

- Les modèles propriétaires surpassent les open-source en moyenne
- **Rule-recall** (connaissance pure) ≠ **rule-application** (raisonnement)
- Les tâches d'*interprétation* et de *rhétorique* sont plus difficiles que l'*issue-spotting*
- Forte variance entre sous-domaines juridiques

## Points forts

- **Construction par des experts juridiques** → tâches réalistes
- **Couverture taxonomique complète** (IRAC + Interpretation + Rhetoric)
- **Échelle massive** (162 tâches) → statistiquement solide
- **Standardisé** (CC BY-4.0, repo public) → reproductibilité
- Devenu la **référence de facto** en NLP juridique

## Limites

- **100% en droit américain common law** — pas transposable directement en civil law
- Tâches conçues pour le raisonnement anglo-saxon (précédent jurisprudentiel contraignant)
- Pas de couverture des spécificités continentales (hiérarchie des normes, codes)
- Prompts uniquement en anglais
- Métriques essentiellement classification / comparaison texte → peu adapté à des tâches génératives longues

## Liens avec mon projet

> [!important] Pertinence pour le KG juridique francais
> LegalBench est **la référence méthodologique** mais **inutilisable directement**. Le découpage IRAC est **transposable en partie** au syllogisme français (majeure/mineure/conclusion) mais les tâches elles-mêmes reposent sur du common law — construire un équivalent FR est un gap réel. C'est une opportunité de contribution.

### Ce que je peux reutiliser
- **Structure IRAC → syllogisme FR** : mapping
  - Issue → identification du problème juridique
  - Rule → fondement (article + JP)
  - Application → qualification des faits
  - Conclusion → solution
- Le **format collaboratif juristes + data scientists** pour concevoir nos tâches
- La **méthodologie de découpage par catégories de raisonnement**
- Le **volume** (viser 50-100 tâches pour une première version FR, pas 162)

### Ce que je dois adapter
- Tâches spécifiques au droit français civiliste :
  - Identification de l'article de loi applicable (pas de précédent)
  - Application de la hiérarchie des normes
  - Gestion des revirements et abrogations
  - Qualification juridique (distinction de droit français)
- Langue : construire les prompts en FR
- Sources : Judilibre + Légifrance (pas de case law US)
- Ajouter une catégorie **temporalité** (validité dans le temps) que LegalBench n'a pas

> [!warning] Attention à la transposition
> La notion de "rule-recall" n'a pas le même sens en FR qu'en US. En common law, la règle = précédent jurisprudentiel. En droit FR, la règle = texte du code. Transposer naïvement = contre-sens.

## Connexions

### Articles lies
- [[Harvard-LIL-2024-Open-French-Law-RAG]] — benchmark RAG sur droit FR, complémentaire
- [[LexGLUE-Chalkidis-2022]] — benchmark NLP juridique anglais, plus ancien
- [[LEXTREME-Niklaus-2023]] — benchmark multilingue (24 langues, dont FR)
- [[CaseHOLD-Zheng-2021]] — benchmark holding prediction sur case law US

### Concepts lies
- [[IRAC framework]]
- [[Syllogisme juridique francais]]
- [[Common law vs civil law]]
- [[Benchmark juridique]]

### Questions soulevees
- Peut-on construire un "LegalBench-FR" adapté au droit civiliste ?
- Quelle granularité de tâches viser pour un premier benchmark FR ?
- Faut-il partir des tâches de LegalBench et les traduire/adapter, ou en concevoir de zéro ?

## Citations cles

> "We present LegalBench, a collaboratively built legal benchmark consisting of 162 tasks."

> "LegalBench tasks were designed by subject matter experts using the IRAC framework, which popularly represents the process of legal reasoning."

## Notes personnelles

- **La méthodologie IRAC est la vraie contribution** — les 162 tâches sont le *livrable*, la taxonomie est le *cadre conceptuel* réutilisable
- Pour notre projet : **adopter IRAC comme squelette** de notre propre benchmark FR, mais peupler avec des tâches civilistes
- Lien direct avec notre slide 2 (syllogisme juridique) — IRAC est la version US du même raisonnement
- Opportunité claire : **construire "FrenchLegalBench"** serait déjà une contribution publiable en soi
- Les 40 co-auteurs signalent l'importance de la collaboration avec des juristes — ne pas vouloir faire ça tout seul
- Piste concrète : identifier des juristes partenaires (Doctrine ? Dalloz ? Fac de droit ?) pour co-construire les tâches
