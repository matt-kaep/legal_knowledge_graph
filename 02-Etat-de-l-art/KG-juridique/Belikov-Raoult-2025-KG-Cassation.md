---
tags: [article, kg-juridique]
categorie: "KG-juridique"
titre_complet: "Knowledge Graphs Construction from Criminal Court Appeals: Insights from the French Cassation Court"
auteurs: "Belikov & Raoult"
annee: 2025
type: "Article (arXiv)"
venue: "arXiv"
url: "arxiv.org/abs/2501.14579"
doi: ""
pdf_local: ""
status: "lu"
pertinence: "haute"
created: 2026-04-09
modified: 2026-04-09
---

# Knowledge Graphs Construction from Criminal Court Appeals: Insights from the French Cassation Court

> [!info] Metadonnees
> **Auteurs** : Belikov & Raoult
> **Annee** : 2025 | **Venue** : arXiv
> **Type** : Article (arXiv)
> **URL** : [Lien](arxiv.org/abs/2501.14579)

## Resume

Exploration de l'utilisation de l'IA pour analyser les decisions judiciaires en droit penal, en se concentrant sur sa capacite a completer et surpasser les methodologies traditionnelles en offrant une perspective big data sur les pratiques judiciaires. Construction de KG a partir de 2820 pourvois penaux francais via Judilibre.

## Contributions principales

1. Proposition d'une ontologie penale sur-mesure, adaptee au droit penal francais
2. Construction de KG a partir de decisions reelles de la Cour de cassation (Judilibre)
3. Comparaison property graph vs RDF triples

## Methodologie

### Donnees
- Source : Judilibre (API Cour de cassation)
- Volume : 2820 pourvois penaux
- Langue : Francais

### Architecture / Pipeline

Construction de KG avec ontologie penale custom. Comparaison de deux representations : property graph et RDF triples.

## Resultats cles

| Metrique | Valeur | Baseline |
|---|---|---|
| Precision | 93% | - |
| Rappel | 89% | - |

> [!warning] Metriques peu documentees
> Le papier manque de details sur les metriques d'accuracy. Les chiffres 93%/89% sont mentionnes mais leur methodologie d'evaluation n'est pas suffisamment detaillee.

## Points forts

- Seul papier travaillant directement sur la jurisprudence penale francaise (Cour de cassation)
- Utilise Judilibre comme source de donnees — directement applicable a notre projet
- Exemple concret de representation : "Person A convicted for Crime B with Punishment C" avec liens vers statuts et precedents

## Limites

- Papier peu documente, manque de details sur les metriques
- Ontologie limitee au droit penal (pas generalisable directement)
- Pas de code ou dataset publie

## Liens avec mon projet

> [!important] Pertinence pour le KG juridique francais
> C'est le papier le plus directement comparable a notre projet : meme source de donnees (Judilibre), meme juridiction (Cour de cassation), meme approche (construction de KG). Cependant, il se limite au droit penal.

### Ce que je peux reutiliser
- L'approche generale : partir de Judilibre pour construire un KG
- L'idee d'une ontologie taillee sur-mesure plutot que generique
- La representation des relations (personne → infraction → peine → texte)

### Ce que je dois adapter
- Elargir au-dela du penal (tous domaines juridiques)
- Construire une ontologie plus riche / plus documentee
- Mieux formaliser les metriques d'evaluation

## Connexions

### Articles lies
- [[LegalRuleML-OASIS]] — cite comme ontologie de base mais jugee insuffisante pour le penal
- [[LKIF-Core-Legal-Knowledge-Interchange]] — cite comme fondation mais manque de granularite pour le droit penal
- [[API-Judilibre-Cour-Cassation]] — source de donnees utilisee

### Concepts lies
- [[Ontologie penale]]
- [[Property graph vs RDF]]

### Questions soulevees
- [[Faut-il creer sa propre ontologie ou adapter l'existant ?]]

## Citations cles

> "KGs can efficiently represent relationships like 'Person A convicted for Crime B with Punishment C,' while simultaneously linking the case to relevant statutes and precedent cases."

> "Ontologies like LegalRuleML and LKIF have provided foundational structures for legal knowledge representation, but they often lack the granularity or domain specificity required for criminal law. This paper builds upon these efforts by proposing a tailored criminal ontology that integrates elements of existing frameworks while addressing the nuances of criminal law."

## Notes personnelles

- Papier peu documente dans l'ensemble, manque de rigueur sur les metriques
- Conclusion importante : **il vaut mieux construire sa propre ontologie** adaptee au domaine plutot que de reutiliser directement les ontologies generiques (LegalRuleML, LKIF). Celles-ci manquent de granularite et de specificite pour un domaine juridique particulier.
- Malgre ses limites, c'est une reference incontournable car c'est le seul travail sur un KG de la Cour de cassation francaise.
- A garder en tete pour la Phase 1 : on pourra s'inspirer de leur approche tout en etant plus rigoureux sur l'evaluation.
