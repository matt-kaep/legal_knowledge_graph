---
tags: [moc, home]
created: 2026-04-09
---

# Knowledge Graph Juridique — Projet de Recherche

> Objectif : Construire un knowledge graph de la jurisprudence francaise permettant a un agent IA de naviguer de maniere structuree dans le droit.

## Navigation

### Projet
- [[01-Projet/Objectifs|Objectifs & Perimetre]]
- [[01-Projet/Roadmap|Roadmap]]
- [[01-Projet/Journal|Journal de recherche]]
- [[01-Projet/Decisions-architecturales|Decisions architecturales]]

### Recherche
- [[02-Etat-de-l-art/MOC-Etat-de-l-art|Etat de l'art (MOC)]]
- [[03-Concepts/MOC-Concepts|Concepts & Modelisation]]
- [[06-Analyses/MOC-Analyses|Analyses & Syntheses]]

### Donnees & Technique
- [[04-Donnees/MOC-Donnees|Sources de donnees]]
- [[05-Technique/MOC-Technique|Stack technique & Prototypes]]

### Redaction
- [[07-Redaction/MOC-Redaction|Redaction & Livrables]]

---

## Tableau de bord (Dataview)

| Metrique | Requete |
|---|---|
| Articles lus | `= length(filter(dv.pages('"02-Etat-de-l-art"'), (p) => p.status == "lu"))` |
| Articles a lire | `= length(filter(dv.pages('"02-Etat-de-l-art"'), (p) => p.status == "a-lire"))` |
| Questions ouvertes | `= length(filter(dv.pages('"06-Analyses/questions-ouvertes"'), (p) => p.statut == "ouverte"))` |
