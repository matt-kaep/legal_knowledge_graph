---
tags: [moc, etat-de-l-art]
created: 2026-04-09
---

# Etat de l'art — Map of Content

## Par thematique

### KG dans le domaine juridique
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/KG-juridique"
SORT annee DESC
```

### Ontologies & Standards
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Ontologies-standards"
SORT annee DESC
```

### GraphRAG
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/GraphRAG"
SORT annee DESC
```

### NLP Juridique & NER
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/NLP-juridique"
SORT annee DESC
```

### Agents LLM Juridiques
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Agents-LLM"
SORT annee DESC
```

### Prediction & Citations
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Prediction-citations"
SORT annee DESC
```

### Construction de KG avec LLMs
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Construction-KG-LLM"
SORT annee DESC
```

### Benchmarks
```dataview
TABLE auteurs as Auteurs, annee as Annee, status as Statut, pertinence as Pertinence
FROM "02-Etat-de-l-art/Benchmarks"
SORT annee DESC
```

## Vues transversales

### Articles a lire en priorite
```dataview
TABLE auteurs as Auteurs, categorie as Cat, pertinence as Pertinence
FROM "02-Etat-de-l-art"
WHERE status = "a-lire" AND pertinence = "haute"
SORT annee DESC
```

### Tous les articles lus
```dataview
LIST
FROM "02-Etat-de-l-art"
WHERE status = "lu"
SORT annee DESC
```
