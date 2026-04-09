---
tags: [moc, analyses]
---

# Analyses & Syntheses

## Comparatifs
```dataview
LIST FROM "06-Analyses/comparatifs" SORT file.name ASC
```

## Syntheses thematiques
```dataview
LIST FROM "06-Analyses/syntheses" SORT file.name ASC
```

## Questions ouvertes
```dataview
TABLE statut, priorite
FROM "06-Analyses/questions-ouvertes"
WHERE statut = "ouverte"
SORT priorite ASC
```
