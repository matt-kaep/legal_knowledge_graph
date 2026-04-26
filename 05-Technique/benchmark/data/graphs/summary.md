# Graphes rapprochements JP — synthèse comparative

Source : `database-judilibre-enrichie/Cour de cassation` — 553,075 arrêts.
Arêtes parsables : 18,901 / 116,196 brutes (16.3 %).

## Comparatif des 3 périmètres

| Périmètre | Nœuds | Arêtes | Densité | Composantes | Isolés | Max degree in | Compute |
|---|---:|---:|---|---:|---:|---:|---:|
| resserre | 5,618 | 5,436 | 1.72e-04 | 884 | 0 | 12 | 1.1 s |
| large | 21,534 | 18,201 | 3.93e-05 | 5,199 | 0 | 13 | 0.3 s |
| tout_cc | 534,600 | 18,201 | 6.37e-08 | 518,265 | 513,066 | 13 | 6.3 s |

## Lecture

- **resserre** : vue centrée benchmark — les 1 532 arrêts-sources + leurs rapprochements directs.
- **large** : tous les arrêts ayant au moins un lien de rapprochement (source ou cible).
- **tout_cc** : le corpus entier (553 k arrêts) — la plupart sont isolés, mais ça donne la densité réelle.

## Fichiers générés

Pour chaque périmètre :
- `rapp-<name>.pkl` (NetworkX picklé, rapide à reloader)
- `rapp-<name>.graphml` (ouvrable dans Gephi)
- `metrics-<name>.md` (métriques détaillées + top 20)