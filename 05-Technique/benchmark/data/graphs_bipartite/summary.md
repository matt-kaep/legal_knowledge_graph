# Graphes bipartites JP × Articles — synthèse

Source : `database-judilibre-enrichie/Cour de cassation`.
Articles uniques dans le corpus : **33,343** (issus de 1,217,069 citations brutes ; 0 malformées rejetées).

## Comparatif des 3 périmètres

| Périmètre | Nœuds | JP | Articles | Arêtes cite | Arêtes rapproche | Composantes | Plus grosse CC | Compute |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| resserre | 9,593 | 5,618 | 3,975 | 18,366 | 5,958 | 2 | 9,586 | 0.2 s |
| large | 31,474 | 21,534 | 9,940 | 73,168 | 18,201 | 31 | 31,390 | 0.8 s |
| tout_cc | 567,943 | 534,600 | 33,343 | 1,196,681 | 18,201 | 72,794 | 493,721 | 14.7 s |

## Format canonique

- Nœud JP : clé = pourvoi normalisé (`10-87525`). Attributs : type, id_judilibre, ecli, chamber, date, solution.
- Nœud Article : clé = `pair_key` au format `code_slug:article_num` (ex. `code_civil:1240`, `code_du_travail:L122-14-3`).
- Arête `rapproche` (Decision→Decision) : héritée Phase A.
- Arête `cite` (Decision→Article) : extraite de `code_article_pairs`.

Conforme à [[Format-Fondement-Juridique]] et [[Format-Jurisprudence]].