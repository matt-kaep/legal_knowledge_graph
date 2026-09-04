# Benchmark rapprochements v1 — stats

- Source : `database-judilibre-enrichie/Cour de cassation`
- Filtre : `>= 3` rapprochements parsables, toutes chambres
- Questions : **1,532**

## Par chambre

| Chambre | N questions |
|---|---:|
| Soc. | 375 |
| Crim. | 370 |
| Civ. 2e | 291 |
| Civ. 1re | 236 |
| Civ. 3e | 181 |
| Com. | 39 |
| Ass. plén. | 26 |
| Ch. mixte | 10 |
| Autre | 4 |

## Pipeline

```
{
  "total": 553075,
  "with_any_rapp": 63695,
  "with_enough_parsable": 1532,
  "rapp_total_seen": 116196,
  "rapp_parsable_seen": 18901,
  "rapp_with_href_id": 3706,
  "questions_total": 1532,
  "by_chamber": {
    "Crim.": 370,
    "Ass. plén.": 26,
    "Civ. 1re": 236,
    "Soc.": 375,
    "Civ. 3e": 181,
    "Com.": 39,
    "Civ. 2e": 291,
    "Autre": 4,
    "Ch. mixte": 10
  },
  "resolution": {
    "matched": 4671,
    "unmatched": 78,
    "href": 1083,
    "matched_and_href": 1083
  }
}
```