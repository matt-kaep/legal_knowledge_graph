# Graphe bipartite `tout_cc` — métriques

## Synthèse

- Nœuds : **567,943** (534,600 JP + 33,343 articles)
- Arêtes : **1,214,882** (18,201 `rapproche` + 1,196,681 `cite`)
- Densité : 3.77e-06
- Composantes connexes (non dirigé) : 72,794 — plus grosse : 493,721
- Temps de calcul : 14.7 s

## Top 15 codes (par nombre d'articles distincts retenus)

- `code_du_travail` : 6,970
- `code_de_la_securite_sociale` : 3,221
- `code_civil` : 3,153
- `code_de_commerce` : 2,504
- `code_de_procedure_penale` : 2,081
- `code_de_procedure_civile` : 1,899
- `code_rural_et_de_la_peche_maritime` : 1,523
- `code_penal` : 1,469
- `code_de_la_sante_publique` : 1,221
- `code_de_la_consommation` : 807
- `code_des_assurances` : 746
- `code_de_l_urbanisme` : 703
- `code_general_des_impots` : 575
- `code_monetaire_et_financier` : 553
- `code_de_la_construction_et_de_l_habitation` : 523

## Top 30 articles les plus cités dans ce périmètre

| pair_key | code | article | in_degree |
|---|---|---|---:|
| `code_de_procedure_civile:700` | code_de_procedure_civile | 700 | 232061 |
| `code_de_procedure_civile:455` | code_de_procedure_civile | 455 | 75604 |
| `code_civil:1134` | code_civil | 1134 | 54101 |
| `code_de_procedure_penale:567-1-1` | code_de_procedure_penale | 567-1-1 | 32688 |
| `code_civil:1382` | code_civil | 1382 | 23784 |
| `code_de_procedure_civile:4` | code_de_procedure_civile | 4 | 21376 |
| `code_de_procedure_civile:16` | code_de_procedure_civile | 16 | 15881 |
| `code_de_procedure_civile:1014` | code_de_procedure_civile | 1014 | 15838 |
| `code_civil:1315` | code_civil | 1315 | 15466 |
| `code_civil:1147` | code_civil | 1147 | 14631 |
| `code_de_procedure_civile:1026` | code_de_procedure_civile | 1026 | 12415 |
| `code_de_procedure_penale:593` | code_de_procedure_penale | 593 | 9122 |
| `code_de_procedure_civile:624` | code_de_procedure_civile | 624 | 8836 |
| `code_civil:1351` | code_civil | 1351 | 6913 |
| `code_de_l_organisation_judiciaire:R431-5` | code_de_l_organisation_judiciaire | R431-5 | 6499 |
| `code_de_procedure_civile:1015` | code_de_procedure_civile | 1015 | 6483 |
| `code_du_travail:L122-14-3` | code_du_travail | L122-14-3 | 6323 |
| `code_de_procedure_penale:618-1` | code_de_procedure_penale | 618-1 | 5972 |
| `code_de_procedure_civile:627` | code_de_procedure_civile | 627 | 5711 |
| `code_de_procedure_civile:12` | code_de_procedure_civile | 12 | 4641 |
| `code_de_procedure_civile:625` | code_de_procedure_civile | 625 | 4379 |
| `code_de_procedure_civile:604` | code_de_procedure_civile | 604 | 3912 |
| `code_de_procedure_civile:452` | code_de_procedure_civile | 452 | 3893 |
| `code_de_procedure_civile:462` | code_de_procedure_civile | 462 | 3508 |
| `code_de_procedure_civile:1009-1` | code_de_procedure_civile | 1009-1 | 3499 |
| `code_civil:1184` | code_civil | 1184 | 3332 |
| `code_du_travail:L1221-1` | code_du_travail | L1221-1 | 3258 |
| `code_de_procedure_penale:575` | code_de_procedure_penale | 575 | 3132 |
| `code_du_travail:L3171-4` | code_du_travail | L3171-4 | 2990 |
| `code_du_travail:L122-14-4` | code_du_travail | L122-14-4 | 2556 |

## Top 15 décisions citant le plus d'articles distincts

| pourvoi | chambre | date | out_degree (articles cités) |
|---|---|---|---:|
| - | Civ. 1re | 1860-08-01 | 62 |
| 10-82938 | Crim. | 2011-05-17 | 40 |
| 14-29179 | Civ. 1re | 2017-03-15 | 37 |
| 15-28683 | Com. | 2017-05-18 | 35 |
| 18-23578 | Civ. 3e | 2022-01-26 | 34 |
| 18-16968 | Civ. 1re | 2022-06-15 | 33 |
| 16-17241 | Soc. | 2017-09-21 | 33 |
| 14-18977 | Soc. | 2016-06-08 | 32 |
| 19-23843 | Soc. | 2022-12-14 | 30 |
| 19-25455 | Civ. 2e | 2021-09-23 | 29 |
| 21-87417 | Crim. | 2022-06-21 | 29 |
| 18-23692 | Soc. | 2020-03-25 | 29 |
| 15-16110 | Civ. 2e | 2016-07-07 | 28 |
| 13-27919 | Civ. 2e | 2015-01-08 | 26 |
| 19-24378 | Soc. | 2021-11-04 | 25 |

## Exports

- `bip-tout_cc.pkl` : 164.66 Mo