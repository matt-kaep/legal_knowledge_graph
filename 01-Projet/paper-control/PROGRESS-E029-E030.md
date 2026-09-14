---
date: 2026-09-14
type: suivi
owner: assainissement
status: judge-six-completions-running-depth-export-ready
tags: [e029, e030, benchmark]
---

# Avancement E029 / E030

## État courant — 2026-09-14

| Livrable | État | Preuve / job | Suite |
|---|---|---|---|
| Inventaire et audit des 17 Judge r6 | terminé | `judge-table-completion-a3-v1/available/audit_receipt_public.json`, SHA `40a3e2e8800c9cd47d8b8cc48fc91415b10b56600aaf83783784796fc146736a` | réutiliser descriptivement |
| Six listes Judge manquantes | figées, contexte vérifié | manifeste SHA `a22c65803bf1cb39a9992f563746b6fb26f564e7fc1fc0a704ebab789bd404ee` | pas de nouvelle sélection |
| Smoke service | réussi sur L40S | 992269, SHA `e86576ecca38f21d4266400459552d31b14582457f6968df0ceaa85b31170e77` | gate satisfait |
| Six calculs Judge | RUNNING observé | 992279–992284 | suivre réponses et reçus |
| Audit global / moyennes trois seeds | PENDING Dependency | 992285, après réussite des six | exiger 23 conditions propres et quatre moyennes avant livraison finale |
| 42 valeurs exactes de profondeur | terminé sans GPU | `reranking-depth-table-a3-v1/reranking_depth.csv`, SHA `8f00ba78e88c1b222fcee9061addbee442b5fdd4cf05fb050cdfd7fb437faf1a` | transmis au Papier |
| Audit juridique aveugle | humain en attente | `lawyer_agreement.json` absent | conserver tous les scores Judge exploratoires |

Chemins de sorties ci-dessus sous `results/benchmark-a3-b1/`. Le README de
`judge-table-completion-a3-v1/` détaille accès distant, identités G6/G7, ports,
hashes, coûts et commandes. Les tableaux suivants sont conservés comme journal historique.

| Livrable | Statut | Dépendance | Job | Sortie / hash | Blocage | Prochaine action |
|---|---|---|---|---|---|---|
| Inventaire Phase 0 | terminé | A3 | aucun | `INVENTAIRE-E029-E030-A3-2026-09-08.md` | aucun | transmettre avant GPU |
| Rankers top-100 sources | terminé | A3 | historiques validés | cosine `7de0504d...`; PPR `8f15c4d3...`; LGC G6 `b1448dd5...` | aucun | sources E029 |
| Structure pénale G1/G6/G7 | terminé | A3 | 985678 historique | manifest `2307ac04...` | graphe complet distinct non matérialisé : aucun artefact source fiable trouvé sur Télécom | attendre une définition et une source explicitement gelées ; ne pas déduire |
| Audit contexte E029 complet | terminé, bloquant | cartes/prompt/modèle | CPU v4 | reçu `d3da604a...`; agrégat `ea1d060e...`; 42/100 conditions compatibles avec textes intégraux | 58 conditions et 24 509 prompts dépassent le budget de 16 128 tokens ; zéro appel modèle | geler une représentation commune dans un manifeste successeur |
| E029 archive 42 cellules | récupéré, incomplet | audit identité | 985512–985516 historiques | manifest `259cff89...`; 31 668 réponses | 58 cellules absentes | figer successeur |
| E029 matrice complète | bloquée avant GPU | règle de représentation + manifeste successeur | aucun | aide à la décision `c516d3bc...` : plafonds 64/96 compatibles à K=100, zéro appel modèle | aucune règle compacte n'est gelée ; GPU interdit | attendre la décision scientifique, puis préflight successeur et soumission complète |
| E030 Judge listes de base | préparation | top-10 hashés + budget | aucun | preflight `0f7c40ce...` | exécution non gelée | geler après E029 |
| E030 Judge reranké | bloqué | E029 complet | aucun | aucun | reranking incomplet | ne pas soumettre |
| Audit avocat | bloqué humain | paquet E016 | aucun | `lawyer_agreement.json` absent | intervention humaine | garder Judge exploratoire |
| Courbes E029 Articles/JP | non démarré | E029 complet | aucun | aucun | cellules manquantes | dériver après complétude |
