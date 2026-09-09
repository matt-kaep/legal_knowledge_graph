# E030 v3 — Preflight de reprise sûre

Ce paquet prépare uniquement la reprise technique des 17 shards E030 v2
`partial_technical`. Il ne contient aucun score E030 et ne remplace ni le
manifeste ni les sorties v2.

## Entrées scellées

- Manifeste v2 : `configs/b2_llm_as_judge_a3_execution_v2_gpu70.json`,
  SHA-256 `7eb5d08758b3f833cfdb0f2c880066b8cbbe481cd2c252f20ed5582c8e3ea16f`.
- Audit public v2 : `results/benchmark-a3-b1/e030-aggregation-audit-v1/aggregation_receipt_public.json`,
  SHA-256 `50cecfddfe450f93353acee101b979b1a53b7bfc4f2921e1a5120871fd57ac9b`.
- Manifeste opérationnel successeur v3-r1 : `configs/b2_llm_as_judge_a3_retry_v3_preflight_r1.json`,
  SHA-256 `505e645ec2f1f1dc7950c22d01d09f1b6b27c68c9b02adb7d48e7a7771978d65`.
  Le premier manifeste v3 (`482a8c45…`) reste archivé : v3-r1 corrige uniquement
  la racine de stockage du reçu de smoke, sans modifier les entrées scientifiques.

Les cinq shards v2 `complete_clean` restent des diagnostics archivés et ne
sont ni rejoués ni agrégés. La liste exacte des dix-sept reprises et leur port
réservé est `retry_shards.csv`.

## Contrôles avant chaque appel

Le smoke GPU démarre Gemma depuis le snapshot figé, mais ne lit aucune liste
de jobs et n’émet aucun jugement. Il doit vérifier : port local libre,
`/health`, `/v1/models`, nom servi `model@revision`, chemin de snapshot se
terminant par la révision figée, GPU et nœud. Son reçu contient ces éléments.
Tout échec écrit un reçu technique et arrête le job ; une erreur HTTP ne peut
être convertie en label, zéro ou jugement.

Le lanceur de 17 shards reste bloqué volontairement jusqu’au reçu smoke vert et
à une autorisation explicite. Chaque reprise devra produire 754 questions × 10
positions, puis une nouvelle agrégation des 22 conditions. Une moyenne
LightGCN reste interdite tant que les trois graines de chaque modalité ne sont
pas complètes et propres. E030 demeure exploratoire jusqu’à
`lawyer_agreement.json`.

## Estimation fondée sur v2

Les sept shards v2 ayant effectivement travaillé ont consommé entre 15 min 44
s et 48 min 53 s (hors Direct LLM JP composé de slots sans candidat). Les
échecs 404 ont quitté en 6–9 s et ne servent pas à estimer un coût. Pour les
17 reprises : enveloppe de 15–60 min GPU par shard, soit environ 4–17 GPU-h ;
à 17 GPU disponibles, durée mur de l’ordre de 1 h après le délai de queue.
Le smoke a une limite Slurm de 30 min.
