# E030 v2 GPU70 aggregation audit

This directory contains lightweight audit exports only. It does not contain prompts, questions, candidate texts, model responses, or corpus data.

The frozen E030 execution had 22 expected shards of 7,540 positions each (754 questions times fixed K=10). Hash, coverage, response-status, and materialization checks found five `complete_clean` shards and 17 `partial_technical` shards. Therefore no E030 method comparison, LightGCN seed mean, or score is reportable from this campaign.

The partial shards contain invalid or error responses. The mass technical failures are recorded as HTTP 404 provider failures; they are not semantic `non_jugeable` decisions. In particular, the all-zero shard outputs are excluded from any score comparison.

`per_shard_audit.csv` is the canonical per-shard evidence. `complete_clean_scores.csv` is a diagnostic subset of technically clean shards only, not a paper table and not a valid cross-method aggregate. The remote immutable aggregation receipt and source hashes are recorded in `05-Technique/benchmark/etape1_embedding_pur/configs/b2_llm_as_judge_a3_aggregation_audit_v1.json`.

E030 remains exploratory even after a technically complete successor, until `lawyer_agreement.json` exists.
