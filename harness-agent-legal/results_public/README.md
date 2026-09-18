# Public aggregate — agentic legal retrieval baseline

`agentic_baseline_pool2runs.csv` is the light, publishable aggregate of an
internal agentic benchmark on 754 French criminal-law questions. It contains no
question text, per-question identifier, ground-truth label, model response or
tool trace.

The executed model was `moonshotai/kimi-k2.6` through OpenRouter at temperature
0. The two conditions were a web-search agent and the same agent with the
OpenLegy / Legifrance MCP tools. Both returned up to ten ordered references.

The historical aggregate keeps the best observed NHit@10 in each cell across
two runs; NDCG@10 and MRR@10 are taken from the same selected run. It therefore
describes an executed **internal agentic baseline**, not a hyperparameter-tuned
or lockbox-confirmed result. Two question identifiers could not be materialized
in the historical runs because their filenames exceeded the filesystem limit;
the public table records 752 exported identifiers out of 754 requested.

The private evidence bundle is intentionally excluded: it contains the ECIR
questions, ground truth, per-question scores and tool traces. Its immutable
hashes and the provenance of this aggregate are recorded in `manifest.json`.
