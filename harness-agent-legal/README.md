# Agentic legal retrieval baseline

This directory contains the reproducible harness and the public aggregate for
an internal comparison of two legal-retrieval agents on the ECIR French
criminal-law evaluation set. The agent receives a question and returns up to
ten ordered references as JSON:

```json
{"references":["Code de procédure pénale, article 63-1"]}
```

## Executed campaign

The historical campaign used `moonshotai/kimi-k2.6` through OpenRouter,
temperature 0, with a two-step budget for Articles and three steps for Judicial
Decisions. It compares:

| Condition | Tool access |
|---|---|
| `web` | Brave Search through `pi-web-access` |
| `openlegy_mcp` | The same web access plus OpenLegy / Legifrance MCP |

The public aggregate is [results_public/agentic_baseline_pool2runs.csv](results_public/agentic_baseline_pool2runs.csv).
It is an executed internal agentic baseline, useful as a complementary
comparison in the paper. It is not a replacement for the frozen A3 retrieval
table: it pools the best cell from two historical runs and historical execution
materialized 752 of 754 question identifiers. No value is imputed for the two
missing questions.

## Public versus private material

Git contains the prompts, orchestration, MCP extension, aggregate scores and
provenance manifest. It deliberately excludes questions, ground truth,
per-question CSVs, model responses, tool traces and credentials. The original
per-question CSVs include expected references and must never be committed.

The aggregate and the private source hashes are documented in
`results_public/manifest.json`.

## Re-running an authorised internal campaign

Install the `pi` harness and `pi-web-access`, then supply an authorised private
input bundle in this directory (`questions_754.tsv`, `ground_truth_754.json`,
`jp_index_full.json`). Set `LKG_REPO` to the repository root when the benchmark
components live elsewhere. Credentials remain environment variables:

```bash
export LKG_REPO="$(cd .. && pwd)"
export OPENLEGY_TOKEN="…"
export BRAVE_SEARCH_API_KEY="…"
python3 run_parallel.py --questions questions_754.tsv --workers 10 --timeout 900 \
  --conditions web,openlegy --modalities articles,jurisprudence
```

`run_agents.py` now hashes per-question artifact filenames deterministically,
while retaining the full QID inside each result record. This prevents the
historical filename-length failure during a future authorised rerun; it does not
alter the historical results in `results_public/`.

For scoring, `export_results.py` requires the private ground truth and the
existing benchmark resolver. It resolves the project through `LKG_REPO` rather
than a developer-specific absolute path.
