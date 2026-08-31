# E021 — reranking comparable

E021 applies one frozen LLM reranker to three independently materialized real
jurisprudence candidate pools: cosine/BGE-M3, PPR, and LightGCN. It is not
LLM-as-a-Judge: here the model changes a candidate order; the judge only
evaluates an already-produced ranking.

The frozen contract is identical for the three pools: 754 questions,
`K_in=20`, `K_out=10`, model revision
`4033b16200f4152e55e100ea12dc388c537df622`, temperature zero, one retry,
and a strict parser which rejects duplicate or out-of-pool identifiers.

## Current evidence

The v5 execution produced 2,249 valid family/question responses out of 2,262:

- cosine/BGE-M3: 747 / 754;
- PPR: 753 / 754;
- LightGCN: 749 / 754.

Thirteen units remain to be recovered. The historic v5 manifest remains
immutable. `manifest_cluster_gpu_runtime_v5_resume_v1.json` is a new resume
manifest: it pins the pre-resume response-history hash, appends only absent or
invalid units, and writes a distinct metrics file and completion receipt. It
does not overwrite the prior partial metrics.

E021 is a supplementary jurisprudence table. It is not part of the main
retrieval comparison, contains no LLM-as-a-Judge column, and must not be used
for a superiority claim until all three families have full coverage and the
Task A registers are updated from the new receipt.

## Télécom execution

From the dedicated reproducibility worktree, set the remote host and checkout
name once. The data-root and Python paths default on Télécom; override them
only when the remote layout differs.

```bash
export REMOTE_HOST='your-account@gpu-gw.enst.fr'
export REMOTE_REPO='legal_knowledge_graph_repro_v1'
export LKG_REPO="$PWD"
ROOT='05-Technique/benchmark/etape1_embedding_pur/scripts'

bash "$ROOT/run_telecom_reproducibility.sh" sync-code
bash "$ROOT/run_telecom_reproducibility.sh" inventory
bash "$ROOT/run_telecom_reproducibility.sh" submit-e021-resume
bash "$ROOT/run_telecom_reproducibility.sh" queue
```

The GPU job requests one L40S, eight CPU cores, 48 GiB RAM and six hours. It
first checks the frozen job hash and the exact pre-resume response-history
hash. It then starts vLLM locally, waits for `/health`, processes only the
missing units, recomputes metrics and writes:

```text
.../_e021_jobs/E021-cluster-gpu-runtime-v5/resume_v1/metrics.json
.../_e021_jobs/E021-cluster-gpu-runtime-v5/resume_v1/completion_receipt.json
```

The receipt exits non-zero if one family remains incomplete, but keeps the
partial output for a later resumable run.
