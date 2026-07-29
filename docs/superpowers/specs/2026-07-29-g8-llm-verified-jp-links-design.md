---
date: 2026-07-29
type: design
status: draft
tags: [benchmark, graphes, jurisprudence, llm, g8]
---

# G8 candidate -- LLM-verified JP-JP legal links

## Goal

Define a conservative JP-JP graph candidate built from semantic neighbors, but keeping only links that a legal LLM can justify as a real legal relation.

This is an exploratory graph-design proposal. It does not change the confirmatory protocol, result registry, or paper claims until a dedicated experiment is registered and validated.

## Real construction spec

This section turns the idea into an implementable graph-building workflow.

### Proposed graph versions

Use a new family name so the result is not confused with raw G4 embedding links:

- `G8-llm-JJ-knn30-issue-rule`
- later hybrid candidate: `G9-citation-G8JJ-knn30-issue-rule`, if combined with G1 or G7

The first implementation should only build `G8-llm-JJ-knn30-issue-rule`.

### Source data

Inputs:

- `G4-knn30` graph artifacts, used only to generate semantic JP-JP candidate pairs.
- G1/G4 node order arrays, to keep graph node indices compatible with existing retrieval code.
- JP identifiers and JP summaries/full text available in the benchmark data.
- A prompt version file, immutable after a run starts.

The builder must not read `eval_rich_retrievable_strict` labels or scores when selecting K, prompts, link types, legal notions, thresholds, or model settings.

### Candidate pair generation

For every JP node with an embedding:

1. Read its top-30 JP neighbors from `G4-knn30`.
2. Keep JP-JP pairs only; drop JP-Article and Article-Article pairs.
3. Drop self-links.
4. Canonicalize each undirected pair as `(min(jp_id_a, jp_id_b), max(jp_id_a, jp_id_b))`.
5. Preserve directional evidence from G4:
   - rank from A to B, if present;
   - rank from B to A, if present;
   - cosine similarity from A to B;
   - cosine similarity from B to A.
6. Deduplicate pairs before LLM calls.

The graph itself is undirected. Candidate generation may be asymmetric because KNN is directional, but the accepted edge is materialized once.

Expected scale:

- approximately `N_jp * 30` directed candidates before deduplication;
- less after undirected deduplication;
- acceptance rate expected to be low by design.

### LLM job records

Each candidate pair becomes one deterministic JSONL job.

Required fields:

```json
{
  "job_id": "sha256(prompt_version|model|jp_id_a|jp_id_b)",
  "prompt_version": "g8_jp_link_v1",
  "candidate_source": "G4-knn30",
  "jp_id_a": "...",
  "jp_id_b": "...",
  "g4_rank_a_to_b": 12,
  "g4_rank_b_to_a": null,
  "g4_similarity_a_to_b": 0.8123,
  "g4_similarity_b_to_a": null,
  "text_a": "...",
  "text_b": "..."
}
```

The job writer must support sharding, for example `jobs/shard-0000.jsonl`, so the run can be executed incrementally or on a cluster.

### Text sent to the LLM

Prefer concise legal summaries when available. If full decisions are used, the prompt should provide:

- decision identifier;
- court/date/chamber if available;
- factual and procedural context;
- legal question or holding;
- relevant reasoning excerpt.

The verifier should not see benchmark questions, gold labels, graph-neighbor ranks, cosine scores, or prior retrieval results. G4 is only a hidden candidate generator.

### Prompt contract

The prompt must force conservative behavior:

- create a link only if both decisions share a precise legal issue or apply the same legal rule/test;
- reject broad thematic similarity;
- reject article-only overlap;
- reject vocabulary-only overlap;
- reject if the shared legal notion cannot be stated precisely;
- output valid JSON only;
- no confidence field;
- no weak link;
- no invented third link type.

Allowed JSON outputs:

```json
{
  "decision": "link",
  "link_type": "same_legal_issue",
  "legal_notion_label": "...",
  "legal_rule_or_test": "...",
  "shared_legal_point": "...",
  "evidence_a": "...",
  "evidence_b": "..."
}
```

```json
{
  "decision": "link",
  "link_type": "same_rule_application",
  "legal_notion_label": "...",
  "legal_rule_or_test": "...",
  "shared_legal_point": "...",
  "evidence_a": "...",
  "evidence_b": "..."
}
```

```json
{
  "decision": "no_link",
  "reason": "..."
}
```

### Output validation

Every LLM response must pass schema validation before it can create an edge.

Reject the response if:

- JSON is invalid;
- `decision` is not `link` or `no_link`;
- `link_type` is missing for a link;
- `link_type` is outside the two allowed values;
- `legal_notion_label` is missing, too broad, or too long;
- `legal_rule_or_test`, `shared_legal_point`, `evidence_a`, or `evidence_b` is empty;
- the response contains a confidence field;
- the response uses a third link type.

Invalid responses are retried with the same job id and a retry counter. After the retry limit, they become `invalid_response` and do not create an edge.

### Accepted edge records

Validated links are written to a canonical edge table:

```csv
edge_id,jp_id_a,jp_id_b,link_type,legal_notion_label,legal_rule_or_test,shared_legal_point,evidence_a,evidence_b,g4_rank_a_to_b,g4_rank_b_to_a,g4_similarity_a_to_b,g4_similarity_b_to_a,prompt_version,model,job_id,response_sha256
```

`edge_id` is deterministic:

```text
sha256(G8-llm-JJ-knn30-issue-rule|jp_id_a|jp_id_b|link_type|canonical_legal_notion_label)
```

### Legal notion normalization

The first graph artifact stores the raw `legal_notion_label`. A separate normalization step creates:

- `legal_notion_raw`
- `legal_notion_canonical`
- `normalization_rule`
- optional `article_refs`

For V1, normalization can be semi-automatic:

1. lowercase and remove punctuation;
2. cluster near-identical labels;
3. manually review high-frequency notions;
4. mark over-broad labels as rejected;
5. rebuild the edge table using canonical notions.

Accepted edges whose labels become over-broad during normalization should be removed or sent to manual review, not silently kept.

### Graph materialization

Create sparse matrices over the same node order as G1/G4:

- `graph_g8_mixed.npz`: all accepted JP-JP links.
- `graph_g8_same_legal_issue.npz`: only `same_legal_issue`.
- `graph_g8_same_rule_application.npz`: only `same_rule_application`.

Initial edge weights:

- binary `1.0` for all accepted links.

Do not use LLM confidence as a weight, because confidence is not part of the method. Later variants may compare binary weights with cosine-preserving weights, but that is a separate ablation.

Required artifact directory:

```text
05-Technique/benchmark/etape1_embedding_pur/data/llm_verified_graphs/G8-llm-JJ-knn30-issue-rule/
```

Required files:

```text
manifest.json
candidate_pairs.csv
jobs/
responses/
accepted_edges_raw.csv
accepted_edges_normalized.csv
rejected_pairs.csv
invalid_responses.csv
notion_dictionary.csv
graph_g8_mixed.npz
graph_g8_same_legal_issue.npz
graph_g8_same_rule_application.npz
jp_ids.npy
article_ids.npy
article_codes.npy
```

### Manifest

`manifest.json` must include:

- graph version;
- source graph version and SHA-256;
- prompt version and SHA-256;
- model name;
- inference provider;
- generation parameters;
- candidate count;
- accepted edge count;
- acceptance rate;
- accepted count by `link_type`;
- accepted count by canonical legal notion;
- response schema version;
- code hashes for builder scripts;
- creation timestamp.

### Pilot before full run

Run a pilot before the full graph:

- sample 200 to 500 candidate pairs;
- stratify by G4 KNN rank buckets: 1-5, 6-10, 11-20, 21-30;
- include reciprocal and non-reciprocal KNN pairs;
- inspect accepted links manually;
- estimate acceptance rate and dominant legal notions;
- revise only the prompt/spec from train-side evidence.

The full run starts only after the pilot prompt is frozen.

### Quality gates

The graph is not usable for benchmark comparison unless:

- every accepted edge has a valid type and legal notion;
- no accepted edge uses a banned broad label;
- all graph matrices match the G1/G4 node order;
- no duplicate undirected edge exists;
- artifacts are reproducible from manifest inputs;
- prompt and model metadata are recorded;
- pilot inspection does not reveal systematic false positives.

### Exploratory evaluation

Exploratory train-side diagnostics:

- gold JP connectivity: do gold JP attached to the same train question become connected or closer?
- notion distribution: which legal notions dominate accepted edges?
- acceptance by KNN rank: are close embedding neighbors more often legally valid?
- neighborhood audit: for selected JP, inspect accepted G8 neighbors against raw G4 neighbors.
- retrieval smoke: run PPR or LightGCN plumbing only on train-side or explicitly exploratory settings.

Confirmatory evaluation requires a new registered `experiment_id` and must respect the existing grouped protocol.

## Link semantics and examples

The graph favors abstention. There is no low/medium confidence field and no weak-link class. Accepted links must use one of the two allowed positive types.

### `same_legal_issue`

Use when both decisions address substantially the same legal question.

Examples:

- Two decisions about the nullity of a geolocation measure for insufficient motivation.
- Two decisions about the admissibility of a nullity request by a person whose protected interest is disputed.
- Two decisions about whether late-produced evidence may be rejected automatically.

Reject if the pair merely shares a broad topic such as nullities, evidence, detention, or geolocation.

### `same_rule_application`

Use when both decisions apply the same rule, test, or legal criterion, even if their facts differ.

Examples:

- Both decisions apply the requirement of prejudice or grievance for procedural nullity.
- Both decisions apply the adversarial-debate requirement for criminal evidence.
- Both decisions apply the criteria making pre-trial detention a measure of last resort.

Reject if the common point is only a shared article citation or a shared vocabulary.

### Legal notion labels

Good labels:

- `exigence de grief en nullite`
- `qualite pour agir en nullite`
- `motivation de la geolocalisation`
- `urgence en geolocalisation`
- `debat contradictoire de la preuve`
- `charge de la preuve`
- `detention provisoire ultime recours`
- `garanties de representation`

Bad labels:

- `procedure penale`
- `nullite`
- `preuve`
- `geolocalisation`
- labels that only describe facts, such as `vehicule vole avec fausses plaques`

## Output examples

Accepted link:

```json
{
  "decision": "link",
  "link_type": "same_rule_application",
  "legal_notion_label": "exigence de grief en nullite",
  "legal_rule_or_test": "Une irregularite procedurale ne justifie l'annulation que si elle porte atteinte aux interets de la partie qui l'invoque.",
  "shared_legal_point": "Les deux decisions appliquent l'exigence de grief pour apprecier une demande de nullite.",
  "evidence_a": "La decision A controle si l'irregularite invoquee a porte atteinte aux interets de la partie.",
  "evidence_b": "La decision B refuse l'annulation faute de grief demontre."
}
```

Rejected pair:

```json
{
  "decision": "no_link",
  "reason": "Les deux decisions relevent de la geolocalisation, mais l'une porte sur la qualite pour agir et l'autre sur l'urgence de la pose du dispositif. Aucun meme critere juridique central n'est etabli."
}
```
