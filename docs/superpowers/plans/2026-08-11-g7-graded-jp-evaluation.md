# G7 Graded JP Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, run, and audit an A--E LLM relevance evaluator for the 7,540 JP positions returned by frozen G7 rankings.

**Architecture:** A preparation script freezes G7 positions and attaches existing Step1 cards without exposing rank or method to the judge. A dedicated append-only runner classifies each unique `(qid, jp_id)` with a strict prompt/schema, an aggregator reconstructs fixed-K scores, and separate audit scripts create a blind lawyer sample and measure agreement. G8 relations remain outside the judge and are deferred to the diagnostic follow-up.

**Tech Stack:** Python 3, pandas/Parquet, psycopg2, OpenAI-compatible vLLM, JSONL, pytest, Slurm, existing `etape1` benchmark utilities.

## Global Constraints

- Experiment ID is `E016`; status remains `exploratoire` until the lawyer gate passes.
- Evaluation scope is exactly 754 `eval_rich_retrievable_strict` questions × G7 JP top-10 = 7,540 positions.
- Method is exactly `LightGCN-trained_K2` from `G7-citation-JJ-cit1-sem025-knn5`.
- LLM input is only question text plus the existing G8 Step1 card; do not regenerate Step1 or add full text.
- Never expose method, rank, Ground Truth, G8 relation, or graph distance in a judge job.
- Labels are `A`, `B`, `C`, `D`, `E`, `non_jugeable`; gains are `1`, `0.5`, `0`, `0`, `0`, `0`.
- Score uses fixed `K=10`: `(n_A + 0.5*n_B)/10`; missing slots and `non_jugeable` keep the denominator fixed.
- Technical failures are not `non_jugeable`; incomplete technical work blocks aggregation.
- Prompt calibration uses train-only examples; the 754 internal-eval questions never tune prompt/model/weights.
- Preserve unrelated dirty work and commit only files belonging to each task.

---

### Task 1: Graded relevance contract, prompt, and schema

**Files:**
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/74_g7_graded_jp_contract.py`
- Create: `05-Technique/benchmark/etape1_embedding_pur/prompts/g7_graded_jp_judge_v1.txt`
- Create: `05-Technique/benchmark/etape1_embedding_pur/schemas/g7_graded_jp_judge_v1.json`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_contract.py`

**Interfaces:**
- Produces: `LABEL_GAIN`, `VALID_LABELS`, `validate_judgment(payload) -> tuple[bool, str]`, `score_labels(labels, k=10) -> float`, `is_generic_justification(text) -> bool`.
- Consumes: no earlier task.

- [ ] **Step 1: Write failing tests for label validation and fixed-K scoring**

```python
def test_score_labels_keeps_fixed_k_denominator():
    assert MODULE.score_labels(["A"] * 4 + ["B"] * 2 + ["C"] * 4, k=10) == 0.5
    assert MODULE.score_labels(["A", "non_jugeable"], k=10) == 0.1

def test_validate_judgment_rejects_generic_justification():
    valid, reason = MODULE.validate_judgment({
        "classe": "A",
        "justification": "La décision applique directement la règle permettant de répondre.",
    })
    assert not valid
    assert reason == "generic_justification"

def test_validate_judgment_accepts_concrete_legal_reason():
    valid, reason = MODULE.validate_judgment({
        "classe": "A",
        "justification": "La Cour exige la présence du ministère public même lorsque la juridiction statue seulement sur l'action civile, ce qui répond à la condition de régularité posée.",
    })
    assert valid
    assert reason == ""
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_contract.py -q`

Expected: FAIL because `74_g7_graded_jp_contract.py` does not exist.

- [ ] **Step 3: Implement the minimal contract**

```python
VALID_LABELS = ("A", "B", "C", "D", "E", "non_jugeable")
LABEL_GAIN = {"A": 1.0, "B": 0.5, "C": 0.0, "D": 0.0, "E": 0.0, "non_jugeable": 0.0}
GENERIC_PATTERNS = (
    "applique directement la règle permettant de répondre",
    "est directement pertinente",
    "apporte une nuance utile",
)

def is_generic_justification(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return not normalized or any(pattern in normalized for pattern in GENERIC_PATTERNS)

def validate_judgment(payload: dict) -> tuple[bool, str]:
    if payload.get("classe") not in VALID_LABELS:
        return False, "invalid_label"
    justification = str(payload.get("justification") or "").strip()
    if not justification:
        return False, "missing_justification"
    if is_generic_justification(justification):
        return False, "generic_justification"
    return True, ""

def score_labels(labels: list[str], *, k: int = 10) -> float:
    if k <= 0 or len(labels) > k:
        raise ValueError("labels must fit a positive fixed K")
    return sum(LABEL_GAIN[label] for label in labels) / k
```

Write the prompt with the approved decision tree and exact output contract. Write a JSON Schema with `additionalProperties: false`, required `classe`/`justification`, and the six-label enum.

- [ ] **Step 4: Run tests and prompt/schema validation**

Run:

```bash
python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_contract.py -q
python3 -m json.tool 05-Technique/benchmark/etape1_embedding_pur/schemas/g7_graded_jp_judge_v1.json >/dev/null
```

Expected: all tests pass; schema parses.

- [ ] **Step 5: Commit Task 1**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/scripts/74_g7_graded_jp_contract.py \
  05-Technique/benchmark/etape1_embedding_pur/prompts/g7_graded_jp_judge_v1.txt \
  05-Technique/benchmark/etape1_embedding_pur/schemas/g7_graded_jp_judge_v1.json \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_contract.py
git commit -m "feat: define G7 graded JP judgment contract"
```

### Task 2: Prepare frozen G7 positions and blind Step1 jobs

**Files:**
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/75_prepare_g7_graded_jp_eval.py`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_preparation.py`

**Interfaces:**
- Consumes: `VALID_LABELS` contract; existing `67_prepare_g8_llm_jp_link_jobs.fetch_decision_cards` behavior and `bench_global.json`.
- Produces: `select_g7_positions(...) -> pd.DataFrame`, `build_blind_jobs(...) -> tuple[list[dict], pd.DataFrame]`, `manifest.json`, `rankings_topk.parquet`, `judge_jobs.jsonl`.

- [ ] **Step 1: Write failing tests for ranking completeness and payload blindness**

```python
def test_select_g7_positions_requires_one_to_k_per_question():
    selected = MODULE.select_g7_positions(rankings(), question_ids={"q1"}, method="LightGCN-trained_K2", k=2)
    assert selected[["qid", "rank", "jp_id"]].to_dict("records") == [
        {"qid": "q1", "rank": 1, "jp_id": "jp1"},
        {"qid": "q1", "rank": 2, "jp_id": "jp2"},
    ]

def test_build_blind_jobs_never_exposes_rank_method_gt_or_g8():
    jobs, positions = MODULE.build_blind_jobs(
        positions=pd.DataFrame([{"qid": "q1", "rank": 1, "jp_id": "jp1"}]),
        questions={"q1": {"enonce": "Question ?", "gold_jp_ids": ["gold"]}},
        cards={"jp1": {"synthese_pour_avocat": "Synthèse"}},
        judge_contract={"model_id": "model", "prompt_sha256": "hash", "prompt_version": "v1"},
    )
    assert set(jobs[0]) == {"job_id", "qid", "jp_id", "question", "decision_card", "judge_contract"}
    assert "rank" not in json.dumps(jobs[0])
    assert "gold" not in json.dumps(jobs[0])
    assert positions.iloc[0]["card_status"] == "available"
```

Add a separate test proving a missing card creates no LLM job but keeps a position with `card_status=missing`.
Add a test proving `--profile calibration` selects only `train_augmented_retrievable_strict`, defaults to `B3-a`, and rejects any path containing `eval_rich_retrievable_strict`.

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_preparation.py -q`

Expected: FAIL because the preparation script does not exist.

- [ ] **Step 3: Implement selection, card attachment, and manifests**

Implement:

```python
def select_g7_positions(rankings: pd.DataFrame, *, question_ids: set[str], method: str, k: int) -> pd.DataFrame: ...
def build_blind_jobs(*, positions: pd.DataFrame, questions: dict[str, dict], cards: dict[str, dict], judge_contract: dict) -> tuple[list[dict], pd.DataFrame]: ...
def sha256(path: Path) -> str: ...
```

Main defaults:

```text
rankings = .../G7-citation-JJ-cit1-sem025-knn5/eval_rich_retrievable_strict/rankings.parquet
bench = .../eval_rich_retrievable_strict/bench_global.json
out = .../eval_rich_retrievable_strict/E016-g7-graded-jp-v1
method = LightGCN-trained_K2
K = 10
```

Add two explicit profiles:

```text
--profile evaluation   -> the frozen G7/eval paths and LightGCN method above
--profile calibration  -> train_augmented_retrievable_strict/rankings.parquet, method B3-a,
                          deterministic --question-limit/--seed selection
```

The calibration profile must fail closed if either input path contains `eval_rich_retrievable_strict`. Fetch cards only for unique JP IDs through existing `jp_decisions.step1_raw`; import the DB helper lazily inside `main()` so unit tests stay DB-free, and never update the database. Hash the source rankings, benchmark, prompt, schema, selected position parquet, card payload, and script.

- [ ] **Step 4: Run tests and a local no-DB fixture smoke test**

Run:

```bash
python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_preparation.py -q
python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g8_semantic_rescue.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/scripts/75_prepare_g7_graded_jp_eval.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_preparation.py
git commit -m "feat: prepare blind G7 graded JP jobs"
```

### Task 3: Append-only vLLM judge runner

**Files:**
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/76_run_g7_graded_jp_judge.py`
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/run_g7_graded_jp_judge_on_cluster.sh`
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_g7_graded_jp_judge.sh`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_runner.py`

**Interfaces:**
- Consumes: `judge_jobs.jsonl`, prompt/schema, `validate_judgment`.
- Produces: append-only `judge_responses.jsonl` or shard files with `status=ok|invalid|error`, response payload, contract, latency.

- [ ] **Step 1: Write failing tests for prompt, retries, cache, and validation**

```python
def test_make_prompt_contains_only_question_and_card():
    prompt = MODULE.make_prompt("Q={question}\nJP={decision_card}", blind_job())
    assert "Question ?" in prompt
    assert "Synthèse" in prompt
    assert "rank" not in prompt
    assert "LightGCN" not in prompt

def test_load_done_retries_invalid_and_error_rows(tmp_path): ...
def test_mock_runner_writes_valid_minimal_payload(tmp_path): ...
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_runner.py -q`

Expected: FAIL because the runner does not exist.

- [ ] **Step 3: Implement the dedicated runner**

Reuse the concurrency and append-under-lock pattern from script 68, but keep the new response contract:

```python
{
  "job_id": "sha256(qid|jp_id)",
  "qid": "...",
  "jp_id": "...",
  "status": "ok|invalid|error",
  "invalid_reason": "",
  "judge_contract": {...},
  "response": {"classe": "A", "justification": "..."},
  "latency_seconds": 1.23
}
```

Use `response_format=json_schema`, `temperature=0`, bounded retries, `--retry-non-ok`, `--mock`, `--limit`, shard-specific response paths, and exact judge-contract verification.

The shell runner starts `cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit` under vLLM, waits for `/v1/models`, then executes the Python runner. The Slurm wrapper uses one GPU and writes job-specific logs without mutating another experiment directory.

- [ ] **Step 4: Run tests and mock end-to-end**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_runner.py -q`

Expected: tests pass; the pytest-managed mock fixture gives every input job exactly one terminal response without depending on a pre-existing `/tmp` directory.

- [ ] **Step 5: Commit Task 3**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/scripts/76_run_g7_graded_jp_judge.py \
  05-Technique/benchmark/etape1_embedding_pur/scripts/run_g7_graded_jp_judge_on_cluster.sh \
  05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_g7_graded_jp_judge.sh \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_runner.py
git commit -m "feat: run resumable G7 graded JP judge"
```

### Task 4: Fixed-K aggregation and exact/graded exports

**Files:**
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/77_summarize_g7_graded_jp_eval.py`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_summary.py`

**Interfaces:**
- Consumes: positions parquet, response JSONL/shards, benchmark questions.
- Produces: `graded_jp_detail.csv`, `graded_jp_per_question.csv`, `graded_jp_summary.json`.

- [ ] **Step 1: Write failing tests for fixed denominator, missing cards, and technical incompleteness**

```python
def test_aggregate_uses_fixed_ten_denominator():
    detail, per_q, summary = MODULE.aggregate(positions_10(), responses_for(["A"] * 4 + ["B"] * 2 + ["C"] * 4), questions())
    assert per_q.iloc[0]["score_gradue_at_10"] == 0.5

def test_missing_card_is_non_judgeable_but_missing_response_blocks(): ...
def test_summary_keeps_exact_hit_separate_from_graded_score(): ...
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_summary.py -q`

Expected: FAIL because the summarizer does not exist.

- [ ] **Step 3: Implement deterministic aggregation**

Implement response deduplication by `job_id`, reject conflicting duplicates, synthesize `non_jugeable` only for `card_status=missing`, and fail if any card-present position lacks an `ok` response. Join `gold_jp_ids` only after judgment. Compute:

```python
score = (count_A + 0.5 * count_B) / 10
```

Macro-average over all 754 questions. Export class distribution over 7,540 positions, `non_jugeable_at_10`, exact-hit flags, hashes, and explicit `status=exploratory_internal_evaluation`.

- [ ] **Step 4: Run tests and deterministic rebuild**

Run:

```bash
python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_summary.py -q
python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_m3_judge_final_champions.py -q
```

Expected: all pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/scripts/77_summarize_g7_graded_jp_eval.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_summary.py
git commit -m "feat: aggregate fixed-K G7 graded JP scores"
```

### Task 5: Blind lawyer sample and agreement report

**Files:**
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/78_select_g7_graded_jp_lawyer_audit.py`
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/79_summarize_g7_graded_jp_lawyer_audit.py`
- Test: `05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_lawyer_audit.py`

**Interfaces:**
- Consumes: graded detail, existing Step1 cards, Judilibre corpora, completed lawyer annotations.
- Produces: blind `lawyer_audit_sample.csv`, private `lawyer_audit_key.csv`, `lawyer_evidence.jsonl`, `lawyer_agreement.json`.

- [ ] **Step 1: Write failing tests for deterministic stratification and blinding**

```python
def test_select_sample_uses_target_allocation_and_seed(): ...
def test_blind_export_omits_llm_label_rank_gt_and_g8(): ...
def test_agreement_reweights_stratified_sample_to_population(): ...
def test_gate_requires_weighted_agreement_and_positive_precision(): ...
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_lawyer_audit.py -q`

Expected: FAIL because the audit scripts do not exist.

- [ ] **Step 3: Implement selection and evidence materialization**

Target 25 A, 20 B, 15 C, 15 D, 15 E, 10 `non_jugeable`; redistribute unavailable quotas deterministically. Within strata, balance ranks 1--3/4--10 and exact/non-exact where possible. Store population size, inclusion probability, and sampling weight in the private key only.

The blind CSV contains `case_id`, question, Step1 card, empty `classe_avocat`, and empty `justification_avocat`. Materialize full Judilibre text in evidence JSONL via the existing corpus loader. Do not expose the LLM class in either lawyer-facing artifact.

- [ ] **Step 4: Implement weighted agreement and gate**

Compute population-reweighted metrics with the inverse inclusion probability stored in the private key:

```python
gain_agreement = 1.0 - weighted_mean(abs(gain_llm - gain_lawyer))
positive_precision = weighted_count(llm in {A, B} and lawyer in {A, B}) / weighted_count(llm in {A, B})
positive_recall = weighted_count(llm in {A, B} and lawyer in {A, B}) / weighted_count(lawyer in {A, B})
mean_absolute_gain_error = weighted_mean(abs(gain_llm - gain_lawyer))
```

Also compute the full weighted confusion matrix. Estimate 95% percentile intervals using 2,000 deterministic stratified bootstrap resamples (`seed=42`), resampling with replacement inside each predicted-label stratum and retaining the original sampling weights. Gates are `gain_agreement >= 0.70` and `positive_precision >= 0.85`. If annotations are incomplete, return `status=incomplete_annotations` rather than partial metrics.

- [ ] **Step 5: Run audit tests**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_lawyer_audit.py -q`

Expected: all pass.

- [ ] **Step 6: Commit Task 5**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/scripts/78_select_g7_graded_jp_lawyer_audit.py \
  05-Technique/benchmark/etape1_embedding_pur/scripts/79_summarize_g7_graded_jp_lawyer_audit.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_lawyer_audit.py
git commit -m "feat: prepare and score G7 lawyer audit"
```

### Task 6: E016 orchestration, registry, and runbook

**Files:**
- Create: `05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py`
- Create: `05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_campaign.py`
- Create: `06-Analyses/comparatifs/e016-g7-graded-jp-2026-08-11/README.md`
- Modify: `01-Projet/paper-control/REGISTRE-EXPERIENCES.csv`
- Modify: `01-Projet/paper-control/ETAT-ASSAINISSEMENT.md`
- Modify: `01-Projet/paper-control/SYNC-ASSAINISSEMENT-VERS-PAPIER.md`

**Interfaces:**
- Consumes: Tasks 1--5 commands/artifacts.
- Produces: preflight/status JSON, reproducible cluster commands, Task A state updates.

- [ ] **Step 1: Write failing campaign-state tests**

```python
def test_status_requires_7540_positions_and_no_open_technical_errors(): ...
def test_status_keeps_lawyer_gate_pending_until_100_annotations(): ...
def test_status_never_calls_internal_eval_confirmatory(): ...
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python3 -m pytest 05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_campaign.py -q`

Expected: FAIL because campaign manager does not exist.

- [ ] **Step 3: Implement status and runbook commands**

Provide subcommands `preflight`, `status`, `prepare`, `summarize`, `select-lawyer-audit`. All commands default to the frozen E016 paths from Task 2. `preflight` is read-only and validates source files, expected G7 method, 754 JP question IDs, prompt/schema validity, and DB configuration without fetching cards. `prepare` invokes the evaluation profile from Task 2. Status reads artifacts only and reports gates without launching GPU jobs. The README documents local preparation, train-only calibration, rsync/cluster launch, polling, response retrieval, aggregation, and lawyer handoff.

- [ ] **Step 4: Register E016 without claiming a result**

Append one E016 row with method `G7_graded_JP_LLM_judge`, graph `G7-citation-JJ-cit1-sem025-knn5`, protocol `internal_graded_judge_v1`, status `exploratoire_en_cours`, metrics `jp_score_gradue_at_10;class_distribution;non_jugeable_at_10;lawyer_agreement`, and the E016 README path.

Update Task A state/outgoing channel with planned scope only; do not add score values before the run completes.

- [ ] **Step 5: Run the focused and neighboring test suites**

Run:

```bash
python3 -m pytest \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_contract.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_preparation.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_runner.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_summary.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_lawyer_audit.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_campaign.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g8_semantic_rescue.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_m3_judge_final_champions.py -q
python3 -m py_compile 05-Technique/benchmark/etape1_embedding_pur/scripts/{74_g7_graded_jp_contract,75_prepare_g7_graded_jp_eval,76_run_g7_graded_jp_judge,77_summarize_g7_graded_jp_eval,78_select_g7_graded_jp_lawyer_audit,79_summarize_g7_graded_jp_lawyer_audit,80_manage_g7_graded_jp_campaign}.py
git diff --check
```

Expected: all tests and compilation pass; no whitespace errors.

- [ ] **Step 6: Commit Task 6**

```bash
git add 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py \
  05-Technique/benchmark/etape1_embedding_pur/tests/test_g7_graded_jp_campaign.py \
  06-Analyses/comparatifs/e016-g7-graded-jp-2026-08-11/README.md \
  01-Projet/paper-control/REGISTRE-EXPERIENCES.csv \
  01-Projet/paper-control/ETAT-ASSAINISSEMENT.md \
  01-Projet/paper-control/SYNC-ASSAINISSEMENT-VERS-PAPIER.md
git commit -m "chore: register E016 G7 graded JP campaign"
```

### Task 7: Execute gates 1--3 and prepare the lawyer handoff

**Files:**
- Generate only under: `05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/G7-citation-JJ-cit1-sem025-knn5/eval_rich_retrievable_strict/E016-g7-graded-jp-v1/`
- Update after evidence exists: `06-Analyses/comparatifs/e016-g7-graded-jp-2026-08-11/README.md`
- Modify after evidence exists: Task A state/outgoing channel files from Task 6.

**Interfaces:**
- Consumes: completed implementation and cluster access.
- Produces: complete E016 raw/summary artifacts and 100-case lawyer package.

- [ ] **Step 1: Run local preflight and preparation**

Run:

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py preflight
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py prepare
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py status
```

Expected: 754 questions, 7,540 positions, card-coverage report, frozen manifest, and no Step1 writes.

- [ ] **Step 2: Run train-only real pilot and inspect it**

Run the deterministic train-only preparation:

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/75_prepare_g7_graded_jp_eval.py \
  --profile calibration --question-limit 30 --seed 42 \
  --out-dir 05-Technique/benchmark/etape1_embedding_pur/data/doctrine_v3plus_bench/calibration/E016-g7-graded-jp-v1
```

Launch the real judge on that bounded bundle, inspect every justification, revise the prompt only if necessary, then create a new prompt version/hash and freeze it. Never inspect E016 internal-eval outputs during calibration.

- [ ] **Step 3: Launch and monitor the full G7 judge**

Transfer jobs/scripts/manifests, then launch with:

```bash
sbatch 05-Technique/benchmark/etape1_embedding_pur/scripts/sbatch_g7_graded_jp_judge.sh
```

Poll without blocking longer than 60 seconds per interaction, retry technical failures with `--retry-non-ok`, and require one `ok` response per card-present unique job.

- [ ] **Step 4: Retrieve and aggregate**

Retrieve response shards, verify manifest contracts, then run:

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py summarize
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py status
```

Run the summarizer twice and confirm byte-identical summary outputs or equal hashes.

- [ ] **Step 5: Materialize the blind lawyer package**

Run:

```bash
python3 05-Technique/benchmark/etape1_embedding_pur/scripts/80_manage_g7_graded_jp_campaign.py select-lawyer-audit --seed 42 --sample-size 100
```

Verify 100 unique cases and blind columns, materialize full texts, and leave annotation columns empty for the lawyer.

- [ ] **Step 6: Update evidence state without overclaiming**

Record the G7-only graded result as exploratory internal evidence, distinguish LLM score from exact Hit@10, mark lawyer validation pending, and do not begin hyperparameter changes for G7.

- [ ] **Step 7: Commit generated reports and Task A state only**

Do not commit large/raw ignored data. Commit the report, manifests/summaries intended for version control, experiment registry/status, and outgoing paper channel with an evidence-scoped message.

---

## Follow-up plan boundary: Chantier 3

After Task 7 and the lawyer gate, write a separate implementation plan for the diagnostic scripts that join G7 A--E outcomes with exact Ground Truth, cosine rankings, G8 relation types, candidate coverage, and graph distance. Do not implement G7 changes in this plan because their hypotheses depend on E016 outcomes.
