#!/usr/bin/env python3
"""Audit frozen Judge evidence and prepare the six authorized table supplements.

No inference in this program. GPU execution uses the immutable r6 launchers.
All output directories are new; historical scores and rankings are read-only.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

BASE = Path('05-Technique/benchmark/etape1_embedding_pur')
R6 = 'b2_llm_as_judge_a3_retry_v3_preflight_r6.json'
R6_SHA = 'b4bb39dcbfff639f3c2cc5a3aecc637f2ff6475c6f1e17dcf61a91a6472746cd'
V2 = 'b2_llm_as_judge_a3_execution_v2_gpu70.json'
V2_SHA = '7eb5d08758b3f833cfdb0f2c880066b8cbbe481cd2c252f20ed5582c8e3ea16f'
EVAL_SHA = '850adae1e411cd83e637ea86061aa742b3c4cd166ad3262ed6a2b8c10b9f5d59'
ROOT = 'doctrine_v3plus_bench/_judge_table_completion_a3_v1_20260914'
RERANK = '_campaign_b2_e029_a3_k70_article192_output512_execution_v2_20260909'
STATUS = 'exploratory_pending_blind_legal_audit'
G6 = 'G6-citation-AA-knn5'
G7 = 'G7-citation-AA-cit1-sem025-knn5'
MISSING = {'cosine-jp', 'ppr-article', 'ppr-jp', 'ppr-g6-jp',
           'lightgcn-seed42-article', 'lightgcn-seed42-jp'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def checked(path, expected):
    if sha(path) != expected:
        raise ValueError(f'hash mismatch: {path}')
    return Path(path)


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows, fields=None):
    with Path(path).open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('.', '_'), Path(__file__).with_name(filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def score_responses(jobs, responses):
    """Independent fixed-K audit: errors cannot become zero-valued judgments."""
    by_id = {r['job_id']: r for r in responses}
    if len(by_id) != len(responses) or set(by_id) != {j['job_id'] for j in jobs}:
        raise ValueError('response identity/coverage mismatch')
    positions, seen, gains = defaultdict(set), defaultdict(set), []
    rubric = {'A': 1, 'B': .5, 'C': 0, 'D': 0, 'E': 0, 'non_jugeable': 0}
    for job in sorted(jobs, key=lambda j: (j['qid'], j['position'])):
        q, rank = job['qid'], job['position']
        if rank in positions[q]:
            raise ValueError('duplicate position')
        positions[q].add(rank)
        response = by_id[job['job_id']]
        if job.get('zero_slot') is True:
            if response['status'] != 'zero_slot' or response.get('judgment') is not None:
                raise ValueError('unresolved-slot contract mismatch')
            gain = 0
        else:
            judgment = response.get('judgment')
            if response['status'] != 'ok' or not isinstance(judgment, dict):
                raise ValueError('technical error is not a judgment')
            if set(judgment) != {'classe', 'justification'} or judgment['classe'] not in rubric or not str(judgment['justification']).strip():
                raise ValueError('invalid rubric response')
            candidate = job['candidate_id_internal']
            gain = 0 if candidate in seen[q] else rubric[judgment['classe']]
            seen[q].add(candidate)
        gains.append(gain)
    if len(positions) != 754 or any(r != set(range(1, 11)) for r in positions.values()) or len(jobs) != 7540:
        raise ValueError('expected 754 questions x positions 1..10')
    return math.fsum(gains) / 7540


def seed_means(rows):
    groups = defaultdict(list)
    for r in rows:
        if r['retriever'] == 'LightGCN' and r['technical_status'] == 'complete_clean':
            groups[(r['task'], r['stage'], r['graph_label'])].append(r)
    means = []
    for _, items in sorted(groups.items()):
        if len(items) != 3 or {str(r['seeds']) for r in items} != {'42', '43', '44'}:
            continue
        row = dict(items[0])
        row.update(condition='lightgcn-mean-' + row['stage'] + '-' + row['task'], seeds='42;43;44',
                   score=math.fsum(float(r['score']) for r in items) / 3,
                   source_ranking_hash=';'.join(r['source_ranking_hash'] for r in sorted(items, key=lambda r: str(r['seeds']))),
                   ranking_identity_hash=';'.join(r['ranking_identity_hash'] for r in sorted(items, key=lambda r: str(r['seeds']))))
        means.append(row)
    return means


class Evidence:
    def __init__(self, repo, data):
        import pandas as pd
        import numpy as np
        self.repo, self.data, self.pd = repo, data, pd
        self.bench = data / 'doctrine_v3plus_bench'
        self.parent_path = checked(repo / BASE / 'configs' / R6, R6_SHA)
        self.parent = read(self.parent_path)
        for ref in self.parent['prompts'].values():
            checked(repo / ref['path'], ref['sha256'])
        for ref in self.parent['code_bundle'].values():
            checked(repo / ref['path'], ref['sha256'])
        self.questions_path = checked(self.bench / 'eval_rich_retrievable_strict/bench_global.json', EVAL_SHA)
        self.freezer = module('111_freeze_b2_e030_judge_jobs.py')
        self.questions = self.freezer._load_questions(self.questions_path)
        self.cards, self.orders, self.cards_path, self.order_path = {}, {}, {}, {}
        self.cards_hash, self.order_hash = {}, {}
        for task, cf, ch, order, oh in [
            ('article', 'articles_all.parquet', '79ed64d48bbd066083d03ab7e8d8b728ed99975c9ba71cda8a98346c19025f6b', 'articles_order_all.npy', '06f5551d6a38f8f7887d760461b7f7460b2e738e062a8bed4b9c598b34c0fbb0'),
            ('jp', 'jp_summaries_penal.parquet', 'e63e71cd7fea04b461bf67d85ed452f922f1967e5659e5082e7925f5c2bd810c', 'jp_summary_order.npy', '985e83bfbee610117fcf8873b8a58eb61ffd1e35bf1423f95985a375a3012c5b')]:
            self.cards_path[task] = checked(data / cf, ch)
            self.order_path[task] = checked(data / order, oh)
            self.cards_hash[task], self.order_hash[task] = ch, oh
            self.cards[task] = self.freezer._load_cards(data / cf, task)
            self.orders[task] = set(map(str, np.load(data / order, allow_pickle=True).tolist()))
        self.ranking_paths = {}
        patterns = [
            '_campaign_b2_e030_a3_top10_projection_preflight_v1_20260909/*parquet',
            '_campaign_b2_direct_llm_a3_v1_20260907/*/materialized/rankings_top10.parquet',
            RERANK + '/shards/*/materialized/rankings_top10.parquet',
            '_judge_table_completion_a3_v1_20260914/inputs/*parquet']
        for pattern in patterns:
            for path in self.bench.glob(pattern):
                self.ranking_paths[sha(path)] = path

    def validate_jobs(self, shard):
        path = checked(self.data / shard['jobs']['path'], shard['jobs']['sha256'])
        jobs = lines(path)
        if len(jobs) != 7540 or len({j['job_id'] for j in jobs}) != 7540:
            raise ValueError('job coverage/identity mismatch')
        task = shard['modality']
        source_sha = jobs[0]['source_ranking_sha256']
        ranking_path = self.ranking_paths[source_sha]
        f = self.pd.read_parquet(ranking_path)
        for col, val in jobs[0].get('ranking_filters', {}).items():
            f = f.loc[f[col].astype(str).eq(str(val))]
        if 'modality' in f:
            f = f.loc[f.modality.isin(['art', 'article'] if task == 'article' else ['jp', 'jurisprudence'])]
        expected = {(str(r.qid), int(r.rank)): None if self.pd.isna(r.item_id) or str(r.item_id).strip() == '' else str(r.item_id) for r in f.itertuples()}
        actual = {(j['qid'], int(j['position'])): j['candidate_id_internal'] for j in jobs}
        if len(expected) != len(f) or expected != actual:
            raise ValueError(f'ranking identity mismatch: {shard["id"]}')
        for j in jobs:
            unhashed = {k: v for k, v in j.items() if k != 'job_id'}
            if digest(unhashed) != j['job_id']:
                raise ValueError('job content hash mismatch')
            if j['source_ranking_sha256'] != source_sha or j['source_questions_sha256'] != EVAL_SHA or j['question'] != self.questions[j['qid']]:
                raise ValueError('question/ranking provenance mismatch')
            if j['model_id'] != self.parent['model']['id'] or j['model_revision'] != self.parent['model']['revision'] or j['temperature'] != 0 or j['prompt_sha256'] != self.parent['prompts'][task]['sha256']:
                raise ValueError('scientific contract mismatch')
            if j['source_cards_sha256'] != self.cards_hash[task] or j['candidate_order_sha256'] != self.order_hash[task]:
                raise ValueError('card/order hash mismatch')
            if j.get('zero_slot'):
                if j['document'] is not None or j['candidate_id_internal'] is not None:
                    raise ValueError('invalid zero slot')
            elif j['candidate_id_internal'] not in self.orders[task] or j['document'] != self.cards[task][j['candidate_id_internal']]:
                raise ValueError('candidate card differs from frozen source')
        if set(j['qid'] for j in jobs) != set(self.questions):
            raise ValueError('A3 question coverage mismatch')
        context = checked(self.data / shard['context_audit']['path'], shard['context_audit']['sha256'])
        cr = read(context)
        if cr['compatible'] is not True or cr['files']['jobs.jsonl'] != sha(path):
            raise ValueError('context receipt mismatch')
        graph = ''
        if 'selected_graph_version' in f:
            values = set(f.selected_graph_version)
            if len(values) != 1:
                raise ValueError('mixed graphs in one condition')
            graph = str(next(iter(values)))
        return jobs, {'source_ranking_path': str(ranking_path), 'source_ranking_hash': source_sha,
                      'ranking_identity_hash': digest(sorted((q, p, c) for (q, p), c in actual.items())),
                      'observed_graph': graph, 'jobs_hash': sha(path), 'context_hash': sha(context)}

    def audit_existing(self):
        auditor = module('123_aggregate_b2_e030_retry_v3.py')
        old_root = self.data / self.parent['outputs']['root']
        receipt = checked(old_root / 'aggregate_audit_r1/aggregation_receipt.json', '14cfda37fcce41bc4bf9b0e0a49da21b85b3d9ca5517655e4f90be950136d413')
        old_csv = checked(old_root / 'aggregate_audit_r1/per_shard_audit.csv', read(receipt)['files']['per_shard_audit.csv'])
        previous = {r['shard']: r for r in csv.DictReader(old_csv.open())}
        evidence = []
        for s in self.parent['shards']:
            jobs, proof = self.validate_jobs(s)
            row = auditor.audit_shard(shard=s, data_root=self.data, output_root=old_root,
                                     manifest_sha=R6_SHA, adapter_sha=sha(Path(__file__).with_name('122_materialize_b2_e030_retry_v3.py')))
            if row['technical_verdict'] != 'complete_clean':
                raise ValueError(f'old shard no longer clean: {s["id"]}')
            for key in ['responses_sha256', 'run_receipt_sha256', 'base_materialization_receipt_sha256', 'retry_materialization_receipt_sha256']:
                if row[key] != previous[s['id']][key]:
                    raise ValueError('old evidence changed since audited receipt')
            score = score_responses(jobs, lines(old_root / 'shards' / s['id'] / 'responses.jsonl'))
            if not math.isclose(score, float(row['judge_score_at_10']), abs_tol=1e-14):
                raise ValueError('stored score differs from independently recomputed gains')
            evidence.append({'shard': s['id'], 'score': score, **proof, **row})
        return evidence


def prepare(args):
    ev = Evidence(args.repo, args.data)
    root = args.data / ROOT
    root.mkdir(parents=True, exist_ok=False)
    old = ev.audit_existing()
    write_json(root / 'reused_evidence_audit.json', old)
    v2 = read(checked(args.repo / BASE / 'configs' / V2, V2_SHA))
    selected = [copy.deepcopy(s) for s in v2['shards'] if s['id'] in MISSING]
    if len(selected) != 5:
        raise ValueError('expected exactly five previously frozen missing conditions')
    raw = checked(ev.bench / '_campaign_b1_a3_ppr_scoped_missing_replays_v2_20260908/ppr_rankings_top100.parquet', 'afd1aa5c70c618cd8b43248e91b14266fc5c72507cfc865553dd2cf04cbc14ec')
    f = ev.pd.read_parquet(raw)
    f = f.loc[(f.selected_graph_version == G6) & (f.modality == 'jp') & (f['rank'] <= 10)].copy()
    (root / 'inputs').mkdir()
    projected = root / 'inputs/ppr_g6_jp_top10.parquet'
    f.to_parquet(projected, index=False)
    ev.ranking_paths[sha(projected)] = projected
    write_json(projected.with_suffix('.receipt.json'), {'source': str(raw), 'source_sha256': sha(raw), 'output_sha256': sha(projected), 'graph': G6, 'filter': 'modality=jp;rank<=10', 'model_calls': 0})
    output = root / 'inputs/ppr_g6_jp.jsonl'
    frozen = ev.freezer.freeze_jobs(rankings_path=projected, questions_path=ev.questions_path,
        texts_path=ev.cards_path['jp'], candidate_order_path=ev.order_path['jp'],
        prompt_path=args.repo / ev.parent['prompts']['jp']['path'], output_path=output,
        family='ppr_g6', modality='jp', model_id=ev.parent['model']['id'], model_revision=ev.parent['model']['revision'])
    import subprocess
    import sys
    context_root = root / 'inputs/ppr_g6_jp_context256'
    subprocess.run([sys.executable, str(Path(__file__).with_name('114_audit_b2_e030_context.py')),
        '--jobs', str(output), '--article-prompt', str(args.repo / ev.parent['prompts']['article']['path']),
        '--jp-prompt', str(args.repo / ev.parent['prompts']['jp']['path']), '--model-snapshot', ev.parent['model']['snapshot'],
        '--max-model-tokens', '16384', '--max-output-tokens', '256', '--out-dir', str(context_root)], check=True)
    context_path = context_root / 'context_audit_receipt.json'
    selected.append({'id': 'ppr-g6-jp', 'modality': 'jp', 'jobs': {'path': str(output.relative_to(args.data)), 'sha256': frozen['jobs_sha256'], 'count': 7540},
                     'context_audit': {'path': str(context_path.relative_to(args.data)), 'sha256': sha(context_path)}})
    inventory = []
    for i, s in enumerate(sorted(selected, key=lambda s: s['id'])):
        jobs, proof = ev.validate_jobs(s)
        expected_graph = G7 if s['id'] == 'ppr-jp' else G6 if s['id'].startswith('ppr') else None
        if expected_graph and proof['observed_graph'] != expected_graph:
            raise ValueError('wrong PPR graph')
        s['vllm_port'] = 18501 + i
        s['provenance'] = proof
        inventory.append({'condition': s['id'], 'decision': 'calculate', 'questions': 754, 'positions': len(jobs), 'port': s['vllm_port'], **proof})
    if {s['id'] for s in selected} != MISSING:
        raise ValueError('unauthorized condition')
    manifest = copy.deepcopy(ev.parent)
    manifest.update(manifest_id='judge-table-completion-a3-v1-2026-09-14', campaign_id='judge-table-completion-a3-v1-2026-09-14',
                    status='frozen_preflight_only', shards=selected,
                    parent={'path': str(BASE / 'configs' / R6), 'sha256': R6_SHA},
                    outputs={'root': ROOT, 'immutable': True},
                    retry_policy={'reuse_r6_scores': 'only after fresh hashes, ranking identity and independent gain audit', 'reuse_v2_scores': False, 'http_failure': 'fail_closed_no_score'},
                    post_execution_audit={'conditions_total': 23, 'reused': 17, 'new': 6, 'questions': 754, 'positions': 7540, 'lightgcn_seed_mean': 'exactly clean seeds 42,43,44 per task and stage'},
                    execution_gate={'smoke_test': 'required; model identity only; no judgments', 'full_submission': 'authorized by user after successful smoke', 'smoke_anomaly': 'stop'},
                    preparation={'script_sha256': sha(Path(__file__)), 'model_calls': 0, 'reused_evidence_audit_sha256': sha(root / 'reused_evidence_audit.json')})
    manifest['service_identity_contract']['ports'] = {s['id']: s['vllm_port'] for s in selected}
    write_json(root / 'judge_table_completion_manifest.json', manifest)
    write_csv(root / 'missing_conditions.csv', inventory)
    write_json(root / 'preparation_receipt.json', {'status': 'pass', 'model_calls': 0, 'reused_clean': len(old), 'new_conditions': len(selected), 'files': {p.name: sha(p) for p in root.iterdir() if p.is_file()}})
    print(json.dumps({'root': str(root), 'manifest_sha256': sha(root / 'judge_table_completion_manifest.json')}))


def table_row(shard, proof, parent, score='', technical_status='pending'):
    sid, task = shard['id'], shard['modality']
    reranked = sid.startswith('reranked-')
    method = 'LightGCN' if 'lightgcn' in sid else 'PPR' if 'ppr' in sid else 'BGE-M3 cosine' if 'cosine' in sid else 'Direct LLM'
    seed = next((str(s) for s in (42, 43, 44) if f'seed{s}' in sid), '')
    graph = 'G6' if method == 'LightGCN' else 'G7-AA' if sid == 'ppr-jp' else 'G6-AA' if method == 'PPR' else 'not_applicable'
    stage = 'reranked_top10' if reranked else 'direct_generation' if method == 'Direct LLM' else 'raw_retrieval'
    return {'condition': sid, 'task': 'statutory_articles' if task == 'article' else 'judicial_decisions',
            'retriever': method, 'graph_label': graph, 'stage': stage,
            'comparison_role': 'reranking_input' if sid == 'ppr-g6-jp' else 'reranking_output' if reranked else 'main_table',
            'k_in': 70 if reranked else '', 'k_out': 10, 'seeds': seed,
            'score': score, 'questions': 754, 'positions': 7540,
            'source_ranking_hash': proof['source_ranking_hash'], 'ranking_identity_hash': proof['ranking_identity_hash'],
            'model': parent['model']['id'], 'revision': parent['model']['revision'], 'temperature': 0,
            'prompt_hash': parent['prompts'][task]['sha256'], 'rubric_hash': digest(parent['grading']),
            'technical_status': technical_status, 'status': STATUS}


def check_run_identity(run, manifest):
    for key, expected in [('model_id', manifest['model']['id']), ('model_revision', manifest['model']['revision']),
                          ('served_model_id', manifest['model']['id'] + '@' + manifest['model']['revision'])]:
        if run.get(key) != expected:
            raise ValueError('run model identity mismatch')


def audit(args):
    ev = Evidence(args.repo, args.data)
    manifest_path = checked(args.data / ROOT / 'judge_table_completion_manifest.json', args.manifest_sha)
    manifest = read(manifest_path)
    if {s['id'] for s in manifest['shards']} != MISSING:
        raise ValueError('unexpected completion conditions')
    for key in ['model', 'prompts', 'grading', 'visibility_contract']:
        if manifest[key] != ev.parent[key]:
            raise ValueError('changed scientific contract')
    output = args.out
    if output is None or output.exists():
        raise ValueError('audit requires a new --out directory')
    previous = {r['shard']: r for r in ev.audit_existing()}
    old_root = args.data / ev.parent['outputs']['root']
    rows, proofs = [], []
    for shard in ev.parent['shards']:
        p = previous[shard['id']]
        run = read(old_root / 'shards' / shard['id'] / 'run_receipt.json')
        check_run_identity(run, ev.parent)
        rows.append(table_row(shard, p, ev.parent, p['score'], 'complete_clean'))
        proofs.append(p)
    adapter = module('122_materialize_b2_e030_retry_v3.py')
    auditor = module('123_aggregate_b2_e030_retry_v3.py')
    new_root = args.data / ROOT
    for shard in manifest['shards']:
        jobs, p = ev.validate_jobs(shard)
        run_root = new_root / 'shards' / shard['id']
        receipt = run_root / 'run_receipt.json'
        if not receipt.exists():
            if args.require_complete:
                raise ValueError(f'incomplete shard: {shard["id"]}')
            rows.append(table_row(shard, p, manifest))
            proofs.append({'shard': shard['id'], 'technical_verdict': 'pending', **p})
            continue
        run = read(receipt)
        check_run_identity(run, manifest)
        if run['port'] != shard['vllm_port'] or run['listener'] != '127.0.0.1':
            raise ValueError('service port mismatch')
        models = read(run_root / 'v1_models.json')
        if run['served_model_id'] not in [x['id'] for x in models['data']]:
            raise ValueError('served identity not in model endpoint receipt')
        responses = lines(run_root / 'responses.jsonl')
        value = score_responses(jobs, responses)
        if not (run_root / 'materialized').exists():
            adapter.materialize_retry_v3(jobs, responses, run_root / 'materialized')
        checked_row = auditor.audit_shard(shard=shard, data_root=args.data, output_root=new_root,
            manifest_sha=args.manifest_sha, adapter_sha=sha(Path(__file__).with_name('122_materialize_b2_e030_retry_v3.py')))
        if checked_row['technical_verdict'] != 'complete_clean' or not math.isclose(value, float(checked_row['judge_score_at_10']), abs_tol=1e-14):
            raise ValueError('new shard failed independent audit')
        rows.append(table_row(shard, p, manifest, value, 'complete_clean'))
        proofs.append({**p, **checked_row, 'v1_models_sha256': sha(run_root / 'v1_models.json')})
    means = seed_means(rows)
    if args.require_complete and (len(rows) != 23 or len(means) != 4):
        raise ValueError('incomplete 23-condition matrix or three-seed means')
    output.mkdir(parents=True)
    (output / 'shards').mkdir()
    for row in rows:
        write_csv(output / 'shards' / (row['condition'] + '.csv'), [row])
    write_csv(output / 'judge_scores_by_condition.csv', rows)
    write_csv(output / 'lightgcn_three_seed_means.csv', means, list(rows[0]))
    summary = [r for r in rows if r['retriever'] != 'LightGCN'] + means
    write_csv(output / 'judge_scores_for_paper.csv', summary)
    write_json(output / 'evidence_manifest.json', {'experiment_id': 'E030', 'parent_manifest_sha256': R6_SHA,
        'completion_manifest_sha256': args.manifest_sha, 'proofs': proofs, 'grading': manifest['grading'],
        'aggregation_script_sha256': sha(Path(__file__)), 'source_root': str(args.data)})
    receipt = {'status': STATUS, 'technical_verdict': 'complete_clean' if all(r['technical_status'] == 'complete_clean' for r in rows) else 'pending_six_conditions',
        'conditions': len(rows), 'complete_clean': sum(r['technical_status'] == 'complete_clean' for r in rows),
        'questions_per_condition': 754, 'positions_per_condition': 7540,
        'lightgcn_complete_means': len(means), 'lightgcn_seeds': [42, 43, 44],
        'v2_scores_reused': False, 'ppr_jp_conditions_distinct': True,
        'legal_audit_required': 'lawyer_agreement.json',
        'files': {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output / 'audit_receipt_public.json', receipt)
    (output / 'README.md').write_text('''---
type: result-export
status: exploratory
owner: assainissement
---

# LLM-as-a-Judge results on the frozen legal retrieval benchmark

All scores remain exploratory until a blind legal audit is completed.
Human agreement must be recorded in `lawyer_agreement.json` before legal
validation or comparative conclusions. Exact retrieval metrics are separate.

`judge_scores_for_paper.csv` contains direct generation, raw retrieval and
reranked top-10 rows in explicit stages. `comparison_role=reranking_input`
identifies PPR G6-AA Judicial Decisions; PPR G7-AA belongs to the main table.
These are distinct conditions and must never be substituted or averaged.
Other raw retrieval rows also serve as their own reranking baselines.

`judge_scores_by_condition.csv` and `shards/` retain each seed separately.
LightGCN means require precisely the clean seeds 42,43,44 for one task and
stage. Semicolon-separated source hashes in means follow that seed order;
they identify three rankings, never a synthetic average ranking.
`lightgcn_three_seed_means.csv` contains only means that passed this gate.
Blank scores mean pending, never zero. The zero Direct LLM decision score
comes from frozen unresolved reference slots, not completed adverse judgments.

The Judge sees only the question and full article text or decision synthese.
Model, revision, prompt hash and rubric hash are included in every row.
Gains are A=1, B=0.5, C/D/E/non_jugeable=0, with fixed K=10. Repeated
candidates after their first occurrence receive zero gain. Frozen unresolved
slots remain zero-valued positions; HTTP errors never become labels or zeros.
Judging the reranked output uses only input depth 70, output depth 10.
No Judge depth sweep, retriever training or new reranking was performed.

The public receipt hashes every result and the evidence manifest supplies
the exact remote source paths, ranking identities and ledger hashes.
''')
    write_json(output / 'package_hashes.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({'out': str(output), 'complete_clean': receipt['complete_clean'], 'receipt_sha256': sha(output / 'audit_receipt_public.json')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'audit'])
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--manifest-sha')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
    else:
        audit(args)


if __name__ == '__main__':
    main()
