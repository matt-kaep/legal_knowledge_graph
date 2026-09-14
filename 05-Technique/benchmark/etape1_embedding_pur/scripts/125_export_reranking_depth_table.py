#!/usr/bin/env python3
"""Export the 42 frozen reranking depth points without model computation."""
import argparse
import csv
import json
import math
from pathlib import Path
import importlib.util

spec = importlib.util.spec_from_file_location('completion', Path(__file__).with_name('124_complete_judge_tables.py'))
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def export(repo, data, tidy, output):
    common.checked(tidy, '27731e70636902acb147a9f48b27652191ede3f728a44be4f859e4bdcdaf0162')
    manifest = common.checked(repo / common.BASE / 'configs/b2_reranking_comparable_a3_k70_article192_output512_execution_v2.json', 'c33f4474aa47224a884eabc7e9f616f7b7b7cd1d6db49113619449dce1522586')
    root = data / 'doctrine_v3plus_bench' / common.RERANK
    receipt = common.checked(root / 'aggregate/aggregation_receipt.json', 'abdf7471e39c18cf7d68c94399031ca64b489c0ed2f9f6b428eef92b0a470153')
    receipt_data = common.read(receipt)
    per_seed = common.checked(root / 'aggregate/per_seed_exact_metrics.csv', '7ba99db5b979c56b8685e6bb231231820716b4c8d6f70e9edd7daa77f4f9b8d6')
    mean = common.checked(root / 'aggregate/seed_mean_exact_metrics.csv', '2fdf5016be10bd2d7116e0f5f02ec78907ae81b18ecd5ed33655a6de055c0a04')
    rows = list(csv.DictReader(tidy.open()))
    seeds = list(csv.DictReader(per_seed.open()))
    means = list(csv.DictReader(mean.open()))
    if len(rows) != 42 or len(seeds) != 70 or len(means) != 42:
        raise ValueError('incomplete depth matrix')
    ranking_proofs = {}
    for src in receipt_data['sources']:
        name = Path(src['root']).name
        material = root / 'shards' / name / 'materialized'
        rec = common.checked(material / 'materialization_receipt.json', src['materialization_receipt_sha256'])
        met = common.checked(material / 'exact_metrics.csv', src['metrics_sha256'])
        rank = common.checked(material / 'rankings_top10.parquet', common.read(rec)['rankings_top10.parquet'])
        ranking_proofs[name] = {'path': str(rank), 'sha256': common.sha(rank), 'receipt_path': str(rec), 'receipt_sha256': common.sha(rec), 'metrics_sha256': common.sha(met)}
    mean_index = {(r['family'], r['modality'], int(r['k_in'])): r for r in means}
    output_rows = []
    seen = set()
    for row in rows:
        family = row['method']
        task = 'article' if row['target'] == 'statutory_articles' else 'jp'
        key = (family, task, int(row['k_in']))
        if key in seen or key not in mean_index:
            raise ValueError('unexpected/duplicate depth point')
        seen.add(key)
        cells = [r for r in seeds if (r['family'], r['modality'], int(r['k_in'])) == key]
        expected = {'42', '43', '44'} if family == 'lightgcn' else {''}
        if len(cells) != len(expected) or {r['replay_seed'] for r in cells} != expected:
            raise ValueError('incomplete seeds')
        value = math.fsum(float(r['hit_at_10']) for r in cells) / len(cells)
        if not math.isclose(value, float(row['nhit_at_10']), abs_tol=1e-14) or not math.isclose(value, float(mean_index[key]['hit_at_10']), abs_tol=1e-14):
            raise ValueError('tidy score differs from sealed seed aggregates')
        names = [f'lightgcn-seed{s}' for s in (42, 43, 44)] if family == 'lightgcn' else [family]
        graph = 'G6-AA' if family == 'ppr' else 'G6' if family == 'lightgcn' else 'not_applicable'
        if family == 'ppr' and row['graph_label'] != 'PPR G6-AA':
            raise ValueError('PPR reranking must use G6-AA for both tasks')
        output_rows.append({'task': row['target'], 'retriever': {'cosine': 'BGE-M3 cosine', 'ppr': 'PPR', 'lightgcn': 'LightGCN'}[family],
            'graph_label': graph, 'k_in': int(row['k_in']), 'k_out': 10,
            'NHit@10_before': row['retriever_nhit_at_10'], 'NHit@10_after': row['nhit_at_10'],
            'source_ranking_hash': ';'.join(ranking_proofs[n]['sha256'] for n in names),
            'status': 'exploratory_internal_A3', 'seeds': '42;43;44' if family == 'lightgcn' else '', 'questions': 754})
    expected_keys = {(m, t, k) for m in ('cosine', 'ppr', 'lightgcn') for t in ('article', 'jp') for k in range(10, 71, 10)}
    if seen != expected_keys:
        raise ValueError('depths must be exactly 10..70')
    output.mkdir(parents=True, exist_ok=False)
    target = output / 'reranking_depth.csv'
    common.write_csv(target, output_rows)
    provenance = {'experiment_id': 'E029', 'status': 'exploratory_internal_A3', 'model_calls': 0,
        'questions': 754, 'rows': 42, 'k_in': list(range(10, 71, 10)), 'k_out': 10,
        'sources': {str(p): common.sha(p) for p in (tidy, manifest, receipt, per_seed, mean)},
        'rankings': ranking_proofs, 'files': {target.name: common.sha(target)},
        'script_sha256': common.sha(Path(__file__)),
        'aggregation': 'Arithmetic mean of independently evaluated seeds 42,43,44 for LightGCN only.',
        'ppr_jp': 'G6-AA reranking pools and their own raw baseline; not G7-AA from the main retrieval table.'}
    common.write_json(output / 'audit_receipt.json', provenance)
    (output / 'README.md').write_text('''---
type: result-export
status: exploratory
owner: assainissement
---

# Reranking depth on the frozen French legal retrieval benchmark

`reranking_depth.csv` contains 42 exact results: two tasks, three retrievers,
input depths 10,20,30,40,50,60,70 and output depth 10, on 754 questions.
No model, retrieval, reranking or Judge call was made for this export.

PPR uses G6-AA for both tasks. Its Judicial Decisions baseline is
0.22822723253757737; it is distinct from G7-AA in the main retrieval table.
LightGCN uses G6 and the arithmetic mean of seeds 42,43,44, checked against
the sealed per-seed metrics. Semicolon-separated ranking hashes follow
that seed order; each identifies the full multi-depth source parquet,
filtered by task and input depth, rather than a fabricated mean ranking.
The source tidy CSV supplies the unrounded before/after values.
The original figure files are unchanged. There are no points beyond 70.

These are exploratory internal-evaluation results. `audit_receipt.json`
records the source paths, exact SHA-256 values and aggregation rules.

Suggested description: “Exact normalized hit at rank 10 before and after
LLM reranking, as a function of input-pool depth, on 754 evaluation queries.
Articles and judicial decisions are evaluated separately. LightGCN results
average three frozen replay seeds.”
''')
    common.write_json(output / 'package_hashes.json', {p.name: common.sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({'output': str(output), 'csv_sha256': common.sha(target), 'receipt_sha256': common.sha(output / 'audit_receipt.json')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--tidy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    export(a.repo, a.data, a.tidy, a.out)
