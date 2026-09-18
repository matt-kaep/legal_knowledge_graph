#!/usr/bin/env python3
"""Submit exactly the six authorized missing conditions after identity smoke."""
import argparse
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess

spec = importlib.util.spec_from_file_location('completion', Path(__file__).with_name('124_complete_judge_tables.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def submit(repo, data, manifest_sha):
    root = data / c.ROOT
    manifest_path = c.checked(root / 'judge_table_completion_manifest.json', manifest_sha)
    manifest = c.read(manifest_path)
    if {s['id'] for s in manifest['shards']} != c.MISSING or len(manifest['shards']) != 6:
        raise ValueError('only the six authorized missing conditions may be submitted')
    smoke = root / 'smoke/cosine-jp/smoke_receipt.json'
    sr = c.read(smoke)
    if sr['status'] != 'pass' or sr['manifest_sha256'] != manifest_sha or sr['model_calls'] != 0 or sr['judgment_rows'] != 0:
        raise ValueError('smoke gate failed')
    c.check_run_identity(sr, manifest)
    if sr['port'] != manifest['service_identity_contract']['ports']['cosine-jp']:
        raise ValueError('smoke port mismatch')
    for ref in manifest['code_bundle'].values():
        c.checked(repo / ref['path'], ref['sha256'])
    for s in manifest['shards']:
        for key in ('jobs', 'context_audit'):
            c.checked(data / s[key]['path'], s[key]['sha256'])
        if (root / 'shards' / s['id']).exists():
            raise FileExistsError('refusing to resubmit a shard with existing outputs')
    submitted = root / 'submission'
    submitted.mkdir(exist_ok=False)
    env = dict(os.environ, LKG_REPO=str(repo), LKG_DATA_ROOT=str(data),
        E030_EXECUTION_MANIFEST=str(manifest_path), E030_EXECUTION_MANIFEST_SHA=manifest_sha,
        E030_SMOKE_RECEIPT=str(smoke), E030_ALLOW_FULL_RETRY='1')
    launcher = repo / manifest['code_bundle']['v3_launcher']['path']
    jobs = []
    for s in sorted(manifest['shards'], key=lambda s: s['id']):
        job = subprocess.check_output(['sbatch', '--parsable', '--job-name=judge-' + s['id'],
            '--output=' + str(root / (s['id'] + '-%j.out')), str(launcher)],
            env=dict(env, E030_SHARD=s['id']), text=True).strip().split(';')[0]
        if not job.isdigit():
            raise ValueError('unexpected Slurm job identity')
        row = {'condition': s['id'], 'job_id': job, 'port': s['vllm_port'], 'jobs_sha256': s['jobs']['sha256']}
        c.write_json(submitted / (s['id'] + '.json'), row)
        jobs.append(row)
    audit_script = Path(__file__).with_name('124_complete_judge_tables.py')
    final = repo / 'results/benchmark-a3-b1/judge-table-completion-a3-v1/final'
    command = ['env', 'LKG_REPO=' + str(repo), 'LKG_DATA_ROOT=' + str(data),
        '/home/ids/kaeppelin-22/work/.venv-benchmark/bin/python', str(audit_script), 'audit',
        '--repo', str(repo), '--data', str(data), '--out', str(final),
        '--manifest-sha', manifest_sha, '--require-complete']
    # Check the aggregation program before execution as well as the input manifest.
    audit_sha = c.sha(audit_script)
    wrapper = 'test "$(sha256sum ' + shlex.quote(str(audit_script)) + ' | cut -d\" \" -f1)" = ' + shlex.quote(audit_sha) + ' && ' + shlex.join(command)
    aggregate_job = subprocess.check_output(['sbatch', '--parsable', '--partition=CPU', '--cpus-per-task=4',
        '--mem=16G', '--time=00:30:00', '--job-name=judge-tables-audit',
        '--dependency=afterok:' + ':'.join(r['job_id'] for r in jobs),
        '--output=' + str(root / 'aggregation-%j.out'), '--wrap', wrapper], text=True).strip().split(';')[0]
    receipt = {'manifest_sha256': manifest_sha, 'smoke_receipt_sha256': c.sha(smoke), 'jobs': jobs,
        'aggregation_job': aggregate_job, 'aggregation_script_sha256': audit_sha,
        'output': str(final), 'submission_script_sha256': c.sha(Path(__file__)), 'status': 'submitted_no_new_scores'}
    c.write_json(root / 'submission_receipt.json', receipt)
    print(c.canonical(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest-sha', required=True)
    a = p.parse_args()
    submit(a.repo, a.data, a.manifest_sha)
