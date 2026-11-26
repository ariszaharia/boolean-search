"""
E3 = E1 ranked by SSR (hypotheses.md, Setting and H9): E1's Boolean queries, unchanged, run by the engine with
shortest substring ranking (C = 42, cpp/src/ssr_scorer.cpp) instead of BM25. Same sets, other order; no model call.

    venv\\Scripts\\python.exe api/run_e3.py --split test

Reads results/runs/E1_<split>_plans.jsonl; writes results/runs/E3_<split>.tsv (top 100), E3_<split>_sizes.tsv,
E3_<split>_failures.tsv and E3_<split>.json.
"""
import argparse
import datetime
import json
import os
import subprocess

from baseline_pyserini import CORPUS, ROOT, WORK, corpus_sha256
from run_baselines import BOOLEAN_CLI, K, git_commit, git_dirty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'test', 'targeted'])
    args = ap.parse_args()
    commit, dirty = git_commit(), git_dirty()
    if dirty:
        print('WARNING: uncommitted changes to tracked files; the run will not match its git commit exactly.')
    runs = os.path.join(ROOT, 'results', 'runs')
    plans = os.path.join(runs, f'E1_{args.split}_plans.jsonl')
    pairs = []
    with open(plans, encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            if p.get('ok'):
                pairs.append((p['query_id'], p['query']))
    os.makedirs(WORK, exist_ok=True)
    queries = os.path.join(WORK, f'boolean_queries_E3_{args.split}.tsv')
    with open(queries, 'w', encoding='utf-8') as f:
        for qid, query in pairs:
            f.write(f'{qid}\t{" ".join(query.split())}\n')
    out = os.path.join(runs, f'E3_{args.split}.tsv')
    stem = os.path.splitext(out)[0]
    subprocess.run([BOOLEAN_CLI, CORPUS, queries, out, stem + '_sizes.tsv', stem + '_failures.tsv', str(K), 'ssr'],
                   check=True)
    meta = {'system': 'E3', 'split': args.split, 'what': 'E1 queries ranked by SSR (C = 42)',
            'e1_plans': os.path.relpath(plans, ROOT), 'n_queries': len(pairs), 'k': K, 'ranking': 'ssr',
            'corpus_sha256': corpus_sha256(), 'engine_binary': os.path.relpath(BOOLEAN_CLI, ROOT),
            'git_commit': commit, 'git_dirty': dirty, 'run_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(stem + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'E3 on {args.split}: {len(pairs)} queries -> {out}')


if __name__ == '__main__':
    main()
