"""
Runs E1 (an LLM writes one Boolean query per request, the engine runs it; api/planner.py) on one split.

Writes  results/runs/E1_<split>.tsv            the run: top 100 by BM25 inside the Boolean set
        results/runs/E1_<split>_sizes.tsv      the size of each query's Boolean set (H1); a failed query has no line
        results/runs/E1_<split>_failures.tsv   queries the engine still rejected after the retries
        results/runs/E1_<split>_plans.jsonl    every request with all its attempts and the model's raw answers
        results/runs/E1_<split>.json           the settings: model and version, prompt fingerprint, git commit, ...

Development: --sample N runs a fixed random sample (seed 20261015) of N train claims that are neither base claims
of the targeted set nor prompt examples, writes to results/dev/ instead, and prints a short summary. The prompt is
developed only this way: never on the targeted set or on test.

Run from the project folder (Ollama running; the engine built with CMake):
    venv\\Scripts\\python.exe api/run_planner.py --split train --sample 30
    venv\\Scripts\\python.exe api/run_planner.py --split train
"""
import argparse
import collections
import datetime
import json
import os
import random
import statistics

from baseline_pyserini import ROOT, corpus_sha256, file_sha256
from planner import (EXAMPLE_CLAIMS, LLM, MAX_RETRIES, MAX_TOKENS, MODELS, SEED, SYSTEM_PROMPT, TEMPERATURE, plan,
                     prompt_sha256)
from run_baselines import BOOLEAN_CLI, K, git_commit, git_dirty, load_queries, queries_path, run_boolean

SAMPLE_SEED = 20261015


def targeted_base_claims():
    with open(os.path.join(ROOT, 'data', 'targeted', 'queries.tsv'), encoding='utf-8') as f:
        rows = [line.rstrip('\r\n').split('\t') for line in f if line.strip()]
    col = rows[0].index('base_claim_id')
    return {r[col] for r in rows[1:]}


def dev_summary(out_path, sizes_path, plans, split):
    """Prints how the run did on the sampled train claims: failures, empty sets, set sizes, relevant documents found."""
    relevant = collections.defaultdict(set)
    with open(os.path.join(ROOT, 'data', f'scifact_qrels_{split}.tsv'), encoding='utf-8') as f:
        for line in f:
            qid, doc, rel = line.rstrip('\r\n').split('\t')
            if int(rel) > 0:
                relevant[qid].add(doc)
    with open(sizes_path, encoding='utf-8') as f:
        sizes = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f if line.strip()}
    ranked = collections.defaultdict(list)
    with open(out_path, encoding='utf-8') as f:
        for line in f:
            qid, _, doc, _ = line.rstrip('\r\n').split('\t')
            ranked[qid].append(doc)
    n = len(plans)
    non_empty = [s for s in sizes.values() if s > 0]
    top10 = sum(1 for q in plans if relevant[q] & set(ranked[q][:10]))
    top100 = sum(1 for q in plans if relevant[q] & set(ranked[q]))
    retried = sum(1 for p in plans.values() if len(p['attempts']) > 1)
    print(f'\n{n} requests: {n - len(sizes)} failed after retries ({retried} needed a retry), '
          f'{sum(1 for s in sizes.values() if s == 0)} empty sets.')
    if non_empty:
        print(f'Non-empty sets: median size {statistics.median(non_empty):g}, largest {max(non_empty)}.')
    print(f'A relevant document in the top 10: {top10} / {n}; in the top {K}: {top100} / {n}.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'test', 'targeted'])
    ap.add_argument('--model', default='qwen2.5-7b', choices=sorted(MODELS))
    ap.add_argument('--sample', type=int, help='development: N random train claims (not targeted, not examples)')
    args = ap.parse_args()
    if args.sample and args.split != 'train':
        raise SystemExit('--sample is for development, on train only.')

    commit, dirty = git_commit(), git_dirty()
    if dirty:
        print('WARNING: uncommitted changes to tracked files; the run will not match its git commit exactly.')
    queries = load_queries(args.split, None)
    out_dir = os.path.join(ROOT, 'results', 'dev' if args.sample else 'runs')
    if args.sample:
        excluded = targeted_base_claims() | set(EXAMPLE_CLAIMS)
        pool = sorted((q for q in queries if q[0] not in excluded), key=lambda q: int(q[0]))
        queries = sorted(random.Random(SAMPLE_SEED).sample(pool, args.sample), key=lambda q: int(q[0]))
    os.makedirs(out_dir, exist_ok=True)
    name = f'E1_{args.split}'
    out_path = os.path.join(out_dir, f'{name}.tsv')

    llm = LLM(args.model)
    print(f'{len(queries)} requests, model {args.model} ({MODELS[args.model]["description"]})')
    plans = plan([(qid, text) for qid, text, _ in queries], llm, BOOLEAN_CLI)
    settings = run_boolean([(qid, plans[qid]['query']) for qid, _, _ in queries], out_path, args.split, 'E1')

    with open(os.path.join(out_dir, f'{name}_plans.jsonl'), 'w', encoding='utf-8') as f:
        for qid, text, _ in queries:
            f.write(json.dumps({'query_id': qid, 'request': text, **plans[qid]}, ensure_ascii=False) + '\n')
    meta = {'system': 'E1', 'split': args.split, 'n_queries': len(queries), 'sample': args.sample,
            'sample_seed': SAMPLE_SEED if args.sample else None, 'k': K, 'corpus': 'data/scifact.tsv',
            'corpus_sha256': corpus_sha256(), 'queries_sha256': file_sha256(queries_path(args.split)),
            'model_alias': args.model, 'model': MODELS[args.model]['model'],
            'model_version': MODELS[args.model]['version'], 'model_description': MODELS[args.model]['description'],
            'temperature': TEMPERATURE, 'seed': SEED, 'max_tokens': MAX_TOKENS, 'max_retries': MAX_RETRIES,
            'prompt_sha256': prompt_sha256(), 'system_prompt': SYSTEM_PROMPT, 'example_claims': EXAMPLE_CLAIMS,
            'model_calls': llm.calls, 'cache_hits': llm.cache_hits, 'k1': 1.2, 'b': 0.75, **settings,
            'git_commit': commit, 'git_dirty': dirty, 'run_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(os.path.join(out_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'E1 on {args.split}: {len(queries)} requests -> {out_path} '
          f'({llm.calls} model calls, {llm.cache_hits} from the cache)')
    if args.sample:
        dev_summary(out_path, os.path.splitext(out_path)[0] + '_sizes.tsv', plans, args.split)


if __name__ == '__main__':
    main()
