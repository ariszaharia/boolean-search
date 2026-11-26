"""
Exploratory (not in any hypothesis, not in the Holm correction): E4's ranking filtered by E2's exclusion.

On the targeted set E2 obeys exclusions (the model writes a (- ...) and the trap leaves the set) but often misses the
relevant abstract; E4 finds it more often but rarely writes a NOT (PROGRESS.md, 2026-11-24). This combines them
without any new model call and without the relevance judgments: for each request, the excluded operands of E2's
final query (the parts after the first operand of a top-level NOT, as the model wrote them) are run on the whole
corpus, and every abstract that matches one of them is removed from E4's ranking. Requests whose E2 query has no
top-level NOT keep E4's ranking unchanged. The order of the remaining abstracts is E4's.

    venv\\Scripts\\python.exe api/combine_e4_not.py            (targeted, budget 10)

With --rerank rr100-medcpt it filters the reranked E4 run instead (filtering and reranking commute: the reranker
scores each abstract on its own, so removing some does not change the order of the others).

Writes results/exploratory/E4-not-b<B>-<model>[-<rerank>]_targeted.tsv and a .json with, per request, what was excluded and
how many abstracts were removed.
"""
import argparse
import json
import os
import subprocess

from baseline_pyserini import CORPUS, ROOT, WORK
from planner import parse_tree, to_query
from run_baselines import BOOLEAN_CLI


def excluded_operands(query):
    """The excluded operands of a top-level NOT, as query text; [] if the query is not a NOT."""
    tree = parse_tree(query)
    if isinstance(tree, list) and tree[0] == '-' and len(tree) > 2:
        return [to_query(operand) for operand in tree[2:]]
    return []


def full_sets(pairs):
    """{key: set of every matching doc id} for [(key, query)], from the engine with an unlimited top k."""
    os.makedirs(WORK, exist_ok=True)
    paths = {n: os.path.join(WORK, f'combine_{n}.tsv') for n in ('queries', 'run', 'sizes', 'failures')}
    with open(paths['queries'], 'w', encoding='utf-8') as f:
        for key, query in pairs:
            f.write(f'{key}\t{query}\n')
    subprocess.run([BOOLEAN_CLI, CORPUS, paths['queries'], paths['run'], paths['sizes'], paths['failures'], '1000000'],
                   check=True, capture_output=True)
    sets = {key: set() for key, _ in pairs}
    with open(paths['run'], encoding='utf-8') as f:
        for line in f:
            if line.strip():
                key, _, doc, _ = line.split('\t')
                sets[key].add(doc)
    with open(paths['sizes'], encoding='utf-8') as f:
        for line in f:
            key, size = line.rstrip('\n').split('\t')
            assert int(size) == len(sets[key]), key   # the whole set came back
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt-6-luna')
    ap.add_argument('--budget', type=int, default=10)
    ap.add_argument('--rerank', default='', help='e.g. rr100-medcpt: filter that reranked E4 run')
    args = ap.parse_args()
    runs = os.path.join(ROOT, 'results', 'runs')
    e2 = {}
    with open(os.path.join(runs, 'E2_targeted_plans.jsonl'), encoding='utf-8') as f:
        for line in f:
            p = json.loads(line)
            e2[p['query_id']] = p['query'] if p.get('ok') else ''
    e4 = {}
    tag = f'-{args.rerank}' if args.rerank else ''
    source = f'E4-b{args.budget}-{args.model}{tag}_targeted.tsv'
    with open(os.path.join(runs, source), encoding='utf-8') as f:
        for line in f:
            qid, rank, doc, score = line.rstrip('\n').split('\t')
            e4.setdefault(qid, []).append((int(rank), doc, score))

    pairs, excluded = [], {}
    for qid, query in e2.items():
        excluded[qid] = excluded_operands(query) if query else []
        pairs += [(f'{qid}#{i}', op) for i, op in enumerate(excluded[qid])]
    sets = full_sets(pairs)

    out_dir = os.path.join(ROOT, 'results', 'exploratory')
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'E4-not-b{args.budget}-{args.model}{tag}_targeted.tsv')
    record = {}
    with open(out, 'w', encoding='utf-8') as f:
        for qid in sorted(e4, key=lambda q: q):
            banned = set().union(*[sets[f'{qid}#{i}'] for i in range(len(excluded.get(qid, [])))])
            kept = [(doc, score) for _, doc, score in sorted(e4[qid]) if doc not in banned]
            for rank, (doc, score) in enumerate(kept, start=1):
                f.write(f'{qid}\t{rank}\t{doc}\t{score}\n')
            record[qid] = {'e2_query': e2.get(qid, ''), 'excluded': excluded.get(qid, []),
                           'removed': len(e4[qid]) - len(kept)}
    with open(os.path.splitext(out)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump({'what': 'E4 ranking minus abstracts matching E2\'s excluded operands (exploratory)',
                   'e4_run': f'results/runs/{source}',
                   'e2_plans': 'results/runs/E2_targeted_plans.jsonl', 'requests': record}, f, indent=1)
    n_not = sum(1 for v in record.values() if v['excluded'])
    print(f'{out}: {n_not} requests with an E2 exclusion, {sum(v["removed"] for v in record.values())} abstracts '
          f'removed in all')


if __name__ == '__main__':
    main()
