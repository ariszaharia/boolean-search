"""Scores one run file.   python -m harness.evaluate --run results/pyserini_top100.tsv --split test
On the targeted split it also gives the means of every group (data.load_groups), which are the numbers to use;
the mean over all 100 queries mixes categories that ask different questions."""
import argparse
import json
import os

from .data import ROOT, SPLITS, load_groups, load_qrels
from .metrics import MEASURES, evaluate_run, means, restrict
from .run_io import read_run, run_name


def means_by_group(per_query, groups):
    """{group: {measure: mean over the group's scored queries}} plus 'n' (how many of them are scored)."""
    result = {}
    for group, qids in groups.items():
        part = restrict(per_query, qids)
        n = len(next(iter(part.values())))
        result[group] = {'n': n, **(means(part) if n else {})}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run file: query_id, rank, doc_id, score (tab-separated)')
    ap.add_argument('--split', required=True, choices=SPLITS)
    ap.add_argument('--name', help='name for the output file, before _<split> (default: the run file name '
                                   'without _<split>, so B1_train.tsv gives B1_train.json)')
    args = ap.parse_args()

    qrels = load_qrels(args.split)
    groups = load_groups(args.split)
    run = read_run(args.run)
    per_query = evaluate_run(run, qrels)
    avg = means(per_query)

    n = len(next(iter(per_query.values())))
    missing = sum(1 for q in next(iter(per_query.values())) if q not in run)
    print(f'Scored {n} queries on split "{args.split}" ({missing} had no results and count as 0).')
    if groups:
        print('All queries pooled (not used by any hypothesis; see the groups below):')
    for m in MEASURES:
        print(f'  {m:10s} {avg[m]:.4f}')
    by_group = means_by_group(per_query, groups)
    for group, g in by_group.items():
        print(f'  {group:22s} n={g["n"]:3d}  ' + '  '.join(f'{m} {g[m]:.4f}' for m in MEASURES if m in g))

    name = args.name or run_name(args.run, args.split)
    out_dir = os.path.join(ROOT, 'results', 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}_{args.split}.json')
    out = {'run': args.run, 'split': args.split, 'n_queries': n,
           'queries_without_results': missing, 'means': avg, 'per_query': per_query}
    if groups:
        out['by_group'] = by_group
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
