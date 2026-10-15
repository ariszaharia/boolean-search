"""Scores one run file.   python -m harness.evaluate --run results/pyserini_top100.tsv --split test"""
import argparse
import json
import os

from .data import ROOT, load_qrels
from .metrics import MEASURES, evaluate_run, means
from .run_io import read_run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='run file: query_id, rank, doc_id, score (tab-separated)')
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--name', help='name for the output file (default: the run file name)')
    args = ap.parse_args()

    qrels = load_qrels(args.split)
    run = read_run(args.run)
    per_query = evaluate_run(run, qrels)
    avg = means(per_query)

    n = len(next(iter(per_query.values())))
    missing = sum(1 for q in next(iter(per_query.values())) if q not in run)
    print(f'Scored {n} queries on split "{args.split}" ({missing} had no results and count as 0).')
    for m in MEASURES:
        print(f'  {m:10s} {avg[m]:.4f}')

    name = args.name or os.path.splitext(os.path.basename(args.run))[0]
    out_dir = os.path.join(ROOT, 'results', 'metrics')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}_{args.split}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'run': args.run, 'split': args.split, 'n_queries': n,
                   'queries_without_results': missing, 'means': avg, 'per_query': per_query}, f, indent=1)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
