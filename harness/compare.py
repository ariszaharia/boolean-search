"""Compares two run files on the same queries (A minus B), with both analyses of hypotheses.md:

  non-empty   only the queries where the reference system returned at least one document (primary)
  empty = 0   every scored query; an empty result counts as 0 (second analysis)

The reference system is A by default (by convention the Boolean one); --nonempty-from both keeps only
the queries where both systems returned something.

    python -m harness.compare --a results/E1.tsv --b results/B1.tsv --split train

Holm correction is not applied here: it is done once over all confirmatory tests (stats.holm).
"""
import argparse
import json
import os

from .data import ROOT, load_qrels, scorable_queries
from .metrics import MEASURES, empty_queries, evaluate_run, restrict
from .run_io import read_run, run_name
from .stats import N_RESAMPLES, SEED, paired_bootstrap


def compare_runs(run_a, run_b, qrels, nonempty_from='a', measures=MEASURES):
    """Returns the comparison as a dict (see main for what is printed)."""
    if nonempty_from not in ('a', 'both'):
        raise ValueError(f"nonempty_from must be 'a' or 'both', got {nonempty_from!r}")
    queries = scorable_queries(qrels)
    per_a, per_b = evaluate_run(run_a, qrels, measures), evaluate_run(run_b, qrels, measures)
    empty_a, empty_b = empty_queries(run_a, qrels), empty_queries(run_b, qrels)

    excluded = set(empty_a) | (set(empty_b) if nonempty_from == 'both' else set())
    nonempty = [q for q in queries if q not in excluded]

    out = {'n_queries': len(queries), 'nonempty_from': nonempty_from,
           'zero_result_rate_a': len(empty_a) / len(queries),
           'zero_result_rate_b': len(empty_b) / len(queries),
           'empty_queries_a': empty_a, 'empty_queries_b': empty_b,
           'n_resamples': N_RESAMPLES, 'seed': SEED, 'measures': {}}
    for m in measures:
        result = {'empty_as_zero': paired_bootstrap(per_a[m], per_b[m])}
        if nonempty:
            result['nonempty'] = paired_bootstrap(restrict(per_a, nonempty)[m], restrict(per_b, nonempty)[m])
        else:
            result['nonempty'] = None
        out['measures'][m] = result
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='run file of system A (by convention the Boolean one)')
    ap.add_argument('--b', required=True, help='run file of system B')
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--nonempty-from', default='a', choices=['a', 'both'])
    ap.add_argument('--name', help='name for the output file, before _<split> (default: A_vs_B, the run file '
                                   'names without _<split>, so B3_train.tsv and B1_train.tsv give B3_vs_B1_train.json)')
    args = ap.parse_args()

    qrels = load_qrels(args.split)
    out = compare_runs(read_run(args.a), read_run(args.b), qrels, args.nonempty_from)
    out.update({'run_a': args.a, 'run_b': args.b, 'split': args.split})

    print(f'A: {args.a}\nB: {args.b}\nSplit "{args.split}", {out["n_queries"]} scored queries. '
          f'Zero results: A {out["zero_result_rate_a"]:.1%}, B {out["zero_result_rate_b"]:.1%}.')
    for m, result in out['measures'].items():
        for analysis in ('nonempty', 'empty_as_zero'):
            r = result[analysis]
            if r is None:
                print(f'  {m:8s} {analysis:13s} no query left')
                continue
            print(f'  {m:8s} {analysis:13s} n={r["n"]:4d}  A {r["mean_a"]:.4f}  B {r["mean_b"]:.4f}  '
                  f'A-B {r["mean_diff"]:+.4f}  95% CI [{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]  '
                  f'p {r["p_value"]:.4f}')

    name = args.name or f'{run_name(args.a, args.split)}_vs_{run_name(args.b, args.split)}'
    out_dir = os.path.join(ROOT, 'results', 'comparisons')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}_{args.split}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
