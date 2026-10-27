"""
Compares the C++ engine with Pyserini on the same queries.
The thresholds below MUST match results/pyserini_thresholds.md (written before the run).

Reads   results/pyserini_top100.tsv   and   results/engine_top100.tsv
        (lines: query_id <TAB> rank <TAB> doc_id <TAB> score)
Writes  results/pyserini_comparison.json  and prints a summary.

Run from the project folder:   python api/compare_runs.py
"""
import json
import os
import statistics
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
RESULTS = os.path.join(ROOT, 'results')

sys.path.insert(0, ROOT)   # so that "harness" can be imported when this file is run directly
from harness.run_io import read_run as load_run   # noqa: E402

THRESHOLDS = {
    'mean_top10_overlap': 0.95,
    'share_queries_overlap_at_least_0.8': 0.90,
    'share_queries_same_top1': 0.90,
    'mean_spearman_top100': 0.95,
    'median_relative_score_difference_max': 0.02,
}

# Deviation 1 (see results/pyserini_thresholds.md): Lucene omits the (k1 + 1) factor of textbook BM25.
# The amended metric 4 divides the engine score by it before comparing.
K1 = 1.2



def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sxx * syy) ** 0.5


def spearman_on_common(engine, pyserini):
    """Spearman correlation over the documents present in both lists."""
    e_docs = [d for d, _ in engine]
    p_docs = [d for d, _ in pyserini]
    common = set(e_docs) & set(p_docs)
    if len(common) < 3:
        return None
    e_order = [d for d in e_docs if d in common]   # rank inside the common set
    p_order = [d for d in p_docs if d in common]
    e_rank = {d: i for i, d in enumerate(e_order)}
    p_rank = {d: i for i, d in enumerate(p_order)}
    docs = list(common)
    return pearson([e_rank[d] for d in docs], [p_rank[d] for d in docs])


def main():
    e_path = os.path.join(RESULTS, 'engine_top100.tsv')
    p_path = os.path.join(RESULTS, 'pyserini_top100.tsv')
    for p in (e_path, p_path):
        if not os.path.isfile(p):
            sys.exit(f'Missing {p}')
    engine, pyserini = load_run(e_path), load_run(p_path)

    qids = sorted(set(engine) & set(pyserini))
    only_e, only_p = set(engine) - set(pyserini), set(pyserini) - set(engine)
    if not qids:
        sys.exit('No query appears in both files.')
    print(f'Queries in both runs: {len(qids)}  (only engine: {len(only_e)}, only Pyserini: {len(only_p)})')

    per_query, score_diffs, scaled_diffs = [], [], []
    for q in qids:
        e, p = engine[q], pyserini[q]
        e10, p10 = e[:10], p[:10]
        e10_docs, p10_docs = [d for d, _ in e10], [d for d, _ in p10]
        overlap = len(set(e10_docs) & set(p10_docs)) / 10
        same_top1 = bool(e and p and e[0][0] == p[0][0])
        p_scores = dict(p10)
        for d, s in e10:
            if d in p_scores and p_scores[d] != 0:
                score_diffs.append(abs(s - p_scores[d]) / abs(p_scores[d]))
                scaled_diffs.append(abs(s / (K1 + 1.0) - p_scores[d]) / abs(p_scores[d]))
        per_query.append({
            'query_id': q, 'top10_overlap': overlap, 'same_top1': same_top1,
            'spearman_top100': spearman_on_common(e, p),
            'engine_top10': e10_docs, 'pyserini_top10': p10_docs,
        })

    spearmans = [r['spearman_top100'] for r in per_query if r['spearman_top100'] is not None]
    measured = {
        'mean_top10_overlap': statistics.mean(r['top10_overlap'] for r in per_query),
        'share_queries_overlap_at_least_0.8': sum(r['top10_overlap'] >= 0.8 for r in per_query) / len(per_query),
        'share_queries_same_top1': sum(r['same_top1'] for r in per_query) / len(per_query),
        'mean_spearman_top100': statistics.mean(spearmans) if spearmans else None,
        'median_relative_score_difference_max': statistics.median(score_diffs) if score_diffs else None,
    }
    passed = {
        'mean_top10_overlap': measured['mean_top10_overlap'] >= THRESHOLDS['mean_top10_overlap'],
        'share_queries_overlap_at_least_0.8': measured['share_queries_overlap_at_least_0.8'] >= THRESHOLDS['share_queries_overlap_at_least_0.8'],
        'share_queries_same_top1': measured['share_queries_same_top1'] >= THRESHOLDS['share_queries_same_top1'],
        'mean_spearman_top100': measured['mean_spearman_top100'] is not None and measured['mean_spearman_top100'] >= THRESHOLDS['mean_spearman_top100'],
        'median_relative_score_difference_max': measured['median_relative_score_difference_max'] is not None and measured['median_relative_score_difference_max'] <= THRESHOLDS['median_relative_score_difference_max'],
    }

    scaled_median = statistics.median(scaled_diffs) if scaled_diffs else None
    amended_passed = dict(passed)
    amended_passed['median_relative_score_difference_max'] = (
        scaled_median is not None and scaled_median <= THRESHOLDS['median_relative_score_difference_max'])

    worst = sorted(per_query, key=lambda r: r['top10_overlap'])[:10]
    out = {'n_queries': len(qids), 'thresholds': THRESHOLDS, 'measured': measured,
           'passed': passed, 'all_passed': all(passed.values()),
           'amended_metric4_median_relative_score_difference_scaled': scaled_median,
           'amended_all_passed': all(amended_passed.values()),
           'worst_10_by_overlap': worst, 'per_query': per_query}
    out_path = os.path.join(RESULTS, 'pyserini_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)

    print()
    for name in THRESHOLDS:
        m = measured[name]
        mark = 'PASS' if passed[name] else 'FAIL'
        cmp_ = '<=' if name.endswith('_max') else '>='
        print(f'[{mark}] {name}: {m:.4f}  (needs {cmp_} {THRESHOLDS[name]})' if m is not None else f'[FAIL] {name}: no data')
    print('\nOVERALL (pre-registered metrics):', 'PASS' if out['all_passed'] else 'FAIL')
    if scaled_median is not None:
        print(f'Amended metric 4 (engine score / (k1+1) vs Pyserini): median {scaled_median:.4f}  '
              f'-> OVERALL {"PASS" if out["amended_all_passed"] else "FAIL"}')
    print('\nWorst 5 queries by top-10 overlap:')
    for r in worst[:5]:
        print(f"  query {r['query_id']}: overlap {r['top10_overlap']:.1f}, spearman {r['spearman_top100']}")
    print(f'\nFull details: {out_path}')


if __name__ == '__main__':
    main()
