"""Checks the data, run-file and scoring code on tiny cases worked out by hand.

Run from the project folder:   python -m harness.selftest
"""
import math
import os
import sys
import tempfile

from .compare import compare_runs
from .data import read_qrels, scorable_queries
from .metrics import empty_queries, evaluate_run
from .run_io import read_run
from .stats import equivalent, holm, paired_bootstrap

failures = 0


def check(name, ok):
    global failures
    if not ok:
        failures += 1
    print(('[PASS] ' if ok else '[FAIL] ') + name)


def raises(function, *args):
    try:
        function(*args)
    except ValueError:
        return True
    return False


def write_temp(text):
    """Writes text to a temporary file (newline='' keeps \\r\\n as written) and returns its path."""
    handle, path = tempfile.mkstemp(suffix='.tsv')
    with os.fdopen(handle, 'w', encoding='utf-8', newline='') as f:
        f.write(text)
    return path


def close(a, b):
    return abs(a - b) < 1e-4


def test_qrels():
    path = write_temp('1\td1\t1\n1\td2\t0\r\n2\td9\t0\n\n3\td3\t1\n')
    qrels = read_qrels(path)
    check('qrels: parsed, CRLF and blank line tolerated',
          qrels == {'1': {'d1': 1, 'd2': 0}, '2': {'d9': 0}, '3': {'d3': 1}})
    check('qrels: a query without a relevant document is not scorable', scorable_queries(qrels) == ['1', '3'])
    check('qrels: a line with 2 columns is rejected', raises(read_qrels, write_temp('1\td1\n')))


def test_run_io():
    path = write_temp('q1\t2\tdB\t4.0\nq1\t1\tdA\t5.0\nq2\t1\tdC\t1.0\n')
    run = read_run(path)
    check('run: lines out of order are sorted by rank', run['q1'] == [('dA', 5.0), ('dB', 4.0)])
    check('run: queries are kept apart', run['q2'] == [('dC', 1.0)])
    check('run: a line with 3 columns is rejected', raises(read_run, write_temp('q1\t1\tdA\n')))
    check('run: ranks 1, 3 (gap) are rejected', raises(read_run, write_temp('q1\t1\tdA\t2\nq1\t3\tdB\t1\n')))
    check('run: ranks starting at 2 are rejected', raises(read_run, write_temp('q1\t2\tdA\t2\n')))
    check('run: the same document twice is rejected', raises(read_run, write_temp('q1\t1\tdA\t2\nq1\t2\tdA\t1\n')))


def test_scoring():
    qrels = {'q1': {'d1': 1, 'd3': 1}, 'q2': {'d9': 1}, 'q3': {'d5': 0}, 'q4': {'d1': 1}}

    # q1 is ranked d2, d1, d3 -> gains 0, 1, 1.
    # DCG = 1/log2(3) + 1/log2(4); the ideal order has gains 1, 1: 1 + 1/log2(3).
    expected_q1 = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))     # 0.6934
    # q4 has one relevant document d1, written second. All scores are equal on purpose: if ties were
    # broken by document id, d1 would move to rank 1. The written order must win: 1/log2(3) / 1.
    expected_q4 = 1 / math.log2(3)                                                    # 0.6309
    run = {'q1': [('d2', 5.0), ('d1', 5.0), ('d3', 5.0)],
           'q4': [('d9', 1.0), ('d1', 1.0)],
           'not_in_qrels': [('d1', 1.0)]}
    per_query = evaluate_run(run, qrels)

    check('nDCG@10 of q1 matches the hand calculation (0.6934)', close(per_query['nDCG@10']['q1'], expected_q1))
    check('nDCG@10 of q4: written order beats document-id order on equal scores (0.6309)',
          close(per_query['nDCG@10']['q4'], expected_q4))
    check('a scored query with no results gets 0 for every measure',
          per_query['nDCG@10']['q2'] == 0.0 and per_query['R@100']['q2'] == 0.0)
    check('a query without a relevant document is not scored', 'q3' not in per_query['nDCG@10'])
    check('a run query that is not in the qrels is ignored', 'not_in_qrels' not in per_query['nDCG@10'])
    check('recall@100 of q1 is 1.0 (both relevant documents found)', close(per_query['R@100']['q1'], 1.0))

    # Recall counts only the top 100: a relevant document at rank 101 does not count.
    deep = {'q2': [(f'x{i}', 1.0) for i in range(100)] + [('d9', 1.0)]}
    check('recall@100 ignores a relevant document at rank 101', evaluate_run(deep, qrels)['R@100']['q2'] == 0.0)
    deep_in = {'q2': [(f'x{i}', 1.0) for i in range(99)] + [('d9', 1.0)]}
    check('recall@100 counts a relevant document at rank 100', evaluate_run(deep_in, qrels)['R@100']['q2'] == 1.0)

    # The same for the N = 30 pool: rank 31 is outside, rank 30 is inside.
    at_31 = {'q2': [(f'x{i}', 1.0) for i in range(30)] + [('d9', 1.0)]}
    at_30 = {'q2': [(f'x{i}', 1.0) for i in range(29)] + [('d9', 1.0)]}
    check('recall@30 ignores a relevant document at rank 31 (but recall@100 counts it)',
          evaluate_run(at_31, qrels)['R@30']['q2'] == 0.0 and evaluate_run(at_31, qrels)['R@100']['q2'] == 1.0)
    check('recall@30 counts a relevant document at rank 30', evaluate_run(at_30, qrels)['R@30']['q2'] == 1.0)
    # q1 has 2 relevant documents; only d1 is in the top 30 -> 1/2.
    half = {'q1': [('d1', 1.0)] + [(f'x{i}', 1.0) for i in range(29)] + [('d3', 1.0)]}
    check('recall@30 of q1 is 0.5 when one of its two relevant documents is at rank 31',
          close(evaluate_run(half, qrels)['R@30']['q1'], 0.5))


def test_bootstrap():
    # Every query has the same difference 0.1: every resample has mean 0.1, so the interval is [0.1, 0.1].
    # No bootstrap mean is 0.1 or more away from 0.1, so the p-value is (1 + 0) / (1 + 10000).
    a = {f'q{i}': 0.6 for i in range(50)}
    b = {f'q{i}': 0.5 for i in range(50)}
    r = paired_bootstrap(a, b)
    check('bootstrap: constant difference gives the interval [0.1, 0.1]',
          close(r['mean_diff'], 0.1) and close(r['ci_low'], 0.1) and close(r['ci_high'], 0.1))
    check('bootstrap: constant difference gives the smallest p-value, 1/10001', close(r['p_value'], 1 / 10001))

    # Differences +1 and -1, half each: observed mean 0, so every bootstrap mean counts and p = 1.
    a = {f'q{i}': (1.0 if i % 2 else 0.0) for i in range(100)}
    b = {f'q{i}': (0.0 if i % 2 else 1.0) for i in range(100)}
    r = paired_bootstrap(a, b)
    check('bootstrap: mean difference 0 gives p = 1 and an interval around 0',
          close(r['p_value'], 1.0) and r['ci_low'] < 0 < r['ci_high'])

    # 400 queries, difference 1 for half of them and 0 for the rest: mean 0.5, standard error
    # sqrt(0.25 / 400) = 0.025, so the interval should be close to 0.5 +- 1.96 * 0.025 = [0.451, 0.549].
    a = {f'q{i}': float(i % 2) for i in range(400)}
    b = {f'q{i}': 0.0 for i in range(400)}
    r = paired_bootstrap(a, b)
    check('bootstrap: interval close to the normal approximation [0.451, 0.549]',
          abs(r['ci_low'] - 0.451) < 0.01 and abs(r['ci_high'] - 0.549) < 0.01)
    check('bootstrap: the same seed gives the same interval', paired_bootstrap(a, b) == r)
    check('bootstrap: different query sets are rejected', raises(paired_bootstrap, {'q1': 1.0}, {'q2': 1.0}))

    check('equivalence: [-0.01, 0.015] is inside +-0.02', equivalent({'ci_low': -0.01, 'ci_high': 0.015}, 0.02))
    check('equivalence: [-0.01, 0.025] is not inside +-0.02',
          not equivalent({'ci_low': -0.01, 'ci_high': 0.025}, 0.02))


def test_holm():
    # Sorted: d 0.005 * 4 = 0.02, a 0.01 * 3 = 0.03, c 0.03 * 2 = 0.06, b 0.04 * 1 = 0.04 -> raised to 0.06.
    r = holm({'a': 0.01, 'b': 0.04, 'c': 0.03, 'd': 0.005})
    check('holm: adjusted p-values match the hand calculation',
          all(close(r[k]['p_adjusted'], v) for k, v in {'d': 0.02, 'a': 0.03, 'c': 0.06, 'b': 0.06}.items()))
    check('holm: only a and d are rejected at 0.05',
          sorted(k for k in r if r[k]['reject']) == ['a', 'd'])
    check('holm: an adjusted p-value is never above 1', holm({'x': 0.6, 'y': 0.7})['x']['p_adjusted'] == 1.0)


def test_empty_results():
    qrels = {'q1': {'d1': 1}, 'q2': {'d2': 1}, 'q3': {'d3': 1}}
    run_a = {'q1': [('d1', 1.0)], 'q2': [('d2', 1.0)]}                     # nothing for q3
    run_b = {'q1': [('x', 2.0), ('d1', 1.0)], 'q2': [('d2', 1.0)], 'q3': [('d3', 1.0)]}
    check('empty: q3 is the only empty query of A', empty_queries(run_a, qrels) == ['q3'])

    # nDCG@10: A = 1, 1, 0 and B = 1/log2(3), 1, 1 for q1, q2, q3.
    out = compare_runs(run_a, run_b, qrels)
    ndcg = out['measures']['nDCG@10']
    nonempty_mean = (1 - 1 / math.log2(3)) / 2                              # q1 and q2 only: 0.1845
    zero_mean = (1 - 1 / math.log2(3) + 0 - 1) / 3                          # all three: -0.2103
    check('empty: zero-result rates are 1/3 for A and 0 for B',
          close(out['zero_result_rate_a'], 1 / 3) and out['zero_result_rate_b'] == 0.0)
    check('empty: the primary analysis leaves out q3 (n = 2, mean +0.1845)',
          ndcg['nonempty']['n'] == 2 and close(ndcg['nonempty']['mean_diff'], nonempty_mean))
    check('empty: the second analysis counts q3 as 0 (n = 3, mean -0.2103)',
          ndcg['empty_as_zero']['n'] == 3 and close(ndcg['empty_as_zero']['mean_diff'], zero_mean))

    run_b_gap = {'q1': [('d1', 1.0)], 'q3': [('d3', 1.0)]}                   # B has nothing for q2
    both = compare_runs(run_a, run_b_gap, qrels, nonempty_from='both')
    check('empty: nonempty_from both leaves out the empty queries of A and of B (n = 1)',
          both['measures']['nDCG@10']['nonempty']['n'] == 1)


def main():
    test_qrels()
    test_run_io()
    test_scoring()
    test_bootstrap()
    test_holm()
    test_empty_results()
    print('\nALL SELF-TESTS PASSED' if failures == 0 else f'\n{failures} SELF-TEST(S) FAILED')
    sys.exit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
