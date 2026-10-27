"""Checks the data, run-file and scoring code on tiny cases worked out by hand.

Run from the project folder:   python -m harness.selftest
"""
import math
import os
import sys
import tempfile

from .data import read_qrels, scorable_queries
from .metrics import evaluate_run
from .run_io import read_run

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


def main():
    test_qrels()
    test_run_io()
    test_scoring()
    print('\nALL SELF-TESTS PASSED' if failures == 0 else f'\n{failures} SELF-TEST(S) FAILED')
    sys.exit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
