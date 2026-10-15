"""Checks the scoring pipeline on tiny hand-computed cases.   python -m harness.selftest"""
import math
import sys

from .metrics import evaluate_run


def close(a, b):
    return abs(a - b) < 1e-4


def main():
    failures = 0

    def check(name, ok):
        nonlocal failures
        failures += 0 if ok else 1
        print(('[PASS] ' if ok else '[FAIL] ') + name)

    qrels = {'q1': {'d1': 1, 'd2': 0, 'd3': 1},
             'q2': {'d9': 1},
             'q3': {'d5': 0}}          # q3 has no relevant document: it must not be scored

    # q1: ranking d2, d1, d3 -> gains 0,1,1.  DCG = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3).
    expected_q1 = (1 / math.log2(3) + 1 / math.log2(4)) / (1 + 1 / math.log2(3))   # 0.6934
    run = {'q1': [('d2', 5.0), ('d1', 5.0), ('d3', 5.0)]}                            # equal scores on purpose
    pq = evaluate_run(run, qrels)

    check('nDCG@10 matches the hand calculation (0.6934)', close(pq['nDCG@10']['q1'], expected_q1))
    check('equal scores do not change the order that was written', close(pq['nDCG@10']['q1'], expected_q1))
    check('a query with no results scores 0', pq['nDCG@10']['q2'] == 0.0)
    check('a query without relevant documents is not scored', 'q3' not in pq['nDCG@10'])
    check('recall@100 counts both relevant documents of q1', close(pq['R@100']['q1'], 1.0))
    # All 3 returned documents of q1 are judged. Whether ir_measures divides by 3 (-> 1.0) or by 10 (-> 0.3)
    # is not asserted here; it is printed so the definition is known before the number is used in a table.
    print('Judged@10 for q1 =', pq['Judged@10']['q1'], '(1.0 = divides by returned results, 0.3 = divides by 10)')

    print('\nALL SELF-TESTS PASSED' if failures == 0 else f'\n{failures} SELF-TEST(S) FAILED')
    sys.exit(0 if failures == 0 else 1)


if __name__ == '__main__':
    main()
