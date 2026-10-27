"""
The development claims: train claims used to develop a component (the planner's prompt in run_planner.py, the
reranker's choice in rerank.py), never the targeted set or test.

The eligible claims are the train claims that are neither base claims of the targeted set nor DEV_EXCLUDED. They
are shuffled once (seed 20261015), always in the same way, so a sample is the same claims every time it is drawn.
"""
import os
import random

from baseline_pyserini import ROOT

SAMPLE_SEED = 20261015
# Claims kept out of the development pool before it is shuffled: the examples of the first prompt (9309c3d). Fixed,
# so the samples stay the same claims when the prompt's examples change. A new example must come from outside the
# part being run (run_planner.py stops if one is inside it).
DEV_EXCLUDED = ['1125', '334', '326', '1209']


def targeted_base_claims():
    with open(os.path.join(ROOT, 'data', 'targeted', 'queries.tsv'), encoding='utf-8') as f:
        rows = [line.rstrip('\r\n').split('\t') for line in f if line.strip()]
    col = rows[0].index('base_claim_id')
    return {r[col] for r in rows[1:]}


def dev_sample(queries, n, start=0):
    """queries: the train queries [(query_id, text, words)]. Returns n of them, the ones at positions start to
    start + n of the shuffled pool, sorted by query id."""
    excluded = targeted_base_claims() | set(DEV_EXCLUDED)
    pool = sorted((q for q in queries if q[0] not in excluded), key=lambda q: int(q[0]))
    random.Random(SAMPLE_SEED).shuffle(pool)
    return sorted(pool[start:start + n], key=lambda q: int(q[0]))
