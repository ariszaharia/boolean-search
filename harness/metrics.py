"""Scores a run against qrels with ir_measures (the metrics are never written by hand).

Rules that keep every system comparable:
  * The scored queries are the ones with at least one relevant document (see data.scorable_queries).
  * A scored query for which the system returned nothing gets 0. An empty Boolean result is a
    failure of that system, not a missing value.
  * The order written in the run file is the order that is scored. The rank is turned into a
    strictly decreasing score, so ties in the original scores can never change the ranking.
"""
from .data import scorable_queries

MEASURES = ['nDCG@10', 'R@100']


def evaluate_run(run, qrels, measures=MEASURES):
    """run: {qid: [(doc_id, score), ...]} in rank order.
    Returns {measure: {qid: value}} with one value for every scored query."""
    import ir_measures

    queries = scorable_queries(qrels)
    judged = {q: qrels[q] for q in queries}
    parsed = {str(ir_measures.parse_measure(m)): m for m in measures}

    ranked = {}
    for qid, items in run.items():
        if qid in judged:
            ranked[qid] = {doc_id: float(len(items) - i) for i, (doc_id, _) in enumerate(items)}

    per_query = {m: {q: 0.0 for q in queries} for m in measures}
    for result in ir_measures.iter_calc(list(ir_measures.parse_measure(m) for m in measures), judged, ranked):
        per_query[parsed[str(result.measure)]][result.query_id] = float(result.value)
    return per_query


def means(per_query):
    return {m: sum(values.values()) / len(values) for m, values in per_query.items()}
