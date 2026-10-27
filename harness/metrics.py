"""Scores a run against qrels with ir_measures (the metrics are never written by hand).

Rules that keep every system comparable:
  * The scored queries are the ones with at least one relevant document (see data.scorable_queries).
  * A scored query for which the system returned nothing gets 0 here. hypotheses.md asks for two analyses:
    the primary one keeps only the queries with a non-empty result (use empty_queries and restrict below),
    the second one keeps every query with empty results counted as 0 (what evaluate_run returns).
  * The order written in the run file is the order that is scored. The rank is turned into a
    strictly decreasing score, so ties in the original scores can never change the ranking.
"""
from .data import scorable_queries

MEASURES = ['nDCG@10', 'R@30', 'R@100']   # R@30 and R@100: recall of the N = 30 and N = 100 pools


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


def set_cutoffs(sizes, qrels, pool_n):
    """H1's cutoff for every scored query (hypotheses.md, H1 precision clarification): k = min(set size, N).
    A query whose set is empty, or that failed (no line in the set-size file), gets k = N: there the Boolean
    system scores 0 and the other system is scored at N."""
    return {q: min(sizes[q], pool_n) if sizes.get(q, 0) > 0 else pool_n for q in scorable_queries(qrels)}


def precision_at(run, qrels, cutoffs):
    """Precision with its own cutoff for each query: {qid: P@cutoffs[qid]}. P@k always divides by k, so ranks the
    run does not fill count as not relevant. ir_measures is called once per distinct cutoff."""
    by_cutoff = {}
    for q, k in cutoffs.items():
        by_cutoff.setdefault(k, []).append(q)
    values = {}
    for k, queries in by_cutoff.items():
        measure = f'P@{k}'
        values.update(evaluate_run(run, {q: qrels[q] for q in queries}, [measure])[measure])
    return values


def empty_queries(run, qrels):
    """Scored queries for which the system returned no document (the query is missing from its run file)."""
    return [q for q in scorable_queries(qrels) if not run.get(q)]


def restrict(per_query, queries):
    """Keeps only the given queries in {measure: {qid: value}}."""
    keep = set(queries)
    return {m: {q: v for q, v in values.items() if q in keep} for m, values in per_query.items()}


def means(per_query):
    return {m: sum(values.values()) / len(values) for m, values in per_query.items()}
