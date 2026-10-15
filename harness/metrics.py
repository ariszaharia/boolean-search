"""Scores a run against qrels with ir_measures (never hand-written metrics).

Rules that keep every system comparable:
  * The query set is the set of qrels queries that have at least one relevant document.
  * A query for which the system returned nothing scores 0 (an empty Boolean result is a failure,
    not a missing value).
  * Ties never matter: the rank in the run file is turned into the score, so the order the
    system wrote is the order that is scored.
"""
from .data import queries_with_relevant_docs

MEASURES = ['nDCG@10', 'R@100', 'Judged@10']


def evaluate_run(run, qrels, measures=MEASURES):
    """run: {qid: [(doc_id, score), ...]} in rank order. Returns {measure: {qid: value}}."""
    import ir_measures

    queries = queries_with_relevant_docs(qrels)
    qrels = {q: qrels[q] for q in queries}
    parsed = {m: ir_measures.parse_measure(m) for m in measures}

    ranked = {q: {d: float(1_000_000 - i) for i, (d, _) in enumerate(items)}
              for q, items in run.items() if q in qrels}

    per_query = {m: {q: 0.0 for q in queries} for m in measures}
    for result in ir_measures.iter_calc(list(parsed.values()), qrels, ranked):
        per_query[str(result.measure)][result.query_id] = float(result.value)
    return per_query


def means(per_query):
    return {m: sum(v.values()) / len(v) for m, v in per_query.items()}
