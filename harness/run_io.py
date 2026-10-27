"""Reading run files. Every system writes its results in the same format, one result per line:

    query_id <TAB> rank <TAB> doc_id <TAB> score
"""
import os


def run_name(path, split):
    """The run file's name without folder, extension and a trailing _<split>: results/runs/B1_train.tsv -> B1.
    Output files add the split back once (B1_train.json), so it is never written twice."""
    name = os.path.splitext(os.path.basename(path))[0]
    suffix = f'_{split}'
    return name[:-len(suffix)] if name.endswith(suffix) and name != suffix else name


def read_run(path):
    """Returns {query_id: [(doc_id, score), ...]} in rank order. All ids stay strings."""
    rows = {}
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 4:
                raise ValueError(f'{path}:{line_no}: expected 4 tab-separated columns, got {len(parts)}')
            qid, rank, doc_id, score = parts
            rows.setdefault(qid, []).append((int(rank), doc_id, float(score)))

    run = {}
    for qid, items in rows.items():
        items.sort()
        ranks = [rank for rank, _, _ in items]
        if ranks != list(range(1, len(items) + 1)):
            raise ValueError(f'{path}: query {qid} has ranks {ranks[:5]}..., expected 1, 2, 3, ...')
        docs = [doc_id for _, doc_id, _ in items]
        if len(set(docs)) != len(docs):
            raise ValueError(f'{path}: query {qid} lists the same document twice')
        run[qid] = [(doc_id, score) for _, doc_id, score in items]
    return run


def read_set_sizes(path):
    """Reads an E run's set-size file into {query_id: size}. One line per query that ran:

        query_id <TAB> set_size

    Size 0 = the query matched nothing. A query that failed (for example ParseError after the retries) has no line.
    """
    sizes = {}
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 2:
                raise ValueError(f'{path}:{line_no}: expected 2 tab-separated columns, got {len(parts)}')
            qid, size = parts
            if qid in sizes:
                raise ValueError(f'{path}:{line_no}: query {qid} appears twice')
            if not size.isdigit():
                raise ValueError(f'{path}:{line_no}: the set size must be a whole number >= 0, got {size!r}')
            sizes[qid] = int(size)
    return sizes


def check_set_sizes(run, sizes, pool_n):
    """Checks that a run file and its set-size file agree, so that empty sets, failed queries and the cutoff
    k = min(set size, N) cannot be mixed up. Raises ValueError at the first disagreement."""
    for qid, items in run.items():
        if qid not in sizes:
            raise ValueError(f'query {qid} has results but no set size (a failed query has neither)')
        if len(items) > sizes[qid]:
            raise ValueError(f'query {qid}: {len(items)} results, but its set has only {sizes[qid]} documents')
        if len(items) < min(sizes[qid], pool_n):
            raise ValueError(f'query {qid}: {len(items)} results, fewer than min(set size {sizes[qid]}, '
                             f'N = {pool_n})')
    for qid, size in sizes.items():
        if size > 0 and qid not in run:
            raise ValueError(f'query {qid}: set size {size}, but no results in the run file')
