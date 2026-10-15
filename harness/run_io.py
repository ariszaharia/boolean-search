"""Reading run files. Format of every system's output:  query_id <TAB> rank <TAB> doc_id <TAB> score"""


def read_run(path):
    """Returns {query_id: [(doc_id, score), ...]} in rank order. All ids are strings."""
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
        ranks = [r for r, _, _ in items]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError(f'{path}: query {qid} has ranks {ranks[:5]}..., expected 1, 2, 3, ...')
        docs = [d for _, d, _ in items]
        if len(set(docs)) != len(docs):
            raise ValueError(f'{path}: query {qid} lists the same document twice')
        run[qid] = [(d, s) for _, d, s in items]
    return run
