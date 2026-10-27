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
