"""SciFact relevance judgments (qrels). Downloaded once, then cached as a small TSV in data/."""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(ROOT, 'data')


def qrels_path(split):
    return os.path.join(DATA_DIR, f'scifact_qrels_{split}.tsv')


def load_qrels(split):
    """{query_id: {doc_id: relevance}} for split 'test' or 'train'. All ids are strings."""
    path = qrels_path(split)
    if not os.path.isfile(path):
        from datasets import load_dataset  # only needed for the first download
        rows = load_dataset('BeIR/scifact-qrels', split=split)
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(f"{r['query-id']}\t{r['corpus-id']}\t{r['score']}\n")
        print(f'Saved {len(rows)} judgments to {path}')
    qrels = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            qid, doc_id, rel = line.rstrip('\r\n').split('\t')
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def queries_with_relevant_docs(qrels):
    """Only these queries can be scored (nDCG is undefined without a relevant document)."""
    return sorted(q for q, docs in qrels.items() if any(r > 0 for r in docs.values()))
