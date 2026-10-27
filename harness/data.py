"""SciFact relevance judgments (qrels), read from the TSV files that api/prep_qrels.py writes.

File format, one judgment per line:  query_id <TAB> doc_id <TAB> relevance
Split names: "train" (our dev set) and "test" (held out, used only for final numbers).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
SPLITS = ('train', 'test')


def read_qrels(path):
    """Returns {query_id: {doc_id: relevance}}. All ids stay strings, relevance is an int."""
    qrels = {}
    with open(path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip('\r\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) != 3:
                raise ValueError(f'{path}:{line_no}: expected 3 tab-separated columns, got {len(parts)}')
            qid, doc_id, rel = parts
            qrels.setdefault(qid, {})[doc_id] = int(rel)
    return qrels


def load_qrels(split):
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}, got {split!r}')
    path = os.path.join(DATA_DIR, f'scifact_qrels_{split}.tsv')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Missing {path}. Run python api/prep_qrels.py first.')
    return read_qrels(path)


def scorable_queries(qrels):
    """Queries with at least one relevant document. nDCG and recall are undefined for the others."""
    return sorted(q for q, docs in qrels.items() if any(rel > 0 for rel in docs.values()))
