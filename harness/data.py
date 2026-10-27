"""SciFact relevance judgments (qrels), read from the TSV files that api/prep_qrels.py writes.

File format, one judgment per line:  query_id <TAB> doc_id <TAB> relevance
Split names: "train" (our dev set) and "test" (held out, used only for final numbers), and "targeted" (the frozen
targeted constraint set, data/targeted/qrels.tsv from api/targeted_build.py; query ids t001 to t100).

The targeted set is always reported by group (load_groups), never as one pool of 100: its four categories ask
different questions (hypotheses.md, H1 to H3).
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(ROOT, 'data')
SPLITS = ('train', 'test', 'targeted')
TARGETED_QUERIES = os.path.join(DATA_DIR, 'targeted', 'queries.tsv')
# "Constraint queries" in hypotheses.md (H1) = the first three categories; the fourth is paraphrase (H2).
CONSTRAINT_CATEGORIES = ('exclusion', 'phrase', 'identifier')
CATEGORIES = CONSTRAINT_CATEGORIES + ('paraphrase',)


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
    if split == 'targeted':
        path = os.path.join(DATA_DIR, 'targeted', 'qrels.tsv')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing {path}. Run python api/targeted_build.py --final first.')
    else:
        path = os.path.join(DATA_DIR, f'scifact_qrels_{split}.tsv')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing {path}. Run python api/prep_qrels.py first.')
    return read_qrels(path)


def groups_from_rows(rows):
    """rows: [(query_id, category, kind)]. Returns {group: sorted query ids}, in this order:
      constraint          every exclusion, phrase and identifier query (H1)
      <category>          each category on its own (H2 uses paraphrase, H3 exclusion and identifier)
      <category>/<kind>   trap and no-trap separately, only for a category that has both
    """
    by_category, by_kind = {}, {}
    for qid, category, kind in rows:
        if category not in CATEGORIES:
            raise ValueError(f'query {qid}: unknown category {category!r}')
        by_category.setdefault(category, []).append(qid)
        by_kind.setdefault((category, kind), []).append(qid)
    groups = {'constraint': sorted(q for c in CONSTRAINT_CATEGORIES for q in by_category.get(c, []))}
    for c in CATEGORIES:
        if c in by_category:
            groups[c] = sorted(by_category[c])
    for c in CATEGORIES:
        kinds = sorted(k for (cat, k) in by_kind if cat == c)
        if len(kinds) > 1:
            for k in kinds:
                groups[f'{c}/{k}'] = sorted(by_kind[(c, k)])
    return {g: qids for g, qids in groups.items() if qids}


def load_groups(split):
    """The groups of the targeted set (groups_from_rows), read from data/targeted/queries.tsv; {} for the
    SciFact splits, which have no categories."""
    if split != 'targeted':
        return {}
    with open(TARGETED_QUERIES, encoding='utf-8') as f:
        lines = [line.rstrip('\r\n').split('\t') for line in f if line.strip()]
    header = lines[0]
    col = {name: header.index(name) for name in ('query_id', 'category', 'kind')}
    return groups_from_rows([(r[col['query_id']], r[col['category']], r[col['kind']]) for r in lines[1:]])


def scorable_queries(qrels):
    """Queries with at least one relevant document. nDCG and recall are undefined for the others."""
    return sorted(q for q, docs in qrels.items() if any(rel > 0 for rel in docs.values()))
