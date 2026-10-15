"""
Pyserini side of the engine comparison (thresholds: results/pyserini_thresholds.md).

What it does
  1. Builds a Lucene index from data/scifact.tsv with stemming OFF and stopwords KEPT
     (the prebuilt BEIR index is stemmed, so it cannot be compared with the exact-match engine).
  2. Loads the SciFact test queries and turns each into a list of lowercase words.
  3. Runs BM25 (k1 = 1.2, b = 0.75) and keeps the top 100 documents per query.

Writes
  results/queries_words.tsv    query_id <TAB> words      (the C++ engine reads this file too)
  results/pyserini_top100.tsv  query_id <TAB> rank <TAB> doc_id <TAB> score

Run from the project folder:   python api/baseline_pyserini.py
Needs Java 21, `pip install pyserini`. Building the index takes a few minutes the first time;
an existing index in results/work/ is reused.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
CORPUS = os.path.join(ROOT, 'data', 'scifact.tsv')
RESULTS = os.path.join(ROOT, 'results')
WORK = os.path.join(RESULTS, 'work')
JSONL_DIR = os.path.join(WORK, 'jsonl')
INDEX_DIR = os.path.join(WORK, 'lucene_index')
OUT_QUERIES = os.path.join(RESULTS, 'queries_words.tsv')
OUT_RUN = os.path.join(RESULTS, 'pyserini_top100.tsv')

K = 100
K1 = 1.2
B = 0.75


def words_of(text):
    """Same rule as the C++ tokenizer: lowercase, split on every non-alphanumeric character."""
    return re.findall(r'[a-z0-9]+', text.lower())


def write_jsonl():
    os.makedirs(JSONL_DIR, exist_ok=True)
    path = os.path.join(JSONL_DIR, 'docs.jsonl')
    count = 0
    with open(CORPUS, encoding='utf-8') as fin, open(path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.rstrip('\r\n')
            if '\t' not in line:
                continue
            doc_id, text = line.split('\t', 1)
            fout.write(json.dumps({'id': doc_id, 'contents': text}) + '\n')
            count += 1
    print(f'Wrote {count} documents to {path}')


def build_index():
    if os.path.isdir(INDEX_DIR) and os.listdir(INDEX_DIR):
        print(f'Reusing existing index in {INDEX_DIR} (delete the folder to rebuild)')
        return
    write_jsonl()
    cmd = [sys.executable, '-m', 'pyserini.index.lucene',
           '--collection', 'JsonCollection',
           '--input', JSONL_DIR,
           '--index', INDEX_DIR,
           '--generator', 'DefaultLuceneDocumentGenerator',
           '--threads', '1',
           '--storePositions', '--storeDocvectors', '--storeRaw',
           '-stemmer', 'none', '-keepStopwords']
    print('Building the index:\n  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def load_queries():
    """SciFact test queries as {query_id: text}. Expected: 300."""
    try:
        from pyserini.search import get_topics
        topics = get_topics('beir-v1.0.0-scifact-test')
        return {str(qid): t['title'] for qid, t in topics.items()}
    except Exception as e:  # noqa: BLE001
        print(f'Could not load the topics from Pyserini ({e}). Trying Hugging Face instead.')
    from datasets import load_dataset
    queries = load_dataset('BeIR/scifact', 'queries', split='queries')
    qrels = load_dataset('BeIR/scifact-qrels', split='test')
    test_ids = {str(r['query-id']) for r in qrels}
    return {str(q['_id']): q['text'] for q in queries if str(q['_id']) in test_ids}


def main():
    if not os.path.isfile(CORPUS):
        sys.exit(f'Missing {CORPUS}. Run python api/prep_scifact.py first.')
    os.makedirs(RESULTS, exist_ok=True)
    build_index()

    from pyserini.analysis import Analyzer, get_lucene_analyzer
    from pyserini.search.lucene import LuceneSearcher

    analyzer = get_lucene_analyzer(stemming=False, stopwords=False)

    # Safety check: refuse to continue if the analyzer stems or removes stopwords,
    # because the comparison would then be meaningless.
    sample = "The T-cell's responses of 3.14 cells"
    tokens = Analyzer(analyzer).analyze(sample)
    print(f'Query analyzer turns  {sample!r}  into  {tokens}')
    if 'the' not in tokens or 'responses' not in tokens:
        sys.exit('The analyzer removes stopwords or stems words. Stopping.')

    searcher = LuceneSearcher(INDEX_DIR)
    searcher.set_analyzer(analyzer)
    searcher.set_bm25(K1, B)

    queries = load_queries()
    print(f'Loaded {len(queries)} queries (expected 300)')

    n_written = 0
    with open(OUT_QUERIES, 'w', encoding='utf-8') as fq, open(OUT_RUN, 'w', encoding='utf-8') as fr:
        for qid in sorted(queries, key=lambda x: int(x) if x.isdigit() else x):
            words = words_of(queries[qid])
            if not words:
                continue
            fq.write(f"{qid}\t{' '.join(words)}\n")
            hits = searcher.search(' '.join(words), k=K)
            for rank, hit in enumerate(hits, start=1):
                fr.write(f'{qid}\t{rank}\t{hit.docid}\t{hit.score:.9g}\n')
            n_written += 1

    print(f'Ran {n_written} queries.')
    print(f'Wrote {OUT_QUERIES}')
    print(f'Wrote {OUT_RUN}')
    print('Next:  compile cpp/src/search_cli.cpp, run it, then python api/compare_runs.py')


if __name__ == '__main__':
    main()
