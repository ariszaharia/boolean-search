"""
Runs the baselines without an LLM on one split and writes their run files.

  B1      BM25 with Pyserini's default analyzer (Porter stemming, English stopwords removed)
  B2      BM25 exact match: the C++ engine with (+ word1 word2 ...)
  B1-lit  a check, not a system: B1 set up like Pyserini's published BEIR baseline, to show that the setup
          reproduces the published BM25 number for SciFact. Not part of any hypothesis; run it on the test
          split at the end, with the final runs.

B1 and B2 get exactly the same words for each query (lowercase, split on every non-alphanumeric
character, as the C++ tokenizer does), the same documents (data/scifact.tsv) and the same BM25
parameters (k1 = 1.2, b = 0.75). The only difference left is the analyzer: B1 stems and drops stopwords,
B2 does neither.
B1-lit differs from B1 in the two ways the published baseline does: k1 = 0.9, b = 0.4, and the raw query
text goes to Lucene's analyzer (so, for example, 0.05 stays one token). Same stemmed index as B1, which
matches BEIR's "flat" index (title and text joined).

Writes   results/runs/<system>_<split>.tsv    query_id <TAB> rank <TAB> doc_id <TAB> score
         results/runs/<system>_<split>.json   the settings of the run (for the thesis and for reruns)

Run from the project folder (needs data/scifact.tsv and data/scifact_queries_<split>.tsv, from
api/prep_scifact.py and api/prep_queries.py; B2 needs the engine built with CMake, see PROGRESS.md):
    venv\\Scripts\\python.exe api/run_baselines.py --system B1 --split train
    venv\\Scripts\\python.exe api/run_baselines.py --system B2 --split train
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
from importlib.metadata import version

from baseline_pyserini import JSONL_DIR, ROOT, WORK, words_of, write_jsonl

CORPUS = os.path.join(ROOT, 'data', 'scifact.tsv')
INDEX_DEFAULT = os.path.join(WORK, 'lucene_index_default')   # stemmed; the unstemmed one is lucene_index
SEARCH_CLI = os.path.join(ROOT, 'cpp', 'build', 'search_cli.exe' if os.name == 'nt' else 'search_cli')

K = 100
K1 = 1.2
B = 0.75
K1_LIT = 0.9   # Pyserini's BM25 setting for its BEIR baselines
B_LIT = 0.4


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def load_queries(split, limit):
    """[(query_id, text, [words])] in file order."""
    path = os.path.join(ROOT, 'data', f'scifact_queries_{split}.tsv')
    if not os.path.isfile(path):
        sys.exit(f'Missing {path}. Run api/prep_queries.py first.')
    queries = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            qid, text = line.rstrip('\r\n').split('\t', 1)
            words = words_of(text)
            if words:
                queries.append((qid, text, words))
    return queries[:limit] if limit else queries


def build_default_index():
    if os.path.isdir(INDEX_DEFAULT) and os.listdir(INDEX_DEFAULT):
        print(f'Reusing existing index in {INDEX_DEFAULT} (delete the folder to rebuild)')
        return
    if not os.path.isfile(os.path.join(JSONL_DIR, 'docs.jsonl')):
        write_jsonl()
    # No -stemmer and no -keepStopwords: Pyserini's default analyzer (Porter stemming, stopwords removed).
    cmd = [sys.executable, '-m', 'pyserini.index.lucene',
           '--collection', 'JsonCollection',
           '--input', JSONL_DIR,
           '--index', INDEX_DEFAULT,
           '--generator', 'DefaultLuceneDocumentGenerator',
           '--threads', '1',
           '--storePositions', '--storeDocvectors', '--storeRaw']
    print('Building the index:\n  ' + ' '.join(cmd))
    subprocess.run(cmd, check=True)


def run_b1(queries, out_path, k1, b, raw_text):
    """B1 (raw_text False: the engine's words) and B1-lit (raw_text True: the query as written)."""
    build_default_index()
    from pyserini.analysis import Analyzer, get_lucene_analyzer
    from pyserini.search.lucene import LuceneSearcher

    # Safety check: B1 must stem and drop stopwords, otherwise it is the same system as B2.
    sample = 'The cells responses'
    tokens = Analyzer(get_lucene_analyzer()).analyze(sample)
    print(f'Default analyzer turns  {sample!r}  into  {tokens}')
    if 'the' in tokens or 'cells' in tokens:
        sys.exit('The default analyzer does not stem or does not remove stopwords. Stopping.')

    searcher = LuceneSearcher(INDEX_DEFAULT)   # its default analyzer is the one the index was built with
    searcher.set_bm25(k1, b)
    with open(out_path, 'w', encoding='utf-8') as out:
        for qid, text, words in queries:
            query = text if raw_text else ' '.join(words)
            for rank, hit in enumerate(searcher.search(query, k=K), start=1):
                out.write(f'{qid}\t{rank}\t{hit.docid}\t{hit.score:.9g}\n')
    return {'k1': k1, 'b': b,
            'query_input': 'raw query text' if raw_text else 'engine words (lowercase, split on non-alphanumeric)',
            'analyzer': 'Pyserini default (Porter stemming, English stopwords removed)',
            'index': os.path.relpath(INDEX_DEFAULT, ROOT), 'pyserini_version': version('pyserini')}


def run_b2(queries, out_path, split):
    if not os.path.isfile(SEARCH_CLI):
        sys.exit(f'Missing {SEARCH_CLI}. Build the engine first (cmake, see PROGRESS.md).')
    words_path = os.path.join(WORK, f'queries_words_{split}.tsv')
    os.makedirs(WORK, exist_ok=True)
    with open(words_path, 'w', encoding='utf-8') as f:
        for qid, _, words in queries:
            f.write(f"{qid}\t{' '.join(words)}\n")
    # search_cli runs (+ word1 word2 ...) with k1 = 1.2, b = 0.75 (fixed in search_cli.cpp).
    subprocess.run([SEARCH_CLI, CORPUS, words_path, out_path, str(K)], check=True)
    return {'k1': K1, 'b': B, 'query_input': 'engine words (lowercase, split on non-alphanumeric)',
            'analyzer': 'engine tokenizer (lowercase, split on non-alphanumeric; no stemming, no stopwords)',
            'query': '(+ word1 word2 ...)', 'engine_binary': os.path.relpath(SEARCH_CLI, ROOT)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', required=True, choices=['B1', 'B2', 'B1-lit'])
    ap.add_argument('--split', required=True, choices=['train', 'test'])
    ap.add_argument('--limit', type=int, help='only the first N queries (for a quick check)')
    ap.add_argument('--out-dir', default=os.path.join(ROOT, 'results', 'runs'))
    args = ap.parse_args()

    if not os.path.isfile(CORPUS):
        sys.exit(f'Missing {CORPUS}. Run api/prep_scifact.py first.')
    queries = load_queries(args.split, args.limit)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f'{args.system}_{args.split}.tsv')

    if args.system == 'B1':
        settings = run_b1(queries, out_path, K1, B, raw_text=False)
    elif args.system == 'B1-lit':
        settings = run_b1(queries, out_path, K1_LIT, B_LIT, raw_text=True)
    else:
        settings = run_b2(queries, out_path, args.split)
    meta = {'system': args.system, 'split': args.split, 'n_queries': len(queries), 'limit': args.limit,
            'k': K, 'corpus': 'data/scifact.tsv', **settings,
            'git_commit': git_commit(), 'run_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(os.path.splitext(out_path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'{args.system} on {args.split}: {len(queries)} queries -> {out_path}')
    print(f'Score it with:  venv\\Scripts\\python.exe -m harness.evaluate --run {out_path} --split {args.split}')


if __name__ == '__main__':
    main()
