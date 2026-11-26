"""
Knowledge-leakage probe (exploratory; plan fixed in PROGRESS.md on 2026-11-26 before any run).

Does the planner model already know SciFact? For each request, the model (no search) is asked for the exact title of
the paper that is the evidence for it. A request is KNOWN if BM25 with that title ranks a relevant abstract first AND
the title matches the start of that abstract (difflib ratio >= MATCH). Then every system is scored on known and
unknown requests separately.

    venv\\Scripts\\python.exe api/leakage_probe.py --split test
    venv\\Scripts\\python.exe api/leakage_probe.py --split targeted

Writes results/exploratory/leakage_<split>.tsv (one line per request) and leakage_<split>.json (the stratified scores).
"""
import argparse
import difflib
import hashlib
import json
import os
import re
import sys

from baseline_pyserini import CORPUS, ROOT, words_of
from planner import LLM, TOTAL_SPEND_LIMIT
from run_baselines import B as BM25_B, INDEX_DEFAULT, K1, build_default_index, load_queries

MODEL = 'gpt-6-luna'
MATCH = 0.7
PROMPT = ('You are an expert on the biomedical literature. Without searching anything, give the exact title of the '
          'published paper (a PubMed abstract) that is the evidence for or against the claim you are given. If you do '
          'not know it, answer unknown. Answer with exactly one line: Title: <the title, or unknown>')


def parse_title(answer):
    m = re.search(r'title\s*:\s*(.+)', answer, flags=re.IGNORECASE)
    title = (m.group(1) if m else answer).strip().strip('"').strip()
    return '' if not title or title.lower().startswith('unknown') else title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['test', 'targeted'])
    ap.add_argument('--max-cost', type=float, default=0.05)
    args = ap.parse_args()
    sys.path.insert(0, ROOT)
    from harness.data import load_groups, load_qrels
    from harness.metrics import evaluate_run, means, restrict
    from harness.run_io import read_run

    queries = load_queries(args.split, None)
    qrels = load_qrels(args.split)
    llm = LLM(MODEL, args.max_cost, TOTAL_SPEND_LIMIT, max_tokens=100)
    llm.shared_len, llm.prompt_hash = 1, hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()

    build_default_index()
    from pyserini.search.lucene import LuceneSearcher
    searcher = LuceneSearcher(INDEX_DEFAULT)
    searcher.set_bm25(K1, BM25_B)
    docs = {}
    with open(CORPUS, encoding='utf-8') as f:
        for line in f:
            doc_id, text = line.rstrip('\n').split('\t', 1)
            docs[doc_id] = text

    rows = []
    for qid, text, _ in queries:
        answer = llm.chat([{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': f'Claim: {text}'}])
        title = parse_title(answer)
        top, ratio, relevant = '', 0.0, False
        if title and words_of(title):
            hits = searcher.search(' '.join(words_of(title)), k=1)
            if hits:
                top = hits[0].docid
                relevant = top in qrels.get(qid, {})
                n = len(words_of(title))
                start = ' '.join(words_of(docs[top])[:n])
                ratio = difflib.SequenceMatcher(None, ' '.join(words_of(title)), start).ratio()
        rows.append({'query_id': qid, 'title': title, 'top_doc': top, 'top_relevant': relevant,
                     'ratio': round(ratio, 3), 'known': relevant and ratio >= MATCH})
    print(f'{llm.calls} model calls, {llm.cache_hits} from the cache, about ${llm.cost:.3f}')

    out_dir = os.path.join(ROOT, 'results', 'exploratory')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f'leakage_{args.split}.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(rows[0]) + '\n')
        for r in rows:
            f.write('\t'.join(str(v).replace('\t', ' ') for v in r.values()) + '\n')

    known = [r['query_id'] for r in rows if r['known'] and r['query_id'] in qrels]
    unknown = [r['query_id'] for r in rows if not r['known'] and r['query_id'] in qrels]
    runs = os.path.join(ROOT, 'results', 'runs')
    systems = {'B1': f'B1_{args.split}', 'B3': f'B3_{args.split}', 'E2': f'E2_{args.split}',
               'E4': f'E4-b10-{MODEL}_{args.split}',
               'B1 reranked': f'B1-rr100-medcpt_{args.split}', 'B3 reranked': f'B3-rr100-medcpt_{args.split}',
               'E2 reranked': f'E2-rr100-medcpt_{args.split}', 'E4 reranked': f'E4-b10-{MODEL}-rr100-medcpt_{args.split}'}
    per = {}
    for name, stem in systems.items():
        path = os.path.join(runs, stem + '.tsv')
        if os.path.isfile(path):
            per[name] = evaluate_run(read_run(path), qrels)['nDCG@10']
    strata = {'known': known, 'unknown': unknown}
    if args.split == 'targeted':
        groups = load_groups('targeted')
        for g in ('exclusion', 'paraphrase'):
            strata[f'{g}, known'] = [q for q in known if q in set(groups[g])]
            strata[f'{g}, unknown'] = [q for q in unknown if q in set(groups[g])]
    scores = {}
    for s, qids in strata.items():
        scores[s] = {'n': len(qids), **{name: (sum(v[q] for q in qids) / len(qids) if qids else None)
                                         for name, v in per.items()}}
    with open(os.path.join(out_dir, f'leakage_{args.split}.json'), 'w', encoding='utf-8') as f:
        json.dump({'model': MODEL, 'prompt': PROMPT, 'match_threshold': MATCH, 'n': len(rows),
                   'n_known': len(known), 'n_answered': sum(1 for r in rows if r['title']),
                   'n_top_relevant': sum(1 for r in rows if r['top_relevant']), 'scores': scores,
                   'cost_usd': round(llm.cost, 4)}, f, indent=1)
    print(f'{len(rows)} requests: {sum(1 for r in rows if r["title"])} answered with a title, '
          f'{sum(1 for r in rows if r["top_relevant"])} with a relevant top hit, {len(known)} KNOWN')
    for s, v in scores.items():
        print(f'  {s:20s} n={v["n"]:3d}  ' + '  '.join(f'{k} {x:.3f}' for k, x in v.items() if k != 'n' and x is not None))


if __name__ == '__main__':
    main()
