"""
Runs E4, the search agent (api/agent.py), on one split, and writes one run per call budget.

    results/<runs or dev>/E4-b<B>-<model>_<split>.tsv        the ranked graded documents within a budget of B calls
    results/<runs or dev>/E4-<model>_<split>_sessions.jsonl  every request: all steps (answer, action, observation),
                                                             the grades and why the search stopped
    results/<runs or dev>/E4-<model>_<split>.json            settings, calls, cost, git commit

Development: --sample N --part a|b|c, as in run_planner.py (train claims only, results/dev/).

    venv\\Scripts\\python.exe api/run_agent.py --split train --sample 30 --part a --model gemini-3.5-flash-lite
"""
import argparse
import collections
import datetime
import json
import os

from agent import (ANSWER_TOKENS, MAX_JUDGED, RANKING, SHOW_N, SNIPPET_WORDS, AGENT_PROMPT, prompt_sha256, ranking,
                   search_agent)
from baseline_pyserini import ROOT, corpus_sha256, file_sha256
from dev_sample import SAMPLE_SEED, dev_sample
from planner import DEV_SPEND_LIMIT, LLM, MODELS, TOTAL_SPEND_LIMIT
from run_baselines import BOOLEAN_CLI, git_commit, git_dirty, load_queries, queries_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', required=True, choices=['train', 'test', 'targeted'])
    ap.add_argument('--model', default='gpt-6-sol', choices=sorted(MODELS))
    ap.add_argument('--budget', type=int, default=10, help='model calls per request (the largest budget)')
    ap.add_argument('--report', default='3,5,10', help='budgets to write runs for (each at most --budget)')
    ap.add_argument('--sample', type=int, help='development: N train claims (not targeted, not examples)')
    ap.add_argument('--part', choices=['a', 'b', 'c'], default='a')
    ap.add_argument('--max-cost', type=float, help='paid models: dollars this run may spend')
    args = ap.parse_args()
    if args.sample and args.split != 'train':
        raise SystemExit('--sample is for development, on train only.')
    budgets = sorted({int(b) for b in args.report.split(',') if int(b) <= args.budget} | {args.budget})

    commit, dirty = git_commit(), git_dirty()
    if dirty:
        print('WARNING: uncommitted changes to tracked files; the run will not match its git commit exactly.')
    queries = load_queries(args.split, None)
    if args.sample:
        queries = dev_sample(queries, args.sample, 'abc'.index(args.part) * args.sample)
    out_dir = os.path.join(ROOT, 'results', 'dev' if args.sample else 'runs')
    os.makedirs(out_dir, exist_ok=True)
    stem = f'E4-{args.model}_{args.split}'

    llm = LLM(args.model, args.max_cost, DEV_SPEND_LIMIT if args.sample else TOTAL_SPEND_LIMIT,
              max_tokens=ANSWER_TOKENS)
    print(f'E4: {len(queries)} requests, model {args.model}, up to {args.budget} calls each')
    try:
        sessions = search_agent([(qid, text) for qid, text, _ in queries], llm, BOOLEAN_CLI, args.budget)
    finally:
        print(f'{llm.calls} model calls, {llm.cache_hits} from the cache'
              + (f', about ${llm.cost:.3f}' if 'price' in MODELS[args.model] else ''))

    paths = {}
    for b in budgets:
        paths[b] = os.path.join(out_dir, f'E4-b{b}-{args.model}_{args.split}.tsv')
        with open(paths[b], 'w', encoding='utf-8') as f:
            for qid, _, _ in queries:
                ranked = ranking(sessions[qid], b)
                for rank, (doc, grade) in enumerate(ranked, start=1):
                    f.write(f'{qid}\t{rank}\t{doc}\t{len(ranked) - rank + 1 + 1000 * grade}\n')
    with open(os.path.join(out_dir, stem + '_sessions.jsonl'), 'w', encoding='utf-8') as f:
        for qid, text, _ in queries:
            s = sessions[qid]
            f.write(json.dumps({'query_id': qid, 'request': text, 'calls': s.calls, 'stop_reason': s.stop_reason,
                                'searches': [{k: v for k, v in x.items() if k != 'docs'} for x in s.searches],
                                'judged': s.judged, 'steps': s.steps}, ensure_ascii=False) + '\n')
    calls = [s.calls for s in sessions.values()]
    meta = {'system': 'E4', 'split': args.split, 'n_queries': len(queries), 'sample': args.sample,
            'part': args.part if args.sample else None, 'sample_seed': SAMPLE_SEED if args.sample else None,
            'corpus': 'data/scifact.tsv', 'corpus_sha256': corpus_sha256(),
            'queries_sha256': file_sha256(queries_path(args.split)),
            'model_alias': args.model, 'model': MODELS[args.model]['model'], 'model_version': MODELS[args.model]['version'],
            'request_settings': llm.request_settings(), 'budget': args.budget, 'budgets_reported': budgets,
            'show_n': SHOW_N, 'snippet_words': SNIPPET_WORDS, 'max_judged': MAX_JUDGED, 'engine_ranking': RANKING,
            'prompt_sha256': prompt_sha256(), 'agent_prompt': AGENT_PROMPT,
            'model_calls': llm.calls, 'cache_hits': llm.cache_hits, 'calls_per_request_mean': sum(calls) / len(calls),
            'stop_reasons': dict(collections.Counter(s.stop_reason for s in sessions.values())),
            'cost_usd': round(llm.cost, 4), 'new_prompt_tokens': llm.prompt_tokens,
            'new_completion_tokens': llm.completion_tokens, 'runs': {b: os.path.relpath(p, ROOT) for b, p in paths.items()},
            'git_commit': commit, 'git_dirty': dirty, 'run_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(os.path.join(out_dir, stem + '.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'Calls per request: mean {meta["calls_per_request_mean"]:.1f}; stopped: {meta["stop_reasons"]}')
    if args.sample:
        import sys
        sys.path.insert(0, ROOT)
        from harness.data import load_qrels
        qrels = load_qrels('train')
        for b in budgets:
            found = sum(1 for qid, _, _ in queries
                        if any(doc in qrels.get(qid, {}) for doc, _ in ranking(sessions[qid], b)[:10]))
            print(f'  budget {b:2d}: a relevant document in the top 10 for {found} / {len(queries)}')


if __name__ == '__main__':
    main()
