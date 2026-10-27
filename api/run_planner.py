"""
Runs E1 or E2 on one split (api/planner.py).
  E1  an LLM writes one Boolean query per request, the engine runs it.
  E2  E1, then relaxation where E1's valid query found nothing (hypotheses.md, clarification of 2026-10-27).
      E2 first runs E1 exactly (from the cache when E1 has run), so the two differ only by the relaxation.

Writes  results/runs/<system>_<split>.tsv            the run: top 100 by BM25 inside the Boolean set
        results/runs/<system>_<split>_sizes.tsv      the size of each query's Boolean set (H1); failed = no line
        results/runs/<system>_<split>_failures.tsv   queries the engine still rejected after the retries
        results/runs/<system>_<split>_plans.jsonl    every request with all attempts (and E2's relaxation rounds)
        results/runs/<system>_<split>.json           the settings: model and version, prompt fingerprint, git commit

Development: --sample N runs N train claims that are neither base claims of the targeted set nor prompt examples,
writes to results/dev/ instead, prints a short summary and adds it to results/prompt_dev_log.tsv (committed: the
history of the prompt). The eligible claims are shuffled once (seed 20261015, api/dev_sample.py); --part a takes
the first N, --part b the next N, so the two never overlap: tune on a, check the final prompt once on b. The
prompt is developed only this way: never on the targeted set or on test.

Run from the project folder (Ollama running; the engine built with CMake):
    venv\\Scripts\\python.exe api/run_planner.py --split train --sample 30
    venv\\Scripts\\python.exe api/run_planner.py --split train
    venv\\Scripts\\python.exe api/run_planner.py --system E2 --split train --sample 30 --part a
"""
import argparse
import collections
import datetime
import json
import os
import statistics

from baseline_pyserini import ROOT, corpus_sha256, file_sha256
from dev_sample import SAMPLE_SEED, dev_sample
from planner import (EXAMPLE_CLAIMS, LLM, MAX_RELAX_ROUNDS, MAX_RETRIES, MAX_TOKENS, MODELS, RELAX_MESSAGE, SEED,
                     SYSTEM_PROMPT, TEMPERATURE, plan, prompt_sha256, relax)
from run_baselines import BOOLEAN_CLI, K, git_commit, git_dirty, load_queries, queries_path, run_boolean

def dev_summary(out_path, sizes_path, qids, retried, split):
    """Prints how the run did on the sampled train claims: failures, empty sets, set sizes, relevant documents found.
    qids: the requests of the run; retried: how many of them needed a parse retry."""
    relevant = collections.defaultdict(set)
    with open(os.path.join(ROOT, 'data', f'scifact_qrels_{split}.tsv'), encoding='utf-8') as f:
        for line in f:
            qid, doc, rel = line.rstrip('\r\n').split('\t')
            if int(rel) > 0:
                relevant[qid].add(doc)
    with open(sizes_path, encoding='utf-8') as f:
        sizes = {line.split('\t')[0]: int(line.split('\t')[1]) for line in f if line.strip()}
    ranked = collections.defaultdict(list)
    with open(out_path, encoding='utf-8') as f:
        for line in f:
            qid, _, doc, _ = line.rstrip('\r\n').split('\t')
            ranked[qid].append(doc)
    n = len(qids)
    non_empty = [s for s in sizes.values() if s > 0]
    summary = {'n': n, 'failed': n - len(sizes), 'retried': retried,
               'empty': sum(1 for s in sizes.values() if s == 0),
               'median_size': statistics.median(non_empty) if non_empty else 0,
               'relevant_top10': sum(1 for q in qids if relevant[q] & set(ranked[q][:10])),
               'relevant_top100': sum(1 for q in qids if relevant[q] & set(ranked[q]))}
    print(f'\n{n} requests: {summary["failed"]} failed after retries ({summary["retried"]} needed a retry), '
          f'{summary["empty"]} empty sets.')
    if non_empty:
        print(f'Non-empty sets: median size {summary["median_size"]:g}, largest {max(non_empty)}.')
    print(f'A relevant document in the top 10: {summary["relevant_top10"]} / {n}; '
          f'in the top {K}: {summary["relevant_top100"]} / {n}.')
    return summary


def log_dev(summary, args, note):
    """One line per development run in results/prompt_dev_log.tsv."""
    path = os.path.join(ROOT, 'results', 'prompt_dev_log.tsv')
    columns = ['time', 'system', 'model', 'prompt_sha256', 'part', 'n', 'failed', 'retried', 'empty', 'median_size',
               'relevant_top10', 'relevant_top100', 'note']
    new = not os.path.isfile(path)
    with open(path, 'a', encoding='utf-8', newline='\n') as f:
        if new:
            f.write('\t'.join(columns) + '\n')
        row = {'time': datetime.datetime.now().isoformat(timespec='seconds'), 'system': args.system,
               'model': args.model,
               'prompt_sha256': prompt_sha256()[:16], 'part': args.part, 'note': note or '', **summary}
        f.write('\t'.join(str(row[c]) for c in columns) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--system', default='E1', choices=['E1', 'E2'])
    ap.add_argument('--split', required=True, choices=['train', 'test', 'targeted'])
    ap.add_argument('--model', default='qwen2.5-7b', choices=sorted(MODELS))
    ap.add_argument('--sample', type=int, help='development: N random train claims (not targeted, not examples)')
    ap.add_argument('--part', choices=['a', 'b'], default='a', help='development: first (a) or next (b) N claims')
    ap.add_argument('--note', help='development: what this prompt version changes (goes into the log)')
    args = ap.parse_args()
    if args.sample and args.split != 'train':
        raise SystemExit('--sample is for development, on train only.')

    commit, dirty = git_commit(), git_dirty()
    if dirty:
        print('WARNING: uncommitted changes to tracked files; the run will not match its git commit exactly.')
    queries = load_queries(args.split, None)
    out_dir = os.path.join(ROOT, 'results', 'dev' if args.sample else 'runs')
    if args.sample:
        queries = dev_sample(queries, args.sample, 0 if args.part == 'a' else args.sample)
        inside = sorted(set(EXAMPLE_CLAIMS) & {q[0] for q in queries})
        if inside:
            raise SystemExit(f'prompt examples {inside} are inside this development part; pick examples elsewhere')
    os.makedirs(out_dir, exist_ok=True)
    name = f'{args.system}_{args.split}'
    out_path = os.path.join(out_dir, f'{name}.tsv')

    llm = LLM(args.model)
    print(f'{args.system}: {len(queries)} requests, model {args.model} ({MODELS[args.model]["description"]})')
    e1, conversations = plan([(qid, text) for qid, text, _ in queries], llm, BOOLEAN_CLI)
    retried = sum(1 for r in e1.values() if len(r['attempts']) > 1)
    if args.system == 'E2':
        plans = relax(e1, conversations, llm, BOOLEAN_CLI)
        records = {qid: {'e1_attempts': e1[qid]['attempts'], **plans[qid]} for qid in plans}
        relaxed = [qid for qid in plans if plans[qid]['rounds']]
        recovered = [qid for qid in relaxed if (plans[qid]['rounds'][-1]['size'] or 0) > 0]
        relax_info = {'max_relax_rounds': MAX_RELAX_ROUNDS, 'relax_message': RELAX_MESSAGE,
                      'n_relaxed': len(relaxed), 'n_recovered': len(recovered),
                      'n_rejected_rounds': sum(1 for q in relaxed for r in plans[q]['rounds'] if r['rejected'])}
        print(f'Relaxed {len(relaxed)} empty queries; {len(recovered)} found documents after relaxing; '
              f'{relax_info["n_rejected_rounds"]} relaxed queries rejected for dropping an exclusion.')
    else:
        plans, records, relax_info = e1, e1, {}
    settings = run_boolean([(qid, plans[qid]['query']) for qid, _, _ in queries], out_path, args.split, args.system)

    with open(os.path.join(out_dir, f'{name}_plans.jsonl'), 'w', encoding='utf-8') as f:
        for qid, text, _ in queries:
            f.write(json.dumps({'query_id': qid, 'request': text, **records[qid]}, ensure_ascii=False) + '\n')
    meta = {'system': args.system, 'split': args.split, 'n_queries': len(queries), 'sample': args.sample,
            'sample_seed': SAMPLE_SEED if args.sample else None, 'k': K, 'corpus': 'data/scifact.tsv',
            'corpus_sha256': corpus_sha256(), 'queries_sha256': file_sha256(queries_path(args.split)),
            'model_alias': args.model, 'model': MODELS[args.model]['model'],
            'model_version': MODELS[args.model]['version'], 'model_description': MODELS[args.model]['description'],
            'temperature': TEMPERATURE, 'seed': SEED, 'max_tokens': MAX_TOKENS, 'max_retries': MAX_RETRIES,
            'prompt_sha256': prompt_sha256(), 'system_prompt': SYSTEM_PROMPT, 'example_claims': EXAMPLE_CLAIMS,
            'model_calls': llm.calls, 'cache_hits': llm.cache_hits, 'k1': 1.2, 'b': 0.75, **relax_info, **settings,
            'git_commit': commit, 'git_dirty': dirty, 'run_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(os.path.join(out_dir, f'{name}.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'{args.system} on {args.split}: {len(queries)} requests -> {out_path} '
          f'({llm.calls} model calls, {llm.cache_hits} from the cache)')
    if args.sample:
        summary = dev_summary(out_path, os.path.splitext(out_path)[0] + '_sizes.tsv', [q[0] for q in queries],
                              retried, args.split)
        log_dev(summary, args, args.note)


if __name__ == '__main__':
    main()
