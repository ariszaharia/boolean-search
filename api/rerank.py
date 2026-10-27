"""
Reranking (step 4): takes the top N of any run file and orders those N documents again with a reranker.

The reranker sees the request as written (the query text of the split, constraint sentence included) and the
document (title and text, as in data/scifact.tsv). It never sees the Boolean query, so every system's pool gets
exactly the same treatment; only the pool differs. The output has exactly the N reranked documents (fewer if
the pool had fewer): the Setting cuts every system to the same top N, so documents below N are not added back.

Which reranker the confirmatory runs use is NOT decided yet (PROGRESS.md, 2026-10-28: Jev and/or a cross-encoder;
it is written into hypotheses.md before any reranked run on the targeted set or on test). For now there is one
entry in RERANKERS, a free local cross-encoder, used for development; until an entry has fixed_in_hypotheses set,
the script reranks train runs only. An API reranker such as Jev becomes one
more entry (with its own scoring function) once there is a key; nothing else changes.

Every score is cached (results/work/rerank_cache/, one file per reranker, revision and device), keyed by the
exact request and document text, so a rerun or N = 30 after N = 100 costs nothing.
Uses the GPU when there is one. The laptop's CPU does about 19 pairs a second (all of train at N = 100 is over an
hour per system), so long runs go to Google Colab: colab/rerank.ipynb.

Writes  results/runs/<run>-rr<N>-<reranker>_<split>.tsv    query_id <TAB> rank <TAB> doc_id <TAB> score
        results/runs/<run>-rr<N>-<reranker>_<split>.json   the settings, and the run that was reranked
For example B1_train.tsv, N = 100 -> B1-rr100-ce-minilm_train.tsv (harness name B1-rr100-ce-minilm).

Run from the project folder:
    venv\\Scripts\\python.exe api/rerank.py --run results/runs/B1_train.tsv --split train --n 100
Commit first: the .json records the git commit, and git_dirty = true if tracked files had uncommitted changes.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
from importlib.metadata import version

from baseline_pyserini import ROOT, WORK, corpus_sha256, file_sha256
from run_baselines import git_commit, git_dirty, load_corpus, load_queries, queries_path

sys.path.insert(0, ROOT)   # so that "harness" can be imported when this file is run directly
from harness.run_io import read_run, run_name   # noqa: E402

RERANKERS = {
    'ce-minilm': {'kind': 'cross-encoder', 'model': 'cross-encoder/ms-marco-MiniLM-L6-v2',
                  'revision': '233902d25c440f23af6f7d6e94d2946bac0bee0a',   # Hugging Face commit, pinned 2026-10-28
                  'max_tokens': 512,
                  'fixed_in_hypotheses': None,   # the date of the hypotheses.md entry that fixes it; None = not fixed
                  'description': 'MiniLM-L6 cross-encoder trained on MS MARCO; the reranker of the BEIR paper '
                                 '(Thakur et al., 2021). Local, free; development only for now'},
}
# Pairs per forward pass. On the laptop's CPU one pair at a time is the fastest (about 19 pairs a second against
# 13 for batches of 32, since nothing is padded). On a GPU (Google Colab) batches are much faster. A score can
# differ in about the 6th decimal between devices and batch sizes, so each device has its own cache file and the
# run's .json records the device.
BATCH = {'cpu': 1, 'cuda': 64}
CACHE_DIR = os.path.join(WORK, 'rerank_cache')


def pair_key(request, document):
    """The cache key: a fingerprint of the exact text pair the reranker scores."""
    return hashlib.sha256((request + '\0' + document).encode('utf-8')).hexdigest()[:32]


class CrossEncoder:
    """A local cross-encoder: one forward pass per (request, document) pair, one relevance score (the logit)."""

    def __init__(self, setting):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.setting = setting
        self.tokenizer = AutoTokenizer.from_pretrained(setting['model'], revision=setting['revision'])
        import torch
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device_name = torch.cuda.get_device_name(0) if self.device == 'cuda' else 'cpu'
        self.model = AutoModelForSequenceClassification.from_pretrained(
            setting['model'], revision=setting['revision']).eval().to(self.device)

    def n_truncated(self, pairs):
        """How many (request, document) pairs are longer than max_tokens (their document gets cut)."""
        lengths = [len(ids) for ids in self.tokenizer([p[0] for p in pairs], [p[1] for p in pairs],
                                                      verbose=False)['input_ids']]
        return sum(n > self.setting['max_tokens'] for n in lengths)

    def score(self, pairs):
        """[(request, document)] -> [score], in the same order."""
        import torch
        scores = []
        with torch.no_grad():
            for start in range(0, len(pairs), BATCH[self.device]):
                batch = pairs[start:start + BATCH[self.device]]
                requests, documents = [p[0] for p in batch], [p[1] for p in batch]
                inputs = self.tokenizer(requests, documents, padding=True, truncation='only_second',
                                        max_length=self.setting['max_tokens'], return_tensors='pt').to(self.device)
                scores.extend(self.model(**inputs).logits[:, 0].tolist())
        return scores


class Cache:
    """Scores already computed, one file per reranker, revision and device: key <TAB> score, one line per pair."""

    def __init__(self, alias, setting, device):
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.path = os.path.join(CACHE_DIR, f'{alias}_{setting["revision"][:12]}_{device}.tsv')
        self.scores = {}
        if os.path.isfile(self.path):
            with open(self.path, encoding='utf-8') as f:
                for line in f:
                    key, score = line.rstrip('\r\n').split('\t')
                    self.scores[key] = float(score)

    def add(self, new):
        """new: {key: score}. Written to the file at once, so an interrupted run keeps what it computed."""
        with open(self.path, 'a', encoding='utf-8', newline='\n') as f:
            for key, score in new.items():
                f.write(f'{key}\t{score!r}\n')
        self.scores.update(new)


def rerank(run, requests, documents, n, reranker, cache):
    """run: {query_id: [(doc_id, score)]} in rank order; requests: {query_id: text}; documents: {doc_id: text}.
    Returns {query_id: [(doc_id, score)]}: the top n of each query, ordered by the reranker's score (equal scores
    keep their order in the input run). Also returns how many scores came from the cache."""
    out, hits = {}, 0
    for i, (qid, items) in enumerate(run.items(), start=1):
        pool = [doc for doc, _ in items[:n]]
        keys = [pair_key(requests[qid], documents[doc]) for doc in pool]
        missing = [(key, doc) for key, doc in zip(keys, pool) if key not in cache.scores]
        hits += len(pool) - len(missing)
        if missing:
            new = reranker.score([(requests[qid], documents[doc]) for _, doc in missing])
            cache.add({key: s for (key, _), s in zip(missing, new)})
        scored = [(doc, cache.scores[key]) for doc, key in zip(pool, keys)]
        # sorted() is stable, so equal scores keep the input run's order.
        out[qid] = sorted(scored, key=lambda pair: -pair[1])
        if i % 100 == 0:
            print(f'  reranked {i} / {len(run)} queries', flush=True)
    return out, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='the run file to rerank, e.g. results/runs/B1_train.tsv')
    ap.add_argument('--split', required=True, choices=['train', 'test', 'targeted'])
    ap.add_argument('--n', type=int, required=True, choices=[30, 100], help='rerank the top N (the Setting: 30, 100)')
    ap.add_argument('--reranker', default='ce-minilm', choices=sorted(RERANKERS))
    args = ap.parse_args()

    commit, dirty = git_commit(), git_dirty()
    if dirty:
        print('WARNING: uncommitted changes to tracked files; the run will not match its git commit exactly.')
    setting = RERANKERS[args.reranker]
    if args.split != 'train' and not setting['fixed_in_hypotheses']:
        sys.exit(f'{args.reranker} is not fixed in hypotheses.md, so it may only rerank train runs (development). '
                 'Write the reranker choice into hypotheses.md first, then set fixed_in_hypotheses.')
    source = run_name(args.run, args.split)
    name = f'{source}-rr{args.n}-{args.reranker}'
    out_dir = os.path.dirname(os.path.abspath(args.run))
    out_path = os.path.join(out_dir, f'{name}_{args.split}.tsv')

    run = read_run(args.run)
    requests = {qid: text for qid, text, _ in load_queries(args.split, None)}
    unknown = sorted(set(run) - set(requests))
    if unknown:
        sys.exit(f'{args.run} has queries that are not in the {args.split} split, e.g. {unknown[:5]}')
    ids, texts = load_corpus()
    documents = dict(zip(ids, texts))

    reranker = CrossEncoder(setting)
    cache = Cache(args.reranker, setting, reranker.device)
    print(f'Reranking the top {args.n} of {len(run)} queries from {args.run} with {args.reranker} on '
          f'{reranker.device_name} ({len(cache.scores)} scores in the cache)')
    reranked, hits = rerank(run, requests, documents, args.n, reranker, cache)
    with open(out_path, 'w', encoding='utf-8') as out:
        for qid, items in reranked.items():
            for rank, (doc, score) in enumerate(items, start=1):
                out.write(f'{qid}\t{rank}\t{doc}\t{score:.9g}\n')

    source_meta = os.path.splitext(args.run)[0] + '.json'
    source_sizes = os.path.splitext(args.run)[0] + '_sizes.tsv'
    n_pairs = sum(len(items) for items in reranked.values())
    n_truncated = reranker.n_truncated([(requests[qid], documents[doc])
                                        for qid, items in reranked.items() for doc, _ in items])
    source_commit = None
    if os.path.isfile(source_meta):
        with open(source_meta, encoding='utf-8') as f:
            source_commit = json.load(f).get('git_commit')
    meta = {'system': name, 'split': args.split, 'n_queries': len(reranked), 'pool_n': args.n,
            'reranked_run': os.path.relpath(args.run, ROOT),
            'reranked_run_sha256': file_sha256(args.run),
            'reranked_run_git_commit': source_commit,
            # An E run's Boolean set sizes stay those of the run that was reranked (reranking only reorders).
            'set_sizes_file': os.path.relpath(source_sizes, ROOT) if os.path.isfile(source_sizes) else None,
            'reranker_alias': args.reranker, **{f'reranker_{k}': v for k, v in setting.items()},
            'reranker_input': 'request text as in the split (constraint sentence included) + document title and '
                              f'text; pairs over {setting["max_tokens"]} tokens have the document cut, never the '
                              'request',
            'device': reranker.device, 'device_name': reranker.device_name,
            'batch_size': BATCH[reranker.device], 'cache_file': os.path.relpath(cache.path, ROOT),
            'pairs_scored': n_pairs, 'pairs_from_cache': hits, 'pairs_truncated': n_truncated,
            'corpus': 'data/scifact.tsv', 'corpus_sha256': corpus_sha256(),
            'queries_sha256': file_sha256(queries_path(args.split)),
            'transformers_version': version('transformers'), 'torch_version': version('torch'),
            'git_commit': commit, 'git_dirty': dirty, 'run_at': datetime.datetime.now().isoformat(timespec='seconds')}
    with open(os.path.splitext(out_path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    print(f'{name} on {args.split}: {len(reranked)} queries, {n_pairs} pairs ({hits} from the cache) -> {out_path}')
    print(f'Score it with:  venv\\Scripts\\python.exe -m harness.evaluate --run {out_path} --split {args.split}')


if __name__ == '__main__':
    main()
