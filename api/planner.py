"""
The planner of E1: an LLM turns a request (a claim, maybe with a constraint sentence) into one Boolean query for
the engine. Used by api/run_planner.py.

  - The model is reached through the OpenAI-style chat API (the openai package). Ollama on this computer offers the
    same API, so moving to OpenAI later changes only MODELS below, not the code.
  - Every model call is cached in results/llm_cache/<model alias>.jsonl, one JSON object per line: the key, the
    messages after the shared prompt, the answer, the token counts and the time. The cache is also the log of
    every call (hypotheses.md: "everything logged and cached"); rerunning with the same settings costs nothing and
    gives the same queries.
  - The answer is cleaned (code fences and extra lines removed, Greek letters written as names, as in the corpus).
  - A query the engine rejects (ParseError) goes back to the model with the engine's message, at most MAX_RETRIES
    times. A query still rejected after that is a failed query: it has no set size (hypotheses.md, H1 clarification).

The prompt's examples are train claims that are not base claims of the targeted set, and their constraint sentences
are worded unlike the targeted set's fixed phrasings (api/targeted_draft.py), so the prompt does not rehearse the
targeted set. Each example query was checked to find its claim's relevant document at rank 1 (2026-10-27).
"""
import datetime
import hashlib
import json
import os
import re
import subprocess
import urllib.request

from baseline_pyserini import ROOT, WORK
from greek import greek_to_names

# Each model the planner can use. version pins the exact weights: for Ollama, the digest of the model files,
# checked against the local copy before any call (check_version).
MODELS = {
    'qwen2.5-7b': {'model': 'qwen2.5:7b', 'base_url': 'http://localhost:11434/v1', 'api_key_env': None,
                   'version': '845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e',
                   'description': 'Qwen2.5-7B-Instruct, Q4_K_M, local through Ollama'},
}
TEMPERATURE = 0
SEED = 1
MAX_TOKENS = 200
MAX_RETRIES = 2
CACHE_DIR = os.path.join(ROOT, 'results', 'llm_cache')

SYSTEM_PROMPT = """You turn a request about scientific findings into ONE Boolean query for an exact-match search engine over scientific abstracts (title and abstract).

Query language (prefix notation; every operator goes inside parentheses):
  (^ A B ...)      AND: documents that contain all of A, B, ...
  (+ A B ...)      OR: documents that contain at least one of A, B, ...
  (- A B ...)      NOT: documents that match A but none of B, ...; it needs at least two operands, and the first one is what you want
  "nitric oxide"   an exact phrase, in double quotes
  vasospasm        one exact word
Words are matched exactly, ignoring case: there is no stemming and there are no synonyms, so write the variants you need with OR, for example (+ tumor tumors tumour tumours). Hyphens and other punctuation separate words, so "g to a" matches G-to-A. Write Greek letters as their names (alpha, beta, kappa).

How to write a good query:
  1. Pick the 2 to 4 concepts that matter most and join them with AND. Leave out common words (effect, patients, increase, study).
  2. Write each concept as one word or a short fixed term of 2 or 3 words, with OR of its variants: plural forms, spellings, abbreviations, close synonyms. Never put a long part of the sentence in quotes.
  3. A negation inside the claim ("does not", "no", "lacks", "fails to") is part of what the claim says: documents that discuss it are the ones wanted, so it is never an exclusion.
  4. Use (- ...) only when the request explicitly rules something out of the results, and then exclude every common form of it.
  5. If the request requires a specific term to appear, make that term one of the AND operands.
Answer with the query only, on one line, and nothing else."""

# (train claim id, request, query). Constraint sentences worded unlike the targeted set's phrasings.
EXAMPLES = [
    ('1125', 'Systemic immunosuppressive (IS) therapy does not enhance the chance of cancer mortality in patients '
             'with inflammatory eye disease (IED)',
     '(^ (+ immunosuppressive immunosuppression) (+ cancer malignancy) mortality '
     '(+ "eye disease" "ocular inflammation" uveitis))'),
    ('334', 'Depletion of nitric oxide is responsible for vasospasm.',
     '(^ "nitric oxide" vasospasm)'),
    ('326', 'Deletion of ATM and Rad3 related protein leads to acute tissue loss. Ignore any work done in rats.',
     '(- (^ (+ atr "atm and rad3 related") (+ deletion knockout)) (+ rat rats))'),
    ('1209', 'The deamination of cytidine to uridine on the minus strand of viral DNA can inactivate the viral genome '
             'by inducing G-to-A mutations. The paper has to discuss APOBEC3G.',
     '(^ (+ deamination deaminase) (+ "g to a" hypermutation) apobec3g)'),
]
EXAMPLE_CLAIMS = [claim_id for claim_id, _, _ in EXAMPLES]


def base_messages():
    """The shared start of every conversation: the system prompt and the examples as earlier turns."""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for _, request, query in EXAMPLES:
        messages += [{'role': 'user', 'content': request}, {'role': 'assistant', 'content': query}]
    return messages


def prompt_sha256():
    """Fingerprint of the system prompt and the examples, recorded with every run."""
    return hashlib.sha256(json.dumps(base_messages(), sort_keys=True).encode('utf-8')).hexdigest()


def check_version(alias):
    """Stops if the local Ollama copy of the model is not the pinned version."""
    config = MODELS[alias]
    if 'localhost:11434' not in config['base_url']:
        return
    with urllib.request.urlopen('http://localhost:11434/api/tags') as r:
        digests = {m['name']: m['digest'] for m in json.load(r)['models']}
    if digests.get(config['model']) != config['version']:
        raise SystemExit(f'{config["model"]}: local digest {digests.get(config["model"])} is not the pinned '
                         f'{config["version"]}. Do not run with other weights; pin the new version first.')


class LLM:
    """Chat completions with a cache. chat() returns the answer text; calls and cache_hits count what happened."""

    def __init__(self, alias):
        from openai import OpenAI
        self.alias, self.config = alias, MODELS[alias]
        check_version(alias)
        key = os.environ[self.config['api_key_env']] if self.config['api_key_env'] else 'ollama'
        self.client = OpenAI(base_url=self.config['base_url'], api_key=key)
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.cache_path = os.path.join(CACHE_DIR, f'{alias}.jsonl')
        self.cache = {}
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, encoding='utf-8') as f:
                for line in f:
                    entry = json.loads(line)
                    self.cache[entry['key']] = entry['answer']
        self.calls, self.cache_hits = 0, 0

    def chat(self, messages):
        settings = {'model': self.config['model'], 'version': self.config['version'], 'temperature': TEMPERATURE,
                    'seed': SEED, 'max_tokens': MAX_TOKENS, 'messages': messages}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode('utf-8')).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        r = self.client.chat.completions.create(model=self.config['model'], messages=messages,
                                                temperature=TEMPERATURE, seed=SEED, max_tokens=MAX_TOKENS)
        answer = r.choices[0].message.content or ''
        self.calls += 1
        shared = len(base_messages())
        entry = {'key': key, 'model': self.config['model'], 'version': self.config['version'],
                 'prompt_sha256': prompt_sha256(), 'messages_after_prompt': messages[shared:], 'answer': answer,
                 'prompt_tokens': r.usage.prompt_tokens if r.usage else None,
                 'completion_tokens': r.usage.completion_tokens if r.usage else None,
                 'time': datetime.datetime.now().isoformat(timespec='seconds')}
        with open(self.cache_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        self.cache[key] = answer
        return answer


def clean(answer):
    """The query from a model answer: code fences removed, the first non-empty line, Greek letters as names."""
    text = re.sub(r'```[a-zA-Z]*', '', answer).strip()
    line = next((l.strip() for l in text.splitlines() if l.strip()), '')
    return greek_to_names(line)


def parse_failures(pairs, boolean_cli):
    """{query_id: message} for the queries the engine cannot parse. Parsing does not depend on the documents, so
    the check runs on a one-document corpus, which takes a moment instead of loading SciFact."""
    os.makedirs(WORK, exist_ok=True)
    corpus = os.path.join(WORK, 'parse_check_corpus.tsv')
    with open(corpus, 'w', encoding='utf-8') as f:
        f.write('1\tx\n')
    paths = {name: os.path.join(WORK, f'parse_check_{name}.tsv') for name in ('queries', 'run', 'sizes', 'failures')}
    with open(paths['queries'], 'w', encoding='utf-8') as f:
        for qid, query in pairs:
            f.write(f'{qid}\t{query}\n')
    subprocess.run([boolean_cli, corpus, paths['queries'], paths['run'], paths['sizes'], paths['failures']],
                   check=True, capture_output=True)
    with open(paths['failures'], encoding='utf-8') as f:
        return {line.split('\t')[0]: line.rstrip('\n').split('\t', 2)[2] for line in f if line.strip()}


def plan(requests, llm, boolean_cli, max_retries=MAX_RETRIES):
    """requests: [(query_id, text)]. Returns {query_id: {'query': the last query written, 'ok': whether it parses,
    'attempts': [{'answer', 'query', 'error'}]}}. A rejected query is sent back with the engine's message."""
    conversations = {qid: base_messages() + [{'role': 'user', 'content': text}] for qid, text in requests}
    result = {qid: {'query': '', 'ok': False, 'attempts': []} for qid, _ in requests}
    pending = [qid for qid, _ in requests]
    for attempt in range(1 + max_retries):
        if not pending:
            break
        for i, qid in enumerate(pending, start=1):
            answer = llm.chat(conversations[qid])
            query = clean(answer)
            result[qid]['attempts'].append({'answer': answer, 'query': query, 'error': None})
            result[qid]['query'] = query
            if i % 25 == 0:
                print(f'  attempt {attempt + 1}: {i} / {len(pending)} queries written', flush=True)
        failures = parse_failures([(qid, result[qid]['query'] or '(') for qid in pending], boolean_cli)
        next_pending = []
        for qid in pending:
            if qid in failures:
                result[qid]['attempts'][-1]['error'] = failures[qid]
                conversations[qid] += [
                    {'role': 'assistant', 'content': result[qid]['attempts'][-1]['answer']},
                    {'role': 'user', 'content': f'The engine rejected this query: {failures[qid]}. '
                                                'Write a corrected query, on one line.'}]
                next_pending.append(qid)
            else:
                result[qid]['ok'] = True
        pending = next_pending
    return result
