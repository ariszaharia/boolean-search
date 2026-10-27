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

from baseline_pyserini import CORPUS, ROOT, WORK, words_of
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
  1. Use 2 or 3 concepts, never more than 3, joined with AND. A concept is a specific thing: a gene, protein, drug, disease, organism, cell type, method or measured quantity.
  2. Never use a general word as a concept on its own, not even inside an OR: effect, role, development, increase, decrease, reduce, upregulation, expression, regulation, level, levels, risk, acute, severe, patients, study, cells, function, activity.
  3. Write each concept as one word, or as an established name of 2 or 3 words that an abstract would use word for word ("nitric oxide", "embryonic stem cells"), with OR of its variants: plural forms, spellings, abbreviations, synonyms. Words that can be written with or without a hyphen or space need both forms: (+ proinflammatory "pro inflammatory"), (+ pd1 "pd 1"). Never put wording copied from the request in quotes.
  4. The claim may say that something does not happen. Words like no, not, lack, absence, without, fails, deficiency, less, lower, negatively are part of the claim: keep the concept they apply to as a normal AND operand, and never write (- ...) for them.
  5. Write (- ...) only when the request adds a separate sentence that rules something out of the results, and then exclude every common form of it. Otherwise the query has no (- ...) at all.
  6. If the request requires a specific term to appear, make that term one of the AND operands (it counts as one of the concepts).
Answer in exactly three lines and nothing else:
  Concepts: the 2 or 3 concepts you chose, separated by ;
  Exclude: what the request explicitly rules out of the results, or none
  Query: the query, on one line (with (- ...) only if Exclude is not none)"""

# (train claim id, request, concepts, exclude, query). Constraint sentences worded unlike the targeted set's phrasings.
EXAMPLES = [
    ('1125', 'Systemic immunosuppressive (IS) therapy does not enhance the chance of cancer mortality in patients '
             'with inflammatory eye disease (IED)',
     'immunosuppressive therapy; cancer; inflammatory eye disease', 'none',
     '(^ (+ immunosuppressive immunosuppression) (+ cancer malignancy) (+ "eye disease" "ocular inflammation" uveitis))'),
    ('334', 'Depletion of nitric oxide is responsible for vasospasm.',
     'nitric oxide; vasospasm', 'none',
     '(^ "nitric oxide" vasospasm)'),
    ('566', 'In S. cerevisiae, the absence of RNA surveillance pathways reduces sensitivity to high iron conditions.',
     'RNA surveillance; iron; S. cerevisiae', 'none',
     '(^ "rna surveillance" iron (+ cerevisiae yeast))'),
    ('326', 'Deletion of ATM and Rad3 related protein leads to acute tissue loss. Ignore any work done in rats.',
     'ATR (ATM and Rad3 related); deletion', 'rats',
     '(- (^ (+ atr "atm and rad3 related") (+ deletion knockout)) (+ rat rats))'),
    ('1209', 'The deamination of cytidine to uridine on the minus strand of viral DNA can inactivate the viral genome '
             'by inducing G-to-A mutations. The paper has to discuss APOBEC3G.',
     'deamination; G-to-A mutation; APOBEC3G (required)', 'none',
     '(^ (+ deamination deaminase) (+ "g to a" hypermutation) apobec3g)'),
]
EXAMPLE_CLAIMS = [example[0] for example in EXAMPLES]


def base_messages():
    """The shared start of every conversation: the system prompt and the examples as earlier turns."""
    messages = [{'role': 'system', 'content': SYSTEM_PROMPT}]
    for _, request, concepts, exclude, query in EXAMPLES:
        answer = f'Concepts: {concepts}\nExclude: {exclude}\nQuery: {query}'
        messages += [{'role': 'user', 'content': request}, {'role': 'assistant', 'content': answer}]
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
    """The query from a model answer: code fences removed; the text after "Query:" if there is such a line,
    otherwise the first non-empty line; Greek letters as names."""
    text = re.sub(r'```[a-zA-Z]*', '', answer).strip()
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    query_lines = [l for l in lines if l.lower().startswith('query:')]
    line = query_lines[0][len('query:'):].strip() if query_lines else (lines[0] if lines else '')
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


def retry_hint(error):
    """A short fixed explanation for the engine errors a model most often cannot fix from the message alone."""
    if "'-' needs at least two operands" in error:
        return 'If the request does not ask to rule anything out, remove the (- ...) part entirely. '
    if 'unexpected text after the query' in error:
        return 'The whole query must be one expression: wrap all parts in one (^ ...), and write nothing after it. '
    return ''


def write_queries(conversations, llm, boolean_cli, max_retries=MAX_RETRIES):
    """The core loop: asks the model for a query at the end of each conversation ({query_id: messages}, each ending
    with a user turn) and sends a rejected query back with the engine's message, at most max_retries times.
    Returns {query_id: {'query', 'ok', 'attempts': [{'answer', 'query', 'error'}]}}; each conversation is extended
    in place with the rejected answers and messages, but not with the last answer."""
    result = {qid: {'query': '', 'ok': False, 'attempts': []} for qid in conversations}
    pending = list(conversations)
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
                                                f'{retry_hint(failures[qid])}Answer again in the same three lines, '
                                                'with a corrected query.'}]
                next_pending.append(qid)
            else:
                result[qid]['ok'] = True
        pending = next_pending
    return result


def plan(requests, llm, boolean_cli, max_retries=MAX_RETRIES):
    """E1. requests: [(query_id, text)]. Returns (result, conversations): result as in write_queries, and the
    conversations (without the last answer), which E2 continues."""
    conversations = {qid: base_messages() + [{'role': 'user', 'content': text}] for qid, text in requests}
    return write_queries(conversations, llm, boolean_cli, max_retries), conversations


# --- E2: relaxation (hypotheses.md, clarification of 2026-10-27) ---

MAX_RELAX_ROUNDS = 2
RELAX_MESSAGE = ('This query is valid but matched no documents. Write a broader query: fewer AND parts, more '
                 'variants and synonyms, shorter phrases. Keep every exclusion (- ...) and every term the request '
                 'requires. Answer again in the same three lines.')
QUERY_TOKEN = re.compile(r'\(|\)|"[^"]*"|[^\s()"]+')


def parse_tree(query):
    """A query as nested lists: (- (^ a "b c") d) -> ['-', ['^', 'a', '"b c"'], 'd']. Only for queries the engine
    has already accepted, so it does not check the syntax again."""
    tokens = QUERY_TOKEN.findall(query)

    def node(i):
        if tokens[i] != '(':
            return tokens[i], i + 1
        operator, children, i = tokens[i + 1], [], i + 2
        while tokens[i] != ')':
            child, i = node(i)
            children.append(child)
        return [operator] + children, i + 1

    return node(0)[0]


def excluded_words(query):
    """Every word inside the excluded operands (all but the first) of every (- ...) in the query, as the engine
    splits them (lowercase, alphanumeric runs)."""
    found = set()

    def leaves(n):
        return [n] if isinstance(n, str) else [leaf for child in n[1:] for leaf in leaves(child)]

    def walk(n):
        if isinstance(n, str):
            return
        if n[0] == '-':
            for operand in n[2:]:
                for leaf in leaves(operand):
                    found.update(words_of(leaf.strip('"')))
        for child in n[1:]:
            walk(child)

    walk(parse_tree(query))
    return found


def set_sizes(pairs, boolean_cli):
    """{query_id: size of the Boolean set on the corpus} for queries the engine accepts."""
    os.makedirs(WORK, exist_ok=True)
    paths = {name: os.path.join(WORK, f'relax_{name}.tsv') for name in ('queries', 'run', 'sizes', 'failures')}
    with open(paths['queries'], 'w', encoding='utf-8') as f:
        for qid, query in pairs:
            f.write(f'{qid}\t{query}\n')
    subprocess.run([boolean_cli, CORPUS, paths['queries'], paths['run'], paths['sizes'], paths['failures'], '1'],
                   check=True, capture_output=True)
    with open(paths['sizes'], encoding='utf-8') as f:
        return {line.split('\t')[0]: int(line.split('\t')[1]) for line in f if line.strip()}


def relax(first, conversations, llm, boolean_cli, rounds=MAX_RELAX_ROUNDS):
    """E2 from E1's result and conversations (plan). For each valid E1 query with an empty set: up to `rounds`
    rounds in which the model is asked for a broader query; the first non-empty set ends them. A relaxed query that
    drops a word E1 excluded is rejected. Returns {query_id: {'query' (E2's), 'ok', 'e1_query', 'e1_size',
    'rounds': [{'attempts', 'query', 'ok', 'rejected', 'size'}]}}."""
    valid = [qid for qid in first if first[qid]['ok']]
    e1_sizes = set_sizes([(qid, first[qid]['query']) for qid in valid], boolean_cli)
    result = {qid: {'query': first[qid]['query'], 'ok': first[qid]['ok'], 'e1_query': first[qid]['query'],
                    'e1_size': e1_sizes.get(qid), 'rounds': []} for qid in first}
    pending = [qid for qid in valid if e1_sizes[qid] == 0]
    note = {qid: '' for qid in pending}
    last_answer = {qid: first[qid]['attempts'][-1]['answer'] for qid in pending}
    for r in range(rounds):
        if not pending:
            break
        print(f'  relaxation round {r + 1}: {len(pending)} empty queries', flush=True)
        for qid in pending:
            conversations[qid] += [{'role': 'assistant', 'content': last_answer[qid]},
                                   {'role': 'user', 'content': note[qid] + RELAX_MESSAGE}]
        written = write_queries({qid: conversations[qid] for qid in pending}, llm, boolean_cli)
        accepted = []
        for qid in pending:
            w = written[qid]
            last_answer[qid] = w['attempts'][-1]['answer']
            entry = {'attempts': w['attempts'], 'query': w['query'], 'ok': w['ok'], 'rejected': None, 'size': None}
            if w['ok']:
                dropped = sorted(excluded_words(result[qid]['e1_query']) - excluded_words(w['query']))
                if dropped:
                    entry['rejected'] = f'dropped the exclusion of: {" ".join(dropped)}'
                    note[qid] = f'Your last query dropped the exclusion of {" ".join(dropped)}; keep it. '
                else:
                    note[qid] = ''
                    accepted.append(qid)
            result[qid]['rounds'].append(entry)
        sizes = set_sizes([(qid, written[qid]['query']) for qid in accepted], boolean_cli)
        next_pending = []
        for qid in pending:
            entry = result[qid]['rounds'][-1]
            if qid in sizes:
                entry['size'] = sizes[qid]
                result[qid]['query'] = written[qid]['query']
            if sizes.get(qid, 0) == 0:
                next_pending.append(qid)
        pending = next_pending
    return result
