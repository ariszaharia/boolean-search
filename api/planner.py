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
import time
import urllib.error
import urllib.request

from openai import OpenAI, RateLimitError

from baseline_pyserini import CORPUS, ROOT, WORK, words_of
from greek import greek_to_names

# Each model the planner can use. version pins the exact weights: for Ollama, the digest of the model files,
# checked against the local copy before any call (check_version).
MODELS = {
    'qwen2.5-7b': {'model': 'qwen2.5:7b', 'base_url': 'http://localhost:11434/v1', 'api_key_env': None,
                   'version': '845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e',
                   'description': 'Qwen2.5-7B-Instruct, Q4_K_M, local through Ollama'},
    # Free tier. The endpoint rejects seed; 'none' reasoning is rejected, so the lowest level is pinned explicitly
    # (tested 2026-11-11: same three-line answers as the default). version is Google's own version string of the
    # stable model id, checked before any call. min_interval: seconds between calls (free-tier requests per minute).
    'gemini-3.5-flash-lite': {'model': 'gemini-3.5-flash-lite',
                              'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
                              'api_key_env': 'GEMINI_API_KEY', 'version': '3.5-flash-lite-07-2026',
                              'params': {'reasoning_effort': 'minimal'}, 'seed': False, 'min_interval': 7,
                              'description': 'Gemini 3.5 Flash-Lite, reasoning effort minimal, Gemini API free tier'},
    # Paid (OpenAI). price: dollars per million (input, output) tokens, for the cost guard and the run record;
    # output includes hidden reasoning tokens. OpenAI lists no dated snapshot for Sol (GET /v1/models, 2026-11-14),
    # so version pins the id with its 'created' time, checked before any call (a replaced model gets a new one);
    # each call also logs the model name the response reports. version None = not pinned: check_version refuses.
    # GPT-6 Sol accepts temperature only with reasoning effort none (OpenAI's migration guide, checked 2026-11-11).
    'gpt-6-sol': {'model': 'gpt-6-sol', 'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY',
                  'version': 'gpt-6-sol created 1789405937', 'params': {'reasoning_effort': 'none'},
                  'price': (2.0, 10.0),
                  'max_tokens_param': 'max_completion_tokens',
                  'description': 'OpenAI GPT-6 Sol, reasoning effort none, temperature 0'},
    # GPT-6 Luna: the small, cheap model of the same family (third-party price listings, 2026-11-14: $0.10 / $0.50
    # per million tokens, 20x below Sol). Same settings as GPT-6 Sol; pinned like it.
    'gpt-6-luna': {'model': 'gpt-6-luna', 'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY',
                   'version': 'gpt-6-luna created 1789406102', 'params': {'reasoning_effort': 'none'},
                   'price': (0.10, 0.50), 'max_tokens_param': 'max_completion_tokens',
                   'description': 'OpenAI GPT-6 Luna, reasoning effort none, temperature 0'},
    # GPT-6.1 Sol has no 'none' effort and then takes no temperature; it needs max_completion_tokens, with room
    # for the reasoning tokens.
    'gpt-6.1-sol': {'model': 'gpt-6.1-sol', 'base_url': 'https://api.openai.com/v1', 'api_key_env': 'OPENAI_API_KEY',
                    'version': None, 'params': {'reasoning_effort': 'low'}, 'temperature': False,
                    'max_tokens_param': 'max_completion_tokens', 'max_tokens': 2000, 'price': (2.0, 10.0),
                    'description': 'OpenAI GPT-6.1 Sol, reasoning effort low, no temperature'},
}
GOOGLE_MODELS = 'https://generativelanguage.googleapis.com/v1beta/models/'
OPENAI_MODELS = 'https://api.openai.com/v1/models/'
DEFAULT_MAX_COST = 1.0  # dollars per run for a priced model, unless run_planner --max-cost says otherwise
# Limits on everything ever spent, summed over the cost_usd of all cache files (Aris's prepaid credit: $5, of which
# $4.75 was left on 2026-11-14 after $0.26 of logged calls). Development may not take the total past
# DEV_SPEND_LIMIT, so the rest stays for the confirmatory runs; nothing runs past TOTAL_SPEND_LIMIT.
DEV_SPEND_LIMIT = 1.80
TOTAL_SPEND_LIMIT = 4.90
RATE_LIMIT_WAITS = [30, 60, 120]  # seconds to wait after each 429 before giving up (a daily limit does not clear)
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


# --- Planner versions. v4: the prompt developed with Qwen and the E2 rule of hypotheses.md (clarification of
# 2026-10-27). v5 (development, 2026-11-14): (a) the request comes with the first words of the top PEEK_K abstracts
# BM25 finds for it, so the model can see the corpus's wording; (b) E2 also relaxes sets smaller than MIN_SET, keeps
# E1's set (the E2 query is the OR of E1's and the relaxed one), restates the request, gives the document count of
# every part of the query, and allows "any 2 of 3 concepts". v5 is a deviation and must be entered in hypotheses.md
# before any run on the targeted set or test.
PLANNER = 'v4'
PEEK_K = 5
PEEK_WORDS = 25
MIN_SET = 2
RULE_7 = ('\n  7. After the request you see the first words of the top abstracts a keyword search finds for it. They '
          'are not necessarily relevant, but they show how abstracts word things: when they name one of your '
          'concepts differently (a drug class, a broader term, a synonym, another spelling), add that wording as an OR '
          'variant. Never add a concept only because it appears there.')


def set_planner(version):
    global PLANNER
    if version not in ('v4', 'v5'):
        raise ValueError(version)
    PLANNER = version


_peek_cache = {}


def peek(text):
    """v5: the first PEEK_WORDS words of the top PEEK_K BM25 abstracts for a request (B1's index and settings)."""
    if text not in _peek_cache:
        if 'searcher' not in _peek_cache:
            from pyserini.search.lucene import LuceneSearcher
            from run_baselines import B as BM25_B, INDEX_DEFAULT, K1, build_default_index
            build_default_index()
            searcher = LuceneSearcher(INDEX_DEFAULT)
            searcher.set_bm25(K1, BM25_B)
            docs = {}
            with open(CORPUS, encoding='utf-8') as f:
                for line in f:
                    doc_id, body = line.rstrip('\n').split('\t', 1)
                    docs[doc_id] = body
            _peek_cache['searcher'], _peek_cache['docs'] = searcher, docs
        hits = _peek_cache['searcher'].search(' '.join(words_of(text)), k=PEEK_K)
        _peek_cache[text] = [' '.join(_peek_cache['docs'][h.docid].split()[:PEEK_WORDS]) for h in hits]
    return _peek_cache[text]


def request_message(text):
    """The user turn for a request: the request alone (v4), or with the BM25 peek (v5)."""
    if PLANNER == 'v4':
        return text
    lines = '\n'.join(f'{i}. {snippet} ...' for i, snippet in enumerate(peek(text), start=1))
    return f'{text}\n\nTop abstracts of a keyword search for this request (first words):\n{lines}'


def base_messages():
    """The shared start of every conversation: the system prompt and the examples as earlier turns."""
    prompt = SYSTEM_PROMPT if PLANNER == 'v4' else SYSTEM_PROMPT.replace('\nAnswer in exactly', RULE_7 + '\nAnswer in exactly')
    messages = [{'role': 'system', 'content': prompt}]
    for _, request, concepts, exclude, query in EXAMPLES:
        answer = f'Concepts: {concepts}\nExclude: {exclude}\nQuery: {query}'
        messages += [{'role': 'user', 'content': request_message(request)}, {'role': 'assistant', 'content': answer}]
    return messages


def prompt_sha256():
    """Fingerprint of the system prompt and the examples, recorded with every run."""
    return hashlib.sha256(json.dumps(base_messages(), sort_keys=True).encode('utf-8')).hexdigest()


def check_version(alias):
    """Stops if the model is not the pinned version: the local Ollama copy's digest, or the version Google reports
    for a Gemini model id, or (OpenAI) that the pinned id exists. Every check is free."""
    config = MODELS[alias]
    if config['base_url'].startswith('https://api.openai.com/'):
        if not config['version']:
            raise SystemExit(f'{alias}: not pinned. Set version to "<id> created <time>" from GET /v1/models/<id> '
                             f'before any paid call.')
        request = urllib.request.Request(OPENAI_MODELS + config['model'],
                                         headers={'Authorization': 'Bearer ' + os.environ[config['api_key_env']]})
        try:
            with urllib.request.urlopen(request) as r:
                info = json.load(r)
        except urllib.error.HTTPError as e:
            raise SystemExit(f'{config["model"]}: OpenAI does not list this model id (HTTP {e.code}).')
        version = f'{info["id"]} created {info["created"]}'
        if version != config['version']:
            raise SystemExit(f'{config["model"]}: OpenAI reports "{version}", not the pinned "{config["version"]}". '
                             f'Do not run with another model; pin the new one first.')
        return
    if config['base_url'].startswith('https://generativelanguage.googleapis.com/'):
        request = urllib.request.Request(GOOGLE_MODELS + config['model'],
                                         headers={'x-goog-api-key': os.environ[config['api_key_env']]})
        with urllib.request.urlopen(request) as r:
            version = json.load(r).get('version')
        if version != config['version']:
            raise SystemExit(f'{config["model"]}: Google reports version {version}, not the pinned '
                             f'{config["version"]}. Do not run with another version; pin the new one first.')
        return
    if 'localhost:11434' not in config['base_url']:
        return
    with urllib.request.urlopen('http://localhost:11434/api/tags') as r:
        digests = {m['name']: m['digest'] for m in json.load(r)['models']}
    if digests.get(config['model']) != config['version']:
        raise SystemExit(f'{config["model"]}: local digest {digests.get(config["model"])} is not the pinned '
                         f'{config["version"]}. Do not run with other weights; pin the new version first.')


class LLM:
    """Chat completions with a cache. chat() returns the answer text; calls and cache_hits count what happened."""

    def __init__(self, alias, max_cost=None, spend_limit=TOTAL_SPEND_LIMIT, max_tokens=None):
        """max_cost: dollars this run may spend on a priced model (default DEFAULT_MAX_COST); spend_limit: what all
        logged calls together may have cost. The run stops before a call once either is reached. Finished calls are
        cached, so a rerun continues from there."""
        self.alias, self.config = alias, MODELS[alias]
        self.max_cost = max_cost if max_cost is not None else DEFAULT_MAX_COST
        self.spend_limit = spend_limit
        self.max_tokens = max_tokens  # overrides the model's answer limit (the agent's answers are longer)
        self.spent_before = 0.0
        for name in os.listdir(CACHE_DIR) if os.path.isdir(CACHE_DIR) else []:
            with open(os.path.join(CACHE_DIR, name), encoding='utf-8') as f:
                self.spent_before += sum(json.loads(line).get('cost_usd') or 0 for line in f if line.strip())
        self.cost, self.prompt_tokens, self.completion_tokens = 0.0, 0, 0
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
        self.last_call = 0.0

    def request_settings(self):
        """What is sent with every call besides the messages. Also part of the cache key; for a model without
        extra params or a seed exception this is exactly the old key, so its cached calls stay valid."""
        settings = {}
        if self.config.get('temperature', True):
            settings['temperature'] = TEMPERATURE
        settings[self.config.get('max_tokens_param', 'max_tokens')] = (getattr(self, 'max_tokens', None)
                                                                       or self.config.get('max_tokens', MAX_TOKENS))
        if self.config.get('seed', True):
            settings['seed'] = SEED
        settings.update(self.config.get('params', {}))
        return settings

    def create(self, messages):
        """One API call, spaced by min_interval; a 429 (too many requests) is waited out RATE_LIMIT_WAITS times,
        then the run stops. Every finished call is already in the cache, so rerunning continues from there."""
        if 'price' in self.config and self.cost >= self.max_cost:
            raise SystemExit(f'Cost cap reached: about ${self.cost:.3f} spent in this run ({self.calls} calls, cap '
                             f'${self.max_cost:.2f}). The calls are cached; rerun with a higher --max-cost to go on.')
        if 'price' in self.config and self.spent_before + self.cost >= self.spend_limit:
            raise SystemExit(f'Spending limit reached: about ${self.spent_before + self.cost:.3f} logged in total '
                             f'(limit ${self.spend_limit:.2f}). Stopped before the next call; the calls are cached.')
        for wait in RATE_LIMIT_WAITS + [None]:
            pause = self.config.get('min_interval', 0) - (time.monotonic() - self.last_call)
            if pause > 0:
                time.sleep(pause)
            try:
                self.last_call = time.monotonic()
                return self.client.chat.completions.create(model=self.config['model'], messages=messages,
                                                           **self.request_settings())
            except RateLimitError as e:
                if wait is None:
                    raise SystemExit(f'Rate limit still reached after {len(RATE_LIMIT_WAITS)} waits ({e}). Probably '
                                     f'the daily limit: rerun later; the {self.calls} calls made so far are cached.')
                print(f'  rate limit (429), waiting {wait} s')
                time.sleep(wait)

    def chat(self, messages):
        settings = {'model': self.config['model'], 'version': self.config['version'], **self.request_settings(),
                    'messages': messages}
        key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode('utf-8')).hexdigest()
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        r = self.create(messages)
        answer = r.choices[0].message.content or ''
        self.calls += 1
        # The agent (agent.py) sets shared_len and prompt_hash for its own prompt; the planner uses its examples.
        shared = getattr(self, 'shared_len', None) or len(base_messages())
        entry = {'key': key, 'model': self.config['model'], 'version': self.config['version'],
                 'prompt_sha256': getattr(self, 'prompt_hash', None) or prompt_sha256(),
                 'messages_after_prompt': messages[shared:], 'answer': answer,
                 'prompt_tokens': r.usage.prompt_tokens if r.usage else None,
                 'completion_tokens': r.usage.completion_tokens if r.usage else None,
                 'time': datetime.datetime.now().isoformat(timespec='seconds')}
        if 'price' in self.config and r.usage:
            details = getattr(r.usage, 'completion_tokens_details', None)
            entry['reasoning_tokens'] = getattr(details, 'reasoning_tokens', None)
            entry['served_model'] = getattr(r, 'model', None)
            entry['cost_usd'] = (r.usage.prompt_tokens * self.config['price'][0]
                                 + r.usage.completion_tokens * self.config['price'][1]) / 1e6
            self.cost += entry['cost_usd']
            self.prompt_tokens += r.usage.prompt_tokens
            self.completion_tokens += r.usage.completion_tokens
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
    conversations = {qid: base_messages() + [{'role': 'user', 'content': request_message(text)}]
                     for qid, text in requests}
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


RELAX_MESSAGE_V5 = ('This query matched {size} documents, too few. The request was: "{request}"\nDocuments in the '
                    'corpus that contain each part of your query: {counts}.\nWrite a broader query for this request: '
                    'replace or drop parts that occur in no or very few documents, add variants and synonyms, shorten '
                    'phrases. With 3 concepts you may also require any 2 of them: (+ (^ A B) (^ A C) (^ B C)). Keep '
                    'every exclusion (- ...) and every term the request requires. Answer again in the same three '
                    'lines.')


def to_query(node):
    """parse_tree's lists back to query text."""
    return node if isinstance(node, str) else '(' + ' '.join([node[0]] + [to_query(c) for c in node[1:]]) + ')'


def part_counts(query, boolean_cli):
    """v5 feedback: 'part: N' for every word or phrase of the query and every operand of its top AND (or of the
    AND inside a top NOT), with N the number of corpus documents that match it."""
    tree = parse_tree(query)
    parts = []

    def leaves(n):
        return [n] if isinstance(n, str) else [leaf for child in n[1:] for leaf in leaves(child)]

    top = tree[1] if isinstance(tree, list) and tree[0] == '-' else tree
    if isinstance(top, list) and top[0] == '^':
        parts += [to_query(c) for c in top[1:] if not isinstance(c, str)]
    parts += [leaf for leaf in leaves(tree) if leaf not in parts]
    sizes = set_sizes([(str(i), p) for i, p in enumerate(parts)], boolean_cli)
    return '; '.join(f'{p}: {sizes.get(str(i), 0)}' for i, p in enumerate(parts))


def relax(first, conversations, llm, boolean_cli, rounds=MAX_RELAX_ROUNDS, requests=None):
    """E2 from E1's result and conversations (plan). v4: for each valid E1 query with an empty set, up to `rounds`
    rounds in which the model is asked for a broader query; the first non-empty set ends them. v5: also sets smaller
    than MIN_SET, with the v5 message (requests: {query_id: text}); the E2 query is the OR of E1's query and the
    relaxed one, so E1's documents stay, and the rounds end once that has MIN_SET documents. Both: a relaxed query
    that drops a word E1 excluded is rejected. Returns {query_id: {'query' (E2's), 'ok', 'e1_query', 'e1_size',
    'rounds': [{'attempts', 'query', 'ok', 'rejected', 'size'}]}}."""
    v5 = PLANNER == 'v5'
    target = MIN_SET if v5 else 1
    valid = [qid for qid in first if first[qid]['ok']]
    e1_sizes = set_sizes([(qid, first[qid]['query']) for qid in valid], boolean_cli)
    result = {qid: {'query': first[qid]['query'], 'ok': first[qid]['ok'], 'e1_query': first[qid]['query'],
                    'e1_size': e1_sizes.get(qid), 'rounds': []} for qid in first}
    pending = [qid for qid in valid if e1_sizes[qid] < target]
    note = {qid: '' for qid in pending}
    last_answer = {qid: first[qid]['attempts'][-1]['answer'] for qid in pending}
    last_query = {qid: first[qid]['query'] for qid in pending}
    size_now = {qid: e1_sizes[qid] for qid in pending}
    for r in range(rounds):
        if not pending:
            break
        print(f'  relaxation round {r + 1}: {len(pending)} queries with fewer than {target} documents', flush=True)
        for qid in pending:
            if v5:
                message = RELAX_MESSAGE_V5.format(size=size_now[qid], request=requests[qid],
                                                  counts=part_counts(last_query[qid], boolean_cli))
            else:
                message = RELAX_MESSAGE
            conversations[qid] += [{'role': 'assistant', 'content': last_answer[qid]},
                                   {'role': 'user', 'content': note[qid] + message}]
        written = write_queries({qid: conversations[qid] for qid in pending}, llm, boolean_cli)
        accepted = []
        for qid in pending:
            w = written[qid]
            last_answer[qid] = w['attempts'][-1]['answer']
            entry = {'attempts': w['attempts'], 'query': w['query'], 'ok': w['ok'], 'rejected': None, 'size': None}
            if w['ok']:
                last_query[qid] = w['query']
                dropped = sorted(excluded_words(result[qid]['e1_query']) - excluded_words(w['query']))
                if dropped:
                    entry['rejected'] = f'dropped the exclusion of: {" ".join(dropped)}'
                    note[qid] = f'Your last query dropped the exclusion of {" ".join(dropped)}; keep it. '
                else:
                    note[qid] = ''
                    accepted.append(qid)
            result[qid]['rounds'].append(entry)

        def combined(qid):
            q = written[qid]['query']
            return f'(+ {result[qid]["e1_query"]} {q})' if v5 and e1_sizes[qid] > 0 else q

        sizes = set_sizes([(qid, combined(qid)) for qid in accepted], boolean_cli)
        next_pending = []
        for qid in pending:
            entry = result[qid]['rounds'][-1]
            if qid in sizes:
                entry['size'] = sizes[qid]
                if not v5 or sizes[qid] > e1_sizes[qid]:
                    result[qid]['query'] = combined(qid)
                size_now[qid] = sizes[qid]
            if sizes.get(qid, size_now[qid]) < target:
                next_pending.append(qid)
        pending = next_pending
    return result
