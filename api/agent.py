"""
E4: a multi-call search agent in the style of Clarke & Smucker, "Boolean queries are all you need?" (arXiv
2607.11362, the Vole), adapted to SciFact. Used by api/run_agent.py; H8 (hypotheses.md) compares it with E2.

As in the paper:
  - The model gets instructions, the request and the transcript so far (append-only, so the provider can cache it),
    and answers each step with exactly one JSON action: run a new Boolean query (same language and engine as E1),
    see more results of an earlier query, read a whole abstract, or stop.
  - Every document it is shown, it grades 0-3 with a short reason. Documents already graded are skipped and the model
    is told how many were skipped.
  - The final ranking is the graded documents, by grade and then by the order they were found. External relevance
    judgments are only used afterwards, by the harness.
Adapted to SciFact (a claim has 1 or 2 relevant abstracts, not dozens; abstracts are short; the budget is small):
  - A query or "more" shows SHOW_N results at a time, each as its first SNIPPET_WORDS words (title and start), not one
    200-word snippet per call.
  - At most max_calls model calls (the run's budget) and MAX_JUDGED graded documents per request.
  - Development (train only, results/agent_dev_log.tsv): v1 searched too narrowly (74 of 144 searches empty, 3.7
    documents graded per claim); v2 tells the model to start broad and narrow only on many results, shows 5 results
    instead of 3, and runs up to 20 calls.
  - The model is not told the number of calls left, so the state after b calls is what a budget of b would have
    given: one run at the largest budget also gives the runs at smaller budgets (H8 compares several budgets).
  - A query's results are ranked by shortest substring ranking (SSR, C = 42), as in the paper (cpp/src/ssr_scorer.cpp,
    boolean_cli's ssr option).
"""
import hashlib
import json
import os
import re
import subprocess

from baseline_pyserini import CORPUS, WORK

SHOW_N = 5          # development v1 (2026-11-24): 3
SNIPPET_WORDS = 60
MAX_JUDGED = 30
ENGINE_K = 100
ANSWER_TOKENS = 400
RANKING = 'ssr'   # how the engine orders a query's results (boolean_cli's last argument)

AGENT_PROMPT = """You are a search agent. You look for the scientific abstracts that are evidence for or against a request (a claim about a scientific finding, sometimes with an extra sentence about which papers to leave out), in a corpus of about 5,000 abstracts (title and abstract). Usually only one or two abstracts are the evidence, and they may word things differently from the request.

Your only search tool is an exact-match Boolean engine. Query language (prefix notation; every operator goes inside parentheses):
  (^ A B ...)      AND: documents that contain all of A, B, ...
  (+ A B ...)      OR: documents that contain at least one of A, B, ...
  (- A B ...)      NOT: documents that match A but none of B, ...; at least two operands, the first one is what you want
  "nitric oxide"   an exact phrase, in double quotes
  vasospasm        one exact word
Words are matched exactly, ignoring case: no stemming and no synonyms, so write the variants you need with OR, for example (+ tumor tumors tumour tumours). Hyphens and other punctuation separate words, so "g to a" matches G-to-A. Write Greek letters as their names (alpha, beta, kappa). The results of a query are ranked by its matches in each document: documents where the parts of the query occur close together, and often, come first.

How to search well:
  - Start broad: AND only the 2 most specific concepts of the request (a gene, protein, drug, disease, organism, cell type, method or measured quantity), each as a large OR of its variants: plurals, spellings, abbreviations, synonyms, broader names (a drug class for a drug). For example (^ urease (+ helicobacter pylori)), not (^ urease (+ helicobacter pylori) (+ subunit subunits) (+ pentamer pentameric)).
  - A search that matches no documents wastes a step. Add a third AND part only when a search matches more than about 50 documents. When a search matches nothing or only a few documents, make the next one broader, never narrower: drop an AND part, add variants, shorten phrases, or use the wording you saw in the abstracts.
  - Look at the results before searching again: "more" shows the next results of a search that worked. Do not repeat a query.
  - If the request rules papers out (for example "Ignore studies in mice"), write (- ...) with every common form of it, and grade such papers 0. A claim that says something does not happen ("no", "not", "lack of") rules nothing out.

Grades, for every document you are shown:
  3  the abstract reports the finding the request is about (it supports or refutes it)
  2  about the same specific question, but not clearly that finding
  1  the same topic only
  0  not relevant, or ruled out by the request

Answer every step with exactly one JSON object and nothing else:
  {"judgments": {"<doc id>": [<grade>, "<reason, at most 10 words>"], ...}, "action": "search", "query": "<Boolean query>"}
  {"judgments": {...}, "action": "more", "search": <number of an earlier search>}
  {"judgments": {...}, "action": "read", "doc": "<doc id>"}
  {"judgments": {...}, "action": "stop"}
"judgments" grades every document shown to you since your last answer (use {} when there are none). "more" shows the next results of that search; "read" shows a whole abstract. Stop when you have found the evidence (grade 3) and more searching is unlikely to find another, or when you cannot find anything more. You have a limited number of steps, so do not waste them."""


def prompt_sha256():
    return hashlib.sha256(AGENT_PROMPT.encode('utf-8')).hexdigest()


def load_docs():
    docs = {}
    with open(CORPUS, encoding='utf-8') as f:
        for line in f:
            doc_id, text = line.rstrip('\n').split('\t', 1)
            docs[doc_id] = text
    return docs


def run_queries(pairs, boolean_cli):
    """{key: (ranked doc ids (top ENGINE_K), set size, error or None)} for [(key, query)], in one engine call."""
    if not pairs:
        return {}
    os.makedirs(WORK, exist_ok=True)
    paths = {name: os.path.join(WORK, f'agent_{name}.tsv') for name in ('queries', 'run', 'sizes', 'failures')}
    with open(paths['queries'], 'w', encoding='utf-8') as f:
        for key, query in pairs:
            f.write(f'{key}\t{" ".join(query.split())}\n')
    subprocess.run([boolean_cli, CORPUS, paths['queries'], paths['run'], paths['sizes'], paths['failures'],
                    str(ENGINE_K), RANKING], check=True, capture_output=True)
    out = {key: ([], 0, None) for key, _ in pairs}
    with open(paths['run'], encoding='utf-8') as f:
        ranked = {}
        for line in f:
            if line.strip():
                key, rank, doc, _ = line.rstrip('\n').split('\t')
                ranked.setdefault(key, []).append((int(rank), doc))
    with open(paths['sizes'], encoding='utf-8') as f:
        for line in f:
            if line.strip():
                key, size = line.rstrip('\n').split('\t')
                out[key] = ([d for _, d in sorted(ranked.get(key, []))], int(size), None)
    with open(paths['failures'], encoding='utf-8') as f:
        for line in f:
            if line.strip():
                key, _, message = line.rstrip('\n').split('\t', 2)
                out[key] = ([], 0, message)
    return out


def parse_action(answer):
    """The JSON object in the model's answer (code fences and text around it ignored), or None."""
    text = re.sub(r'```[a-zA-Z]*', '', answer)
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None
    try:
        action = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return action if isinstance(action, dict) else None


class Session:
    """One request's search: the transcript, the searches, the graded documents and what the model must grade."""

    def __init__(self, qid, request):
        self.qid, self.request = qid, request
        self.messages = [{'role': 'system', 'content': AGENT_PROMPT}, {'role': 'user', 'content': f'Request: {request}'}]
        self.searches = []          # [{'query', 'docs', 'size', 'error', 'next'}]
        self.judged = {}            # doc -> {'grade', 'reason', 'order', 'call'}
        self.shown = []             # docs shown since the last answer, to be graded now
        self.calls, self.done, self.stop_reason = 0, False, None
        self.steps = []             # per call: {'answer', 'action', 'observation'}

    def record_judgments(self, action):
        judgments = action.get('judgments') if isinstance(action.get('judgments'), dict) else {}
        for doc, value in judgments.items():
            doc = str(doc)
            if doc in self.judged or doc not in self.shown:
                continue
            grade, reason = (value[0], value[1] if len(value) > 1 else '') if isinstance(value, list) else (value, '')
            try:
                grade = max(0, min(3, int(grade)))
            except (TypeError, ValueError):
                continue
            self.judged[doc] = {'grade': grade, 'reason': str(reason)[:200], 'order': len(self.judged),
                                'call': self.calls}
        self.shown = []

    def show(self, number, docs_text):
        """Observation with the next SHOW_N unjudged results of search `number` (1-based)."""
        s = self.searches[number - 1]
        if s['error']:
            return f'Search {number} was rejected by the engine: {s["error"]}'
        if s['size'] == 0:
            return f'Search {number}: {s["query"]}\nIt matched no documents.'
        lines, skipped = [], 0
        while s['next'] < len(s['docs']) and len(lines) < SHOW_N:
            doc = s['docs'][s['next']]
            s['next'] += 1
            if doc in self.judged or doc in self.shown:
                skipped += 1
                continue
            self.shown.append(doc)
            snippet = ' '.join(docs_text[doc].split()[:SNIPPET_WORDS])
            lines.append(f'[doc {doc}] (result {s["next"]}) {snippet} ...')
        head = f'Search {number}: {s["query"]}\n{s["size"]} documents match.'
        if skipped:
            head += f' Skipped {skipped} you already graded.'
        if not lines:
            return head + ' No more new results for this search.'
        return head + '\n' + '\n'.join(lines)


def search_agent(requests, llm, boolean_cli, max_calls):
    """E4 for [(query_id, request)]. All sessions advance one call at a time together, so each step's new queries
    run in one engine call. Returns {query_id: Session}."""
    docs_text = load_docs()
    llm.shared_len, llm.prompt_hash = 1, prompt_sha256()
    sessions = {qid: Session(qid, text) for qid, text in requests}
    for step in range(max_calls):
        active = [s for s in sessions.values() if not s.done]
        if not active:
            break
        print(f'  step {step + 1}: {len(active)} active searches', flush=True)
        pending_queries = []
        for s in active:
            answer = llm.chat(s.messages)
            s.calls += 1
            s.messages.append({'role': 'assistant', 'content': answer})
            action = parse_action(answer)
            s.steps.append({'answer': answer, 'action': action, 'observation': None})
            if action is None:
                s.steps[-1]['observation'] = ('Your answer was not one valid JSON object. Answer again with exactly '
                                              'one JSON object as described.')
                continue
            s.record_judgments(action)
            kind = action.get('action')
            if kind == 'stop':
                s.done, s.stop_reason = True, 'model stopped'
            elif kind == 'search' and isinstance(action.get('query'), str) and action['query'].strip():
                s.searches.append({'query': ' '.join(action['query'].split()), 'docs': [], 'size': 0,
                                   'error': None, 'next': 0})
                pending_queries.append((f'{s.qid}#{len(s.searches)}', s.searches[-1]['query']))
            elif kind == 'more' and isinstance(action.get('search'), int) and 1 <= action['search'] <= len(s.searches):
                s.steps[-1]['observation'] = s.show(action['search'], docs_text)
            elif kind == 'read' and str(action.get('doc')) in docs_text:
                s.steps[-1]['observation'] = f'[doc {action["doc"]}] {docs_text[str(action["doc"])]}'
            else:
                s.steps[-1]['observation'] = ('That action could not be carried out (unknown action, missing query, '
                                              'no such search number or document). Answer with a valid action.')
            if len(s.judged) >= MAX_JUDGED and not s.done:
                s.done, s.stop_reason = True, f'{MAX_JUDGED} documents graded'
        results = run_queries(pending_queries, boolean_cli)
        for key, (docs, size, error) in results.items():
            qid, number = key.rsplit('#', 1)
            s = sessions[qid]
            s.searches[int(number) - 1].update({'docs': docs, 'size': size, 'error': error})
            s.steps[-1]['observation'] = s.show(int(number), docs_text)
        for s in active:
            if not s.done:
                s.messages.append({'role': 'user', 'content': s.steps[-1]['observation']})
    for s in sessions.values():
        if not s.done:
            s.done, s.stop_reason = True, f'budget of {max_calls} calls'
    return sessions


def ranking(session, budget=None):
    """The session's ranked documents: graded ones by grade, then by discovery order; with budget b, only the
    grades given within the first b calls (what a budget of b would have produced)."""
    judged = [(doc, j) for doc, j in session.judged.items() if budget is None or j['call'] <= budget]
    judged.sort(key=lambda dj: (-dj[1]['grade'], dj[1]['order']))
    return [(doc, j['grade']) for doc, j in judged]
