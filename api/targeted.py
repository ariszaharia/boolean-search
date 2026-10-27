"""
Shared definitions for the targeted constraint set (targeted_set_design.md, decided 2026-10-27).
Used by api/targeted_candidates.py now, and by the build/check script later, so both apply exactly the
same rules. Everything here reads only the corpus, the train claims and the train qrels: never a run file.
"""
import bisect
import collections
import os
import re

from baseline_pyserini import ROOT, words_of

CORPUS = os.path.join(ROOT, 'data', 'scifact.tsv')
CLAIMS = os.path.join(ROOT, 'data', 'scifact_queries_train.tsv')
QRELS = os.path.join(ROOT, 'data', 'scifact_qrels_train.tsv')
TARGETED_DIR = os.path.join(ROOT, 'data', 'targeted')

# The stopword list of Lucene's default English analyzer, the one B1 uses (so it is not chosen by us).
STOPWORDS = frozenset('a an and are as at be but by for if in into is it no not of on or such that the their '
                      'then there these they this to was will with'.split())


def load_corpus():
    """{doc_id: text} in file order. The text is title and abstract, Greek letters already as names (D7)."""
    docs = {}
    with open(CORPUS, encoding='utf-8') as f:
        for line in f:
            doc_id, text = line.rstrip('\r\n').split('\t', 1)
            docs[doc_id] = text
    return docs


def load_claims():
    """{claim_id: text} of the train claims (D1: the targeted set is built from train only)."""
    with open(CLAIMS, encoding='utf-8') as f:
        return dict(line.rstrip('\r\n').split('\t', 1) for line in f if line.strip())


def load_relevant():
    """{claim_id: set of relevant doc ids} from the train qrels."""
    relevant = collections.defaultdict(set)
    with open(QRELS, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                qid, doc_id, rel = line.rstrip('\r\n').split('\t')
                if int(rel) > 0:
                    relevant[qid].add(doc_id)
    return dict(relevant)


def usable_claims(claims, relevant):
    """D10: each claim text at most once. A text that appears under several ids keeps its lowest id if all of
    them have the same relevant documents, and is dropped if they differ (the ground truth would be ambiguous).
    Claims without a relevant document are dropped too. Returns {claim_id: text}."""
    by_text = collections.defaultdict(list)
    for qid, text in claims.items():
        if relevant.get(qid):
            by_text[text.strip().lower()].append(qid)
    usable = {}
    for qids in by_text.values():
        qids.sort(key=int)
        if all(relevant[q] == relevant[qids[0]] for q in qids):
            usable[qids[0]] = claims[qids[0]]
    return dict(sorted(usable.items(), key=lambda item: int(item[0])))


def content_words(text):
    """The words of a text (words_of: lowercase, split on non-alphanumeric) without stopwords and numbers,
    each once, in order of first appearance."""
    seen = []
    for w in words_of(text):
        if w not in STOPWORDS and not w.isdigit() and w not in seen:
            seen.append(w)
    return seen


def word_stats(docs):
    """(doc_words, df, postings) over the corpus, with words_of: {doc_id: set of words}, {word: number of
    documents}, {word: set of doc ids}."""
    doc_words = {d: set(words_of(t)) for d, t in docs.items()}
    df = collections.Counter(w for ws in doc_words.values() for w in ws)
    postings = collections.defaultdict(set)
    for d, ws in doc_words.items():
        for w in ws:
            postings[w].add(d)
    return doc_words, df, postings


def rarest_content_word(text, df):
    """The text's content word found in the fewest documents (ties: alphabetical), or None."""
    known = [w for w in content_words(text) if df[w] > 0]
    return min(known, key=lambda w: (df[w], w)) if known else None


def distractors(rule_docs, rarest, postings, relevant_docs):
    """D3 fallback exclusions: documents that match the rule and contain the claim's rarest content word,
    other than the relevant ones."""
    return (rule_docs & postings[rarest]) - relevant_docs


def overlap(text, relevant_texts):
    """D4: (share, words) = the share of the text's content words that occur in at least one relevant document,
    and those words. A paraphrase must at least halve the share of its claim."""
    words = content_words(text)
    in_relevant = set()
    for t in relevant_texts:
        in_relevant |= set(words_of(t))
    shared = [w for w in words if w in in_relevant]
    return (len(shared) / len(words) if words else 0.0), shared


def term_pieces(term):
    """Runs of letters or numbers, lowercase (a decimal point stays inside its number): "IL-6" -> il, 6."""
    return re.findall(r'[a-z]+|\d+(?:\.\d+)?', term.lower())


def term_rule(term):
    """A rule (D2) from a phrase or identifier, as a suggestion: "IL-6" -> \\bil[ -]*6\\b,
    "randomized controlled trial" -> \\brandomized[ -]*controlled[ -]*trial\\b. Between pieces, spaces and
    hyphens are optional, so IL-6, IL 6 and IL6 all match. Spelling variants (randomised) are added by hand."""
    return r'\b' + r'[ -]*'.join(re.escape(p) for p in term_pieces(term)) + r'\b'


def term_docs(term, matcher):
    """The documents term_rule(term) matches, found quickly: every match contains all of the term's pieces
    literally, so only documents containing all of them are searched with the rule."""
    return matcher.docs(term_rule(term), must_contain=term_pieces(term))


class RuleMatcher:
    """Finds which documents a rule matches: case-insensitive, on the raw corpus text (D2).
    The whole corpus is searched as one string, one document per line, so a rule runs once, not once per
    document. A rule must not match across a line break (two documents); docs() raises ValueError if one
    does, for example a hand-written rule with \\s or . in it (write a space or [ -] instead)."""

    def __init__(self, docs):
        self.docs_text = docs
        self.lower = {d: t.lower() for d, t in docs.items()}
        self.ids = list(docs)
        self.text = '\n'.join(docs[d] for d in self.ids)
        self.starts, pos = [], 0
        for d in self.ids:
            self.starts.append(pos)
            pos += len(docs[d]) + 1
        self.cache = {}

    def docs(self, rule, must_contain=()):
        """The set of doc ids the rule matches. must_contain (lowercase pieces of text that every match contains)
        makes it faster: only the documents containing all of them are searched. A search over the whole corpus
        takes about 0.2 s; with must_contain, a few milliseconds."""
        if rule not in self.cache:
            found = set()
            if must_contain:
                for d in self.ids:
                    if all(p in self.lower[d] for p in must_contain) and re.search(rule, self.docs_text[d],
                                                                                    re.IGNORECASE):
                        found.add(d)
            else:
                for m in re.finditer(rule, self.text, re.IGNORECASE):
                    if '\n' in m.group():
                        raise ValueError(f'rule {rule!r} matched across two documents: {m.group()[:60]!r}')
                    found.add(self.ids[bisect.bisect_right(self.starts, m.start()) - 1])
            self.cache[rule] = found
        return self.cache[rule]
