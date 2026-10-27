"""
Candidate lists for the targeted constraint set (targeted_set_design.md, step b of PROGRESS.md).

Reads only the corpus, the train claims and the train qrels (D1, D3): never a run file, so the candidates
cannot be chosen where some system fails. Aris picks from these lists and writes the queries and rules by
hand (D5); the build/check script then verifies them. A rule here is only a suggestion: add spelling variants
where needed (randomised, tumour, ...) and check that the claim with the constraint still makes sense.

Writes data/targeted/candidates/ (one row per candidate, tab-separated, with a header):
  exclusion_trap.tsv      claims with >= 2 relevant documents where an exclusion removes some of them but not
                          all (D3 "trap"). relevant_kept = what stays relevant; relevant_removed = the traps.
  exclusion_fallback.tsv  claims with 1 relevant document that the exclusion does not match, with at least 3
                          distractors (documents matching the rule and the claim's rarest content word).
                          Use only if fewer than 25 trap queries can be written. Up to 3 per claim.
  multi_relevant.tsv      every claim with >= 2 relevant documents and the words found in some but not all of
                          them (most common first): ideas for exclusions that are not in the list below.
  phrase.tsv              phrases (2 or 3 words) and entities (HIV, mTOR) from the relevant documents.
  identifier.tsv          identifiers with letters and digits (IL-6, KLF4, p53) and numbers with a unit (10 mg).
  paraphrase.tsv          every claim, with the share of its content words found in its relevant documents
                          (D4: a paraphrase must at least halve it) and those words.
Columns used in several files: claim_id, claim, n_relevant, rule, corpus_docs (documents the rule matches),
relevant_matching (relevant documents it matches), trap (yes = it matches some but not all relevant documents),
in_claim (yes = the rule already matches the claim itself).

Run from the project folder:   venv\\Scripts\\python.exe api/targeted_candidates.py
"""
import collections
import os
import re

from baseline_pyserini import words_of
from targeted import (STOPWORDS, TARGETED_DIR, RuleMatcher, content_words, load_claims, load_corpus,
                      load_relevant, overlap, term_docs, term_pieces, term_rule, usable_claims)

OUT_DIR = os.path.join(TARGETED_DIR, 'candidates')

# Natural exclusion constraints and their suggested rules (raw text, case-insensitive; no \s, see RuleMatcher).
EXCLUSIONS = {
    'mice': r'\b(mice|mouse|murine)\b',
    'rats': r'\brats?\b',
    'zebrafish': r'\bzebrafish\b',
    'Drosophila': r'\b(drosophila|fruit fly|fruit flies)\b',
    'yeast': r'\b(yeasts?|saccharomyces)\b',
    'non-human primates': r'\b(monkeys?|macaques?)\b',
    'pigs': r'\b(pigs?|porcine|swine)\b',
    'dogs': r'\b(dogs?|canine)\b',
    'in vitro': r'\bin vitro\b',
    'cell lines': r'\bcell lines?\b',
    'children': r'\b(child|children|childhood|infants?|pediatric|paediatric|adolescents?)\b',
    'older adults': r'\b(elderly|older adults|older people)\b',
    'women': r'\b(women|woman|females?)\b',
    'men': r'\b(men|males?)\b',
    'pregnancy': r'\b(pregnant|pregnancy|pregnancies)\b',
    'randomized trials': r'\brandomi[sz]ed\b',
    'meta-analyses': r'\bmeta[ -]?analys[ie]s\b',
    'systematic reviews': r'\bsystematic reviews?\b',
    'cohort studies': r'\bcohorts?\b',
    'case-control studies': r'\bcase[ -]control\b',
    'cross-sectional studies': r'\bcross[ -]sectional\b',
    'cancer': r'\b(cancers?|tumou?rs?|carcinomas?)\b',
    'diabetes': r'\b(diabetes|diabetic)\b',
    'obesity': r'\b(obesity|obese)\b',
    'HIV': r'\bhiv\b',
    'China': r'\b(china|chinese)\b',
    'United States': r'\b(united states|usa)\b',
    'Europe': r'\b(europe|european)\b',
    'Africa': r'\b(africa|african)\b',
    'Japan': r'\b(japan|japanese)\b',
    'United Kingdom': r'\b(united kingdom|uk|britain|british|england)\b',
}

# Candidate-list limits (they only keep the lists readable; they are not rules of the design).
MAX_DOCS = 100         # a phrase or entity in more documents than this is too common to be a constraint
PER_CLAIM_PHRASE = 8
PER_CLAIM_IDENTIFIER = 6
FALLBACK_PER_CLAIM = 3
SPLIT_WORDS_SHOWN = 15

IDENTIFIER = re.compile(r'\b(?=[A-Za-z0-9-]*\d)(?=[A-Za-z0-9-]*[A-Za-z])[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b')
NUMBER_UNIT = re.compile(r'\b\d+(?:\.\d+)? ?(?:mg|kg|g|ml|mmol|mol|nm|mm|cm|years?|months?|weeks?|days?|hours?)\b',
                         re.IGNORECASE)
NOT_IDENTIFIER = re.compile(r'^\d+(st|nd|rd|th|s)$', re.IGNORECASE)   # ordinals and decades: 1st, 1990s
ENTITY = re.compile(r'\b[A-Za-z]*[A-Z][A-Za-z]*\b')


def yes(flag):
    return 'yes' if flag else 'no'


def phrases_in(text):
    """Lowercase 2- and 3-word phrases of a text: words of letters only, separated only by spaces or hyphens
    (never across punctuation), not starting or ending with a stopword or a word shorter than 3 letters."""
    words = list(re.finditer(r'[A-Za-z]+', text))
    found = set()
    for i in range(len(words)):
        for n in (2, 3):
            run = words[i:i + n]
            if len(run) < n:
                continue
            if any(not re.fullmatch(r'[ -]+', text[a.end():b.start()]) for a, b in zip(run, run[1:])):
                continue
            ws = tuple(w.group().lower() for w in run)
            if any(w in STOPWORDS or len(w) < 3 for w in (ws[0], ws[-1])):
                continue
            found.add(ws)
    return found


def entities_in(text):
    """Words written with capitals in a way plain words are not (HIV, mTOR, BRCA): at least 3 letters, no digits,
    all capitals or a capital after the first letter."""
    return {m.group() for m in ENTITY.finditer(text)
            if len(m.group()) >= 3 and (m.group().isupper() or any(c.isupper() for c in m.group()[1:]))}


def identifiers_in(text):
    found = {m.group() for m in IDENTIFIER.finditer(text) if len(m.group()) <= 20 and not NOT_IDENTIFIER.match(m.group())}
    return found | {m.group() for m in NUMBER_UNIT.finditer(text)}


def compact(term):
    """A term without spaces, hyphens and decimal points ("IL-6" -> il6). Used only for the quick estimates:
    a document "contains" a term if some run of its consecutive words, written together, gives the same text,
    so IL-6, IL 6 and IL6 count alike, as the rule treats them."""
    return ''.join(term_pieces(term)).replace('.', '')


def runs_in(words, wanted, max_words):
    """The compact terms from `wanted` that some run of 1 to max_words consecutive words spells."""
    found = set()
    for i in range(len(words)):
        for n in range(1, max_words + 1):
            joined = ''.join(words[i:i + n])
            if joined in wanted:
                found.add(joined)
    return found


def term_row(qid, claim, rel, term, kind, matcher):
    """One phrase/entity/identifier candidate, or None if it breaks D3 (no relevant document, < 2 documents)."""
    rule = term_rule(term)
    docs = term_docs(term, matcher)
    matching = len(docs & rel)
    if matching == 0 or len(docs) < 2:
        return None
    return {'claim_id': qid, 'claim': claim, 'n_relevant': len(rel), 'kind': kind, 'term': term, 'rule': rule,
            'relevant_matching': matching, 'trap': yes(0 < matching < len(rel)),
            'in_claim': yes(re.search(rule, claim, re.IGNORECASE)), 'corpus_docs': len(docs)}


def best(rows, limit, specificity):
    """Traps first, then terms not already in the claim, then the most specific (lowest specificity(term))."""
    rows = [r for r in rows if r]
    unique = {r['rule']: r for r in rows}.values()
    return sorted(unique, key=lambda r: (r['trap'] != 'yes', r['in_claim'] == 'yes', specificity(r['term']),
                                         r['term']))[:limit]


def write(name, rows, columns):
    path = os.path.join(OUT_DIR, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(columns) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[c]) for c in columns) + '\n')
    print(f'{len(rows):5d} rows, {len({r["claim_id"] for r in rows}):4d} claims -> {os.path.relpath(path)}')


def main():
    docs = load_corpus()
    relevant = load_relevant()
    claims = usable_claims(load_claims(), relevant)
    matcher = RuleMatcher(docs)
    doc_words = {d: set(words_of(t)) for d, t in docs.items()}
    df = collections.Counter(w for ws in doc_words.values() for w in ws)
    postings = collections.defaultdict(set)
    for d, ws in doc_words.items():
        for w in ws:
            postings[w].add(d)
    print(f'{len(claims)} usable train claims (D10), {len(docs)} documents')

    trap, fallback, multi = [], [], []
    for qid, claim in claims.items():
        rel = relevant[qid]
        if len(rel) >= 2:
            for name, rule in EXCLUSIONS.items():
                hit = matcher.docs(rule) & rel
                if 0 < len(hit) < len(rel) and not re.search(rule, claim, re.IGNORECASE):
                    trap.append({'claim_id': qid, 'claim': claim, 'n_relevant': len(rel), 'exclude': name,
                                 'rule': rule, 'relevant_kept': len(rel) - len(hit), 'relevant_removed': len(hit),
                                 'corpus_docs': len(matcher.docs(rule))})
            word_sets = [doc_words[d] for d in rel]
            split = (set.union(*word_sets) - set.intersection(*word_sets)) - STOPWORDS
            split = sorted((w for w in split if len(w) >= 4 and not w.isdigit()), key=lambda w: (-df[w], w))
            multi.append({'claim_id': qid, 'claim': claim, 'n_relevant': len(rel),
                          'words_in_some_relevant_docs': ' '.join(split[:SPLIT_WORDS_SHOWN])})
        else:
            known = [w for w in content_words(claim) if df[w] > 0]
            if not known:
                continue
            rarest = min(known, key=lambda w: (df[w], w))
            rows = []
            for name, rule in EXCLUSIONS.items():
                if matcher.docs(rule) & rel or re.search(rule, claim, re.IGNORECASE):
                    continue
                distractors = (matcher.docs(rule) & postings[rarest]) - rel
                if len(distractors) >= 3:
                    rows.append({'claim_id': qid, 'claim': claim, 'exclude': name, 'rule': rule,
                                 'rarest_word': rarest, 'distractors': len(distractors)})
            fallback += sorted(rows, key=lambda r: -r['distractors'])[:FALLBACK_PER_CLAIM]

    # Phrases, entities, identifiers: ranked on quick estimates (compact, runs_in) first; only the best
    # 2 x limit per claim are checked with the real rule, because one rule search costs milliseconds.
    # An all-capitals word whose lowercase form is an ordinary word in >= 5 documents is a heading of a structured
    # abstract (BACKGROUND, SETTINGS), not an entity; acronyms (HIV, AMP) are rarely written in lowercase.
    lowercase_df = collections.Counter(w for t in docs.values() for w in set(re.findall(r'\b[a-z]+\b', t)))
    phrase_terms, id_terms = {}, {}
    for qid in claims:
        rel_texts = [docs[d] for d in relevant[qid]]
        phrase_terms[qid] = ({(' '.join(p), 'phrase') for t in rel_texts for p in phrases_in(t)}
                             | {(e, 'entity') for t in rel_texts for e in entities_in(t)
                                if not (e.isupper() and lowercase_df[e.lower()] >= 5)})
        id_terms[qid] = {(i, 'identifier') for t in rel_texts for i in identifiers_in(t)}
    all_terms = [t for terms in (*phrase_terms.values(), *id_terms.values()) for t, _ in terms]
    wanted = {compact(t) for t in all_terms}
    max_words = min(6, max(len(term_pieces(t)) for t in all_terms))
    doc_runs = {d: runs_in(words_of(t), wanted, max_words) for d, t in docs.items()}
    df_est = collections.Counter(c for cs in doc_runs.values() for c in cs)

    def shortlist(qid, claim, terms, limit, specificity, max_docs):
        rel = relevant[qid]
        claim_runs = runs_in(words_of(claim), wanted, max_words)
        ranked = []
        for term, kind in terms:
            c = compact(term)
            if df_est[c] < 2 or df_est[c] > max_docs:
                continue
            matching = sum(c in doc_runs[d] for d in rel)
            ranked.append((not 0 < matching < len(rel), c in claim_runs, specificity(term), term, kind))
        ranked.sort()
        rows = [term_row(qid, claim, rel, term, kind, matcher) for *_, term, kind in ranked[:2 * limit]]
        return best(rows, limit, specificity)

    # Specificity (lower = more specific). A phrase is only as specific as its most common word, so "smooth
    # muscle" comes before "key role". An identifier: the fewer documents it is in, the more specific.
    def phrase_specificity(term):
        return max(df[w] for w in words_of(term))

    def identifier_specificity(term):
        return df_est[compact(term)]

    phrase_rows, identifier_rows = [], []
    for qid, claim in claims.items():
        phrase_rows += shortlist(qid, claim, phrase_terms[qid], PER_CLAIM_PHRASE, phrase_specificity, MAX_DOCS)
        identifier_rows += shortlist(qid, claim, id_terms[qid], PER_CLAIM_IDENTIFIER, identifier_specificity,
                                     len(docs))

    paraphrase = []
    for qid, claim in claims.items():
        share, shared = overlap(claim, [docs[d] for d in relevant[qid]])
        paraphrase.append({'claim_id': qid, 'claim': claim, 'n_relevant': len(relevant[qid]),
                           'content_words': ' '.join(content_words(claim)), 'overlap_share': f'{share:.2f}',
                           'overlapping_words': ' '.join(shared)})
    paraphrase.sort(key=lambda r: (-float(r['overlap_share']), int(r['claim_id'])))

    os.makedirs(OUT_DIR, exist_ok=True)
    term_columns = ['claim_id', 'claim', 'n_relevant', 'kind', 'term', 'rule', 'relevant_matching', 'trap',
                    'in_claim', 'corpus_docs']
    write('exclusion_trap.tsv', trap, ['claim_id', 'claim', 'n_relevant', 'exclude', 'rule', 'relevant_kept',
                                       'relevant_removed', 'corpus_docs'])
    write('exclusion_fallback.tsv', fallback, ['claim_id', 'claim', 'exclude', 'rule', 'rarest_word', 'distractors'])
    write('multi_relevant.tsv', multi, ['claim_id', 'claim', 'n_relevant', 'words_in_some_relevant_docs'])
    write('phrase.tsv', phrase_rows, term_columns)
    write('identifier.tsv', identifier_rows, term_columns)
    write('paraphrase.tsv', paraphrase, ['claim_id', 'claim', 'n_relevant', 'content_words', 'overlap_share',
                                         'overlapping_words'])


if __name__ == '__main__':
    main()
