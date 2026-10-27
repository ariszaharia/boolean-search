"""
Checks the hand-written targeted queries and rules against the design, and builds their qrels
(targeted_set_design.md D2, D3, D4, D8, D10). Uses the same definitions as api/targeted_candidates.py
(api/targeted.py). Reads only the corpus, the train claims and the train qrels.

Input, in data/targeted/ (tab-separated, first line = the header exactly as below):
  queries.tsv   query_id  category  base_claim_id  kind  text
                query_id: t001 ... t100. category: exclusion, phrase, identifier, paraphrase.
                kind: trap or no-trap (exclusion, phrase, identifier), paraphrase (paraphrase).
                text: what every system gets. Greek letters written as names (alpha, not α).
  rules.tsv     query_id  type  rule  in_claim
                type: exclude (exclusion), require (phrase, identifier), none (paraphrase; rule and in_claim "-").
                rule: case-insensitive regular expression on the raw corpus text; a space or [ -] between
                words, never \\s or . (they could match across two documents).
                in_claim: yes or no, whether the rule matches the base claim itself (the script checks it).

What is checked (every error is listed with its query id; nothing is written while there is one):
  - format: columns, unique ids t001..., known category/kind/type, one rule per query, matching categories;
  - D1, D10: the base claim is a usable train claim, and each claim is used at most once;
  - text: no Greek letters (D7); in the constraint categories it contains every content word of its claim
    (D4: the claim plus a constraint sentence);
  - D2, D3: exclusion: keeps >= 1 relevant document; trap removes >= 1; no-trap has 1 relevant document, the
    rule does not match it, and >= 3 distractors; the rule does not match the claim.
    phrase, identifier: >= 1 relevant document matches, the rule matches >= 2 documents; trap = some but not
    all relevant documents match, no-trap = all match; in_claim is correct.
    paraphrase: the overlap share at most half of the claim's (and the claim's is above 0);
  - with --final: exactly 25 queries per category (the freeze).

Output (only without errors): data/targeted/qrels.tsv   query_id <TAB> doc_id <TAB> 1

Run from the project folder:   venv\\Scripts\\python.exe api/targeted_build.py [--final]
"""
import argparse
import collections
import os
import re
import sys

from greek import GREEK
from targeted import (TARGETED_DIR, RuleMatcher, content_words, distractors, load_claims, load_corpus,
                      load_relevant, overlap, rarest_content_word, usable_claims, word_stats)

CATEGORIES = {'exclusion': 'exclude', 'phrase': 'require', 'identifier': 'require', 'paraphrase': 'none'}
KINDS = {'exclusion': {'trap', 'no-trap'}, 'phrase': {'trap', 'no-trap'}, 'identifier': {'trap', 'no-trap'},
         'paraphrase': {'paraphrase'}}
QUERY_COLUMNS = ['query_id', 'category', 'base_claim_id', 'kind', 'text']
RULE_COLUMNS = ['query_id', 'type', 'rule', 'in_claim']
PER_CATEGORY = 25


def read_table(path, columns, errors):
    """Rows of a tab-separated file with the given header, as dicts; problems are added to errors."""
    name = os.path.basename(path)
    if not os.path.isfile(path):
        errors.append(f'{name}: missing')
        return []
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\r\n') for line in f]
    if not lines or lines[0].split('\t') != columns:
        errors.append(f'{name}: the first line must be the header: {" <TAB> ".join(columns)}')
        return []
    rows = []
    for line_no, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split('\t')
        if len(parts) != len(columns):
            errors.append(f'{name}:{line_no}: expected {len(columns)} tab-separated columns, got {len(parts)}')
            continue
        rows.append(dict(zip(columns, (p.strip() for p in parts)), line=f'{name}:{line_no}'))
    return rows


def check(directory, final):
    errors = []
    queries = read_table(os.path.join(directory, 'queries.tsv'), QUERY_COLUMNS, errors)
    rules = read_table(os.path.join(directory, 'rules.tsv'), RULE_COLUMNS, errors)

    docs = load_corpus()
    relevant = load_relevant()
    claims = usable_claims(load_claims(), relevant)
    matcher = RuleMatcher(docs)
    _, df, postings = word_stats(docs)

    rule_of = {}
    for r in rules:
        if r['query_id'] in rule_of:
            errors.append(f'{r["line"]}: {r["query_id"]} has a second rule')
        rule_of[r['query_id']] = r
    seen_ids, used_claims, qrels, summary = set(), {}, {}, collections.defaultdict(collections.Counter)

    for q in queries:
        qid, cat, base, kind, text = q['query_id'], q['category'], q['base_claim_id'], q['kind'], q['text']
        where = f'{q["line"]} ({qid})'

        def err(message):
            errors.append(f'{where}: {message}')

        if not re.fullmatch(r't\d{3}', qid):
            err('query_id must be t followed by 3 digits (t001)')
        if qid in seen_ids:
            err('query_id used twice')
        seen_ids.add(qid)
        if cat not in CATEGORIES:
            err(f'unknown category {cat!r} (one of {", ".join(CATEGORIES)})')
            continue
        if kind not in KINDS[cat]:
            err(f'kind for {cat} must be one of {", ".join(sorted(KINDS[cat]))}, got {kind!r}')
        if base not in claims:
            err(f'base claim {base!r} is not a usable train claim (see api/targeted.py usable_claims)')
            continue
        if base in used_claims:
            err(f'base claim {base} already used by {used_claims[base]} (D10: each claim once)')
        used_claims[base] = qid
        if not text:
            err('empty text')
            continue
        if any(ch in GREEK for ch in text):
            err('the text has a Greek letter; write its name (alpha, beta, ...), as the corpus does (D7)')

        claim, rel = claims[base], relevant[base]
        r = rule_of.get(qid)
        if r is None:
            err('no line in rules.tsv')
            continue
        if r['type'] != CATEGORIES[cat]:
            err(f'rule type for {cat} must be {CATEGORIES[cat]}, got {r["type"]!r}')
            continue

        if cat == 'paraphrase':
            if r['rule'] != '-' or r['in_claim'] != '-':
                err('a paraphrase has no rule: write - in the rule and in_claim columns')
            rel_texts = [docs[d] for d in rel]
            before, _ = overlap(claim, rel_texts)
            after, shared = overlap(text, rel_texts)
            if before == 0:
                err('the claim shares no content word with its relevant documents; nothing to halve')
            elif after > before / 2:
                err(f'overlap share {after:.2f} is more than half of the claim\'s {before:.2f} '
                    f'(words still shared: {" ".join(shared)})')
            qrels[qid] = set(rel)
            summary[cat][kind] += 1
            continue

        missing = [w for w in content_words(claim) if w not in content_words(text)]
        if missing:
            err(f'the text must contain the whole claim (D4); missing words: {" ".join(missing)}')
        if any(ch in GREEK for ch in r['rule']):
            err('the rule has a Greek letter; the corpus has names (alpha), so it would never match')
        if re.search(r'\\s|(?<!\\)\.', r['rule']):
            err('the rule uses \\s or an unescaped . (they can match across documents); write a space or [ -], '
                'and \\. for a real dot')
        try:
            re.compile(r['rule'])
        except re.error as e:
            err(f'the rule is not a valid regular expression: {e}')
            continue
        try:
            rule_docs = matcher.docs(r['rule'])
        except ValueError as e:
            err(str(e))
            continue
        matching = rule_docs & rel
        in_claim = bool(re.search(r['rule'], claim, re.IGNORECASE))

        if cat == 'exclusion':
            kept = rel - matching
            if in_claim:
                err('the rule matches the claim itself: the claim would exclude what it states')
            if r['in_claim'] != 'no':
                err('in_claim must be no for an exclusion')
            if not kept:
                err('the rule removes every relevant document; nothing stays relevant')
            if kind == 'trap' and not matching:
                err('trap: the rule must remove at least 1 relevant document (it removes none)')
            if kind == 'no-trap':
                if len(rel) != 1 or matching:
                    err('no-trap: needs a claim with 1 relevant document that the rule does not match')
                rarest = rarest_content_word(claim, df)
                n = len(distractors(rule_docs, rarest, postings, rel)) if rarest else 0
                if n < 3:
                    err(f'no-trap: needs at least 3 distractors, found {n} (rarest content word: {rarest})')
            qrels[qid] = kept
        else:
            if r['in_claim'] not in ('yes', 'no'):
                err('in_claim must be yes or no')
            elif (r['in_claim'] == 'yes') != in_claim:
                err(f'in_claim says {r["in_claim"]}, but the rule {"does" if in_claim else "does not"} match the claim')
            if not matching:
                err('the rule matches no relevant document')
            if len(rule_docs) < 2:
                err(f'the rule matches {len(rule_docs)} document(s) in the corpus; at least 2 are needed (D3)')
            is_trap = 0 < len(matching) < len(rel)
            if matching and kind == 'trap' and not is_trap:
                err('trap: the rule must match some but not all relevant documents (it matches all)')
            if kind == 'no-trap' and is_trap:
                err('no-trap: the rule matches only some relevant documents; this is a trap')
            qrels[qid] = matching
        summary[cat][kind] += 1

    for qid in rule_of:
        if qid not in seen_ids:
            errors.append(f'rules.tsv ({qid}): no such query in queries.tsv')
    counts = {cat: sum(summary[cat].values()) for cat in CATEGORIES}
    if final:
        for cat, n in counts.items():
            if n != PER_CATEGORY:
                errors.append(f'--final: {cat} has {n} queries, {PER_CATEGORY} are needed')
    return errors, qrels, summary, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--final', action='store_true', help='also require exactly 25 queries per category (freeze)')
    ap.add_argument('--dir', default=TARGETED_DIR, help='folder with queries.tsv and rules.tsv')
    args = ap.parse_args()

    errors, qrels, summary, counts = check(args.dir, args.final)
    for cat in CATEGORIES:
        kinds = ', '.join(f'{k} {n}' for k, n in sorted(summary[cat].items())) or '-'
        print(f'  {cat:11s} {counts[cat]:3d} / {PER_CATEGORY}   ({kinds})')
    if errors:
        print(f'\n{len(errors)} problem(s); qrels.tsv not written:')
        for e in errors:
            print(f'  {e}')
        sys.exit(1)
    path = os.path.join(args.dir, 'qrels.tsv')
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for qid in sorted(qrels):
            for doc_id in sorted(qrels[qid], key=int):
                f.write(f'{qid}\t{doc_id}\t1\n')
    sizes = [len(v) for v in qrels.values()]
    print(f'\nAll checks pass. {len(qrels)} queries, {sum(sizes)} relevant documents '
          f'(1 to {max(sizes) if sizes else 0} per query) -> {os.path.relpath(path)}')


if __name__ == '__main__':
    main()
