"""
Drafts the 100 targeted queries (Deviation of 2026-10-27 in hypotheses.md: drafted by Claude, not by Aris).
The choice of claims and the wording of the 75 constraint queries are mechanical, fixed here before any query
was drafted; the only hand-made inputs are:
  data/targeted/skips.tsv        category <TAB> claim_id <TAB> term <TAB> reason
                                 a candidate that is not used, and why. term = * skips the whole claim.
  data/targeted/paraphrases.tsv  claim_id <TAB> text      the 25 paraphrases, written by Claude
Writes data/targeted/queries.tsv and rules.tsv, then run api/targeted_build.py to check them.

The fixed rules:
  - Claims are visited in one fixed random order (usable train claims, seed 20261015), the same for every
    category. Categories are filled in the order exclusion, identifier, phrase, paraphrase (fewest candidates
    first), and a claim used by one category is not used again (D10).
  - exclusion: the claim's trap candidates (exclusion_trap.tsv); the one whose rule matches the most documents
    (the most general constraint, ties by name). Query kind: trap.
  - identifier, phrase: the claim's first candidate in identifier.tsv / phrase.tsv (they are ranked there:
    traps first, then terms not in the claim, then the most specific). Query kind from its trap column.
  - paraphrase: claims whose overlap share is above 0 (otherwise nothing can be halved), in the same order.
  - The constraint sentence is the next phrasing of the category's list below, in turn (query 1 of the category
    gets phrasing 1, query 6 gets phrasing 1 again).
Query ids: t001-t025 exclusion, t026-t050 phrase, t051-t075 identifier, t076-t100 paraphrase.

Run from the project folder:   venv\\Scripts\\python.exe api/targeted_draft.py
"""
import os
import random

from targeted import TARGETED_DIR, load_claims, load_relevant, usable_claims

SEED = 20261015
PER_CATEGORY = 25
CANDIDATES = os.path.join(TARGETED_DIR, 'candidates')

# {c} = the claim ending with its full stop; {c0} = without it. {x} = what is excluded, {t} = the required term.
EXCLUDE_PHRASINGS = ['{c0}, but not {x}.', '{c} Exclude {x}.', '{c} I am not interested in {x}.',
                     '{c} Leave out {x}.', '{c} No {x}, please.']
REQUIRE_PHRASINGS = ['{c} Only documents that mention {t}.', '{c} It must mention {t}.',
                     '{c} Show me only studies that talk about {t}.', '{c} The study should mention {t}.',
                     '{c0}, specifically mentioning {t}.']
# How each exclusion of api/targeted_candidates.py is named in a query.
EXCLUDE_NAMES = {
    'mice': 'studies in mice', 'rats': 'studies in rats', 'zebrafish': 'zebrafish studies',
    'Drosophila': 'Drosophila studies', 'yeast': 'yeast studies', 'non-human primates': 'studies in monkeys',
    'pigs': 'studies in pigs', 'dogs': 'studies in dogs', 'in vitro': 'in vitro studies',
    'cell lines': 'cell line studies', 'children': 'studies of children', 'older adults': 'studies of older adults',
    'women': 'studies of women', 'men': 'studies of men', 'pregnancy': 'studies in pregnancy',
    'randomized trials': 'randomized trials', 'meta-analyses': 'meta-analyses',
    'systematic reviews': 'systematic reviews', 'cohort studies': 'cohort studies',
    'case-control studies': 'case-control studies', 'cross-sectional studies': 'cross-sectional studies',
    'cancer': 'cancer studies', 'diabetes': 'diabetes studies', 'obesity': 'obesity studies',
    'HIV': 'HIV studies', 'China': 'studies from China', 'United States': 'studies from the United States',
    'Europe': 'European studies', 'Africa': 'African studies', 'Japan': 'Japanese studies',
    'United Kingdom': 'UK studies',
}


def read_tsv(path, required=True):
    """Rows of a tab-separated file with a header line, as dicts."""
    if not os.path.isfile(path):
        if required:
            raise SystemExit(f'Missing {path}')
        return []
    with open(path, encoding='utf-8') as f:
        lines = [line.rstrip('\r\n') for line in f if line.strip()]
    header = lines[0].split('\t')
    return [dict(zip(header, line.split('\t'))) for line in lines[1:]]


def with_stop(claim):
    """(claim ending with a full stop, claim without its final full stop)."""
    c0 = claim.rstrip().rstrip('.')
    return c0 + '.', c0


def main():
    claims = usable_claims(load_claims(), load_relevant())
    order = sorted(claims, key=int)
    random.Random(SEED).shuffle(order)

    skips = read_tsv(os.path.join(TARGETED_DIR, 'skips.tsv'), required=False)
    skipped_claims = {(s['category'], s['claim_id']) for s in skips if s['term'] == '*'}
    skipped_terms = {(s['category'], s['claim_id'], s['term']) for s in skips if s['term'] != '*'}
    paraphrase_text = {p['claim_id']: p['text'] for p in read_tsv(os.path.join(TARGETED_DIR, 'paraphrases.tsv'),
                                                                   required=False)}

    def by_claim(name):
        rows = {}
        for r in read_tsv(os.path.join(CANDIDATES, name)):
            rows.setdefault(r['claim_id'], []).append(r)
        return rows

    traps, identifiers, phrases = (by_claim('exclusion_trap.tsv'), by_claim('identifier.tsv'),
                                   by_claim('phrase.tsv'))
    overlap_share = {r['claim_id']: float(r['overlap_share']) for r in read_tsv(os.path.join(CANDIDATES,
                                                                                            'paraphrase.tsv'))}
    used, picked = set(), {}

    def usable(category, qid, candidates, term_key):
        """The candidates of this claim for this category that are not skipped, or [] if the claim cannot be used."""
        if qid in used or (category, qid) in skipped_claims:
            return []
        return [r for r in candidates.get(qid, []) if (category, qid, r[term_key]) not in skipped_terms]

    picked['exclusion'] = []
    for qid in order:
        rows = usable('exclusion', qid, traps, 'exclude')
        if rows and len(picked['exclusion']) < PER_CATEGORY:
            row = max(rows, key=lambda r: (int(r['corpus_docs']), r['exclude']))
            picked['exclusion'].append(row)
            used.add(qid)
    for category, candidates in (('identifier', identifiers), ('phrase', phrases)):
        picked[category] = []
        for qid in order:
            rows = usable(category, qid, candidates, 'term')
            if rows and len(picked[category]) < PER_CATEGORY:
                picked[category].append(rows[0])
                used.add(qid)
    picked['paraphrase'], missing = [], []
    for qid in order:
        if len(picked['paraphrase']) == PER_CATEGORY:
            break
        if qid in used or ('paraphrase', qid) in skipped_claims or overlap_share.get(qid, 0) == 0:
            continue
        picked['paraphrase'].append({'claim_id': qid})
        used.add(qid)
        if qid not in paraphrase_text:
            missing.append(qid)

    queries, rules = [], []
    number = 0
    for category in ('exclusion', 'phrase', 'identifier', 'paraphrase'):
        for i, row in enumerate(picked[category]):
            number += 1
            tid, qid = f't{number:03d}', row['claim_id']
            c, c0 = with_stop(claims[qid])
            if category == 'exclusion':
                text = EXCLUDE_PHRASINGS[i % len(EXCLUDE_PHRASINGS)].format(c=c, c0=c0, x=EXCLUDE_NAMES[row['exclude']])
                queries.append((tid, category, qid, 'trap', text))
                rules.append((tid, 'exclude', row['rule'], 'no'))
            elif category == 'paraphrase':
                queries.append((tid, category, qid, 'paraphrase', paraphrase_text.get(qid, '')))
                rules.append((tid, 'none', '-', '-'))
            else:
                text = REQUIRE_PHRASINGS[i % len(REQUIRE_PHRASINGS)].format(c=c, c0=c0, t=row['term'])
                queries.append((tid, category, qid, 'trap' if row['trap'] == 'yes' else 'no-trap', text))
                rules.append((tid, 'require', row['rule'], row['in_claim']))

    for name, header, rows in (('queries.tsv', 'query_id\tcategory\tbase_claim_id\tkind\ttext', queries),
                               ('rules.tsv', 'query_id\ttype\trule\tin_claim', rules)):
        with open(os.path.join(TARGETED_DIR, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write(header + '\n')
            for row in rows:
                f.write('\t'.join(row) + '\n')
    for category in ('exclusion', 'phrase', 'identifier', 'paraphrase'):
        print(f'  {category:11s} {len(picked[category]):3d} / {PER_CATEGORY}')
    print(f'Skips used: {len(skips)}. Wrote {len(queries)} queries to data/targeted/queries.tsv and rules.tsv.')
    if missing:
        print(f'Paraphrases still to write (claim ids, in order): {" ".join(missing)}')


if __name__ == '__main__':
    main()
