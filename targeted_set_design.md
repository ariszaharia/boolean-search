# Targeted constraint query set: design (decided 2026-10-27)

Status: drafted by Claude; on 2026-10-27 Aris adopted every recommendation (D1 to D10). The decisions are
summarized in hypotheses.md as a dated clarification; this file holds the details and the reasons.
Order: this design, then the queries, then the planner prompt. No system runs on the set before it is frozen.

What hypotheses.md already fixes (not open here): about 100 queries in four categories (exclusions, exact
phrases and entities, numeric or identifier constraints, paraphrase control); the ground truth is "a relevant
set from existing qrels, filtered by a text rule"; results on the set are evidence about constraint handling
only; categories are reported separately.

## Feasibility (counted 2026-10-27 from qrels and corpus text only, no system output)
- Relevant documents per claim, train: 1 for 732 claims, 2 for 53, 3 for 16, 4 for 7, 5 for 1. Test: 277 of 300
  have 1. So most filtered relevant sets will have 1 to 3 documents.
- 77 train claims have 2 or more relevant documents. In 47 of them, a short list of natural constraint terms
  (mice, rats, children, women, in vitro, cancer, randomized, meta-analysis, cohort) matches some but not all
  of the relevant documents. Example: claim 85 ("Adult tissue-resident macrophages are seeded before birth.")
  has 2 relevant documents and only one mentions mice.
- 506 of 809 train claims have a letters-and-digits identifier (IL6, KLF4, GLP-1, ...) in a relevant document.
- 508 of 5,183 documents contain a Greek letter (see D7).
- Some claims appear twice with the same text under different ids (85 and 86).

---

## D1. Which claims the queries are built from
Options: (a) train claims; (b) test claims; (c) both.
RECOMMENDED: (a) train. Test stays untouched for H7. Every train claim used as a base is listed in the set's
files, and those claims are never used while developing the planner (no prompt examples, no prompt tuning on
them). The planner is developed on the remaining train claims.
Why: building the set from test would mean reading test claims and their relevant documents now. Building it
from train is safe as long as the base claims are kept out of planner development.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D2. Ground truth: the text rule
For a targeted query built from claim q with relevant documents R(q):
- exclusion:          relevant = documents of R(q) that do NOT match the rule
- phrase, identifier: relevant = documents of R(q) that DO match the rule
- paraphrase control: relevant = R(q) (no rule; only the wording of the query changes)
Every document outside R(q) is not relevant, as in the qrels (SciFact's judgments are sparse; this affects every
system the same way).
The rule is a case-insensitive regular expression with word boundaries, applied to the raw document text
(title and abstract, data/scifact.tsv), NOT to the engine's tokens. Spelling variants are listed explicitly in
the rule and fixed before any run, e.g. exclusion "mice": \b(mice|mouse|murine)\b.
RECOMMENDED: as above.
Why raw text and not the engine's tokenizer: if the rule used the engine's tokens, the engine would match the
ground truth by construction even where its tokenizer is wrong (Greek letters, decimals). On raw text, those
mistakes count against the engine, which is the honest direction.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D3. Which claims qualify (selection rules, from qrels and corpus only)
Selection must never look at any system's output (not B1, not B3, not a test run of the planner); otherwise
the set is chosen where a system fails.
- Exclusions: |R(q)| >= 2, and the rule removes at least 1 and keeps at least 1 relevant document. The removed
  ones are "traps": on topic, judged relevant for the claim, but excluded by the constraint. This is the case
  H3 is about (BM25 and dense cannot express NOT).
  Fallback if fewer than 25 qualify after D10: |R(q)| = 1, the relevant document does not match the rule, and
  at least 3 other documents match both the rule and the claim's rarest content word (distractors exist).
  Every query records which kind it is (trap / no-trap); both are reported.
- Exact phrases and entities: the phrase or entity (a multi-word term, a gene, drug or organism name) occurs in
  at least 1 relevant document. Preferred: |R(q)| >= 2 and it occurs in some but not all (traps again).
  It must occur in at least 2 documents of the corpus, so the constraint is not a lookup of one document.
- Numeric or identifier constraints: same rules as phrases, for a dose, a number with its unit, a gene or
  protein symbol with digits, or a trial id.
- Paraphrase control: any claim; see D4 for the rewording check.
For phrases and identifiers, record whether the term already appears in the claim (in_claim = yes/no):
when it does, BM25 gets it for free through idf, which is the reason H3 expects a small gain there.
RECOMMENDED: as above.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D4. What the query text looks like
Constraint categories: the claim, then one natural sentence with the constraint. Vary the wording across
queries ("but not in mice", "excluding mouse studies", "only studies that mention KLF4"). The variant list of
D2 stays in the rule; the query names the constraint the way a person would.
Paraphrase control: the claim rewritten with the same meaning but different words. Check (mechanical): the
share of the claim's content words (not stopwords) that occur in its relevant documents must at least halve.
Every system gets the same text. B1 and B2 get its words as usual, so "not in mice" adds "mice" to the BM25
query and pushes mouse documents up. That is BM25's real limitation, not an artifact, and it is reported.
RECOMMENDED: as above.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D5. Who writes the queries
Options: (a) Aris by hand, from a candidate list made by a script (qrels and corpus only); (b) an LLM writes,
Aris checks; it must then be a different model family from the planner and be disclosed.
RECOMMENDED: (a). 100 short sentences is about 5 to 8 hours. An LLM-written set risks wording that suits an
LLM planner, and it adds an AI-authorship question to the data itself. The examples in this file were written
by Claude to illustrate the design and are not used in the set.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D6. Size
RECOMMENDED: 25 per category, 100 in all, as hypotheses.md says. Expected precision: per-query differences
of about 0.3 standard deviation give a 95% interval of about +-0.12 on 25 queries. That can show the predicted
exclusion effect (H3: about +0.15) but probably not the identifier one (about +0.05), as the prediction
already says. If more than 25 exclusion claims qualify as traps, up to 40 exclusions make that test stronger.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D7. Greek letters in the tokenizer (decide before writing identifier queries)
Today "TNF-α" is indexed as "tnf": non-ASCII bytes separate words, so the α is lost. 508 documents are affected.
Options:
(a) keep as is: identifier queries with Greek letters are hard for the engine, and that is reported;
(b) replace Greek letters with their names (α -> alpha) once, in api/prep_scifact.py, so data/scifact.tsv says
    "TNF-alpha" and every system (B1, B2, B3, E) sees the same text;
(c) treat UTF-8 letters as word characters in the engine.
RECOMMENDED: (b). It is how people type these names, it keeps the engine ASCII, and because it happens in the
corpus file all systems stay comparable. Cost: rebuild the corpus and both Lucene indexes, rerun B1 to B3 on
train (minutes; no hypothesis is tested on train), and a dated clarification, because data/scifact.tsv changes
after the commit. The Pyserini check of Phase 0 stays valid for the old corpus; rerun it or note this.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.
Implemented 2026-10-27 in api/greek.py. Detail fixed with it: each Greek letter becomes a separate word (a space
is added where the name would touch a letter or digit: "IFNγ" -> "IFN gamma", "TGFβ1" -> "TGF beta 1"), as the
engine already does at a hyphen; the micro sign µ, which looks the same as μ, is also written "mu". Write the
rules of the identifier queries against the converted corpus (for example \btnf-alpha\b, not \btnf-α\b).

## D8. Files and format
- data/targeted/queries.tsv     query_id, category, base_claim_id, kind (trap / no-trap / paraphrase), text
- data/targeted/rules.tsv       query_id, type (exclude / require / none), rule (regular expression), in_claim
- data/targeted/qrels.tsv       written by a script from rules.tsv and the SciFact qrels, in the usual qrels
                                format (query_id, doc_id, relevance), so the harness reads it as a third split
Query ids "t001" to "t100", so they never collide with SciFact ids. A check script verifies: every constraint
query keeps at least 1 relevant document; every trap query removes at least 1; phrase and identifier terms
occur in at least 2 documents; paraphrases pass the overlap check; no base claim is used twice (D10).
The harness gets: the split "targeted" in load_qrels, and per-category results read from queries.tsv.
RECOMMENDED: as above.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

## D9. Freeze
1. Aris fills in D1 to D10; the decisions go into hypotheses.md as a dated clarification; commit and push.
2. Candidate script; Aris writes the queries and rules; qrels built; check script passes; commit and push.
   This is the freeze: after it, no query or rule is edited. Mistakes found later go under Deviations.
3. Only then: the planner prompt, and any system run on the set.

## D10. Duplicate and reused claims
Some SciFact claims appear twice with the same text (85 and 86). Use each claim text at most once, in one
category only, so the 100 queries are 100 different needs.
RECOMMENDED: as above.
DECISION (Aris, 2026-10-27): the recommendation above, adopted.

---

## Examples (illustrations only, written by Claude; not part of the set)
- Exclusion, trap (claim 85, 2 relevant, 1 mentions mice):
  "Adult tissue-resident macrophages are seeded before birth, but not in mice."
  rule: exclude \b(mice|mouse|murine)\b -> relevant = the relevant document without mice; the other is a trap.
- Phrase, trap (claim 108, 3 relevant, 1 mentions a randomized trial):
  "Amitriptyline is an effective treatment for chronic tension-type headaches. Only randomized trials."
  rule: require \brandomi[sz]ed\b
- Identifier (claim 15, 1 relevant; KLF4 occurs in it, not in the claim):
  "50% of patients exposed to radiation have activated markers of mesenchymal stem cells; studies mentioning
  KLF4." rule: require \bklf4\b (in_claim = no)
- Paraphrase (claim 267): "Cognitive behavioral therapy is an ineffective treatment for insomnia" ->
  "Talking therapy that changes thoughts and sleep habits does not help people who cannot sleep."

## Not part of this design, still open before E runs
- E0: hypotheses.md says "flat AND/OR of the question words"; fix whether it is AND or OR.
- H4: which non-empty subset (see PROGRESS.md).
- H2's second source of paraphrase queries ("SciFact queries analysed by vocabulary overlap"): how overlap is
  measured and where the cut is.
