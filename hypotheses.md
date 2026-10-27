# Hypotheses (pre-registration)

Status: written 2026-10-15; predictions and open values filled in 2026-10-27, before any experiment.
On 2026-10-27 Aris adopted the reference estimates (below) as the predictions and the values Claude suggested
for kappa (H5), M (H7) and the H8 threshold; each "MY PREDICTION" line says so.
Committed on: 2026-10-27 16:22 (+0200), git commit 9cf3a800fc2a5089df42dceb8231fbfb3192a280.
Rule: no experiment may run before this file is committed. After that, the text of H1 to H9 is not edited.
Anything that has to change is added at the bottom under "Deviations" with a date and the reason.
"REFERENCE ESTIMATE (Claude, 2026-10-27)" lines are a second, independent prediction written by the AI assistant
(Claude Opus 5.5) before any experiment. They are not Aris's predictions; both are reported against the results.

## Setting (fixed in advance)
- Systems: B1 BM25 stemmed (Pyserini default analyzer) | B2 BM25 exact match | B3 dense (one named embedding model)
  | B4 BM25 with LLM-expanded terms | B5 dense with the same LLM-expanded query
  | E0 engine, flat AND/OR of the question words | E1 engine, LLM-written query, single shot
  | E2 E1 plus relaxation loop | E3 E1 ranked by SSR | E4 E2 as a multi-call agent.
- Data: SciFact (main), TREC-COVID (latency and size), targeted constraint set (about 100 queries in four
  categories: exclusions, exact phrases and entities, numeric or identifier constraints, paraphrase control).
  Datasets are reported separately, never pooled.
- "Constraint queries" = the first three categories of the targeted set. "Paraphrase queries" = the fourth
  category, plus SciFact queries analysed by vocabulary overlap with their relevant documents.
- Pool: the Boolean result is cut to the top N by BM25 (N fixed per experiment, 30 and 100); every system is
  compared on the same N.
- Statistics: paired per-query comparisons; 95% bootstrap confidence interval (10,000 resamples) of the mean
  difference; Holm correction over all confirmatory tests below; effect sizes reported with their intervals.
- p-value (clarification added 2026-10-27, before any experiment; needed by the Holm correction): two-sided
  bootstrap test from the same 10,000 resamples. p = (1 + number of bootstrap means whose distance from the observed
  mean difference is at least the observed difference's distance from 0) / (1 + 10,000). This is the bootstrap
  distribution shifted to the null hypothesis "mean difference = 0" (Efron and Tibshirani, An Introduction to the
  Bootstrap, 1993, ch. 16). The interval is the percentile interval (2.5th and 97.5th percentiles of the bootstrap
  means). The random seed is fixed (20261015), so every rerun gives the same numbers. Implemented in
  harness/stats.py (paired_bootstrap, holm).
- Empty Boolean results: precision is undefined when a candidate set is empty. Primary analysis uses the queries
  with a non-empty set; a second analysis counts empty sets as 0. Both are reported, with the zero-result rate.
- Caveat for the targeted set: its ground truth is derived mechanically (a relevant set from existing qrels, filtered
  by a text rule), which favours lexical constraints by construction. Results on it are evidence about
  constraint handling only, not about retrieval in general.
- BM25 settings (clarification added 2026-10-27, before any baseline run):
  - B1 and B2 use BM25 with k1 = 1.2 and b = 0.75 (the textbook defaults, chosen in advance and not tuned on
    SciFact), the same documents (data/scifact.tsv) and the same query words (lowercase, split on every
    non-alphanumeric character). The only difference between them is the analyzer, so B1 minus B2 measures
    stemming and stopword removal together.
  - B1 uses Pyserini's default analyzer (Porter stemming, English stopwords removed) on a Lucene index built
    from data/scifact.tsv, not the prebuilt BEIR index.
  - B1-lit (k1 = 0.9, b = 0.4, raw query text, the same stemmed index) is run only as a check that the setup
    reproduces the published BEIR BM25 result for SciFact. It is run on the test split at the end, with the
    final runs, and is not part of any hypothesis or of the Holm correction.

Core hypotheses: H1 to H5, H7. Stretch (only if time allows): H6, H8, H9.

---

## H1. Precision on constraint queries
Statement: On constraint queries, the Boolean candidate set (E1) has higher precision than the BM25 (B1) and
dense (B3) top-k, where k equals the size of that query's Boolean set.
Primary test: mean paired difference in precision, E1 minus B1, on constraint queries (E1 minus B3 also reported).
Falsified if: the 95% interval of E1 minus B1 includes 0 (BM25 matches within the interval).
MY PREDICTION (adopted from the reference estimate, 2026-10-27): E1 higher than B1 by about 20 precision points
(range +15 to +30), and higher than B3 by about 25. H1 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): E1 minus B1 is +15 to +30 precision points (about +20); E1 minus B3
about +25. Reason: the targeted set's ground truth is a text rule, so a correct Boolean query nearly matches it by
construction; most E1 errors will come from the LLM writing a wrong query, not from the engine.

## H2. Recall on paraphrase queries
Statement: On paraphrase queries, the Boolean candidate set has lower recall than BM25 and dense.
Primary test: mean paired difference in recall of the top-N pool (N = 100), E1 minus B1, on paraphrase queries
(E1 minus B3 also reported).
Falsified if: the 95% interval of E1 minus B1 includes 0 or is positive.
MY PREDICTION (adopted from the reference estimate, 2026-10-27): E1 lower than B1 by about 20 recall@100 points
(range -15 to -25), and lower than B3 by about 25 (range -20 to -30). H2 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): E1 minus B1 is -15 to -25 recall@100 points; E1 minus B3 is -20 to -30.
Reason: a strict AND over paraphrased wording drops many relevant documents; dense retrieval handles synonyms best.

## H3. nDCG on exclusions and exact identifiers
Statement: After the same reranker on the same pool, E2 beats BM25 (B1) and dense (B3) in nDCG@10 on exclusion
queries and on exact-identifier queries.
Primary test: mean paired difference in nDCG@10, E2 minus B1, separately for each of the two categories.
Falsified if: no significant gain for either category after Holm correction.
MY PREDICTION (adopted from the reference estimate, 2026-10-27): exclusions, E2 higher than B1 by about 0.15
nDCG@10 (range +0.10 to +0.20), significant; exact identifiers, about +0.05 (range +0.03 to +0.08), possibly not
significant. H3 holds through the exclusion category.
REFERENCE ESTIMATE (Claude, 2026-10-27): exclusions, E2 minus B1 is +0.10 to +0.20 nDCG@10; exact identifiers,
+0.03 to +0.08, possibly not significant. Reason: BM25 cannot express NOT at all, but it already ranks rare
identifiers highly through idf, so the gain there is small.

## H4. Relaxation recovers recall
Statement: The relaxation loop (E2) recovers part of the recall lost by the strict query (E1), at some cost in
precision.
Primary test: mean paired difference in recall of the top-N pool, E2 minus E1, on SciFact test and on the
targeted set. The precision change is reported next to it.
Falsified if: the recall gain is not significant after Holm correction.
MY PREDICTION (adopted from the reference estimate, 2026-10-27): recall gain of about 10 points (range +8 to
+15), precision cost of about 5 points (range 3 to 8). H4 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): recall gain of +8 to +15 points, precision cost of 3 to 8 points.
Reason: relaxation mostly helps the queries that returned zero or very few documents.

## H5. Failures of Boolean can be traced from the query
Statement: Most Boolean failures can be diagnosed from the generated query and the `why-not` output alone;
dense failures cannot be diagnosed from their output.
Primary test: on 50 failures per system type, an annotator names the cause using only the query and `why-not`
(Boolean) or only the ranked output (dense). Compare the two proportions with a 95% interval. A second annotator
labels 30 to 50 of them; Cohen's kappa is reported.
Falsified if: the two proportions differ by less than the annotator disagreement (the interval of the difference
includes 0, or kappa is below 0.6).
MY PREDICTION (adopted from the reference estimate, 2026-10-27): share diagnosable, Boolean about 70 %, dense
about 30 % (range 25 to 35 %); kappa at least 0.6. H5 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): Boolean about 70 %, dense about 25 to 35 %; kappa threshold 0.6.
Reason: the query plus `why-not` usually shows the cause (missing synonym, over-strict AND); a dense ranked list
alone rarely does.

## H6. Faithfulness end to end (stretch)
Statement: End to end, answer faithfulness is no worse for Boolean when the evidence is retrieved, but answer
coverage is lower.
Primary test: claim-level support rate, E2 pipeline minus B3 pipeline, with the LLM judge spot-checked by hand
on 50 items.
Falsified if: faithfulness is significantly lower for Boolean.
MY PREDICTION (adopted from the reference estimate, 2026-10-27): faithfulness difference about 0 (within 3 points
either way); coverage 5 to 10 points lower for Boolean. H6 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): faithfulness difference about 0 (within 3 points either way); coverage
5 to 10 points lower for Boolean. Reason: given the same evidence the generator behaves the same; Boolean simply
retrieves less evidence.

## H7. Null on the standard set
Statement: After reranking, nDCG@10 on the standard SciFact test queries differs little between the pipelines
(E2, B1, B3, each with the same reranker).
Primary test: equivalence. The 95% interval of each paired difference must lie inside plus or minus the margin
M = 0.03 nDCG points (decided 2026-10-27; first suggested 0.02, raised because on 300 queries the 95% interval
is expected to be about +-0.025 wide on each side, so 0.02 could not be met even by two equal systems).
Falsified if: Boolean is clearly ahead, i.e. the interval of E2 minus B1 lies above +M.
Note: kept as a null on purpose. Reporting it honestly makes H3 more credible.
MY PREDICTION (adopted from the reference estimate, 2026-10-27): E2 minus B1 about -0.02 nDCG@10 (range -0.01 to
-0.03), E2 minus B3 about -0.03. H7 is not falsified (Boolean is not ahead), but equivalence within +-0.03 is
probably not shown either, because the interval is expected to reach below -0.03.
REFERENCE ESTIMATE (Claude, 2026-10-27): E2 minus B1 is -0.01 to -0.03 nDCG@10; E2 minus B3 about -0.03.
Reason: BM25 scores about 0.665 nDCG@10 on SciFact; the reranker narrows the gaps but cannot win back recall lost
before reranking. Caution on M: per-query nDCG@10 differences typically have a standard deviation of about 0.20 to
0.25, so on 300 queries the 95% interval is about +-0.025 wide on each side. With M = 0.02 the equivalence test
likely fails even if the systems are equal; M of 0.03 to 0.05, or stating the low power in advance, avoids that.

## H8. Cost versus quality of the planner (stretch)
Statement: Single shot plus relaxation (E2) reaches most of the multi-call agent's (E4) nDCG@10 with a fraction
of the model calls.
Primary test: nDCG@10 of E2 divided by nDCG@10 of E4 at several call budgets; calls per query reported.
Falsified if: nDCG@10 keeps rising steeply with the call budget (E2 reaches less than 80 % of E4).
MY PREDICTION (adopted from the reference estimate, 2026-10-27): E2 reaches about 90 % of E4's nDCG@10. H8 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): E2 reaches about 90 % of E4. Reason: extra model calls help only a
minority of hard queries.

## H9. SSR versus BM25 ranking (stretch)
Statement: On the same Boolean candidate set, SSR (shortest-substring ranking, C = 42) and BM25 rank similarly.
Primary test: mean paired difference in nDCG@10, E3 minus E1.
Falsified if: one is significantly better after Holm correction.
MY PREDICTION (adopted from the reference estimate, 2026-10-27): E3 minus E1 within 0.02 nDCG@10 either way,
BM25 (E1) slightly ahead, not significant. H9 holds.
REFERENCE ESTIMATE (Claude, 2026-10-27): the difference lies within 0.02 nDCG@10 either way, BM25 slightly ahead.
Reason: both rank the same candidate set, and SSR uses no idf.

---

## Clarifications
Details the text above left open, fixed after the commit of 2026-10-27 but before any run they concern.
They add to the text above; they do not change it. Format: date, what is fixed, why.

- 2026-10-27, B3 (the Setting says "one named embedding model"): B3 = BAAI/bge-base-en-v1.5 at Hugging Face
  revision a5beb1e3e68b9ab74eb54cfd186867f64f240e1a. Queries get the prefix the model card recommends for
  retrieval ("Represent this sentence for searching relevant passages: "); documents are title and text from
  data/scifact.tsv, at most 512 tokens. Embedding = normalized CLS vector; score = cosine similarity, exact search
  over all documents, top 100 kept. Why this model: a strong, widely used open model that runs on a CPU, so B3 is
  a fair rival, not a weak one. Its model card reports nDCG@10 = 0.740 on SciFact test (MTEB). The card does not
  list its training data, so it cannot be confirmed from the card that SciFact was not used in training.
  Timing (added 2026-10-27): this clarification was not "before any run" as the heading requires. B3 was first
  run on the train split at 16:59, from code not yet committed (its run file named commit 9cf3a80, which has no
  B3 code); this clarification and the B3 code were committed together at 19:16 (commit 30500f9). The settings
  recorded in that run's file are the ones above. B3 was then run again on train from the commit that adds this note, so that
  its run file names the exact code. No B3 test run had been made.
- 2026-10-27, H1 precision (H1 says k = the size of the query's Boolean set, but not what happens when the set is
  larger than the pool N, nor which N; fixed before any E1 run): let S be the query's Boolean set. Precision is
  computed at k = min(|S|, N) with N = 100, as in H2. E1's precision is that of the top k documents of S ranked by
  BM25 (ties by doc id); B1's and B3's is that of their own top k for the same query. A system that returns fewer
  than k documents counts the missing ranks as not relevant. When |S| <= N this is exactly the text of H1.
  N = 30 is also reported, but it is not a confirmatory test and is not in the Holm correction.
  Empty sets (|S| = 0): the primary analysis leaves the query out, as the Setting says; the second analysis gives
  E1 precision 0 and compares it with B1's and B3's precision at k = N.
  Why: the Setting cuts every Boolean result to the top N and compares every system on the same N, and only the
  top N reaches the reranker; one fixed N avoids two tests for one hypothesis.
  Run files: E runs write the usual run file cut at N, plus a file with one line per query, query_id <TAB>
  set_size. Set size 0 means the query matched nothing. A query that failed (for example ParseError after the
  retries) has no line; it is logged and reported separately, and is treated like an empty set in both analyses
  (left out of the primary one, 0 in the second), so failures cannot silently leave the second analysis.
- 2026-10-27, targeted constraint set (the Setting fixes its size, categories and kind of ground truth, not the
  details; fixed before any query of the set is written and before the planner exists). Details and reasons in
  targeted_set_design.md (D1 to D10, recommendations drafted by Claude, adopted by Aris).
  - Built from SciFact train claims. Every base claim is listed in the set's files and is never used while
    developing the planner. No SciFact test claim is used.
  - Ground truth: for a query built from claim q with relevant documents R(q): exclusion = the documents of R(q)
    that do not match the rule; phrase and identifier = the documents of R(q) that match it; paraphrase = R(q).
    The rule is a case-insensitive regular expression with word boundaries and explicit spelling variants, applied
    to the raw text of data/scifact.tsv (not to the engine's tokens), written before any run.
  - Selection uses only qrels and corpus text, never a system's output. Exclusions: |R(q)| >= 2 and the rule
    removes at least one relevant document and keeps at least one ("trap"); if fewer than 25 qualify, the rest
    are |R(q)| = 1 queries whose relevant document does not match and with at least 3 distractors (documents
    matching the rule and the claim's rarest content word). Each query records trap or no-trap; both reported.
    Phrases and identifiers: the term occurs in at least 1 relevant document and in at least 2 documents of the
    corpus; whether it already occurs in the claim is recorded. Paraphrases: the share of the claim's content
    words that occur in its relevant documents at least halves.
  - Query text: the claim plus one natural sentence with the constraint; paraphrases are rewritten claims.
    Written by Aris by hand from a candidate list made by a script; no LLM writes queries.
  - 25 queries per category (100); each claim text used at most once, in one category.
  - Greek letters: replaced by their names (alpha, beta, ...) once in the corpus and query preparation, so
    data/scifact.tsv and the query files change. B1, B2 and B3 are rerun on train after that; no test run had
    been made. The Phase 0 Pyserini check was made on the earlier corpus.
  - Files: data/targeted/queries.tsv, rules.tsv, qrels.tsv (qrels built by a script, checked by a script).
  - Frozen when committed; after that, changes only under Deviations. No system runs on the set before that.
- 2026-10-27, E0 and H4 (fixed before any E run; recommendations drafted by Claude, adopted by Aris):
  - E0 = the engine with (^ word1 word2 ...): an AND of exactly the words B2 uses for the query (lowercase,
    split on every non-alphanumeric character, Greek letters as names), ranked by BM25 with k1 = 1.2, b = 0.75.
    Why AND: an OR of the same words is B2 itself. Many E0 results are expected to be empty; the zero-result
    rate is reported next to both analyses.
  - H4 (E2 minus E1): the primary analysis keeps the queries where E2 is non-empty (E2 is system A); the second
    counts empty results as 0. Not "both non-empty": if relaxation runs only when E1 is empty, E2 = E1 on every
    query where E1 is non-empty, and the difference would be 0 by construction. When relaxation is triggered
    (only on zero results, or also on few) is fixed with the planner design, before any E2 run.

## Deviations
(None yet. Format: date, what changed, why, what the original text said.)
