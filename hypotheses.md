# Hypotheses (pre-registration)

Status: written 2026-10-15; predictions and open values filled in 2026-10-27, before any experiment.
On 2026-10-27 Aris adopted the reference estimates (below) as the predictions and the values Claude suggested
for kappa (H5), M (H7) and the H8 threshold; each "MY PREDICTION" line says so.
Committed on: ____ (date and git commit id).
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

## Deviations
(None yet. Format: date, what changed, why, what the original text said.)
