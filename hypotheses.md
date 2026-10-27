# Hypotheses (pre-registration)

Status: DRAFT written 2026-10-15. Aris fills every "MY PREDICTION" line in his own words and numbers,
then commits this file. Committed on: ____ (date and git commit id).
Rule: no experiment may run before this file is committed. After that, the text of H1 to H9 is not edited.
Anything that has to change is added at the bottom under "Deviations" with a date and the reason.

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
- Empty Boolean results: precision is undefined when a candidate set is empty. Primary analysis uses the queries
  with a non-empty set; a second analysis counts empty sets as 0. Both are reported, with the zero-result rate.
- Caveat for the targeted set: its ground truth is derived mechanically (a relevant set from existing qrels, filtered
  by a text rule), which favours lexical constraints by construction. Results on it are evidence about
  constraint handling only, not about retrieval in general.

Core hypotheses: H1 to H5, H7. Stretch (only if time allows): H6, H8, H9.

---

## H1. Precision on constraint queries
Statement: On constraint queries, the Boolean candidate set (E1) has higher precision than the BM25 (B1) and
dense (B3) top-k, where k equals the size of that query's Boolean set.
Primary test: mean paired difference in precision, E1 minus B1, on constraint queries (E1 minus B3 also reported).
Falsified if: the 95% interval of E1 minus B1 includes 0 (BM25 matches within the interval).
MY PREDICTION (direction and size, e.g. "E1 higher by ____ points"): ____

## H2. Recall on paraphrase queries
Statement: On paraphrase queries, the Boolean candidate set has lower recall than BM25 and dense.
Primary test: mean paired difference in recall of the top-N pool (N = 100), E1 minus B1, on paraphrase queries
(E1 minus B3 also reported).
Falsified if: the 95% interval of E1 minus B1 includes 0 or is positive.
MY PREDICTION: ____

## H3. nDCG on exclusions and exact identifiers
Statement: After the same reranker on the same pool, E2 beats BM25 (B1) and dense (B3) in nDCG@10 on exclusion
queries and on exact-identifier queries.
Primary test: mean paired difference in nDCG@10, E2 minus B1, separately for each of the two categories.
Falsified if: no significant gain for either category after Holm correction.
MY PREDICTION (per category): ____

## H4. Relaxation recovers recall
Statement: The relaxation loop (E2) recovers part of the recall lost by the strict query (E1), at some cost in
precision.
Primary test: mean paired difference in recall of the top-N pool, E2 minus E1, on SciFact test and on the
targeted set. The precision change is reported next to it.
Falsified if: the recall gain is not significant after Holm correction.
MY PREDICTION (recall gain of ____ points, precision cost of ____ points): ____

## H5. Failures of Boolean can be traced from the query
Statement: Most Boolean failures can be diagnosed from the generated query and the `why-not` output alone;
dense failures cannot be diagnosed from their output.
Primary test: on 50 failures per system type, an annotator names the cause using only the query and `why-not`
(Boolean) or only the ranked output (dense). Compare the two proportions with a 95% interval. A second annotator
labels 30 to 50 of them; Cohen's kappa is reported.
Falsified if: the two proportions differ by less than the annotator disagreement (the interval of the difference
includes 0, or kappa is below ____).
MY PREDICTION (share diagnosable: Boolean ____ %, dense ____ %): ____

## H6. Faithfulness end to end (stretch)
Statement: End to end, answer faithfulness is no worse for Boolean when the evidence is retrieved, but answer
coverage is lower.
Primary test: claim-level support rate, E2 pipeline minus B3 pipeline, with the LLM judge spot-checked by hand
on 50 items.
Falsified if: faithfulness is significantly lower for Boolean.
MY PREDICTION: ____

## H7. Null on the standard set
Statement: After reranking, nDCG@10 on the standard SciFact test queries differs little between the pipelines
(E2, B1, B3, each with the same reranker).
Primary test: equivalence. The 95% interval of each paired difference must lie inside plus or minus the margin
M = ____ nDCG points (suggested: 0.02; Aris decides before the run).
Falsified if: Boolean is clearly ahead, i.e. the interval of E2 minus B1 lies above +M.
Note: kept as a null on purpose. Reporting it honestly makes H3 more credible.
MY PREDICTION: ____

## H8. Cost versus quality of the planner (stretch)
Statement: Single shot plus relaxation (E2) reaches most of the multi-call agent's (E4) nDCG@10 with a fraction
of the model calls.
Primary test: nDCG@10 of E2 divided by nDCG@10 of E4 at several call budgets; calls per query reported.
Falsified if: nDCG@10 keeps rising steeply with the call budget (E2 reaches less than ____ % of E4).
MY PREDICTION: ____

## H9. SSR versus BM25 ranking (stretch)
Statement: On the same Boolean candidate set, SSR (shortest-substring ranking, C = 42) and BM25 rank similarly.
Primary test: mean paired difference in nDCG@10, E3 minus E1.
Falsified if: one is significantly better after Holm correction.
MY PREDICTION: ____

---

## Deviations
(None yet. Format: date, what changed, why, what the original text said.)
