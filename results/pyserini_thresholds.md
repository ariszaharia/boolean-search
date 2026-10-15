# Pyserini comparison: pass thresholds

Written: 2026-10-15, BEFORE the comparison was run.
Status: DRAFT. Aris edits the numbers, then commits this file. Only after that is the comparison run.
Rule: thresholds are not changed after seeing results. If one is changed anyway, it is added at the bottom as a "Deviation" with the reason.

## Purpose
Show that the C++ engine ranks documents like Lucene/Pyserini BM25 when both use the same text,
the same tokenization settings and the same BM25 parameters. This is a sanity check on the engine,
not a claim about retrieval quality.

## Setup (fixed in advance)
- Corpus: the same 5,183-document SciFact export (data/scifact.tsv) for both systems.
- Pyserini side: a Lucene index built from that file with stemming OFF and stopwords KEPT
  (`--stemmer none --keepStopwords`). NOT the prebuilt stemmed index.
- BM25 on both sides: k1 = 1.2, b = 0.75.
- Engine side: no stemming, no stopwords.
- Queries: the SciFact test queries (300 expected). Each query is lowercased and split on
  non-alphanumeric characters into words. Pyserini gets the words as a plain query;
  the engine gets the same words as (+ w1 w2 ...).
- Compared: top 10 for overlap and score agreement, top 100 for rank correlation.
  Ties in the engine are ordered by ascending doc id.

## Metrics and thresholds (PASS needs all of them)
1. Top-10 overlap per query = |engine top 10 ∩ Pyserini top 10| / 10.
   PASS: mean >= 0.95 AND at least 90% of queries have overlap >= 0.8.
2. Same top-1 document: PASS if at least 90% of queries.
3. Spearman rank correlation over documents present in both top-100 lists, per query.
   PASS: mean >= 0.95.
4. Score agreement for documents present in both top-10 lists:
   PASS: median relative difference |engine - pyserini| / pyserini <= 2%.

## If it fails
List the 10 queries with the lowest overlap and sort the causes (tokenization differences,
Lucene's length-normalization rounding, a bug) BEFORE touching any threshold.

## Known reasons for small differences (so they are not a surprise)
- Lucene's tokenizer is not identical to "split on every non-alphanumeric character"
  (apostrophes, decimal numbers). Expected to affect a minority of queries.
- Lucene stores document length in a rounded form, so scores can differ slightly.
  These two points are expectations, not verified facts. The run will show how large they are.

## Results (run 2026-10-15, 300 SciFact test queries; details in results/pyserini_comparison.json)
As pre-registered, with the engine's raw scores:

| Metric | Measured | Threshold | |
|---|---|---|---|
| Mean top-10 overlap | 0.959 | >= 0.95 | PASS |
| Queries with overlap >= 0.8 | 97.3% | >= 90% | PASS |
| Same top-1 document | 96.7% | >= 90% | PASS |
| Mean Spearman (top 100) | 0.985 | >= 0.95 | PASS |
| Median relative score difference (shared top-10 documents) | 118.2% | <= 2% | FAIL |

Overall result under the pre-registered rules: FAIL (metric 4).

## Deviation 1 (written after seeing the results, 2026-10-15)
Cause of the metric 4 failure: the engine uses the textbook BM25 form with the factor (k1 + 1) in the
numerator; Lucene (version 8 and later) omits that factor. With k1 = 1.2 the engine's scores are 2.2 times
larger. For 98.9% of the 2,878 compared document pairs the ratio engine/Pyserini is between 2.0 and 2.4.
The factor is identical for every term and document, so it cannot change the ranking, which the passing
metrics 1 to 3 confirm. The expectation written above (Lucene's rounded document lengths) was incomplete:
this constant factor was the main cause and was missed in advance.

Decision: the engine's formula is kept unchanged (textbook form). Metric 4 is redefined as the median
relative difference between (engine score / (k1 + 1)) and the Pyserini score. The threshold (<= 2%) is not changed.
Result: median 0.84% -> PASS. Share of pairs within 1% / 2% / 5% / 10%: 57.5% / 82.9% / 97.4% / 99.1%.
Overall result under the amended metric 4: PASS. Both versions of the metric are reported.

Remaining differences (diagnosed, one example checked by hand): Lucene keeps a decimal such as 0.05 as a single
token, while the engine splits it into 0 and 05. This changes document frequencies (the token "0" occurs in
951 documents under the engine's rule and in 125 under a decimals-kept-together rule) and document lengths.
Example, query 1, document 10608397: engine-style tokenization gives 4.485, decimals kept together gives 5.997,
Pyserini gives 6.103 (all without the (k1 + 1) factor). The lowest-overlap queries (0.3 to 0.5) were not all
inspected; this explanation is verified for one query only.
