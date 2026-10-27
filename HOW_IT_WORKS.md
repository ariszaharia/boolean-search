# How the project works (everything after the engine)

Written 2026-10-27. The C++ engine (parser, index, BM25) is assumed known and is not explained here.
Each part is marked **BUILT** or **PLANNED**. For the order of the remaining work, see `PROGRESS.md`.

---

## 1. The big picture

The thesis compares *ways of finding documents for a question*. Each way is called a **system**. Every system
gets the same questions, returns a ranked list of documents for each one, and every list is scored against the
same answer key. All the code after the engine exists to make that comparison fair and repeatable.

```
                       SciFact (questions + documents + answer key)
                                         |
        +--------------------------------+--------------------------------+
        |                                |                                |
  Baselines, no LLM                Boolean systems                   (later) reranker
  B1 BM25 stemmed (Pyserini)       E0 flat AND of the words           reorders the top N
  B2 BM25 exact (our engine)       E1 LLM writes a Boolean query      of every system
  B3 dense (embedding model)       E2 E1 + relaxation loop
        |                                |                                |
        +---------------> run file (one ranked list per question) <-------+
                                         |
                              harness: score each run
                         (nDCG@10, recall, against the qrels)
                                         |
                       harness: compare two runs, per question
                  (difference, bootstrap interval, p-value, Holm)
                                         |
                   hypotheses.md decides what counts as confirmed
```

The important idea is that systems never talk to the harness directly. They write a **run file**, and the
harness only reads run files. Any system (the C++ engine, Pyserini, a dense model, the LLM agent) can be
evaluated the same way, as long as it writes that file.

---

## 2. Where things are

| Path | What it is | Status |
|---|---|---|
| `api/prep_scifact.py` | Downloads the SciFact documents and writes `data/scifact.tsv` | BUILT |
| `api/prep_qrels.py` | Downloads the SciFact answer key and writes `data/scifact_qrels_{train,test}.tsv` | BUILT |
| `api/baseline_pyserini.py` | Pyserini side of the engine check (unstemmed BM25) | BUILT |
| `cpp/src/search_cli.cpp` | Runs the engine on a file of queries and writes a run file | BUILT |
| `api/compare_runs.py` | Engine-versus-Pyserini check (agreement of rankings) | BUILT |
| `harness/` | Scoring and statistics for every experiment | BUILT (partly) |
| `hypotheses.md` | The pre-registration: what is tested and how, written before any experiment | BUILT, predictions still blank |
| `results/pyserini_thresholds.md` | Pre-registration of the engine check, with its results | BUILT |
| `results/` | Run files and result files | BUILT |
| `api/agent.py` | The LLM planner (writes Boolean queries) | PLANNED (empty file) |
| `_phase0_reference/` | An older, bigger engine (stemmer, stopwords). Not built, kept only for reference | — |
| `venv/` | The Python environment to use: `venv\Scripts\python.exe` (has pyserini, ir_measures, numpy) | — |

---

## 3. The data: SciFact

SciFact is a public benchmark of scientific claims. It has three parts:

1. **Corpus**: 5,183 scientific abstracts (title + text). These are the documents being searched.
2. **Queries**: short scientific claims, e.g. *"0-dimensional biomaterials show inductive properties."*
3. **Qrels** ("query relevance judgments", the answer key): for each query, the documents that experts judged
   relevant. In SciFact, relevance is always 1 (relevant) and most queries have one or two relevant documents.

### `api/prep_scifact.py` → `data/scifact.tsv`
One document per line: `doc_id <TAB> title text`. Documents are sorted by numeric id, because the engine
requires increasing ids. Tabs and newlines inside the text are replaced by spaces, so each document fits on one line.

### `api/prep_qrels.py` → `data/scifact_qrels_train.tsv` and `data/scifact_qrels_test.tsv`
One judgment per line: `query_id <TAB> doc_id <TAB> relevance`. Example: `1  31715818  1` means "for query 1,
document 31715818 is relevant".

### Train and test (the two splits)
SciFact comes with two official sets of queries. The project uses them like this:

- **train (809 queries) = development set.** Use it freely to build things and tune them: the LLM prompt,
  the relaxation rules, sanity checks. Looking at results here is fine.
- **test (300 queries) = held out.** Only for the final numbers in the thesis, after everything is frozen.
  If you tune anything after looking at test results, the test numbers no longer show how well the method
  works on unseen questions; they only show how well it was fitted to those 300.

---

## 4. The run file: what every system produces

Every system writes its results in the same format, one result per line:

```
query_id <TAB> rank <TAB> doc_id <TAB> score
1        1      10608397  6.103
1        2      43385013  4.594
```

- `rank` starts at 1 for the best document and increases without gaps.
- `score` is the system's own score (BM25, cosine similarity, ...). The harness **does not use it for ranking**;
  the written rank order is what gets scored (so ties in score can never reorder anything).
- **A query that is missing from the file means "this system returned nothing for it"** (an empty result).
  This matters for Boolean systems, where an AND query can match zero documents.

`harness/run_io.py` reads this format and refuses broken files: wrong number of columns, ranks with gaps,
the same document twice for one query.

---

## 5. The engine check against Pyserini (done, 2026-10-15)

**Question it answers:** does our C++ engine compute BM25 correctly? If it does not, every later result is suspect.

**Method:** run the same queries through our engine and through Pyserini (a well-known library built on Lucene,
the search engine behind Elasticsearch), with the same settings, and check that both rank documents the same way.

**Steps:**
1. `api/baseline_pyserini.py`
   - converts `data/scifact.tsv` to JSON lines and builds a Lucene index in `results/work/` with **stemming off
     and stopwords kept**, to match our engine (a stemmed index would give different results for a known reason,
     which would hide real bugs);
   - refuses to continue if the analyzer stems or removes stopwords (safety check);
   - takes the 300 test queries, lowercases them and splits them into words, the same way our tokenizer does,
     and writes them to `results/queries_words.tsv` so our engine gets exactly the same words;
   - runs BM25 (k1 = 1.2, b = 0.75) and writes the top 100 per query to `results/pyserini_top100.tsv`.
2. `cpp/build/search_cli data/scifact.tsv results/queries_words.tsv results/engine_top100.tsv 100`
   - turns each query into `(+ word1 word2 ...)` and writes the engine's top 100 per query.
3. `api/compare_runs.py`
   - compares the two run files: top-10 overlap, same first document, Spearman rank correlation, and how
     close the scores are. Writes `results/pyserini_comparison.json`.

**Pass thresholds** were written in `results/pyserini_thresholds.md` *before* running (see section 8).

**Result:** the rankings agree (top-10 overlap 0.959, same #1 document in 96.7% of queries, Spearman 0.985).
The scores were 2.2 times larger, because textbook BM25 multiplies by (k1 + 1) = 2.2 and Lucene does not.
Since the factor is the same everywhere, it cannot change any ranking. This was logged as "Deviation 1" in
the thresholds file. The remaining small differences come from tokenization: Lucene keeps `0.05` as one
token, while the engine splits it into `0` and `05`.

**Note:** this is a correctness check of the engine, not a hypothesis test. Unstemmed Pyserini is not one of the
thesis systems.

---

## 6. The evaluation harness (`harness/`)

Every experiment goes through this code. Run it with the project venv from the project folder:
`venv\Scripts\python.exe -m harness.<module>`.

### 6.1 `harness/data.py`: loading the answer key
- `load_qrels('train')` or `load_qrels('test')` → `{query_id: {doc_id: relevance}}`.
- `scorable_queries(qrels)`: the queries that have at least one relevant document. Only these are scored,
  because nDCG and recall cannot be computed without a relevant document. In SciFact that is all of them.

### 6.2 `harness/run_io.py`: reading run files
- `read_run(path)` → `{query_id: [(doc_id, score), ...]}` in rank order, with the checks from section 4.

### 6.3 `harness/metrics.py`: scoring one run
The numbers are computed by the `ir_measures` library, not by hand-written code, so they match what other
papers report. Three measures for now:

- **nDCG@10** ("normalized discounted cumulative gain at 10"). Looks at the top 10 only. A relevant document
  counts for more the higher it is ranked: rank 1 gives full credit, and lower ranks give less
  (1/log2(rank + 1)). The total is divided by the best possible total, so 1.0 = perfect ranking and
  0 = nothing relevant in the top 10. This is the standard *quality of the ranking* measure.
  Example: one relevant document at rank 2 → 1/log2(3) = 0.63.
- **R@100** (recall at 100): the share of the relevant documents that appear anywhere in the top 100.
  It ignores order. This is the *did we find it at all* measure, and it matters because a reranker can only
  reorder documents it was given.
- **R@30**: the same, for the top 30. R@30 and R@100 are the recall of the two pool sizes (N = 30 and N = 100,
  section 7): how much the reranker can possibly find in each.

Functions:
- `evaluate_run(run, qrels)` → `{measure: {query_id: value}}`, a value for every scorable query.
  **An empty result gets 0.**
- `empty_queries(run, qrels)`: the queries this system returned nothing for.
- `restrict(per_query, queries)`: keeps only the given queries (used for the primary analysis, below).
- `means(per_query)`: averages over queries.

### 6.4 `harness/evaluate.py`: command for one run
```
venv\Scripts\python.exe -m harness.evaluate --run results/X.tsv --split train
```
Prints the mean nDCG@10, R@30 and R@100, how many queries had no result, and saves per-query values to
`results/metrics/X_train.json`.

### 6.5 Why per-query comparisons and statistics are needed
A difference in averages (say, 0.66 versus 0.64) can be luck: with a different sample of 300 questions, the
order might flip. To make a claim, you need to know how much the difference would vary from sample to sample.
The statistics fixed in `hypotheses.md` do this in four steps:

1. **Paired differences.** For each query, compute *system A's score minus system B's score*. Pairing removes
   the fact that some questions are hard for every system; only the difference between the systems remains.
2. **Bootstrap confidence interval (95%, 10,000 resamples).** Pretend the 300 queries are the whole world.
   Draw 300 of them *with replacement* (some twice, some not at all), average their differences, and repeat
   10,000 times. The middle 95% of those 10,000 averages is the interval. Reading it: "the true difference
   is plausibly anywhere in here". If the interval does not contain 0, the difference is unlikely to be luck.
3. **p-value.** One number for "how surprising is this difference if the systems were really equal".
   It is computed from the same 10,000 averages (the share that lie at least as far from the observed mean as
   the observed mean lies from 0). It is needed only for step 4.
4. **Holm correction.** The thesis runs several confirmatory tests. With many tests, some look significant
   by luck alone. Holm makes the bar stricter for each test: sort the p-values, multiply the smallest by the
   number of tests, the next by one fewer, and so on. A test counts as confirmed only if its adjusted value
   stays at or below 0.05. It is applied **once, over all confirmatory tests together**, at the end.

**Equivalence (only H7).** H7 claims that the systems are *about the same* on the standard queries. That needs a
different test: the whole interval must lie inside plus or minus a margin M chosen in advance (for example
[-0.03, +0.03]). "Not significantly different" is not enough, because a small sample is never significantly
different from anything.

### 6.6 `harness/stats.py`: the statistics code
- `paired_bootstrap(a, b)` → `n`, both means, `mean_diff` (A minus B), `ci_low`, `ci_high`, `p_value`.
  `a` and `b` are `{query_id: value}` for the same queries; different query sets are refused. The random seed is
  fixed, so every rerun gives exactly the same interval.
- `holm({test_name: p})` → for each test, the adjusted p-value and whether it is rejected (confirmed).
- `equivalent(result, M)` → True if the interval lies inside [-M, +M].

### 6.7 Empty results: the two analyses
A Boolean query can match no document. Then precision (the share of returned documents that are relevant) is
0 divided by 0, so it is undefined. `hypotheses.md` therefore asks for two analyses, always reported together:

- **Primary (non-empty):** only the queries where the Boolean system returned something.
  This answers *"when Boolean returns something, how good is it?"*
- **Second (empty counted as 0):** every query, with an empty result scored 0.
  This answers *"over all questions, including the ones it gave up on, how good is it?"*
- And next to both: the **zero-result rate** (the share of queries that came back empty).

### 6.8 `harness/compare.py`: command for two runs
```
venv\Scripts\python.exe -m harness.compare --a results/E1.tsv --b results/B1.tsv --split train
```
For each measure and for both analyses, it prints n, the two means, A minus B, the 95% interval and the p-value,
plus each system's zero-result rate. It saves everything to `results/comparisons/E1_vs_B1_train.json`.

- `--a` is, by convention, the Boolean system; the primary analysis keeps the queries where **A** is non-empty.
- `--nonempty-from both` keeps only the queries where both are non-empty.
- Holm is **not** applied here (it needs all tests at once, see 6.5).

**H1's precision (`--sizes`, `--pool-n`).** An E run writes, next to its run file, a set-size file with one line
per query: `query_id <TAB> set_size`. Size 0 = the query matched nothing; a query that failed (for example a
ParseError after the retries) has no line. With
```
venv\Scripts\python.exe -m harness.compare --a E1_test.tsv --b B1_test.tsv --split test --sizes E1_test_sizes.tsv
```
the comparison gets one more measure, `P@min(|S|,100)`: for each query, k = min(set size, N) with N = 100
(`--pool-n 30` for the reported N = 30), and both systems are scored by precision at that k. Ranks a system does
not fill count as not relevant. An empty set or a failed query gets k = N: A scores 0 and B is scored at N. Both
count as empty in the two analyses (6.7), and the failed ones are also listed on their own (`failed_queries_a`).
Before scoring, the set-size file is checked against the run file (no results for an empty set or a failed
query, never more results than the set has, at least min(set size, N) results), so the three cases cannot be
mixed up. The output is saved as `E1_vs_B1_N100_test.json`. The rule itself is the H1 precision clarification
in `hypotheses.md`.

### 6.9 `harness/selftest.py`: checks of the harness itself
```
venv\Scripts\python.exe -m harness.selftest
```
67 small cases whose answers were worked out by hand: file parsing, nDCG and recall on tiny examples, the
bootstrap on cases where the interval is known, Holm on a textbook example, the two empty-result analyses,
output file names, H1's precision with set sizes (including every way a set-size file can disagree with
its run file), and the groups of the targeted set.
Run it after any change to `harness/`.

### 6.10 The targeted set: results by group
`--split targeted` scores against `data/targeted/qrels.tsv` (the frozen set, section 9.4). Its 100 queries are
never used as one pool: `evaluate` and `compare` also report every group on its own queries, each with its own
interval and p-value, and save them under `by_group`:

| group | queries | used by |
|---|---|---|
| `constraint` | exclusion + phrase + identifier (75) | H1 ("constraint queries") |
| `exclusion`, `phrase`, `identifier` | 25 each | H3 (exclusion, identifier) |
| `paraphrase` | 25 | H2 |
| `phrase/trap`, `phrase/no-trap`, `identifier/trap`, `identifier/no-trap` | 1, 24, 1, 24 | description only |

The pooled numbers are still printed, labelled "not used by any hypothesis". A trap group with 1 query has no
meaningful interval; it is there only so the split by kind is visible.

### 6.11 Not built yet
- **The final Holm script**, which collects the p-values of all confirmatory tests. Waits for the experiments.

---

## 7. The systems compared

| Name | What it is | Needs an LLM | Status |
|---|---|---|---|
| **B1** | BM25 with Pyserini's default analyzer (stemming and stopword removal), on an index built from our own `data/scifact.tsv` | no | built (`api/run_baselines.py`); full run waits for the predictions |
| **B2** | BM25 exact match: our engine with `(+ all question words)` | no | built (`api/run_baselines.py`); full run waits for the predictions |
| **B3** | Dense retrieval: one named embedding model turns questions and documents into vectors; the nearest vectors win | no | PLANNED, model not chosen |
| **B4** | BM25 on the question plus extra words suggested by the LLM | yes | PLANNED |
| **B5** | Dense retrieval on the same LLM-expanded question | yes | PLANNED |
| **E0** | Engine, flat AND of exactly B2's words (`(^ w1 w2 ...)`), no LLM | no | DONE (train, targeted) |
| **E1** | The LLM writes one Boolean query (`(^ ...)`, `(- ...)`, phrases); the engine runs it | yes | PLANNED |
| **E2** | E1 plus the relaxation loop (section 9.2) | yes | PLANNED |
| **E3** | E1 ranked by SSR instead of BM25 (stretch) | yes | PLANNED |
| **E4** | E2 as a multi-call agent (stretch) | yes | PLANNED |

**Why B1 versus B2:** they get the same words, the same documents and the same BM25 settings, and differ only
in the analyzer. B1 stems ("cells" = "cell") and ignores stopwords ("the", "of"); B2 does neither. The gap
between them shows how much exact matching costs, separately from anything the LLM does. Because the
default analyzer does both things at once, the gap covers stemming and stopwords together, not stemming alone.

**How to run them** (from the project folder, after `api/prep_queries.py`):
`venv\Scripts\python.exe api/run_baselines.py --system B1 --split train`, the same with `--system B2`.
Each writes `results/runs/<system>_<split>.tsv` and a `.json` with the exact settings, git commit and time.

**B1-lit (a check, not a system):** `--system B1-lit` runs B1 the way Pyserini's published BEIR baseline does
(k1 = 0.9, b = 0.4, raw query text). Its only job is to show that the setup reproduces the published BM25
number for SciFact, so a reader can see B1 is not accidentally weak. It is run on the test split at the end,
with the final runs, and is not part of any hypothesis or of the Holm correction.

**The pool and N:** for the comparisons after reranking, each system's result is cut to its top N (N = 30 and
N = 100) and the reranker reorders those N. Every system gets the same N, so no system wins just by being
allowed more documents.

---

## 8. Pre-registration: writing the rules before the results

**The problem it solves:** after seeing the results, it is always possible to find a threshold, a subset or
a metric under which your method looks good, even without meaning to. Writing down the hypotheses, tests and
pass criteria *before* running, and committing them to git (which timestamps them), makes the results credible.

**Files:**
- `results/pyserini_thresholds.md`: the rules for the engine check (section 5), already used.
- `hypotheses.md`: the rules for the thesis experiments. For each hypothesis H1 to H9: the statement, the
  exact test, what would falsify it, **your prediction**, and a separate reference estimate from Claude.

**The rule:** no experiment runs before `hypotheses.md` is complete and committed. After that, the text is not
edited. Anything that has to change goes under "Deviations" at the bottom, with the date and the reason
(as was done for Deviation 1 of the engine check).

**Still to do before any experiment:** fill in every "MY PREDICTION" line, the margin M (H7), the kappa
threshold (H5), the percentage in H8 and "Committed on"; add how the p-value is computed (section 6.5); commit.

### The hypotheses in one line each
- **H1:** on constraint queries, the Boolean set is more precise than BM25 or dense.
- **H2:** on paraphrase queries, Boolean finds less (lower recall).
- **H3:** after reranking, Boolean wins on exclusions and exact identifiers.
- **H4:** relaxation wins back part of the lost recall, at some cost in precision.
- **H5:** Boolean failures can be explained from the query; dense failures cannot.
- **H6 (stretch):** answers built on Boolean results are just as faithful, but cover less.
- **H7:** on the normal SciFact queries, all pipelines end up about the same (equivalence test).
- **H8 (stretch):** a single LLM call plus relaxation gets most of what a multi-call agent gets.
- **H9 (stretch):** SSR and BM25 rank the same Boolean set about equally well.

---

## 9. Planned components

### 9.1 The planner ("the agent", `api/agent.py`)
The part that turns a natural-language question into a Boolean query for the engine.

- **Input:** a question, e.g. *"Does vitamin D reduce fractures in elderly people, excluding studies on children?"*
- **Output:** a query in the engine's language, e.g.
  `(- (^ "vitamin d" (+ fracture fractures) (+ elderly aged)) children)`.
- **How:** an OpenAI model with a **pinned version** (so it cannot change under you), **temperature 0** (as
  repeatable as possible), and a prompt containing the grammar and a few worked examples.
- **Retries:** if the engine rejects the query with a `ParseError`, the error message (which includes the
  position) is sent back and the model tries again, a limited number of times. Failures are logged separately:
  a query that never parsed must not look like a query with zero results.
- **Caching and logging:** every prompt and answer is saved. Reruns cost nothing and are exactly reproducible,
  and the logs are the evidence for the error analysis (H5).
- **Tuning on train only.**

### 9.2 The relaxation loop (E2)
A strict Boolean query often matches nothing. The relaxation loop then loosens it step by step, for example
by asking the LLM for a less strict version, turning an AND into an OR, or dropping the least important
condition, until the result is non-empty. H4 measures how much recall this wins back and how much precision
it costs.

### 9.3 The reranker
A model that reads the question and each candidate document together, and scores how well they match. It is
slower than BM25 but more accurate, so it is only used on the top N of each system. Every system gets the same
reranker on the same N, so the comparison is about *which documents each system brought*, not about the reranker.

Built (2026-10-28): `api/rerank.py` reranks the top N (30 or 100) of any run file and writes
`<run>-rr<N>-<reranker>_<split>.tsv` next to it (for example `B1-rr100-ce-minilm_train.tsv`). The reranker reads
the request as written (constraint sentence included), never the Boolean query. The rerankers are entries in
`RERANKERS`; the only one now is a free local cross-encoder (`cross-encoder/ms-marco-MiniLM-L6-v2`, revision pinned),
for development. Which reranker the confirmatory runs use (Jev and/or a cross-encoder) is not decided; until an
entry is marked as fixed in hypotheses.md, the script reranks train runs only. Scores are cached in
`results/work/rerank_cache/`, one file per device. On this laptop's CPU it scores about 19 pairs a second (809
train queries at N = 100 take over an hour per system), so long runs go to a free Colab GPU: open
`colab/rerank.ipynb` in Colab (push first; it clones the repository), run all, unzip the downloaded
`rerank_results.zip` into the project folder. The run's `.json` records the device.

### 9.4 The targeted constraint query set
SciFact's own queries rarely contain exclusions or exact identifiers, which are exactly where Boolean should
win. So about 100 extra queries are written, in four categories:

1. **Exclusions** ("... but not in mice")
2. **Exact phrases and entities** (a named gene, a drug name)
3. **Numeric or identifier constraints** (a dose, a trial id)
4. **Paraphrase control** (the same need in different words, where Boolean should lose)

The relevant documents come from the existing qrels, filtered by a text rule (e.g. "relevant, and does not
mention mice"). As `hypotheses.md` admits, that favours lexical methods by construction, so results on this set
only show how well each system handles constraints.

**Order matters:** this set must be written and committed *before* the planner prompt exists. Otherwise you
may, without meaning to, write queries that your prompt already handles well.

### 9.5 `why-not(doc, query)` in the engine
Given a document and a Boolean query, it explains why the document was not returned: for example, "the
AND failed because the document says 'fractures', not 'fracture'". This is the auditability claim of the
thesis (H5): a Boolean failure can be traced, a dense failure cannot.

### 9.6 Stretch items (cut from the bottom if time runs short)
SSR ranking (E3, H9), TREC-COVID (a larger dataset, for latency and size), the multi-call agent (E4, H8),
end-to-end answers and their faithfulness (H6).

---

## 10. How one experiment will run, end to end (example: H3)

1. Each system (E2, B1, B3) runs on the targeted set and writes a run file.
2. Each run is cut to its top N; the reranker reorders those N; the reranked lists are written as new run files.
3. `harness.compare --a E2_reranked.tsv --b B1_reranked.tsv` gives, for nDCG@10, the per-query differences,
   the 95% interval and the p-value, separately for the exclusion queries and the identifier queries,
   for both empty-result analyses.
4. The p-values of all confirmatory tests (H1 to H5, H7) go into `stats.holm` together.
5. H3 counts as supported only if at least one category stays significant after Holm. The result is compared
   with your prediction and with the reference estimate, and both are reported.

---

## 11. Status at a glance

| Part | Status |
|---|---|
| Engine + Pyserini check | done |
| Data, train/test qrels | done |
| Harness: scoring (nDCG@10, R@30, R@100), statistics, empty results | done, 36 self-tests pass |
| Harness: precision, categories, final Holm script | not built (waiting, see 6.10) |
| Pre-registration (`hypotheses.md`) | committed, but predictions still blank |
| Baselines B1, B2, B3 on train | next (after the predictions) |
| Targeted query set | before the planner |
| Planner, relaxation (E1, E2) | after the targeted set |
| Reranker, `why-not`, ablations, error analysis | after that |
| Stretch items, thesis text | last; deadline June 2027 |

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **qrels** | The answer key: which documents are relevant for which query. |
| **run / run file** | One system's ranked results for every query, in the format of section 4. |
| **split** | A fixed set of queries: train (for development) or test (held out, final numbers only). |
| **BM25** | The classic word-matching ranking formula used by the engine and Pyserini. |
| **stemming** | Cutting words to a common root ("cells", "cell" → "cell"). The engine does not stem. |
| **dense retrieval** | Questions and documents become vectors from a neural model; the closest vectors are returned. Finds synonyms, but cannot explain itself or obey a NOT. |
| **reranker / cross-encoder** | A model that reads the question and a document together and scores them; accurate but slow, so used only on a top N. |
| **pool, N** | The top N documents of a system that the reranker gets to reorder. |
| **precision** | Of the documents returned, the share that are relevant. |
| **recall, R@30, R@100** | Of the relevant documents, the share found in the top 30 or top 100. |
| **nDCG@10** | Quality of the top 10, rewarding relevant documents near the top; 1 = perfect. |
| **paired difference** | System A's score minus system B's score on the same query. |
| **bootstrap** | Resampling the queries with replacement many times to see how much a mean would vary. |
| **confidence interval** | The range the true difference plausibly lies in (here 95%). |
| **p-value** | How surprising the observed difference would be if the systems were really equal. |
| **Holm correction** | A stricter bar when several tests are run, so luck does not produce false wins. |
| **equivalence test** | Shows two systems are *about the same*: the whole interval lies inside a margin ±M. |
| **effect size** | How big a difference is (here simply the mean difference, with its interval). |
| **Cohen's kappa** | Agreement between two human annotators beyond chance (H5); 1 = perfect, 0 = chance. |
| **pre-registration** | Writing hypotheses, tests and predictions before the experiment, and committing them. |
| **deviation** | A logged, dated change to a pre-registered rule, with its reason. |
| **zero-result rate** | The share of queries for which a Boolean system returned nothing. |
| **relaxation** | Loosening a Boolean query that matched nothing until it matches something. |
| **temperature 0** | LLM setting that makes the answers as repeatable as possible. |
| **pinned model version** | Naming an exact model snapshot, so the LLM cannot change during the project. |
