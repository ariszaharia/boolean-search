# Thesis progress summary (as of 2026-10-15)

Plan: write the thesis text only after all the code and experiments are finished.

## The thesis in one paragraph
Strict Boolean retrieval versus dense vector search in agentic RAG. An LLM writes an S-expression
Boolean query, a custom C++ engine (positional inverted index, BM25 ranking) runs it, and the result is
reranked. The question is NOT "does Boolean beat dense?" but "WHEN does strict Boolean win (exclusions,
exact phrases and identifiers, auditability, cost) and when does it lose (paraphrase, LLM cost)?"
Full plan: https://claude.ai/code/artifact/7900fa8b-6823-4f65-a9b1-4641a002b011 (do not edit unless asked).
Related work that started the idea: Clarke & Smucker, "Boolean queries are all you need?", arXiv 2607.11362.

## What is done (Phase 0, the engine)
Query language: `(^ a b)` AND, `(+ a b)` OR, `(- A B)` NOT (A minus the rest), `"exact phrase"`, bare term.

- Parser: strict errors with a position (`ParseError`); nesting deeper than 500 is rejected (it used to crash).
- Query terms are normalized like the indexed text: `Cat` finds "cat"; `T-cell` is matched as the phrase "t cell".
- Phrase matching by word positions; works inside AND / OR / NOT.
- BM25 scored from the parsed query: a phrase is one unit (phrase frequency, summed idf); NOT scores only its first
  operand; ties are broken by doc id, so results are always in the same order. Term lookup is a binary search.
- Textbook BM25 including the (k1+1) factor, k1 = 1.2, b = 0.75 (tests pass these explicitly).
- Tests: `test_engine-v1.cpp` (6), `test_phrase.cpp` (25), `test_scoring.cpp` (12), `test_index_io.cpp` (12:
  UTF-8 text, terms glued to `(`/`"`, save/load). All pass. Build and run them with CMake (from `cpp/`):
  `cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release`, `cmake --build build`,
  `ctest --test-dir build --output-on-failure`. Executables are linked statically, because otherwise another
  program's libstdc++ DLL on PATH makes them stop at start (exit code 0xc0000139).
- Tokenizer: non-ASCII (UTF-8) bytes separate words, so "α-synuclein" is indexed as "synuclein". Fixed on
  2026-10-27 to pass bytes as unsigned char (it was undefined behaviour); SciFact results are byte-for-byte
  the same as before. Whether to keep Greek letters etc. as part of words is still open (matters for H3).
- Full SciFact corpus: `python api/prep_scifact.py` -> `data/scifact.tsv`, 5,183 documents, ids increasing.
- Pyserini comparison (300 SciFact test queries, Pyserini built without stemming/stopwords, same k1/b):
  top-10 overlap 0.959, same #1 document 96.7%, Spearman 0.985. Scores differ by exactly the factor (k1+1) = 2.2,
  because Lucene drops it (no effect on ranking); after dividing, median difference 0.84%. Logged as Deviation 1 in
  `results/pyserini_thresholds.md`. Remaining differences: Lucene keeps decimals like 0.05 as one token, the engine
  splits them (checked by hand on one query only).

How to rerun the comparison (from the project folder):
`python api/baseline_pyserini.py`, then build with CMake (above) and run
`cpp\build\search_cli data\scifact.tsv results\queries_words.tsv results\engine_top100.tsv 100`, then
`python api/compare_runs.py`.
Python: use the project venv, `venv\Scripts\python.exe` (it has pyserini, ir_measures, numpy). Plain `python`
on this machine opens the Microsoft Store shortcut.

Evaluation harness (`harness/`, self-tests: `venv\Scripts\python.exe -m harness.selftest`, 36 checks, all pass):
- `python -m harness.evaluate --run R.tsv --split train` scores one run (nDCG@10, R@30, R@100; R@30 and R@100
  are the recall of the N = 30 and N = 100 pools).
- `python -m harness.compare --a A.tsv --b B.tsv --split train` compares two runs, A minus B: paired 95% bootstrap
  interval (10,000 resamples, fixed seed) and a bootstrap p-value, for both analyses of hypotheses.md
  (non-empty queries only = primary; empty counted as 0 = second) plus the zero-result rate of each system.
- `harness.stats.holm` does the Holm correction, once, over all confirmatory tests; `stats.equivalent` is H7.
- An "empty" result means the query is missing from the run file. The planner must log failures (for example
  repeated ParseError) separately, otherwise they look like zero results.
- Open, decide before the run: for H4 (E2 minus E1), the non-empty subset can be E2's (`--a E2`, default) or
  both systems' (`--nonempty-from both`). If relaxation runs only when E1 returns nothing, then E2 = E1 on every
  query where E1 is non-empty, so "both" gives a difference of exactly 0 by construction. For H4 the meaningful
  numbers are A = E2 (non-empty) and "empty counted as 0".

Baselines without an LLM (built and checked on 5 train queries 2026-10-27; full runs NOT run yet, they wait
for the predictions in hypotheses.md):
- `venv\Scripts\python.exe api/prep_queries.py` -> `data/scifact_queries_{train,test}.tsv` (809 / 300 queries).
- `venv\Scripts\python.exe api/run_baselines.py --system B1 --split train` (and `--system B2`) ->
  `results/runs/B1_train.tsv` plus `B1_train.json` with the settings, git commit and time.
- B1: Pyserini 2.4.0, index built from the same `data/scifact.tsv` with the default analyzer (Porter stemming
  AND English stopwords removed), in `results/work/lucene_index_default`. B2: `cpp/build/search_cli`, (+ words).
- Same words for both (lowercase, split on non-alphanumeric), same documents, same k1 = 1.2, b = 0.75.
- k1/b (decided 2026-10-27, written in the Setting of hypotheses.md): B1 and B2 keep k1 = 1.2,
  b = 0.75, so B1 minus B2 is only the analyzer. `--system B1-lit` is a check, not a system: k1 = 0.9, b = 0.4
  and the raw query text, like Pyserini's published BEIR baseline. Run it on test at the end, to show the setup
  reproduces the published SciFact BM25 number (about 0.67 nDCG@10). Not in any hypothesis, not in Holm.
  Checked: runs on all 809 train and 300 test queries without errors (not scored).
- Not built yet: precision of the Boolean set and top-k with k = set size (H1), per-category reporting
  (needs the targeted set's format), the final script that collects all confirmatory p-values for Holm.
- Open, decide before building E1 (blocks H1's precision): run files are cut at the top N, so they lose the size
  of the Boolean set. A query matching 2,000 documents looks the same as one matching exactly 100, but H1 needs
  k = the true set size. Decide (a) whether the engine writes the full set, or the top N plus a small file
  `query_id <TAB> set_size`, and (b) what "precision of the Boolean set" means when the set is larger than N:
  over the whole set, or over the top N. H1's text does not say; write the choice into hypotheses.md
  (clarification or Deviation, dated) before any E1 run.

## Decisions taken
- No stemming and no stopword removal in the engine for now: the engine is a pure exact-match engine.
  Baseline B1 (Pyserini default analyzer) versus B2 (exact match) shows what stemming plus stopword removal
  change together (the default analyzer does both; corrected 2026-10-27, this said "stemming alone").
- Keep the textbook BM25 formula; compare with Pyserini after dividing by (k1+1).
- Work step by step; Aris reads every diff and must be able to explain every line.

## Things to remember / unresolved
- `bm25scorer.hpp` now defaults to k1 = 1.2 (was 1.5 until 2026-10-27); experiments still pass it explicitly.
- `_phase0_reference/` holds an older, bigger version of the engine (Porter stemmer, stopwords, positions with gaps,
  save format v4, 139 tests). Not part of the build. Use it only if stemming/stopwords are wanted later.
- Something outside Claude removed comments from `inverted_index.hpp/.cpp` while editing. Cause unknown (editor
  plugin or formatter?). Happened again before 2026-10-27 (comments put back by hand). Still unexplained.
- `.git` contains harmless leftover files named `stale-index-lock-claude*` (can be deleted).
- Check whether the university requires disclosing AI-written code, before submitting.
- The tokenizer difference (decimals) is only verified on one query.

## What is left, in order
1. Hypotheses H1 to H9: predictions filled in 2026-10-27 (Claude's reference estimates, adopted by Aris and
   marked as such; kappa 0.6, M = 0.03, H8 threshold 80 %). Left: commit, then write the date and commit id in
   "Committed on" (a second small commit). Only after that may the baseline runs start.
2. Evaluation harness: SciFact qrels, dev/test split, `ir_measures` (nDCG@10, recall), bootstrap confidence
   intervals, Holm correction, `results/` layout. Run baselines without an LLM first: B1 (Pyserini stemmed),
   B2 (exact-match BM25), B3 (dense, one named embedding model).
3. Planner: an OpenAI model with a pinned version, temperature 0, prompt with grammar and examples, everything
   logged and cached; retries on `ParseError`; relaxation loop for zero results (E0, E1, E2).
4. Reranker (Jev and an open cross-encoder) on the same top-N pool for every system.
5. Targeted constraint query set (about 100 queries in 4 categories, written before any system runs) and the
   engine's `why-not(doc, query)` function.
6. Ablations, error analysis (about 100 failures), auditability study.
7. Stretch, cut from the bottom if time runs short: SSR ranking, TREC-COVID and latency, multi-call agent,
   end-to-end faithfulness.
8. Thesis text. Deadline June 2027; about 10 hours a week.
