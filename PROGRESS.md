# Thesis progress summary (as of 2026-10-27)

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
NOT needs at least two operands (since 2026-10-27): `(- A)` is a ParseError, because it used to return A itself,
the opposite of what it seems to say. A NOT operand can be any subquery: `(- A (+ mice mouse murine))`. A bare
word matches only that exact word (no stemming), and NOT removes a document wherever the word appears in it.

- Parser: strict errors with a position (`ParseError`); nesting deeper than 500 is rejected (it used to crash).
- Query terms are normalized like the indexed text: `Cat` finds "cat"; `T-cell` is matched as the phrase "t cell".
- Phrase matching by word positions; works inside AND / OR / NOT.
- BM25 scored from the parsed query: a phrase is one unit (phrase frequency, summed idf); NOT scores only its first
  operand; ties are broken by doc id, so results are always in the same order. Term lookup is a binary search.
- Textbook BM25 including the (k1+1) factor, k1 = 1.2, b = 0.75 (tests pass these explicitly).
- Tests: `test_engine-v1.cpp` (6), `test_phrase.cpp` (25), `test_scoring.cpp` (12), `test_index_io.cpp` (18:
  UTF-8 text, terms glued to `(`/`"`, NOT with one operand, save/load). All pass. Build and run them with CMake (from `cpp/`):
  `cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release`, `cmake --build build`,
  `ctest --test-dir build --output-on-failure`. Executables are linked statically, because otherwise another
  program's libstdc++ DLL on PATH makes them stop at start (exit code 0xc0000139).
- Tokenizer: non-ASCII (UTF-8) bytes separate words. Fixed on 2026-10-27 to pass bytes as unsigned char (it was
  undefined behaviour); SciFact results are byte-for-byte the same as before.
- Greek letters (D7, done 2026-10-27): `api/greek.py` writes every Greek letter, and the micro sign, as its
  name and as a separate word ("TNF-α" -> "TNF-alpha", "IFNγ" -> "IFN gamma", "TGFβ1" -> "TGF beta 1"). Applied
  in prep_scifact.py and prep_queries.py (523 documents, 16 train and 6 test claims changed; checked: the new
  files equal greek_to_names(old files) on every row, nothing else changed) and in `words_of`. The engine itself
  is unchanged. Anything that sends text to a system must pass it through greek_to_names too: the planner's
  queries later, and the targeted set's rules are written against the converted corpus. Other non-ASCII
  characters still separate words (for example the "ß" typed for β in train claim 740; it is the only one).
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

Evaluation harness (`harness/`, self-tests: `venv\Scripts\python.exe -m harness.selftest`, 67 checks, all pass):
- `python -m harness.evaluate --run R.tsv --split train` scores one run (nDCG@10, R@30, R@100; R@30 and R@100
  are the recall of the N = 30 and N = 100 pools).
- `python -m harness.compare --a A.tsv --b B.tsv --split train` compares two runs, A minus B: paired 95% bootstrap
  interval (10,000 resamples, fixed seed) and a bootstrap p-value, for both analyses of hypotheses.md
  (non-empty queries only = primary; empty counted as 0 = second) plus the zero-result rate of each system.
- `harness.stats.holm` does the Holm correction, once, over all confirmatory tests; `stats.equivalent` is H7.
- An "empty" result means the query is missing from the run file. The planner must log failures (for example
  repeated ParseError) separately, otherwise they look like zero results.
- Decided 2026-10-27 (clarification in hypotheses.md): H4 is run as `--a E2 --b E1` with the default
  `--nonempty-from a` (E2's non-empty queries), plus "empty counted as 0". Not "both": if relaxation runs only
  when E1 returns nothing, E2 = E1 wherever E1 is non-empty, so the difference would be 0 by construction.
  Still open, with the planner: when relaxation is triggered (only on zero results, or also on few).
- Decided 2026-10-27: E0 = `(^ word1 word2 ...)` of exactly B2's words, BM25 k1 = 1.2, b = 0.75 (an OR of the
  same words would be B2 itself).

Baselines without an LLM (B1, B2, B3 run on all 809 train queries 2026-10-27, after hypotheses.md was committed;
test runs NOT made yet, they come at the end with the final runs):
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
- B3: BAAI/bge-base-en-v1.5 at a pinned revision, query prefix from the model card, CLS vector normalized, exact
  cosine search (`--system B3`). Fixed in the Clarifications of hypotheses.md. Needs `transformers` and `torch`
  in the venv. Document embeddings are cached in `results/work/b3_docs_<revision>_<corpus hash>.npz`; 455 of
  5,183 documents are longer than 512 tokens and are cut. The first B3 run used uncommitted code; it was rerun
  from a commit (the timing is disclosed in the B3 clarification).
- Every run file's .json has `git_dirty`: true means tracked files outside `results/` had uncommitted changes
  when the run started, so `git_commit` alone is not the code that ran. Commit before every run that counts.
  It also has `corpus_sha256` and `queries_sha256`, the fingerprints of the exact corpus and query file used.
- Caches are tied to the corpus: each Lucene index has a stamp `<index>.corpus_sha256` and is deleted and rebuilt
  when the corpus changes; the B3 cache file name contains the corpus hash. (Before 2026-10-27 both were reused
  whenever they existed, which after D7 would have silently used the old text.)
- Train results (`results/metrics/`, `results/comparisons/`; train only, before reranking, not a test of any
  hypothesis):
  | system | nDCG@10 | R@30 | R@100 | before D7 (nDCG@10) |
  |---|---|---|---|---|
  | B1 | 0.699 | 0.869 | 0.933 | 0.697 |
  | B2 | 0.667 | 0.854 | 0.907 | 0.667 |
  | B3 | 0.769 | 0.938 | 0.972 | 0.771 |
  After D7 (rerun 2026-10-27, corpus_sha256 7865b98e...): B1 minus B2 nDCG@10 +0.032 (95% interval +0.020 to
  +0.044); B3 minus B1 +0.071 (+0.049 to +0.092). D7 moved 9 queries for B1, 12 for B2 and 14 for B3, both ways.
  It helps BM25 slightly and costs dense slightly: on claim 846 ("IL-1beta") B1 went 0.631 -> 1.0 and B3
  1.0 -> 0.631 (the embedding model presumably learned the Greek letter itself). Small, but worth one sentence
  in the thesis: the Greek-letter choice interacts with the system type.
  B1 returns fewer than 100 documents for a few queries (80,827 rows instead of 80,900).
- Output names (fixed 2026-10-27): the split appears once. evaluate/compare drop a trailing `_<split>` from the
  run file name and add it back once: `results/runs/B1_train.tsv` -> `results/metrics/B1_train.json`, and
  B3 against B1 -> `results/comparisons/B3_vs_B1_train.json` (`harness.run_io.run_name`). The old
  `B1_train_train.json`, `B2_train_train.json` and `B2_train_vs_B1_train_train.json` were regenerated under the
  new names (contents byte-for-byte the same) and removed.
- H1's precision (built 2026-10-27): `harness.compare --sizes E1_test_sizes.tsv [--pool-n 30]` adds
  `P@min(|S|,N)`, computed by ir_measures once per distinct k, after checking the set-size file against the
  run file. Checked on real runs: with every set size = the number of results, it equals plain P@100 exactly.
  On standard SciFact queries P@100 is about 0.01 for B1 and B2 (about 1 relevant document per query).
- Targeted split (built 2026-10-27): `--split targeted` in evaluate and compare reads `data/targeted/qrels.tsv`
  and reports every group on its own (constraint = H1, each category, trap/no-trap), saved under `by_group`;
  HOW_IT_WORKS.md 6.10. Checked: groups match the frozen set (75 / 25 x 4 / 24 + 1), a perfect run scores 1.0
  in every group, train outputs unchanged.
- Not built yet: the final script that collects all confirmatory p-values for Holm. The E runner must write the
  set-size file (format in HOW_IT_WORKS.md 6.8).
- Decided 2026-10-27 (H1 precision clarification in hypotheses.md, before any E1 run): precision at
  k = min(|S|, N), N = 100 primary (N = 30 reported, not tested); E1 = top k of its set by BM25, B1/B3 = their
  own top k. E runs write the usual run file cut at N plus `query_id <TAB> set_size` (0 = matched nothing,
  no line = failed; failures are reported separately and count like empty sets). Second analysis: an empty E1
  set scores 0 against B1/B3 at k = N.

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
1. DONE 2026-10-27: hypotheses H1 to H9 committed in 9cf3a80 (16:22), recorded in "Committed on".
2. Evaluation harness: built (see above): scoring, statistics, H1 precision, targeted groups. Baselines B1, B2,
   B3 run and scored on train. Left: the Holm script (needs the experiments).
3. Planner: an OpenAI model with a pinned version, temperature 0, prompt with grammar and examples, everything
   logged and cached; retries on `ParseError`; relaxation loop for zero results (E0, E1, E2).
4. Reranker (Jev and an open cross-encoder) on the same top-N pool for every system.
5. Targeted constraint query set (about 100 queries in 4 categories, written before any system runs) and the
   engine's `why-not(doc, query)` function.
   Keep in mind: SciFact has about 1 relevant document per query, and precision at k = 100 can be at most
   (number of relevant documents) / 100 for every system. If the targeted set's relevant sets are this small,
   large precision gaps (H1 predicts about +20 points) are only possible on queries with small Boolean sets.
   Design DECIDED 2026-10-27 (`targeted_set_design.md`, all recommendations adopted; summary in the
   clarifications of hypotheses.md). Moved before the planner (step 3). Order from here:
   a. DONE 2026-10-27: D7 (Greek letters -> names), caches tied to the corpus, B1 to B3 rerun on train.
      The unstemmed index of the Phase 0 check (`results/work/lucene_index`) has no stamp yet; it is rebuilt
      the next time api/baseline_pyserini.py runs. The Phase 0 numbers above were made on the earlier corpus.
   b. Candidate script DONE 2026-10-27: `venv\Scripts\python.exe api/targeted_candidates.py` (about 2 minutes)
      -> `data/targeted/candidates/` (columns explained at the top of the script). Shared definitions in
      `api/targeted.py` (stopwords = Lucene's English list, content words, overlap share, term -> rule, rule
      matching on raw text), which the build/check script will use too. 806 usable train claims (D10: 86 is a
      duplicate of 85; 89 and 90 have the same text but different relevant documents, both dropped).
      Available: 52 claims with a trap exclusion (118 rows; mice 19, in vitro 18, ...), 211 fallback claims,
      phrases/entities for 806 claims, identifiers for 566, paraphrase overlap for all 806.
      Build/check script DONE 2026-10-27: `venv\Scripts\python.exe api/targeted_build.py` checks
      `data/targeted/queries.tsv` and `rules.tsv` (templates with the header are there; formats and every check
      explained at the top of the script) against D1 to D10 and writes `data/targeted/qrels.tsv` only when
      nothing is wrong. Run it as often as you like while writing; `--final` also requires 25 per category.
      Tested on two fixtures outside the repo (5 valid queries pass; 9 planted mistakes are all reported).
      CHANGED 2026-10-27 (Deviation in hypotheses.md): Aris decided not to write the queries; Claude drafted
      them with `api/targeted_draft.py` (seeded claim order, fixed rule per category, fixed phrasings; inputs
      `skips.tsv`, 22 skips each with a reason, and `paraphrases.tsv`, 25 paraphrases written by Claude with no
      padding). `targeted_build.py --final` passes: 100 queries, 113 relevant documents (1 to 3 per query).
      Exclusions are all traps; phrases and identifiers are 24 no-trap + 1 trap each, because the exclusions
      used up almost all claims with several relevant documents (D3 only prefers traps there). Some phrase
      terms are lowercase as extracted ("preventable ades", "enhancer rnas").
      FROZEN 2026-10-27: Aris reviewed all 100 and accepted them without changes (recorded in the Deviation).
      From here on the set is not edited; any change goes under Deviations in hypotheses.md.
   c. DONE 2026-10-27: harness split "targeted" with results by group (HOW_IT_WORKS.md 6.10).
   Baselines on the targeted set DONE 2026-10-27 (`--split targeted`; runs from fdcda6a, clean; first system
   runs on the frozen set). Descriptive only: no hypothesis compares baselines with each other.
   | group | B1 nDCG@10 | B2 | B3 | B1 R@100 | B2 | B3 |
   |---|---|---|---|---|---|---|
   | constraint (75) | 0.731 | 0.705 | 0.749 | 0.980 | 0.960 | 0.987 |
   | exclusion | 0.363 | 0.352 | 0.518 | 0.940 | 0.880 | 1.000 |
   | phrase | 0.895 | 0.817 | 0.763 | 1.000 | 1.000 | 0.960 |
   | identifier | 0.935 | 0.945 | 0.965 | 1.000 | 1.000 | 1.000 |
   | paraphrase | 0.328 | 0.286 | 0.659 | 0.500 | 0.480 | 0.980 |
   - Exclusions are hard for every baseline, as designed: a trap is ranked above the document that stays
     relevant in 15/25 (B1), 14/25 (B2), 11/25 (B3) queries; that document is first in only 4 to 5 of 25. BM25
     is worst because the constraint sentence adds the excluded word ("mice") to its query. B3 minus B1 nDCG@10
     +0.155 (95% interval +0.048 to +0.260).
   - Phrases and identifiers are easy (0.76 to 0.97): they are almost all no-trap, so there is little room for
     a Boolean system to gain there (H3 already expects a small identifier gain). B3 is below B1 on phrases
     (-0.131, interval -0.267 to -0.015): exact wording favours BM25.
   - Paraphrases: BM25 loses half the relevant documents (R@100 0.50), dense keeps them (0.98); B3 minus B1
     R@100 +0.48 (+0.30 to +0.66). Expected by construction (lexical overlap at least halved). Caveat for H2:
     the paraphrases were written by an LLM (Claude), and dense retrieval is also a neural model.
   d. E0 DONE 2026-10-27 (`run_baselines.py --system E0`, runs from 50c17d3, clean). New program
      `cpp/build/boolean_cli` runs any Boolean query and writes the run, `<run>_sizes.tsv` and
      `<run>_failures.tsv` (ParseError with position; a failed query has no size line); E1/E2 will use it via
      `run_baselines.run_boolean`. Checked: B2's (+ words) queries through boolean_cli reproduce B2_train.tsv
      byte for byte. Results (descriptive, E0 is in no hypothesis):
      - train: 787 of 809 claims (97.3%) match nothing. The 22 that match each find exactly 1 document, and 21
        of the 22 are relevant (precision 0.95): an AND of every word finds the claim's own source abstract when
        the wording matches, and nothing otherwise. Empty counted as 0: nDCG@10 0.026 (B2 0.667).
      - targeted: all 100 match nothing (the constraint sentence's words are required too, "not", "mice", ...).
      Strict Boolean without an LLM: near-perfect precision, almost no recall. This is what E1 has to fix.
   e. Planner infrastructure BUILT 2026-10-27 (step 3, development model):
      - `api/planner.py`: OpenAI-style chat API (openai package), so the model is one entry in MODELS. Now:
        Qwen2.5-7B-Instruct (Q4_K_M) local through Ollama, pinned by its digest 845dbda0... and checked before
        any call; temperature 0, seed 1. Every call cached and logged in `results/llm_cache/<alias>.jsonl`
        (committed: it is the record of every model call). Answers cleaned (fences, extra lines, Greek letters
        as names). Rejected queries go back with the engine's ParseError, at most 2 retries; still rejected =
        failed query (no set size). Prompt: grammar, 5 rules (2 to 4 concepts, OR of variants, a negation inside
        the claim is not a NOT, NOT only when asked, a required term is ANDed) and 4 examples from train claims
        1125, 334, 326, 1209 (not targeted base claims; constraint sentences worded unlike the targeted set's
        phrasings; each example query finds its relevant document at rank 1).
      - `api/run_planner.py`: E1 on a split, through run_baselines.run_boolean (run, sizes, failures), plus
        `E1_<split>_plans.jsonl` (all attempts and raw answers) and the settings. `--sample N`: development on
        N random train claims that are neither targeted base claims nor examples, written to `results/dev/`
        (not committed; reproducible from the cache). The prompt is developed only this way.
      - First check, 10 dev claims: all parse, no retries, but 9 of 10 sets are empty and no relevant document
        is found: too many AND operands, the claim's wording in quotes, and "negatively related" turned into a
        NOT.
      - Prompt development 2026-10-27 (log: `results/prompt_dev_log.tsv`, committed). Two fixed development
        parts of 30 train claims each (shuffled once, seed 20261015, without targeted base claims and the first
        examples): tuned on part a only, final prompt checked once on part b. Results (empty sets, relevant doc
        found in the top 100, out of 30):
        v1 first prompt: 19 empty (+3 failed), 4 found | v2 concrete rules, banned general words, negation word
        list, example 566, retry hints: 19, 6 (failures 3 -> 0) | v3 three-line answer (Concepts, Exclude,
        Query): 20, 5, but false NOTs 6 -> 1 | v4 = v3 + hyphen/space variants: 20, 6 | v4 on part b: 18, 6
        (3 false NOTs). Plateau from v2 on (differences of 1 claim are noise): stopped, to avoid fitting the
        prompt to part a. Current prompt: v4 (fingerprint e3eff158...).
      - What is left is mostly vocabulary, not the prompt: e.g. the relevant document says "macrolides" where
        the claim says "erythromycin"; abstracts write "proinflammatory" more often than "pro-inflammatory".
        A strict AND cannot recover from that by itself: this is what relaxation (E2) is for.
      - E2 BUILT 2026-10-27 (rule fixed first, in hypotheses.md, commit 913e246: relax only when E1's valid
        query finds nothing; the model rewrites in the same conversation; at most 2 rounds; every word E1
        excluded must stay excluded, checked mechanically). `run_planner.py --system E2` runs E1 (from the cache)
        and then `planner.relax`. Development, same prompt v4:
        | | E1 empty | E2 empty | E1 relevant in top 10 | E2 relevant in top 10 |
        |---|---|---|---|---|
        | part a | 20 / 30 | 10 / 30 | 6 | 12 |
        | part b | 18 / 30 | 13 / 30 | 6 | 10 |
        Relaxation drops a part, adds variants, or shortens phrases; sometimes it turns the AND into an OR (one
        set of 783 documents): the precision cost H4 expects. Side effect of the exclusion check: it cannot tell
        a requested exclusion from a false one, so E1's false NOTs (from "no relation" claims) are kept; on
        part b 3 relaxed queries were rejected for that. It protects the targeted set's real exclusions.
      - NEXT: decide whether a 7B model is enough: compare Sol and Astra on parts a and b (E1 and E2, about $1
        with Astra) when the API key exists; then fix the model and prompt before any confirmatory run.
   f. Reranker (step 4), 2026-10-28: for now an open cross-encoder (free, runs locally), because there is no
      budget for an API yet. Which reranker the confirmatory runs use is NOT decided: candidates are Jev
      (TypeSafe AI, a "System One" classification model used as a reranker; about $0.04 per million input
      tokens, numbered versions such as 1.14, no free tier) and/or the cross-encoder. Aris is interested in
      Jev because it is unusual. The choice is written into hypotheses.md before any reranked run on the
      targeted set or test. The reranking step is built so that the reranker is one setting.
      Built 2026-10-28: api/rerank.py (cross-encoder/ms-marco-MiniLM-L6-v2, revision 233902d2, pinned). It
      refuses targeted and test until a RERANKERS entry is marked as fixed in hypotheses.md. Uses the GPU when
      there is one; long runs go to Google Colab (colab/rerank.ipynb), since the laptop's CPU does ~19 pairs/s.
      First runs (development, train, 2026-10-28; Colab T4 from 5cf45f6, about 25 minutes). nDCG@10, 809 claims:
        B1 0.699 -> rr100 0.711, rr30 0.708 | B2 0.667 -> 0.702, 0.703 | B3 0.769 -> 0.716, 0.727.
      Paired, rr100 minus the run it reranked: B1 +0.012 [-0.005, +0.028]; B2 +0.035 [+0.018, +0.051];
      B3 -0.054 [-0.072, -0.036]. B1-rr100 minus B3-rr100: -0.005. The same without the 100 targeted base
      claims (709 claims): B1 +0.009, B2 +0.036, B3 -0.059. GPU and CPU scores differ by at most 0.000007.
      Reading: this cross-encoder is weaker than B3's dense model on SciFact (as in BEIR, where it moves BM25
      from 0.665 to 0.688). It makes every pool score about the same, and it makes B3 worse. With it, H3 and H7
      would mostly measure the reranker's ceiling, not what each system's pool contains. A stronger reranker
      (Jev, or a bigger open one) should be tried on train before the choice is fixed.
      Fingerprints (2026-10-28): file_sha256 now ignores line endings (Colab has LF, this laptop CRLF), so both
      record the same value. data/scifact.tsv is 8003e6c7...; run files written before this change recorded the
      Windows copy, 7865b98e..., which is the same text. The B3 cache and the Lucene stamp were relabelled.
      Pass rule for a reranker (written 2026-10-28, before any bge-m3 score exists; a development decision, not
      a confirmatory test). Data: the first 200 claims of the development sample (api/dev_sample.py: train,
      never a base claim of the targeted set), rerank.py --sample 200. A reranker passes if, at N = 100:
        (1) the mean nDCG@10 of B3 reranked minus B3 alone is >= 0 (it does not make the best first stage worse),
        (2) the mean nDCG@10 of B1 reranked minus B1 alone is > 0 (it improves BM25).
      Reported with both: 95% paired intervals, and N = 30. The same rule and the same 200 claims are used for
      every reranker tried later (Jev included). ce-minilm is scored on the same 200 claims from its train runs.
      First candidate: BAAI/bge-reranker-v2-m3 (bge-m3 in RERANKERS, 16-bit on the GPU), on Colab.
6. Ablations, error analysis (about 100 failures), auditability study.
7. Stretch, cut from the bottom if time runs short: SSR ranking, TREC-COVID and latency, multi-call agent,
   end-to-end faithfulness.
8. Thesis text. Deadline June 2027; about 10 hours a week.
