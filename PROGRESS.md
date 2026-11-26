# Thesis progress summary (as of 2026-11-14)

Plan: write the thesis text only after all the code and experiments are finished.

## Next steps (written 2026-11-14)
1. Fix the planner model (GPT-6 Luna, see e. below) and write it into hypotheses.md as a clarification before any
   confirmatory E1/E2 run.
2. DONE 2026-11-24: MedCPT rule (1) again with the new B3 (TAS-B), same 200 claims, Colab T4 from d43e7cc:
   B3 reranked minus B3 +0.168 [+0.124, +0.214] at N = 100 (0.633 -> 0.801), +0.146 [+0.106, +0.188] at N = 30.
   Passes. (Colab zip unpacked into a rerank_results/ subfolder again; files moved to results/dev/ and the
   Colab cache appended to results/work/rerank_cache/medcpt_71caf65d4927_cuda.tsv, not overwritten.)
3. Rerank B1, B2, B3 on the targeted set with medcpt (N = 100 and 30), on Colab: colab/rerank.ipynb with
   RERANKER = 'medcpt', SYSTEMS = ['B1', 'B2', 'B3'], SAMPLE = None, SPLIT = 'targeted' (about 15 minutes).
   Then score by group (exclusion, phrase, identifier, paraphrase), with B3 unreranked as the reference line.
4. E4 (the search agent, api/agent.py) development on dev part a with Luna (about $0.05).
5. DONE 2026-11-26: the final analysis, `python -m harness.final_analysis` -> results/analysis/confirmatory.md
   and .json (all confirmatory tests, Holm over H1, H2, H3 x 2, H4 x 2, H9; H7 and H8 by their own rules;
   exploratory kept apart). Holm changes no verdict: H1, H2, H3 falsified; H4, H8, H9 hold; H7 not falsified,
   equivalence not shown. H2's reversal (Boolean recall higher on paraphrases) stays significant after Holm
   (p 0.040). Left: H5 sampling and annotation-sheet tooling; rerun the analysis when H5 (and H6) exist.
6. Then: the confirmatory runs (E1/E2/E4 on targeted and test, everything reranked with medcpt, test baselines,
   B1-lit at the end), H5 annotation (Aris + a second annotator for 30 to 50 items: find that person early),
   error analysis, thesis text. Optional: TREC-COVID as a second dataset for E4 (stretch, already in the Setting).
7. Settled 2026-11-14 (Aris): AI-written code is allowed.
Budget: $5 of OpenAI credit bought 2026-11-14; about $0.28 logged by 2026-11-14 (results/llm_cache/*.jsonl,
cost_usd). planner.py stops development once all logged calls reach $1.80 and everything at $4.90.

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
- B3: TAS-B (`sentence-transformers/msmarco-distilbert-base-tas-b`) at a pinned revision, no query prefix, CLS
  vector, dot product, exact search (`--system B3`). Fixed in the Clarifications of hypotheses.md. Needs
  `transformers` and `torch` in the venv. Document embeddings are cached in
  `results/work/b3_docs_<revision>_<corpus hash>.npz`. Chosen 2026-11-14 and run on train and targeted the same
  day (CPU); 455 of 5,183 documents are longer than 512 tokens and are cut. First run from uncommitted code,
  rerun from the commit that adds it (embeddings reused from the cache).
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
  | B3 (TAS-B, 2026-11-14) | 0.644 | 0.854 | 0.905 | - |
  B3 minus B1 nDCG@10 -0.055 (-0.076 to -0.033), R@100 -0.029 (-0.049 to -0.009): with TAS-B, BM25 is ahead
  on SciFact, as in BEIR.
  After D7 (rerun 2026-10-27, corpus_sha256 7865b98e...): B1 minus B2 nDCG@10 +0.032 (95% interval +0.020 to
  +0.044). D7 moved 9 queries for B1 and 12 for B2, both ways. It helps BM25 slightly: on claim 846 ("IL-1beta")
  B1 went 0.631 -> 1.0.
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
4. Reranker on the same top-N pool for every system: MedCPT (fixed in hypotheses.md, 2026-10-28; see f. below).
   Jev, if added later, is exploratory only.
5. Targeted constraint query set (about 100 queries in 4 categories, written before any system runs) and the
   engine's `why-not(doc, query)` function. why-not BUILT 2026-10-28: InvertedIndex::explain / render and
   cpp/build/why_not_cli (verdicts not_in_set / below_top_k / in_top_k, a [yes]/[no] tree for every part of the
   query). Tests: tests/test_why_not.cpp (explain = search on 120 pairs, plus the evidence texts); on 1,416 real
   pairs from the dev E1/E2 queries every verdict, set size and rank matched boolean_cli. Left for H5: the
   tooling that samples 50 failures per system and makes the annotation sheet (query + why-not for Boolean,
   ranked list for dense).
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
   | constraint (75) | 0.731 | 0.705 | 0.645 | 0.980 | 0.960 | 0.913 |
   | exclusion | 0.363 | 0.352 | 0.414 | 0.940 | 0.880 | 0.820 |
   | phrase | 0.895 | 0.817 | 0.703 | 1.000 | 1.000 | 0.960 |
   | identifier | 0.935 | 0.945 | 0.817 | 1.000 | 1.000 | 0.960 |
   | paraphrase | 0.328 | 0.286 | 0.464 | 0.500 | 0.480 | 0.760 |
   B3 = TAS-B (run 2026-11-14). B3 minus B1 nDCG@10: constraint -0.086 (-0.156 to -0.019), exclusion +0.051
   (-0.072 to +0.176), phrase -0.192 (-0.312 to -0.089); dense still wins the paraphrases.
   - Exclusions are hard for every baseline, as designed: a trap is ranked above the document that stays
     relevant in 15/25 (B1), 14/25 (B2) queries; that document is first in only 4 to 5 of 25. BM25 suffers because
     the constraint sentence adds the excluded word ("mice") to its query.
   - Phrases and identifiers are easy (0.82 to 0.95): they are almost all no-trap, so there is little room for
     a Boolean system to gain there (H3 already expects a small identifier gain).
   - Paraphrases: BM25 loses half the relevant documents (R@100 0.50). Expected by construction (lexical overlap
     at least halved). Caveat for H2:
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
      - Planner models compared 2026-11-11 to 2026-11-14 (same prompt v4, parts a + b = 60 dev claims, no
        reranking; harness on the 60 claims; part b has now been seen with four models). Models: Qwen2.5-7B
        (local), Gemini 3.5 Flash-Lite (free tier, reasoning minimal, no seed), GPT-6 Sol and GPT-6 Luna (OpenAI,
        reasoning none, temperature 0; no dated snapshot exists, so each is pinned by id + created time).
        | | nDCG@10 | R@100 | P@min(|S|,100) | empty | cost per call |
        |---|---|---|---|---|---|
        | E1 Qwen / Flash-Lite / Sol / Luna | 0.175 / 0.328 / 0.365 / 0.268 | 0.187 / 0.347 / 0.392 / 0.289 | 0.151 / 0.281 / 0.298 / 0.223 | 38 / 30 / 28 / 34 | 0 / 0 / $0.0026 / $0.00013 |
        | E2 Qwen / Flash-Lite / Sol / Luna | 0.310 / 0.522 / 0.487 / 0.604 | 0.351 / 0.587 / 0.526 / 0.693 | 0.273 / 0.384 / 0.416 / 0.366 | 23 / 7 / 10 / 4 | |
        Same 60 claims: B1 0.696 / 0.958, B3 (TAS-B) 0.654 / 0.961 (nDCG@10 / R@100).
        Paired: E2 Luna minus Sol nDCG@10 +0.117 (+0.019 to +0.217); E1 Luna minus Sol -0.097 (-0.175 to -0.031).
        Reading: 7B is not enough; beyond a small hosted model the planner matters little for E1, and Luna's
        bolder relaxation (5 sets over 100 documents, largest 1,525) makes it the best E2. Miss analysis of
        Flash-Lite's 23 E2 misses with why_not (all filtering failures, none ranking): 11 vocabulary (the relevant
        abstract uses other words), 3 claim terms absent from the corpus (refuted claims), 3 borderline, 6 model
        errors; Sol fixed 1 of the 6. Relaxation sometimes answers a prompt example instead of the request
        (Flash-Lite 2 of 45 rounds, Qwen 1 of 64). Proposed planner: Luna (step 1 of Next steps).
      - Set aside 2026-11-14: a planner v5 (BM25 peek, corpus counts in relaxation, relax sets < 2, any-2-of-3).
        Written, never run, removed from the code on 2026-11-27 (it is in git history, commit 2ea624c); E4 does what
        it aimed at.
   h. FIRST CONFIRMATORY RUNS, 2026-11-24: E1 and E2 with GPT-6 Luna on the targeted set (runs from 8b4c1fd,
      clean; 164 calls, $0.022). E1: 46 of 100 sets empty; E2: 8. Harness, primary analysis = queries with a
      non-empty E1 set (Setting); second analysis = empty counts as 0. Holm not applied yet (analysis script).
      - H1 (precision at k = min(|S|, 100), constraint queries): E1 minus B1 +0.069 [-0.015, +0.167], p 0.14,
        n = 41 of 75 (34 empty). The interval includes 0: H1 is FALSIFIED as stated (prediction +0.15 to +0.30 not
        met). Also reported: E1 minus B3 +0.118 [+0.021, +0.232], p 0.031. Second analysis: E1 minus B1 +0.033
        [-0.012, +0.087]; E1 minus B3 +0.060 [+0.007, +0.126].
      - H2 (recall@100, paraphrase queries): E1 minus B1 +0.346 [+0.115, +0.615], p 0.008, n = 13 of 25 (12
        empty). Positive, so H2 is FALSIFIED: where E1 returns a set, it finds MORE than BM25 (the LLM writes the
        synonyms the paraphrase replaced). Second analysis: -0.100 [-0.360, +0.160]. E1 minus B3: 0.000 (primary),
        -0.360 [-0.560, -0.160] (second). Caveat: the paraphrases were written by an LLM (Claude), so another LLM may
        find the original words more easily than a person would.
      - E2 preview, not reranked (H3 is tested after MedCPT; not a test): nDCG@10 E2 / B1 / B3 by group: exclusion
        0.358 / 0.363 / 0.414, phrase 0.610 / 0.895 / 0.703, identifier 0.745 / 0.935 / 0.817, paraphrase
        0.676 / 0.328 / 0.464 (E2 minus B1 +0.348 [+0.140, +0.552]).
      - Exclusions diagnosed (E2): every one of the 25 queries has the (- ...) and the trap is out of the set in 24;
        the relevant abstract is missing from the set in 12 of 25. The NOT works; the loss is recall of the claim
        part (vocabulary), the problem E4 addresses. One relaxed query lost its claim part entirely (t024:
        (- (+ tuberculosis tb) women ...)).
   i. E4 (agent v2, Luna) on the targeted set, 2026-11-24 (from f1377c8, clean; 514 calls, $0.105, 5.1 calls per
      request; reported, not a test: H3 tests E2 and H8 is on test). nDCG@10 / R@100, not reranked:
      | group | B1 | B3 | E2 | E4 (10 calls) |
      |---|---|---|---|---|
      | constraint (75) | 0.731 / 0.980 | 0.645 / 0.913 | 0.571 / 0.649 | 0.678 / 0.789 |
      | exclusion | 0.363 / 0.940 | 0.414 / 0.820 | 0.358 / 0.507 | 0.517 / 0.727 |
      | phrase | 0.895 / 1.000 | 0.703 / 0.960 | 0.610 / 0.680 | 0.780 / 0.800 |
      | identifier | 0.935 / 1.000 | 0.817 / 0.960 | 0.745 / 0.760 | 0.737 / 0.840 |
      | paraphrase | 0.328 / 0.500 | 0.464 / 0.760 | 0.676 / 0.780 | 0.785 / 0.800 |
      E4 minus B1 nDCG@10: exclusion +0.155 [-0.005, +0.307] p 0.052, paraphrase +0.457 [+0.243, +0.659],
      phrase -0.115 [-0.249, +0.000], identifier -0.198 [-0.362, -0.058]. E4 minus B3: exclusion +0.104
      [-0.062, +0.260], paraphrase +0.322 [+0.115, +0.514]. E4 minus E2: constraint +0.107 [+0.008, +0.207].
      Exclusions in detail: the relevant abstract is found in 20 of 25 (E2: 13) and ranked first in 9; but the
      agent used a NOT in only 2 of 25 queries and the trap is in its output in 17 of 25 (top 3 in 12). Its gain
      on exclusions comes from recall, not from obeying the exclusion; E2 obeys it (trap out in 24 of 25) but
      finds less.
      - EXPLORATORY, decided after seeing the E4 result above (not in any hypothesis, not in Holm), 2026-11-24:
        E4+NOT = E4's ranking minus every abstract that matches an excluded operand of E2's final query (the
        model's own (- ...), run on the whole corpus; no new model call, no relevance judgments). api/
        combine_e4_not.py -> results/exploratory/E4-not-b10-gpt-6-luna_targeted.tsv (+ .json: per request what was
        excluded and how many abstracts were removed). E2 has a top-level NOT in all 25 exclusion requests and in no
        other request, so only the exclusions change; 56 abstracts removed in all.
        Exclusions: nDCG@10 0.570 (E4 0.517, B3 0.414, B1 0.363); R@100 0.727 (unchanged: the removed abstracts
        are not relevant); relevant abstract found in 20 of 25 and first in 11 (E4: 9); trap left in 1 of 25
        (E4: 17). E4+NOT minus B1 +0.208 [+0.042, +0.368] p 0.013; minus B3 +0.157 [-0.023, +0.327] p 0.078; minus
        E4 +0.053 [+0.011, +0.106]. Constraint group 0.696 (E4 0.678). Not reranked.
        Reading: the two halves combine as hoped: the agent finds the abstract, the single-shot query supplies an
        exclusion that holds. Being post hoc and on 25 requests, it is a lead for the thesis's discussion (and a
        candidate system for future work), not a confirmed result.
   j. H3 SCORED, 2026-11-24. MedCPT reranking of B1, B2, B3 and E2 on the targeted set (Colab T4, all from baec927,
      clean; zip again unpacked into a "rerank_results (1)" subfolder, files moved to results/runs/, cache appended).
      N: H3 does not name one; following the H1 clarification (one primary N = 100, "as in H2"), N = 100 is primary
      and N = 30 is reported. Empty sets as in the Setting (primary: requests where E2 is non-empty).
      nDCG@10 after reranking, N = 100 (B3 without reranking in brackets):
      | group | B1 | B2 | B3 | E2 |
      |---|---|---|---|---|
      | exclusion | 0.465 | 0.448 | 0.433 (0.414) | 0.416 |
      | identifier | 0.970 | 0.970 | 0.945 (0.817) | 0.760 |
      | phrase | 0.910 | 0.925 | 0.893 (0.703) | 0.642 |
      | paraphrase | 0.470 | 0.445 | 0.630 (0.464) | 0.725 |
      | constraint (75) | 0.782 | 0.781 | 0.757 (0.645) | 0.606 |
      - H3 (E2 minus B1, primary): exclusion -0.048 [-0.167, +0.065] p 0.42 (n = 25, no empty); identifier -0.103
        [-0.273, +0.038] p 0.21 (n = 22 of 25). No significant gain in either category already before Holm, so H3
        is FALSIFIED (prediction: exclusions about +0.15). Second analysis (empty = 0): identifier -0.210 [-0.396,
        -0.046]. E2 minus B3 (reported): exclusion -0.016, identifier -0.074, both not significant. N = 30: exclusion
        E2 minus B1 +0.005, identifier -0.120.
      - Why: reranking lifts BM25 on exclusions from 0.363 to 0.465 (MedCPT reads the whole request, including
        "Ignore studies in mice"), while it cannot bring back the relevant abstract E2's set does not contain (12 of
        25 missing). E2's NOT holds; its recall loses the comparison.
      - Paraphrase (reported): E2 minus B1 +0.255 [+0.065, +0.450] p 0.011; E2 is still the best system there.
      - Context, exploratory (not reranked, E4's ranking is the model's own grades): E4+NOT on exclusions 0.570, above
        every reranked baseline (B1 0.465, B3 0.433). Not a test; the fair confirmatory comparison would need E4 and
        E4+NOT reranked too.
      - Done 2026-11-26, exploratory: E4 (budgets 3, 5, 10) reranked with MedCPT (Colab T4 from 3f3bfbb, clean), and
        E4+NOT on the reranked E4 (api/combine_e4_not.py --rerank rr100-medcpt; filtering and reranking commute).
        All systems reranked, N = 100, nDCG@10:
        | group | B1 | B3 | E2 | E4 | E4+NOT |
        |---|---|---|---|---|---|
        | exclusion | 0.465 | 0.433 | 0.416 | 0.441 | 0.526 |
        | paraphrase | 0.470 | 0.630 | 0.725 | 0.750 | 0.750 |
        | phrase | 0.910 | 0.893 | 0.642 | 0.765 | 0.765 |
        | identifier | 0.970 | 0.945 | 0.760 | 0.825 | 0.825 |
        | constraint (75) | 0.782 | 0.757 | 0.606 | 0.677 | 0.705 |
        E4+NOT minus B1: exclusion +0.061 [-0.064, +0.176] p 0.32 (not significant); paraphrase +0.280 [+0.060,
        +0.500] p 0.013. E4 minus B1: paraphrase +0.280, constraint -0.105 [-0.186, -0.028].
        Reading: with the same reranker for everyone, E4+NOT is still the best system on exclusions, but its lead
        over reranked BM25 is small and not significant; the unreranked 0.570 came partly from the agent's own grades,
        which rank its results better than MedCPT does on exclusions (0.570 vs 0.526). On paraphrases the agent stays
        clearly best. Phrases and identifiers stay BM25's.
   k. TEST SPLIT, 2026-11-26 (all from e65199e, clean). Baselines, nDCG@10 / R@100: B1 0.684 / 0.928, B2 0.663 /
      0.883, B3 (TAS-B) 0.643 / 0.891, E0 0.018 / 0.018. B1-lit (the check against Pyserini's published BEIR
      BM25 for SciFact) 0.679: the setup reproduces it. E1 and E2 with Luna: 500 calls, $0.065; E1 empty on 158 of
      300, E2 on 15 (143 recovered by relaxation).
      - H4 (recall@100 of the pool, E2 minus E1, primary = requests where E2 is non-empty): test +0.334 [+0.280,
        +0.390] p 0.0001 (n = 285; 0.387 -> 0.721); targeted +0.288 [+0.201, +0.380] p 0.0001 (n = 92). Second
        analysis: test +0.317, targeted +0.265. H4 HOLDS (p far below any Holm threshold for about 11 tests); the
        gain is about three times the prediction (+0.08 to +0.15). Precision change, reported next to it: +0.108
        [+0.079, +0.139] on test, +0.092 on targeted, i.e. no precision cost as measured: in this comparison an E1
        query with an empty set has precision 0, so recovering it raises both recall and precision. The predicted
        cost (3 to 8 points) does not appear.
      - E4 (agent v2, Luna) on test: 1,555 calls, $0.301, 5.2 calls per request (stopped by itself in 261 of 300).
        nDCG@10 / R@100, not reranked: budget 3 0.637 / 0.683, budget 5 0.694 / 0.760, budget 10 0.725 / 0.808
        (E2 0.581 / 0.685). E4 (10 calls) minus B1 +0.042 [+0.004, +0.080] p 0.031; minus B3 +0.083 [+0.038,
        +0.128] p 0.0003; minus E2 +0.144 [+0.101, +0.187] p 0.0001. These are reported, not tests (H8 compares
        E2 with E4); the agent is the best system on the standard test claims, above BM25 and dense, before
        reranking.
      - H8 (E2's nDCG@10 divided by E4's, test, not reranked; primary budget 10): 0.913 at 3 calls, 0.837 at 5,
        0.802 at 10. Falsified if below 0.80: at 0.802 H8 HOLDS, by the narrowest margin (prediction about 0.90).
        Bootstrap 95% interval of the ratio at 10 calls (paired, 10,000 resamples, seed 20261015): [0.746, 0.857];
        46% of resamples fall below 0.80, so the verdict is a coin flip and is reported as such. nDCG@10 keeps
        rising with the budget (0.637 -> 0.694 -> 0.725), but more slowly.
      - H9 (E3 = E1's queries ranked by SSR, C = 42; api/run_e3.py, from 16ad991, clean; same sets as E1, checked).
        The split is not named in H9; test is used as primary, as for H7 and H8, the targeted set is reported.
        nDCG@10, E3 minus E1, primary (E1 non-empty): test +0.005 [-0.014, +0.024] p 0.60 (n = 142); targeted
        -0.003 [-0.023, +0.014] p 0.80 (n = 54). H9 HOLDS: within 0.02 either way and not significant (prediction:
        within 0.02, BM25 slightly ahead; SSR is in fact very slightly ahead on test). Caveat for the thesis: E1's
        sets are mostly tiny (median 1 to 2 documents), so the order inside them can hardly matter; H9 says little
        about SSR on large sets. SSR's real use in this project is inside E4, where the agent pages through results.
      - H7, 2026-11-26. MedCPT reranking of B1, B3, E2 and E4 (10 calls) on test (Colab T4 from 0c9cb44, clean; E2
        has 285 requests, 15 stayed empty; E4 297). N = 100 primary, as for H1 to H3. nDCG@10 (all 300, empty = 0):
        B1 0.787, B3 0.751, E2 0.635, E4 0.741 (N = 30: 0.774, 0.731, 0.629, 0.741).
        Equivalence margin M = 0.03. E2 minus B1 -0.122 [-0.164, -0.082] (primary, n = 285); E2 minus B3 -0.090
        [-0.133, -0.048]. Neither interval lies inside +-0.03, so equivalence is NOT shown; neither lies above +0.03,
        so H7 is NOT FALSIFIED (Boolean is not ahead). The prediction (E2 about -0.02 against B1, H7 not falsified,
        equivalence probably not shown) has the right verdict but the gap is six times larger. B1 minus B3 +0.036
        [+0.013, +0.059]: after reranking BM25 is also ahead of dense.
        Reported, not tested: E4 reranked minus B1 -0.039 [-0.072, -0.008] p 0.017; minus B3 -0.006 [-0.042, +0.030]
        p 0.75; minus E2 +0.107 [+0.069, +0.148]. Reranking lifts BM25 from 0.684 to 0.787 and the agent only from
        0.725 to 0.741 (the agent's list is at most 30 abstracts, already ordered by its own grades), so the agent's
        lead over BM25 before reranking (+0.042) becomes a small deficit after it; the agent ends level with dense.
      - Exploratory robustness check, 2026-11-26: the agent (v2, unchanged; prompt developed with Luna) with GPT-6 Sol
        on development part c (the same 30 claims as Luna's held-out check). nDCG@10 Sol / Luna: 3 calls 0.705 /
        0.659, 5 calls 0.771 / 0.801, 10 calls 0.800 / 0.792; Sol minus Luna at 10 calls +0.009 [-0.112, +0.133]
        p 0.90. Same behaviour (5.3 calls per claim; relevant abstract in the top 10 for 27 vs 26 of 30). Cost: Sol
        $0.630, Luna $0.034 (about 19x). Decision (Aris, rule agreed beforehand: switch only if Sol is clearly
        better): Luna stays the model of E1, E2 and E4. For the thesis: a model about 20x more expensive gave no
        measurable gain for the agent (30 claims, exploratory).
      - Knowledge-leakage probe, PLAN FIXED BEFORE RUNNING (2026-11-26; exploratory, not in Holm): does the planner
        already know SciFact? For every test claim and every targeted request, GPT-6 Luna (pinned, same settings,
        no search) is asked for the exact title of the paper that is the evidence for it, or "unknown"
        (api/leakage_probe.py). A request counts as KNOWN if both hold: (1) BM25 (B1's index and settings) with the
        given title as query ranks a relevant abstract first, and (2) the given title matches the start of that
        abstract (difflib ratio >= 0.7 between the lowercased title words and the abstract's first as many words,
        where SciFact puts the title). Everything else, "unknown" included, counts as UNKNOWN. Then nDCG@10 of B1,
        B3, E2 and E4 (not reranked and reranked) is reported on known and unknown requests separately, with the
        E4 minus B1 and E2 minus B1 differences in each. Reading fixed in advance: if the Boolean systems' advantages
        also hold on the unknown requests, leakage is not what drives them; if they appear only on known requests,
        it may be.
        RESULT (from 24c6d09; 400 calls, $0.007): test: a title was given for 102 of 300 claims, its BM25 top hit was
        relevant for 62, and only 6 (2%) are KNOWN; targeted: 25 titles, 16 relevant top hits, 2 KNOWN. On the 294
        unknown test claims nDCG@10 is B1 0.677, B3 0.635, E2 0.573, E4 0.720 (reranked 0.783, 0.746, 0.627, 0.735),
        the same pattern as on all 300. Targeted, unknown only: paraphrase (24) B1 0.300, B3 0.441, E2 0.662, E4
        0.776; exclusion (25, none known) unchanged. Reading: near-verbatim recall of the evidence paper is rare, and
        every Boolean advantage holds on the requests the model does not know, so this form of leakage does not
        drive the results. Limit, for the thesis: the probe detects knowing the paper by its title, not vaguer
        familiarity with the topic (62 of 300 guessed titles led BM25 to a relevant abstract, mostly by topic).
   g. SSR ranking (E3, H9) and the search agent (E4, H8), built 2026-11-14:
      - cpp/src/ssr_scorer.cpp: shortest substring ranking (Clarke & Smucker 2026, arXiv 2607.11362, section 3):
        per document the shortest stretches satisfying the query, score = sum of 1 / (42 + q - p). NOT works per
        document, as in the engine. boolean_cli takes a last argument bm25 (default, unchanged output) or ssr.
        cpp/tests/test_ssr.cpp: 15 checks computed by hand; all 6 test programs pass.
      - api/agent.py + api/run_agent.py: E4 in the style of the paper's Vole. Each step one JSON action (search,
        more, read, stop); every document shown is graded 0-3; final ranking by grade, then discovery order;
        results ranked by SSR. Adapted to SciFact: 3 results per view as 60-word snippets, at most 10 calls and 30
        grades per request; one run at budget 10 also gives the runs at budgets 3 and 5 (H8). Tested offline with
        a scripted model; a Flash-Lite dev run was stopped (not used).
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
        B1 0.699 -> rr100 0.711, rr30 0.708 | B2 0.667 -> 0.702, 0.703.
      Paired, rr100 minus the run it reranked: B1 +0.012 [-0.005, +0.028]; B2 +0.035 [+0.018, +0.051].
      The same without the 100 targeted base claims (709 claims): B1 +0.009, B2 +0.036. GPU and CPU scores
      differ by at most 0.000007.
      Reading: this cross-encoder is weak on SciFact (as in BEIR, where it moves BM25 from 0.665 to 0.688). It
      makes every pool score about the same. With it, H3 and H7
      would mostly measure the reranker's ceiling, not what each system's pool contains. A stronger reranker
      (Jev, or a bigger open one) should be tried on train before the choice is fixed.
      Fingerprints (2026-10-28): file_sha256 now ignores line endings (Colab has LF, this laptop CRLF), so both
      record the same value. data/scifact.tsv is 8003e6c7...; run files written before this change recorded the
      Windows copy, 7865b98e..., which is the same text. The Lucene stamp was relabelled.
      Pass rule for a reranker (written 2026-10-28, before any bge-m3 score exists; a development decision, not
      a confirmatory test). Data: the first 200 claims of the development sample (api/dev_sample.py: train,
      never a base claim of the targeted set), rerank.py --sample 200. A reranker passes if, at N = 100:
        (1) the mean nDCG@10 of B3 reranked minus B3 alone is >= 0 (it does not make the best first stage worse),
        (2) the mean nDCG@10 of B1 reranked minus B1 alone is > 0 (it improves BM25).
      Reported with both: 95% paired intervals, and N = 30. The same rule and the same 200 claims are used for
      every reranker tried later (Jev included). ce-minilm is scored on the same 200 claims from its train runs.
      First candidate: BAAI/bge-reranker-v2-m3 (bge-m3 in RERANKERS, 16-bit on the GPU), on Colab.
      Result (Colab T4 from c22c907, about 12 minutes; runs in results/dev/, not committed). nDCG@10 on the 200
      claims: B1 0.713 | ce-minilm: B1-rr100 0.725 | bge-m3: B1-rr100 0.757, B1-rr30 0.749. Paired, reranked
      minus not reranked:
        bge-m3     B1 rr100 +0.044 [+0.005, +0.084]; B1 rr30 +0.036 [-0.001, +0.074]
        ce-minilm  B1 rr100 +0.012 [-0.021, +0.045]
      bge-m3 passes rule (2); rule (1) is to be checked again with the current B3 (2026-11-14). 16-bit (GPU) and 32-bit (CPU) bge-m3 scores differ by at most 0.007 (60 shared pairs).
      Second candidate (added 2026-10-28, before any of its scores exist): ncbi/MedCPT-Cross-Encoder (medcpt in
      RERANKERS, 32-bit), trained on PubMed search logs, i.e. SciFact's domain. Same rule, same 200 claims.
      Result (Colab T4 from 86b7b4c, about 15 minutes). nDCG@10 on the 200 claims: B1-rr100 0.815, B1-rr30 0.795
      (B1 alone 0.713). Paired, reranked minus not reranked:
        medcpt     B1 rr100 +0.102 [+0.065, +0.141]; B1 rr30 +0.082 [+0.051, +0.115]
      medcpt passes rule (2) and beats bge-m3 at N = 100 on B1 (+0.058); rule (1) is to be checked again with the
      current B3 (2026-11-14). Weights: the pinned revision has only pytorch_model.bin; transformers also
      fetched Hugging Face's converted model.safetensors during the run. All 201 tensors are bit-identical in
      both files, so the pinned revision describes exactly what ran.
      To note in the thesis: MedCPT was trained on PubMed search logs, and SciFact's documents are PubMed
      abstracts. That is training on the domain (SciFact's claims and qrels are not PubMed logs), not on the
      test labels, but it should be stated. Three rerankers were compared on the same 200 claims; also stated.
      DECIDED 2026-10-28 (Aris): MedCPT is the reranker of H3 and H7, written into hypotheses.md as a
      clarification before any reranked run on the targeted set or test. B3 without reranking is reported next
      to every reranked comparison. Jev, if added later, is exploratory (same rule, same 200 claims, not in the
      Holm correction). rerank.py now lets medcpt rerank the targeted set and test.
6. Ablations, error analysis (about 100 failures), auditability study.
7. Stretch, cut from the bottom if time runs short: SSR ranking, TREC-COVID and latency, multi-call agent,
   end-to-end faithfulness.
8. Thesis text. Deadline June 2027; about 10 hours a week.
