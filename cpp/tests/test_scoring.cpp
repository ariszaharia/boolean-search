// Ranking tests: ties, phrase scoring, NOT, malformed queries.
// Each test prints PASS or FAIL; the program exits with 1 if any fail.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

static int failures = 0;

static void report(const std::string& name, bool ok, const std::string& detail = "") {
    if (!ok) failures++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << (ok || detail.empty() ? "" : "   " + detail) << "\n";
}

static bool close_to(double a, double b) { return std::fabs(a - b) < 1e-9; }

static std::vector<SearchResult> run(const InvertedIndex& ix, const std::string& q) {
    BM25Scorer scorer(ix, 1.2, 0.75);
    return scorer.score(ix.search(q), q);
}

static double score_of(const std::vector<SearchResult>& r, int doc_id) {
    for (const auto& x : r) if (x.doc_id == doc_id) return x.score;
    return -1.0;
}

// BM25 for one word (or phrase), written out here independently of the scorer.
static double bm25(double idf, int tf, int dl, double avgdl, double k1 = 1.2, double b = 0.75) {
    return idf * tf * (k1 + 1.0) / (tf + k1 * (1.0 - b + b * dl / avgdl));
}
static double idf_of(int N, int df) { return std::log(1.0 + (N - df + 0.5) / (df + 0.5)); }

int main() {
    // --- 1. Ties are broken by doc_id (ascending), every time ---
    {
        InvertedIndex ix;
        for (int id = 1; id <= 50; id++) ix.add_document(id, "same text here");
        auto r = run(ix, "text");
        bool ok = r.size() == 50;
        for (size_t i = 0; ok && i < r.size(); i++) ok = (r[i].doc_id == (int)i + 1);
        report("50 equal scores come out in doc_id order", ok);
    }

    // --- 2. Higher score still comes first; ties only matter among equals ---
    {
        InvertedIndex ix;
        ix.add_document(1, "cat dog bird fish");     // 4 words
        ix.add_document(2, "cat");                   // 1 word  -> shorter -> higher score
        ix.add_document(3, "cat dog bird fish");     // ties with doc 1
        auto r = run(ix, "cat");
        report("shorter document ranks first, ties by id",
               r.size() == 3 && r[0].doc_id == 2 && r[1].doc_id == 1 && r[2].doc_id == 3);
    }

    // --- 3. Phrase score = phrase frequency with the summed idf of its words ---
    {
        InvertedIndex ix;
        ix.add_document(1, "stem cells aa bb");          // phrase once
        ix.add_document(2, "stem cells stem cells");     // phrase twice
        ix.add_document(3, "zz yy xx ww");               // no match
        auto r = run(ix, "\"stem cells\"");
        double avgdl = 4.0;
        double idf = idf_of(3, 2) + idf_of(3, 2);        // stem: 2 docs, cells: 2 docs
        report("phrase score matches the formula (tf = 1)", close_to(score_of(r, 1), bm25(idf, 1, 4, avgdl)));
        report("phrase score matches the formula (tf = 2)", close_to(score_of(r, 2), bm25(idf, 2, 4, avgdl)));
        report("phrase twice ranks above phrase once", r.size() == 2 && r[0].doc_id == 2);
    }

    // --- 4. Words far apart do not earn phrase score ---
    {
        InvertedIndex ix;
        ix.add_document(1, "stem cells aa bb");
        ix.add_document(2, "stem aa cells bb");   // has both words, but not the phrase
        auto r = run(ix, "(+ \"stem cells\" aa)");
        auto only_aa = run(ix, "aa");
        report("doc without the phrase scores only from the other word",
               close_to(score_of(r, 2), score_of(only_aa, 2)));
        report("doc with the phrase ranks above it", r.size() == 2 && r[0].doc_id == 1);
    }

    // --- 5. Hyphenated term is scored like the phrase of its words ---
    {
        InvertedIndex ix;
        ix.add_document(1, "T-cell response");
        ix.add_document(2, "other text here");
        auto a = run(ix, "T-cell");
        auto b = run(ix, "\"t cell\"");
        report("T-cell scores the same as \"t cell\"",
               a.size() == 1 && b.size() == 1 && close_to(a[0].score, b[0].score) && a[0].score > 0);
    }

    // --- 6. NOT: excluded words add nothing to the score ---
    {
        InvertedIndex ix;
        ix.add_document(1, "cat mouse");
        ix.add_document(2, "cat dog");
        ix.add_document(3, "bird");
        auto with_not = run(ix, "(- cat dog)");
        auto plain = run(ix, "cat");
        report("NOT result is only doc 1", with_not.size() == 1 && with_not[0].doc_id == 1);
        report("NOT score equals the score of the first operand alone",
               close_to(score_of(with_not, 1), score_of(plain, 1)));
    }

    // --- 7. AND adds the scores of its parts ---
    {
        InvertedIndex ix;
        ix.add_document(1, "cat dog");
        ix.add_document(2, "cat");
        ix.add_document(3, "dog");
        auto both = run(ix, "(^ cat dog)");
        auto c = run(ix, "cat"), d = run(ix, "dog");
        report("AND score = score(cat) + score(dog)",
               both.size() == 1 && close_to(both[0].score, score_of(c, 1) + score_of(d, 1)));
    }

    // --- 8. Malformed query is an error, not silently scored ---
    {
        InvertedIndex ix;
        ix.add_document(1, "cat");
        BM25Scorer scorer(ix);
        bool threw = false;
        try { scorer.score({1}, "(^ cat"); } catch (const ParseError&) { threw = true; }
        report("scorer throws ParseError for a malformed query", threw);
    }

    std::cout << (failures == 0 ? "\nALL SCORING TESTS PASSED\n"
                                : "\n" + std::to_string(failures) + " SCORING TEST(S) FAILED\n");
    return failures == 0 ? 0 : 1;
}
