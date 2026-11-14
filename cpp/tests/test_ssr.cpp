// SSR tests: shortest substrings per operator and the score sum of 1 / (42 + q - p), computed here by hand.
// Each test prints PASS or FAIL; the program exits with 1 if any fail.
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "inverted_index.hpp"
#include "ssr_scorer.hpp"

static int failures = 0;

static void report(const std::string& name, bool ok, const std::string& detail = "") {
    if (!ok) failures++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << (ok || detail.empty() ? "" : "   " + detail) << "\n";
}

static bool close_to(double a, double b) { return std::fabs(a - b) < 1e-12; }

static std::string show(const std::vector<Interval>& v) {
    std::string s;
    for (const Interval& iv : v) s += "[" + std::to_string(iv.first) + "," + std::to_string(iv.second) + "]";
    return s.empty() ? "(none)" : s;
}

static double score_of(const InvertedIndex& ix, const std::string& q, int doc_id) {
    SSRScorer ssr(ix);
    for (const auto& r : ssr.score(ix.search(q), q)) if (r.doc_id == doc_id) return r.score;
    return -1.0;
}

int main() {
    // Positions:             0 1 2 3 4 5 6
    InvertedIndex ix;
    ix.add_document(1, "a x b y y y a");
    ix.add_document(2, "a b");
    ix.add_document(3, "a x x x x b");
    ix.add_document(4, "nitric oxide and nitric acid");
    ix.add_document(5, "a c");
    ix.add_document(6, "a b c");
    ix.add_document(7, "b a");
    SSRScorer ssr(ix);

    // --- 1. A word: every position, length 0 ---
    report("word: every position", show(ssr.shortest_substrings("a", 1)) == "[0,0][6,6]",
           show(ssr.shortest_substrings("a", 1)));
    report("word: score 2/42", close_to(score_of(ix, "a", 1), 2.0 / 42));

    // --- 2. AND: shortest stretches holding both words; neither contains the other ---
    report("AND: [0,2] and [2,6]", show(ssr.shortest_substrings("(^ a b)", 1)) == "[0,2][2,6]",
           show(ssr.shortest_substrings("(^ a b)", 1)));
    report("AND: score 1/44 + 1/46", close_to(score_of(ix, "(^ a b)", 1), 1.0 / 44 + 1.0 / 46));
    report("AND: order of the words does not matter", show(ssr.shortest_substrings("(^ a b)", 7)) == "[0,1]",
           show(ssr.shortest_substrings("(^ a b)", 7)));

    // --- 3. More matches and closer matches rank higher ---
    {
        auto r = ssr.score(ix.search("(^ a b)"), "(^ a b)");
        // doc 1: two matches, 1/44 + 1/46 = 0.0445; docs 2, 6, 7: one match of length 1, 1/43 = 0.0233 (tie, by doc
        // id); doc 3: one match of length 5, 1/47 = 0.0213.
        bool ok = r.size() == 5 && r[0].doc_id == 1 && r[1].doc_id == 2 && r[2].doc_id == 6
                  && r[3].doc_id == 7 && r[4].doc_id == 3;
        std::string order;
        for (auto& x : r) order += std::to_string(x.doc_id) + " ";
        report("AND: ranking by number and closeness of matches, ties by doc id (1 2 6 7 3)", ok, order);
    }

    // --- 4. Phrase: each exact occurrence ---
    report("phrase: [0,1] only", show(ssr.shortest_substrings("\"nitric oxide\"", 4)) == "[0,1]",
           show(ssr.shortest_substrings("\"nitric oxide\"", 4)));
    report("phrase: score 1/43", close_to(score_of(ix, "\"nitric oxide\"", 4), 1.0 / 43));

    // --- 5. OR: a stretch that contains another one is dropped ---
    report("OR: (+ a \"a b\") keeps [0,0] only", show(ssr.shortest_substrings("(+ a \"a b\")", 2)) == "[0,0]",
           show(ssr.shortest_substrings("(+ a \"a b\")", 2)));
    report("OR: (+ a b) keeps both words", show(ssr.shortest_substrings("(+ a b)", 2)) == "[0,0][1,1]",
           show(ssr.shortest_substrings("(+ a b)", 2)));

    // --- 6. NOT works per document; the stretches come from the kept part ---
    report("NOT: doc with the excluded word is not scored", score_of(ix, "(- a b)", 2) < 0);
    report("NOT: score of the kept part, 1/42", close_to(score_of(ix, "(- a b)", 5), 1.0 / 42));

    // --- 7. Nested: (^ (+ a c) b) in "a b c": [0,1] (a,b) and [1,2] (b,c) ---
    report("nested AND of OR", show(ssr.shortest_substrings("(^ (+ a c) b)", 6)) == "[0,1][1,2]",
           show(ssr.shortest_substrings("(^ (+ a c) b)", 6)));

    // --- 8. Three-way AND: one stretch covering a, b and c ---
    report("AND of three words", show(ssr.shortest_substrings("(^ a b c)", 6)) == "[0,2]",
           show(ssr.shortest_substrings("(^ a b c)", 6)));

    // --- 9. minimal() on its own: duplicates and nested intervals ---
    report("minimal: drops duplicates and containers",
           show(SSRScorer::minimal({{0, 5}, {1, 2}, {1, 2}, {3, 4}, {0, 9}})) == "[1,2][3,4]",
           show(SSRScorer::minimal({{0, 5}, {1, 2}, {1, 2}, {3, 4}, {0, 9}})));

    std::cout << (failures ? "SOME TESTS FAILED\n" : "ALL TESTS PASSED\n");
    return failures ? 1 : 0;
}
