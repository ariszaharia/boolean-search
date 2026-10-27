// why-not tests. Each test prints PASS or FAIL; the program exits with 1 if any fail.
// The main check: for every document and every query, explain() says "holds" exactly when search() returns the
// document, so the explanation can never disagree with the engine.
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "inverted_index.hpp"

static int failures = 0;

static void report(const std::string& name, bool ok) {
    if (!ok) failures++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
}

static bool contains(const std::string& text, const std::string& piece) {
    return text.find(piece) != std::string::npos;
}

int main() {
    InvertedIndex ix;
    ix.add_document(1, "Vitamin D deficiency and tuberculosis in humans");
    ix.add_document(2, "vitamin D3 and TB risk in rats");
    ix.add_document(3, "T-cell response to tuberculosis, vitamin d levels");
    ix.add_document(4, "stem cells and T cell therapy in rats and mice");
    ix.add_document(5, "deficiency of vitamin a, not d");
    ix.add_document(6, "an unrelated document about weather");

    const std::vector<int> docs = {1, 2, 3, 4, 5, 6};
    const std::vector<std::string> queries = {
        "vitamin", "tb", "banana", "\"vitamin d\"", "\"d vitamin\"", "T-cell", "\"t cell\"",
        "(^ vitamin tuberculosis)", "(+ tb tuberculosis)", "(^ \"vitamin d\" (+ tuberculosis tb))",
        "(- vitamin rats)", "(- vitamin rats mice)", "(- (+ vitamin cells) rats)",
        "(^ (- vitamin rats) (+ deficiency tuberculosis))", "(- (^ vitamin d) \"vitamin d\")",
        "(+ (^ stem cells) (- tuberculosis humans))", "(^ vitamin banana)", "(+ banana \"weather report\")",
        "(- vitamin (+ rats humans) mice)", "(^ (+ a b) (- vitamin \"vitamin a\"))",
    };

    // --- 1. explain agrees with search, for every document and every query ---
    int checked = 0, disagreements = 0;
    for (const std::string& q : queries) {
        std::vector<int> found = ix.search(q);
        for (int d : docs) {
            bool in_set = std::find(found.begin(), found.end(), d) != found.end();
            if (ix.explain(q, d).holds != in_set) {
                disagreements++;
                std::cout << "  disagreement: doc " << d << ", query " << q << "\n";
            }
            checked++;
        }
    }
    report("explain agrees with search on " + std::to_string(checked) + " (document, query) pairs",
           disagreements == 0);

    // --- 2. the evidence names the right cause ---
    {
        Explanation e = ix.explain("\"vitamin d\"", 2);   // doc 2 has "vitamin d3", not "vitamin d"
        report("phrase missing a word: says which word is missing",
               !e.holds && contains(e.reason, "not in the document") && contains(e.reason, "d 0x"));
    }
    {
        Explanation e = ix.explain("\"d vitamin\"", 1);   // both words occur, in the other order
        report("phrase with every word present but not adjacent",
               !e.holds && contains(e.reason, "never next to each other"));
    }
    {
        Explanation e = ix.explain("banana", 1);
        report("word not in the corpus at all", !e.holds && contains(e.reason, "not in any document of the corpus"));
    }
    {
        Explanation e = ix.explain("tb", 1);
        report("word in the corpus but not in this document",
               !e.holds && contains(e.reason, "it is in 1 document of the corpus"));
    }
    {
        Explanation e = ix.explain("T-cell", 3);
        report("a hyphenated term is explained as the phrase it is searched as",
               e.holds && contains(e.reason, "searched as the phrase \"t cell\""));
    }
    {
        Explanation e = ix.explain("(- vitamin rats mice)", 4);   // doc 4 has no "vitamin"
        report("NOT: the kept part does not hold", !e.holds && contains(e.reason, "the first part (the one kept)"));
        e = ix.explain("(- vitamin rats mice)", 2);               // doc 2 has vitamin and rats
        report("NOT: names the exclusion that removed the document",
               !e.holds && contains(e.reason, "excluded by: rats") && !contains(e.reason, "mice"));
        report("NOT: children are labelled keep / without",
               e.children.size() == 3 && e.children[0].part == "keep: vitamin" &&
               e.children[1].part == "without: rats" && e.children[2].part == "without: mice");
    }
    {
        Explanation e = ix.explain("(^ \"vitamin d\" (+ tuberculosis tb))", 2);
        report("AND: every part is checked, none skipped",
               !e.holds && e.children.size() == 2 && !e.children[0].holds && e.children[1].holds &&
               contains(e.reason, "1 of 2 hold"));
    }

    // --- 3. rendering, errors ---
    {
        std::string text = InvertedIndex::render(ix.explain("(^ vitamin tuberculosis)", 2));
        report("render: one line per part, indented",
               text.rfind("[no]  (^ ... 2 parts)", 0) == 0 && contains(text, "\n  [yes] vitamin") &&
               contains(text, "\n  [no]  tuberculosis"));
    }
    {
        bool threw = false;
        try { ix.explain("vitamin", 99); } catch (const std::out_of_range&) { threw = true; }
        report("unknown document: std::out_of_range", threw);
        threw = false;
        try { ix.explain("(^ vitamin", 1); } catch (const ParseError&) { threw = true; }
        report("bad query: ParseError", threw);
    }

    std::cout << (failures == 0 ? "\nALL WHY-NOT TESTS PASSED\n"
                                : "\n" + std::to_string(failures) + " WHY-NOT TEST(S) FAILED\n");
    return failures == 0 ? 0 : 1;
}
