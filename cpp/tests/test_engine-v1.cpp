// These tests use assert, which a Release build (NDEBUG) turns off. Keep it on.
#undef NDEBUG
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <string>
#include <stdexcept>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

bool is_close(float a, float b, float tolerance = 1e-4) {
    return std::abs(a - b) < tolerance;
}

void run_tests() {
    InvertedIndex index;
    
    // 1. Corpus de 3 documente
    index.add_document(1, "stem cells are critical for immune response");
    index.add_document(2, "stem cells mutation causes disease");
    index.add_document(3, "immune response to mutation in cells");

    // 2. Initializare Scorer (k1 = 1.2, b = 0.75, conform tezei)
    BM25Scorer scorer(index, 1.2, 0.75);

    std::cout << "Running Phase 0 Engine Gates Tests...\n";

    // --- TEST 1: BM25 Calculat Manual ---
    auto results_term = scorer.score(index.search("mutation"), "mutation");
    float expected_score_doc2 = 0.50439f; 
    assert(results_term.size() == 2);
    
    bool found_doc2 = false;
    for(const auto& res : results_term) {
        if(res.doc_id == 2) {
            assert(is_close(res.score, expected_score_doc2)); 
            found_doc2 = true;
        }
    }
    assert(found_doc2);
    std::cout << "[PASS] BM25 Manual Calculation\n";

    // --- TEST 2: Fraze Exacte ---
    auto results_phrase = index.search("\"stem cells\"");
    assert(results_phrase.size() == 2);
    assert(results_phrase[0] == 1 && results_phrase[1] == 2);
    std::cout << "[PASS] Phrase Operator\n";

    // --- TEST 3: Operatorul NOT ---
    auto results_not = index.search("(- \"stem cells\" mutation)");
    assert(results_not.size() == 1);
    assert(results_not[0] == 1);
    std::cout << "[PASS] NOT Operator\n";

    // --- TEST 4: Erori de Parsare Explicite ---
    bool caught_error = false;
    try {
        index.search("(^ stem cells"); 
    } catch (const std::exception& e) { 
        caught_error = true;
    }
    assert(caught_error);
    std::cout << "[PASS] Parser Error Handling\n";

    // --- TEST 5: Interogare prea adanc imbricata (nu trebuie sa crape programul) ---
    std::string deep_query;
    for (int i = 0; i < 100000; i++) deep_query += "(+ ";
    deep_query += "stem";
    for (int i = 0; i < 100000; i++) deep_query += ")";
    bool caught_deep = false;
    try {
        index.search(deep_query);
    } catch (const ParseError& e) {
        caught_deep = true;
    }
    assert(caught_deep);
    std::cout << "[PASS] Deep Nesting Rejected\n";

    // --- TEST 6: Termenii din interogare sunt normalizati ca textul indexat ---
    InvertedIndex norm_index;
    norm_index.add_document(1, "Cat");
    norm_index.add_document(2, "T-cell response");
    norm_index.add_document(3, "cell and T");
    auto r_upper = norm_index.search("CAT");
    assert(r_upper.size() == 1 && r_upper[0] == 1);
    auto r_hyphen = norm_index.search("T-cell");   // = fraza "t cell", nu doar cuvintele separate
    assert(r_hyphen.size() == 1 && r_hyphen[0] == 2);
    auto r_inside = norm_index.search("(^ T-cell response)");
    assert(r_inside.size() == 1 && r_inside[0] == 2);
    std::cout << "[PASS] Query Term Normalization\n";

    std::cout << "ALL PHASE 0 TESTS PASSED!\n";
}

int main() {
    run_tests();
    return 0;
}