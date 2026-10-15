// Engine test suite.
//
// Build (from the cpp/ folder):
//   g++ -std=c++17 -Wall -Wextra -Iinclude src/inverted_index.cpp src/tokenizer.cpp
//       src/bm25scorer.cpp tests/test_engine.cpp -o test_engine   (one command, one line)
// Run:
//   ./test_engine            (all tests)
//   ./test_engine phrase     (only tests whose name contains "phrase")
//
// Unlike assert(), a failed check here is reported and the run continues, so
// one run shows every failing test.
//
// Expected BM25 numbers and stems come from independent Python code (a separate
// BM25 implementation, and NLTK's Porter stemmer in its reference-C mode), not from
// the C++ under test. Scoring uses k1 = 1.2, b = 0.75:
//   idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))
//   score  = sum over query clauses of idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
//   phrase : tf = number of occurrences of the phrase, idf = sum of the words' idfs
//   NOT    : operands after the first add nothing to the score

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "bm25scorer.hpp"
#include "inverted_index.hpp"
#include "tokenizer.hpp"

// ---------------------------------------------------------------- mini framework

namespace {

int g_checks = 0;
int g_failed_checks = 0;
bool g_current_failed = false;

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, void (*fn)()) { registry().push_back({name, fn}); }
};

void fail(const char* file, int line, const std::string& msg) {
    g_current_failed = true;
    ++g_failed_checks;
    std::cerr << "    " << file << ":" << line << ": " << msg << "\n";
}

std::string show(const std::string& s) { return "\"" + s + "\""; }
std::string show(const char* s) { return show(std::string(s)); }
std::string show(const std::vector<int>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << v[i];
    o << "]";
    return o.str();
}
std::string show(const std::vector<std::string>& v) {
    std::ostringstream o;
    o << "[";
    for (size_t i = 0; i < v.size(); ++i) o << (i ? ", " : "") << "\"" << v[i] << "\"";
    o << "]";
    return o.str();
}
template <typename T>
std::string show(const T& x) {
    std::ostringstream o;
    o.precision(17);
    o << x;
    return o.str();
}

}  // namespace

#define TEST(name)                                  \
    static void name();                             \
    static Registrar registrar_##name(#name, name); \
    static void name()

#define CHECK(cond)                                           \
    do {                                                      \
        ++g_checks;                                           \
        if (!(cond)) fail(__FILE__, __LINE__, "CHECK(" #cond ")"); \
    } while (0)

#define CHECK_EQ(a, b)                                                                     \
    do {                                                                                   \
        ++g_checks;                                                                        \
        auto va_ = (a);                                                                    \
        auto vb_ = (b);                                                                    \
        if (!(va_ == vb_))                                                                 \
            fail(__FILE__, __LINE__,                                                       \
                 std::string(#a " == " #b "   got ") + show(va_) + " vs " + show(vb_));    \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                              \
    do {                                                                                   \
        ++g_checks;                                                                        \
        double va_ = (a);                                                                  \
        double vb_ = (b);                                                                  \
        if (!(std::fabs(va_ - vb_) <= (tol)))                                              \
            fail(__FILE__, __LINE__,                                                       \
                 std::string(#a " ~= " #b "   got ") + show(va_) + " vs " + show(vb_));    \
    } while (0)

// Expects idx.search(query) to throw ParseError. expected_pos < 0 means "do not check position".
#define CHECK_PARSE_ERROR(idx, query, expected_pos)                                           \
    do {                                                                                      \
        ++g_checks;                                                                           \
        try {                                                                                 \
            (idx).search(query);                                                              \
            fail(__FILE__, __LINE__, std::string("no ParseError for query ") + show(query));  \
        } catch (const ParseError& e_) {                                                      \
            if ((expected_pos) >= 0 && static_cast<long>(e_.position) != (long)(expected_pos)) \
                fail(__FILE__, __LINE__,                                                      \
                     std::string("wrong error position for ") + show(query) + ": got " +      \
                         show((long)e_.position) + ", expected " + show((long)(expected_pos)) + \
                         " (" + e_.what() + ")");                                             \
        } catch (const std::exception& e_) {                                                  \
            fail(__FILE__, __LINE__,                                                          \
                 std::string("wrong exception type for ") + show(query) + ": " + e_.what());  \
        }                                                                                     \
    } while (0)

#define CHECK_THROWS(stmt, ExceptionType)                                                   \
    do {                                                                                    \
        ++g_checks;                                                                         \
        try {                                                                               \
            stmt;                                                                           \
            fail(__FILE__, __LINE__, "no exception from: " #stmt);                          \
        } catch (const ExceptionType&) {                                                    \
        } catch (const std::exception& e_) {                                                \
            fail(__FILE__, __LINE__, std::string("wrong exception from: " #stmt ": ") + e_.what()); \
        }                                                                                   \
    } while (0)

#define CHECK_NO_THROW(stmt)                                                                \
    do {                                                                                    \
        ++g_checks;                                                                         \
        try {                                                                               \
            stmt;                                                                           \
        } catch (const std::exception& e_) {                                                \
            fail(__FILE__, __LINE__, std::string("unexpected exception from: " #stmt ": ") + e_.what()); \
        }                                                                                   \
    } while (0)

// ---------------------------------------------------------------- helpers

using Docs = std::vector<std::pair<int, std::string>>;

static InvertedIndex make_index(const Docs& docs) {
    InvertedIndex idx;
    for (const auto& d : docs) idx.add_document(d.first, d.second);
    return idx;
}

// Corpus A: the three-document corpus used for the hand-computed BM25 values.
static InvertedIndex corpus_a() {
    return make_index({{1, "stem cells are critical for immune response"},
                       {2, "stem cells mutation causes disease"},
                       {3, "immune response to mutation in cells"}});
}

// Corpus B: length normalization, tf saturation and exact ties.
static InvertedIndex corpus_b() {
    return make_index({{1, "apple banana"},
                       {2, "apple apple banana"},
                       {3, "apple banana cherry date"},
                       {4, "cherry date elderberry fig"}});
}

// Corpus C: negated phrase must not contribute to scores.
static InvertedIndex corpus_c() {
    return make_index({{1, "cancer stem cells grow"},
                       {2, "stem therapy for cancer cells"},
                       {3, "cancer cells grow fast"}});
}

// Corpus D: phrase frequency (2) differs from the word frequencies (stem 2, cells 3).
static InvertedIndex corpus_d() {
    return make_index({{1, "stem cells and cells stem cells"},
                       {2, "stem cells only"},
                       {3, "cells stem only"}});
}

// Corpus F: phrase matching edge cases.
static InvertedIndex corpus_f() {
    return make_index({{1, "stem cells divide"},
                       {2, "stem and cells divide"},
                       {3, "cells stem divide"},
                       {4, "the quick brown fox jumps"},
                       {5, "quick brown fox"},
                       {6, "quick brown dog fox"},
                       {7, "a a a"},
                       {8, "Stem Cells DIVIDE"}});
}

static std::vector<int> S(const InvertedIndex& idx, const std::string& q) { return idx.search(q); }

static std::vector<SearchResult> ranked(const InvertedIndex& idx, const BM25Scorer& sc, const std::string& q) {
    return sc.score(idx.search(q), q);
}

static std::vector<int> ids_of(const std::vector<SearchResult>& r) {
    std::vector<int> out;
    for (const auto& x : r) out.push_back(x.doc_id);
    return out;
}

static double score_of(const std::vector<SearchResult>& r, int doc_id) {
    for (const auto& x : r)
        if (x.doc_id == doc_id) return x.score;
    return -1.0;
}

static bool sorted_unique(const std::vector<int>& v) {
    for (size_t i = 1; i < v.size(); ++i)
        if (!(v[i - 1] < v[i])) return false;
    return true;
}

static std::string temp_file(const std::string& name) {
    return (std::filesystem::temp_directory_path() / name).string();
}

static const double TOL = 1e-9;

// ================================================================ tokenizer

TEST(tokenizer_lowercases) {
    Tokenizer t;
    CHECK_EQ(t.tokenize("Stem CELLS Divide"), (std::vector<std::string>{"stem", "cells", "divide"}));
}

TEST(tokenizer_splits_on_punctuation_and_hyphens) {
    Tokenizer t;
    CHECK_EQ(t.tokenize("T-cell, (NF-kB)!"), (std::vector<std::string>{"t", "cell", "nf", "kb"}));
}

TEST(tokenizer_keeps_digits) {
    Tokenizer t;
    CHECK_EQ(t.tokenize("IL-6 and 2024"), (std::vector<std::string>{"il", "6", "and", "2024"}));
}

TEST(tokenizer_empty_whitespace_and_punctuation_only) {
    Tokenizer t;
    CHECK(t.tokenize("").empty());
    CHECK(t.tokenize("   \t\n ").empty());
    CHECK(t.tokenize("!!! ... ---").empty());
}

TEST(tokenizer_collapses_any_whitespace) {
    Tokenizer t;
    CHECK_EQ(t.tokenize("a\tb\nc   d"), (std::vector<std::string>{"a", "b", "c", "d"}));
}

TEST(tokenizer_non_ascii_bytes_do_not_crash_and_leave_clean_tokens) {
    Tokenizer t;
    std::vector<std::string> toks;
    CHECK_NO_THROW(toks = t.tokenize("na\xC3\xAFve caf\xC3\xA9 \xCE\xB1-synuclein"));
    bool has_synuclein = false;
    bool all_clean = true;
    for (const auto& tok : toks) {
        if (tok == "synuclein") has_synuclein = true;
        for (unsigned char c : tok)
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) all_clean = false;
    }
    CHECK(has_synuclein);
    CHECK(all_clean);
}

// ================================================================ index basics

TEST(index_empty_state) {
    InvertedIndex idx;
    CHECK_EQ(idx.get_total_docs(), 0);
    CHECK_NEAR(idx.get_avg_doc_length(), 0.0, TOL);
    CHECK_EQ(idx.get_doc_length(1), 0);
    CHECK_EQ(idx.get_doc_frequency("anything"), 0);
    CHECK_EQ(idx.get_term_frequency("anything", 1), 0);
    CHECK(idx.search("anything").empty());
}

TEST(index_counts_and_lengths) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(idx.get_total_docs(), 3);
    CHECK_EQ(idx.get_doc_length(1), 7);
    CHECK_EQ(idx.get_doc_length(2), 5);
    CHECK_EQ(idx.get_doc_length(3), 6);
    CHECK_NEAR(idx.get_avg_doc_length(), 6.0, TOL);
}

TEST(index_doc_frequency) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(idx.get_doc_frequency("cells"), 3);
    CHECK_EQ(idx.get_doc_frequency("mutation"), 2);
    CHECK_EQ(idx.get_doc_frequency("disease"), 1);
    CHECK_EQ(idx.get_doc_frequency("absent"), 0);
}

TEST(index_term_frequency) {
    InvertedIndex idx = corpus_d();
    CHECK_EQ(idx.get_term_frequency("stem", 1), 2);
    CHECK_EQ(idx.get_term_frequency("cells", 1), 3);
    CHECK_EQ(idx.get_term_frequency("stem", 2), 1);
    CHECK_EQ(idx.get_term_frequency("absent", 1), 0);
    CHECK_EQ(idx.get_term_frequency("stem", 99), 0);
}

TEST(index_get_document_returns_original_text_and_throws_for_unknown_id) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(idx.get_document(2), std::string("stem cells mutation causes disease"));
    CHECK_THROWS(idx.get_document(99), std::out_of_range);
}

TEST(index_rejects_equal_or_lower_doc_ids_and_keeps_state) {
    InvertedIndex idx;
    idx.add_document(5, "alpha beta");
    CHECK_THROWS(idx.add_document(5, "gamma"), std::invalid_argument);
    CHECK_THROWS(idx.add_document(3, "gamma"), std::invalid_argument);
    CHECK_EQ(idx.get_total_docs(), 1);
    CHECK_EQ(idx.get_doc_frequency("gamma"), 0);
    CHECK_NO_THROW(idx.add_document(6, "gamma"));
    CHECK_EQ(idx.get_total_docs(), 2);
}

TEST(index_accepts_sparse_increasing_ids) {
    InvertedIndex idx;
    idx.add_document(4983, "alpha");
    idx.add_document(5836, "alpha beta");
    idx.add_document(46602807, "beta");
    CHECK_EQ(S(idx, "alpha"), (std::vector<int>{4983, 5836}));
    CHECK_EQ(S(idx, "beta"), (std::vector<int>{5836, 46602807}));
}

TEST(index_empty_document_counts_toward_totals_and_does_not_break_search) {
    InvertedIndex idx;
    idx.add_document(1, "alpha beta");
    idx.add_document(2, "");
    idx.add_document(3, "alpha");
    CHECK_EQ(idx.get_total_docs(), 3);
    CHECK_EQ(idx.get_doc_length(2), 0);
    CHECK_NEAR(idx.get_avg_doc_length(), 1.0, TOL);
    CHECK_EQ(S(idx, "alpha"), (std::vector<int>{1, 3}));
}

TEST(index_punctuation_only_document_has_length_zero) {
    InvertedIndex idx;
    idx.add_document(1, "!!! ---");
    CHECK_EQ(idx.get_doc_length(1), 0);
    CHECK_EQ(idx.get_total_docs(), 1);
}

// ================================================================ single terms

TEST(search_single_term) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "mutation"), (std::vector<int>{2, 3}));
    CHECK_EQ(S(idx, "disease"), (std::vector<int>{2}));
    CHECK_EQ(S(idx, "cells"), (std::vector<int>{1, 2, 3}));
}

TEST(search_term_is_case_insensitive) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "MUTATION"), (std::vector<int>{2, 3}));
    CHECK_EQ(S(idx, "Mutation"), (std::vector<int>{2, 3}));
}

TEST(search_unknown_term_returns_empty) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "zzz").empty());
}

TEST(search_term_with_hyphen_is_matched_as_adjacent_words) {
    InvertedIndex idx = make_index({{1, "a T-cell response"}, {2, "the t cell receptor"}, {3, "t and cell separate"}});
    CHECK_EQ(S(idx, "T-cell"), (std::vector<int>{1, 2}));
}

TEST(search_term_surrounded_by_whitespace) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "   mutation \t\n"), (std::vector<int>{2, 3}));
}

TEST(search_results_are_always_sorted_and_unique) {
    InvertedIndex idx = corpus_a();
    const char* queries[] = {"cells", "(+ stem immune)", "(+ cells cells)", "(^ cells stem)",
                             "(+ (^ stem cells) (^ immune response))", "(- cells disease)"};
    for (const char* q : queries) CHECK(sorted_unique(S(idx, q)));
}

// ================================================================ AND

TEST(and_two_terms) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ stem cells)"), (std::vector<int>{1, 2}));
    CHECK_EQ(S(idx, "(^ stem mutation)"), (std::vector<int>{2}));
}

TEST(and_three_terms) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ immune response mutation)"), (std::vector<int>{3}));
    CHECK(S(idx, "(^ stem immune mutation)").empty());
}

TEST(and_with_no_overlap_is_empty) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "(^ disease critical)").empty());
}

TEST(and_with_unknown_term_is_empty) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "(^ stem zzz)").empty());
    CHECK(S(idx, "(^ zzz stem)").empty());
}

TEST(and_single_operand_returns_that_operand) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ stem)"), (std::vector<int>{1, 2}));
}

TEST(and_operand_order_does_not_matter) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ stem cells)"), S(idx, "(^ cells stem)"));
    CHECK_EQ(S(idx, "(^ immune mutation cells)"), S(idx, "(^ cells mutation immune)"));
}

TEST(and_duplicate_operand_is_idempotent) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ stem stem)"), S(idx, "stem"));
}

TEST(and_with_phrase_operand) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ \"stem cells\" mutation)"), (std::vector<int>{2}));
    CHECK_EQ(S(idx, "(^ \"stem cells\" immune)"), (std::vector<int>{1}));
}

// ================================================================ OR

TEST(or_union_of_two_terms) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ disease critical)"), (std::vector<int>{1, 2}));
    CHECK_EQ(S(idx, "(+ stem immune)"), (std::vector<int>{1, 2, 3}));
}

TEST(or_has_no_duplicate_documents) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ stem stem)"), (std::vector<int>{1, 2}));
    CHECK_EQ(S(idx, "(+ cells stem immune)"), (std::vector<int>{1, 2, 3}));
}

TEST(or_single_operand_returns_that_operand) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ disease)"), (std::vector<int>{2}));
}

TEST(or_ignores_unknown_terms_but_not_when_all_are_unknown) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ zzz disease)"), (std::vector<int>{2}));
    CHECK_EQ(S(idx, "(+ disease zzz)"), (std::vector<int>{2}));
    CHECK(S(idx, "(+ zzz yyy)").empty());
}

TEST(or_with_phrase_operand) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ \"stem cells\" critical)"), (std::vector<int>{1, 2}));
    CHECK_EQ(S(idx, "(+ \"immune response\" disease)"), (std::vector<int>{1, 2, 3}));
}

// ================================================================ nesting

TEST(nested_and_inside_or) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ (^ stem mutation) critical)"), (std::vector<int>{1, 2}));
}

TEST(nested_or_inside_and) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ (+ disease critical) stem)"), (std::vector<int>{1, 2}));
    CHECK_EQ(S(idx, "(^ (+ disease critical) immune)"), (std::vector<int>{1}));
}

TEST(nested_three_levels) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ (^ (+ stem immune) mutation) critical)"), (std::vector<int>{1, 2, 3}));
}

TEST(deeply_nested_query_is_handled) {
    InvertedIndex idx = corpus_a();
    std::string q = "stem";
    for (int i = 0; i < 200; ++i) q = "(^ " + q + ")";
    std::vector<int> r;
    CHECK_NO_THROW(r = S(idx, q));
    CHECK_EQ(r, (std::vector<int>{1, 2}));
}

TEST(wide_query_with_many_operands) {
    InvertedIndex idx = corpus_a();
    std::string q = "(+";
    for (int i = 0; i < 500; ++i) q += " zzz" + std::to_string(i);
    q += " disease)";
    CHECK_EQ(S(idx, q), (std::vector<int>{2}));
}

// ================================================================ NOT  ( (- A B C) = A and not (B or C) )

TEST(not_removes_second_operand_from_first) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- cells mutation)"), (std::vector<int>{1}));
    CHECK_EQ(S(idx, "(- stem mutation)"), (std::vector<int>{1}));
}

TEST(not_phrase_as_first_operand) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- \"stem cells\" mutation)"), (std::vector<int>{1}));
}

TEST(not_phrase_as_excluded_operand) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- cells \"stem cells\")"), (std::vector<int>{3}));
}

TEST(not_with_several_excluded_operands) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- cells disease critical)"), (std::vector<int>{3}));
    CHECK(S(idx, "(- cells stem mutation)").empty());
}

TEST(not_excluding_an_unknown_term_changes_nothing) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- stem zzz)"), (std::vector<int>{1, 2}));
}

TEST(not_with_unknown_first_operand_is_empty) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "(- zzz stem)").empty());
}

TEST(not_excluding_everything_is_empty) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "(- cells cells)").empty());
    CHECK(S(idx, "(- disease cells)").empty());
}

TEST(not_with_single_operand_returns_it_unchanged) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- stem)"), (std::vector<int>{1, 2}));
}

TEST(not_nested_inside_and) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^ (- cells stem) immune)"), (std::vector<int>{3}));
}

TEST(not_nested_inside_or) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+ (- stem mutation) disease)"), (std::vector<int>{1, 2}));
}

TEST(not_with_boolean_expressions_on_both_sides) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(- (^ cells immune) (^ stem mutation))"), (std::vector<int>{1, 3}));
    CHECK_EQ(S(idx, "(- (+ disease critical) critical)"), (std::vector<int>{2}));
}

TEST(not_is_not_commutative) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "(- stem cells)").empty());
    CHECK_EQ(S(idx, "(- cells stem)"), (std::vector<int>{3}));
}

// ================================================================ phrases

TEST(phrase_matches_adjacent_words_only) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"stem cells\""), (std::vector<int>{1, 8}));
}

TEST(phrase_order_matters) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"cells stem\""), (std::vector<int>{3}));
}

TEST(phrase_with_gap_does_not_match) {
    InvertedIndex idx = corpus_f();
    std::vector<int> r = S(idx, "\"stem cells\"");
    CHECK(std::find(r.begin(), r.end(), 2) == r.end());
}

TEST(phrase_of_three_words) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"quick brown fox\""), (std::vector<int>{4, 5}));
}

TEST(phrase_is_case_insensitive_on_both_sides) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"STEM cells\""), S(idx, "\"stem CELLS\""));
    CHECK_EQ(S(idx, "\"stem cells divide\""), (std::vector<int>{1, 8}));
}

TEST(phrase_with_internal_punctuation_behaves_like_separate_words) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"stem-cells\""), (std::vector<int>{1, 8}));
}

TEST(phrase_of_one_word_equals_the_term) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"divide\""), S(idx, "divide"));
}

TEST(phrase_with_unknown_word_is_empty) {
    InvertedIndex idx = corpus_f();
    CHECK(S(idx, "\"zzz cells\"").empty());
    CHECK(S(idx, "\"stem zzz\"").empty());
}

TEST(phrase_longer_than_any_document_is_empty) {
    InvertedIndex idx = corpus_f();
    CHECK(S(idx, "\"stem cells divide fast tomorrow\"").empty());
}

TEST(phrase_repeated_in_one_document_lists_the_document_once) {
    InvertedIndex idx = corpus_d();
    CHECK_EQ(S(idx, "\"stem cells\""), (std::vector<int>{1, 2}));
}

TEST(phrase_with_repeated_words) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"a a\""), (std::vector<int>{7}));
    CHECK_EQ(S(idx, "\"a a a\""), (std::vector<int>{7}));
    CHECK(S(idx, "\"a a a a\"").empty());
}

TEST(phrase_overlapping_a_repeated_word) {
    InvertedIndex idx = corpus_d();  // doc 1: "stem cells and cells stem cells"
    CHECK_EQ(S(idx, "\"cells stem cells\""), (std::vector<int>{1}));
    CHECK_EQ(S(idx, "\"cells and cells\""), (std::vector<int>{1}));
}

TEST(phrase_does_not_match_across_documents) {
    InvertedIndex idx = make_index({{1, "stem"}, {2, "cells"}});
    CHECK(S(idx, "\"stem cells\"").empty());
}

TEST(phrase_surrounded_by_spaces_inside_quotes) {
    InvertedIndex idx = corpus_f();
    CHECK_EQ(S(idx, "\"  stem   cells  \""), (std::vector<int>{1, 8}));
}

// ================================================================ valid syntax variants

TEST(syntax_whitespace_variants_are_accepted) {
    InvertedIndex idx = corpus_a();
    std::vector<int> expected{1, 2};
    CHECK_EQ(S(idx, "(^ stem cells)"), expected);
    CHECK_EQ(S(idx, "( ^ stem cells )"), expected);
    CHECK_EQ(S(idx, "(^stem cells)"), expected);
    CHECK_EQ(S(idx, "(^   stem    cells)"), expected);
    CHECK_EQ(S(idx, "(^\tstem\ncells)"), expected);
    CHECK_EQ(S(idx, "  (^ stem cells)  "), expected);
    CHECK_EQ(S(idx, "(^ stem cells )"), expected);
}

TEST(syntax_no_space_between_nested_groups) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(+(^ stem mutation)(^ immune response))"), (std::vector<int>{1, 2, 3}));
}

TEST(syntax_phrase_directly_after_operator) {
    InvertedIndex idx = corpus_a();
    CHECK_EQ(S(idx, "(^\"stem cells\" mutation)"), (std::vector<int>{2}));
}

// ================================================================ parse errors (message and position)

TEST(parse_error_empty_and_blank_queries) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "", 0);
    CHECK_PARSE_ERROR(idx, "   ", 3);
}

TEST(parse_error_missing_closing_parenthesis) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "(^ stem cells", 0);
    CHECK_PARSE_ERROR(idx, "(^ stem (+ cells", 8);
    CHECK_PARSE_ERROR(idx, "(^ a (+ b)", 0);
    CHECK_PARSE_ERROR(idx, "(^", 0);
}

TEST(parse_error_missing_operator) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "(", 1);
    CHECK_PARSE_ERROR(idx, "()", 1);
    CHECK_PARSE_ERROR(idx, "(stem cells)", 1);
}

TEST(parse_error_unknown_operator) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "(* stem cells)", 1);
    CHECK_PARSE_ERROR(idx, "(& stem cells)", 1);
    CHECK_PARSE_ERROR(idx, "(AND stem cells)", 1);
}

TEST(parse_error_operator_without_operands) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "(^)", 0);
    CHECK_PARSE_ERROR(idx, "(+ )", 0);
    CHECK_PARSE_ERROR(idx, "(-)", 0);
    CHECK_PARSE_ERROR(idx, "(^ stem ())", 9);
}

TEST(parse_error_stray_closing_parenthesis) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, ")", 0);
    CHECK_PARSE_ERROR(idx, "(^ stem cells))", 14);
    CHECK_PARSE_ERROR(idx, "stem)", 4);
}

TEST(parse_error_text_after_the_query) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "stem cells", 5);
    CHECK_PARSE_ERROR(idx, "(^ stem cells) mutation", 15);
    CHECK_PARSE_ERROR(idx, "\"stem cells\" mutation", 13);
}

TEST(parse_error_quote_problems) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "\"unterminated", 0);
    CHECK_PARSE_ERROR(idx, "(^ stem \"cells)", 8);
    CHECK_PARSE_ERROR(idx, "\"\"", 0);
    CHECK_PARSE_ERROR(idx, "\"   \"", 0);
    CHECK_PARSE_ERROR(idx, "\"!!\"", 0);
}

TEST(parse_error_term_with_no_searchable_characters) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "!!!", 0);
    CHECK_PARSE_ERROR(idx, "(^ stem ---)", 8);
}

TEST(parse_error_is_a_runtime_error_with_a_readable_message) {
    InvertedIndex idx = corpus_a();
    bool caught = false;
    try {
        idx.search("(^ stem cells");
    } catch (const std::runtime_error& e) {
        caught = true;
        CHECK(std::string(e.what()).size() > 0);
        CHECK(std::string(e.what()).find("position") != std::string::npos);
    }
    CHECK(caught);
}

TEST(parse_error_leaves_the_index_usable) {
    InvertedIndex idx = corpus_a();
    CHECK_PARSE_ERROR(idx, "(^ stem cells", -1);
    CHECK_EQ(S(idx, "(^ stem cells)"), (std::vector<int>{1, 2}));
}

TEST(parse_error_comes_from_search_not_from_scoring) {
    // Scoring only ever receives documents the search already produced.
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    CHECK_THROWS(sc.score(idx.search("(^ stem cells"), "(^ stem cells"), ParseError);
}

// ================================================================ BM25 (k1 = 1.2, b = 0.75, values from the Python reference)

TEST(bm25_single_term_manual_values) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "mutation");
    CHECK_EQ(r.size(), (size_t)2);
    CHECK_NEAR(score_of(r, 2), 0.5043941387027407, TOL);   // dl 5
    CHECK_NEAR(score_of(r, 3), 0.47000362924573563, TOL);  // dl 6 = avgdl
}

TEST(bm25_term_present_in_every_document_has_positive_idf) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "cells");
    CHECK_EQ(r.size(), (size_t)3);
    CHECK_NEAR(score_of(r, 1), 0.12500811224423392, TOL);
    CHECK_NEAR(score_of(r, 2), 0.14330198232875596, TOL);
    CHECK_NEAR(score_of(r, 3), 0.13353139262452257, TOL);
    for (const auto& x : r) CHECK(x.score > 0.0);
}

TEST(bm25_and_sums_the_scores_of_its_terms) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "(^ immune response)");
    CHECK_EQ(ids_of(r), (std::vector<int>{3, 1}));
    CHECK_NEAR(score_of(r, 1), 0.8800067951835052, TOL);
    CHECK_NEAR(score_of(r, 3), 0.9400072584914713, TOL);
}

TEST(bm25_or_document_matching_more_terms_outranks_one_matching_fewer) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "(+ mutation immune)");
    CHECK_EQ(ids_of(r), (std::vector<int>{3, 2, 1}));
    CHECK_NEAR(score_of(r, 3), 0.9400072584914713, TOL);  // both terms
    CHECK_NEAR(score_of(r, 2), 0.5043941387027407, TOL);  // mutation only
    CHECK_NEAR(score_of(r, 1), 0.4400033975917526, TOL);  // immune only
}

TEST(bm25_unknown_query_terms_add_nothing) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto with_unknown = ranked(idx, sc, "(+ mutation zzz)");
    auto plain = ranked(idx, sc, "mutation");
    CHECK_EQ(ids_of(with_unknown), ids_of(plain));
    for (const auto& x : plain) CHECK_NEAR(score_of(with_unknown, x.doc_id), x.score, TOL);
}

TEST(bm25_scoring_is_case_insensitive) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto lower = ranked(idx, sc, "mutation");
    auto upper = ranked(idx, sc, "MUTATION");
    CHECK_EQ(ids_of(lower), ids_of(upper));
    for (const auto& x : lower) CHECK_NEAR(score_of(upper, x.doc_id), x.score, TOL);
}

TEST(bm25_results_are_sorted_by_descending_score) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    const char* queries[] = {"cells", "(+ mutation immune)", "(+ stem immune response mutation)"};
    for (const char* q : queries) {
        auto r = ranked(idx, sc, q);
        for (size_t i = 1; i < r.size(); ++i) CHECK(r[i - 1].score >= r[i].score);
    }
}

TEST(bm25_every_returned_document_comes_from_the_match_list) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    std::vector<int> matches = S(idx, "(^ stem cells)");
    auto r = sc.score(matches, "(^ stem cells)");
    CHECK_EQ(r.size(), matches.size());
    std::vector<int> got = ids_of(r);
    std::sort(got.begin(), got.end());
    CHECK_EQ(got, matches);
}

TEST(bm25_empty_match_list_gives_empty_result) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    CHECK(sc.score({}, "mutation").empty());
}

TEST(bm25_on_empty_index_is_safe) {
    InvertedIndex idx;
    BM25Scorer sc(idx, 1.2, 0.75);
    CHECK(sc.score({}, "mutation").empty());
    CHECK(sc.score({1, 2}, "mutation").empty());
}

TEST(bm25_shorter_document_scores_higher_at_equal_term_frequency) {
    InvertedIndex idx = corpus_b();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "apple");
    CHECK_NEAR(score_of(r, 1), 0.42327399986090236, TOL);  // dl 2, tf 1
    CHECK_NEAR(score_of(r, 3), 0.32590745676190885, TOL);  // dl 4, tf 1
    CHECK(score_of(r, 1) > score_of(r, 3));
}

TEST(bm25_term_frequency_saturates) {
    InvertedIndex idx = corpus_b();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "apple");
    CHECK_NEAR(score_of(r, 2), 0.5012728941841645, TOL);  // tf 2, dl 3
    CHECK(score_of(r, 2) > score_of(r, 1));
    CHECK(score_of(r, 2) < 2.0 * score_of(r, 1));
    CHECK_EQ(ids_of(r), (std::vector<int>{2, 1, 3}));
}

TEST(bm25_custom_parameters_b_zero_removes_length_normalization) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.0);
    auto r = ranked(idx, sc, "mutation");
    CHECK_NEAR(score_of(r, 2), 0.47000362924573563, TOL);
    CHECK_NEAR(score_of(r, 3), 0.47000362924573563, TOL);
}

TEST(bm25_custom_parameters_k1_zero_makes_tf_binary) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 0.0, 0.75);
    auto r = ranked(idx, sc, "mutation");
    CHECK_NEAR(score_of(r, 2), 0.47000362924573563, TOL);
    CHECK_NEAR(score_of(r, 3), 0.47000362924573563, TOL);
}

TEST(bm25_custom_parameters_k1_2_b_half) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 2.0, 0.5);
    auto r = ranked(idx, sc, "mutation");
    CHECK_NEAR(score_of(r, 2), 0.49765090155430824, TOL);
    CHECK_NEAR(score_of(r, 3), 0.47000362924573563, TOL);
}

TEST(bm25_repeated_calls_give_identical_results) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto first = ranked(idx, sc, "(+ mutation immune)");
    for (int i = 0; i < 20; ++i) {
        auto again = ranked(idx, sc, "(+ mutation immune)");
        CHECK_EQ(ids_of(again), ids_of(first));
        for (const auto& x : first) CHECK_NEAR(score_of(again, x.doc_id), x.score, 0.0);
    }
}

// ---- ranking details: ties, phrase scoring, excluded branches

TEST(bm25_equal_scores_are_ordered_by_ascending_doc_id) {
    InvertedIndex idx = corpus_b();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "cherry");
    CHECK_NEAR(score_of(r, 3), 0.6333549317576496, TOL);
    CHECK_NEAR(score_of(r, 4), 0.6333549317576496, TOL);
    CHECK_EQ(ids_of(r), (std::vector<int>{3, 4}));
    for (int i = 0; i < 20; ++i) CHECK_EQ(ids_of(ranked(idx, sc, "cherry")), (std::vector<int>{3, 4}));
}

TEST(bm25_ties_with_many_documents_are_ordered_by_doc_id) {
    // 60 identical documents: every score is equal. More than 16 elements, so an
    // unstable std::sort would reorder them; only an explicit tie-break keeps doc_id order.
    InvertedIndex idx;
    std::vector<int> expected;
    for (int id = 1; id <= 60; ++id) {
        idx.add_document(id * 7, "same words here");
        expected.push_back(id * 7);
    }
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "words");
    CHECK_EQ(ids_of(r), expected);
    auto r2 = ranked(idx, sc, "(+ words here)");
    CHECK_EQ(ids_of(r2), expected);
}

TEST(bm25_phrase_scores_with_phrase_frequency_and_summed_idf) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "\"stem cells\"");
    CHECK_EQ(r.size(), (size_t)2);
    CHECK_NEAR(score_of(r, 1), 0.5650115098359865, TOL);
    CHECK_NEAR(score_of(r, 2), 0.6476961210314967, TOL);
    CHECK_EQ(ids_of(r), (std::vector<int>{2, 1}));
}

TEST(bm25_phrase_frequency_is_not_the_word_frequencies) {
    // Doc 1 has the phrase twice, but "stem" twice and "cells" three times.
    InvertedIndex idx = corpus_d();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "\"stem cells\"");
    CHECK_EQ(r.size(), (size_t)2);
    CHECK_NEAR(score_of(r, 1), 0.32193870002624625, TOL);  // phrase tf 2
    CHECK_NEAR(score_of(r, 2), 0.2974876595179237, TOL);   // phrase tf 1
}

TEST(bm25_and_with_a_phrase_adds_both_clauses) {
    InvertedIndex idx = corpus_c();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "(^ cancer \"stem cells\")");
    CHECK_EQ(ids_of(r), (std::vector<int>{1}));
    CHECK_NEAR(score_of(r, 1), 0.7610144207419036, TOL);
}

TEST(bm25_excluded_branches_do_not_contribute) {
    // (- cancer "stem cells") keeps docs 2 and 3. Doc 2 contains "stem" and "cells"
    // separately; those words are inside the excluded phrase and must add nothing.
    InvertedIndex idx = corpus_c();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "(- cancer \"stem cells\")");
    CHECK_EQ(r.size(), (size_t)2);
    CHECK_NEAR(score_of(r, 2), 0.1256249285875443, TOL);
    CHECK_NEAR(score_of(r, 3), 0.1378699577278464, TOL);
}

TEST(bm25_excluded_single_term_does_not_contribute) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    auto with_not = ranked(idx, sc, "(- cells disease)");
    auto only_first = sc.score(S(idx, "(- cells disease)"), "cells");
    CHECK_EQ(ids_of(with_not), ids_of(only_first));
    for (const auto& x : only_first) CHECK_NEAR(score_of(with_not, x.doc_id), x.score, TOL);
}

// ================================================================ save / load

TEST(saveload_roundtrip_preserves_search_scores_and_text) {
    InvertedIndex idx = corpus_a();
    std::string path = temp_file("engine_test_roundtrip.idx");
    idx.save(path);

    InvertedIndex loaded;
    loaded.load(path);
    std::remove(path.c_str());

    CHECK_EQ(loaded.get_total_docs(), 3);
    CHECK_NEAR(loaded.get_avg_doc_length(), 6.0, TOL);
    CHECK_EQ(loaded.get_doc_length(1), 7);
    CHECK_EQ(loaded.get_document(3), std::string("immune response to mutation in cells"));
    CHECK_EQ(loaded.get_doc_frequency("cells"), 3);
    CHECK_EQ(loaded.get_term_frequency("mutation", 2), 1);

    const char* queries[] = {"mutation", "(^ stem cells)", "\"stem cells\"", "(- cells mutation)",
                             "(+ immune disease)"};
    BM25Scorer a(idx, 1.2, 0.75), b(loaded, 1.2, 0.75);
    for (const char* q : queries) {
        CHECK_EQ(S(loaded, q), S(idx, q));
        auto ra = ranked(idx, a, q);
        auto rb = ranked(loaded, b, q);
        CHECK_EQ(ids_of(ra), ids_of(rb));
        for (const auto& x : ra) CHECK_NEAR(score_of(rb, x.doc_id), x.score, 0.0);
    }
}

TEST(saveload_empty_index_roundtrip) {
    InvertedIndex idx;
    std::string path = temp_file("engine_test_empty.idx");
    idx.save(path);
    InvertedIndex loaded;
    CHECK_NO_THROW(loaded.load(path));
    std::remove(path.c_str());
    CHECK_EQ(loaded.get_total_docs(), 0);
    CHECK(loaded.search("anything").empty());
}

TEST(saveload_missing_file_throws) {
    InvertedIndex idx;
    CHECK_THROWS(idx.load(temp_file("engine_test_does_not_exist.idx")), std::runtime_error);
}

TEST(saveload_unwritable_path_throws) {
    InvertedIndex idx = corpus_a();
    CHECK_THROWS(idx.save(temp_file("no_such_folder_xyz/engine_test.idx")), std::runtime_error);
}

TEST(saveload_wrong_format_version_throws) {
    std::string path = temp_file("engine_test_badversion.idx");
    {
        std::ofstream out(path, std::ios::binary);
        uint32_t bad_version = 999;
        out.write(reinterpret_cast<const char*>(&bad_version), sizeof(bad_version));
    }
    InvertedIndex idx;
    CHECK_THROWS(idx.load(path), std::runtime_error);
    std::remove(path.c_str());
}

TEST(saveload_load_replaces_existing_content) {
    InvertedIndex saved = make_index({{1, "alpha beta"}});
    std::string path = temp_file("engine_test_replace.idx");
    saved.save(path);

    InvertedIndex idx = corpus_a();
    idx.load(path);
    std::remove(path.c_str());

    CHECK_EQ(idx.get_total_docs(), 1);
    CHECK_NEAR(idx.get_avg_doc_length(), 2.0, TOL);
    CHECK(idx.search("mutation").empty());
    CHECK_EQ(S(idx, "alpha"), (std::vector<int>{1}));
}

TEST(saveload_add_after_load_still_enforces_increasing_ids) {
    InvertedIndex saved = make_index({{10, "alpha"}, {20, "beta"}});
    std::string path = temp_file("engine_test_ids.idx");
    saved.save(path);

    InvertedIndex idx;
    idx.load(path);
    std::remove(path.c_str());

    CHECK_THROWS(idx.add_document(15, "gamma"), std::invalid_argument);
    CHECK_THROWS(idx.add_document(20, "gamma"), std::invalid_argument);
    CHECK_NO_THROW(idx.add_document(21, "gamma"));
    CHECK_EQ(S(idx, "gamma"), (std::vector<int>{21}));
}

// ================================================================ end to end

TEST(end_to_end_search_then_rank) {
    InvertedIndex idx = corpus_a();
    BM25Scorer sc(idx, 1.2, 0.75);
    std::string q = "(^ (+ mutation immune) cells)";
    std::vector<int> docs = idx.search(q);
    CHECK_EQ(docs, (std::vector<int>{1, 2, 3}));
    auto r = sc.score(docs, q);
    CHECK_EQ(r.size(), (size_t)3);
    CHECK_EQ(ids_of(r).front(), 3);  // matches immune, mutation and cells
}

TEST(end_to_end_strict_query_returns_fewer_documents_than_loose_query) {
    InvertedIndex idx = corpus_a();
    CHECK(S(idx, "(^ stem cells immune)").size() <= S(idx, "(+ stem cells immune)").size());
    CHECK(S(idx, "\"stem cells\"").size() <= S(idx, "(^ stem cells)").size());
}

// ================================================================ tokenizer options: stemming and stopwords

static std::vector<std::string> terms_of(const std::vector<AnalyzedToken>& v) {
    std::vector<std::string> out;
    for (const auto& t : v) out.push_back(t.term);
    return out;
}

static std::vector<int> positions_of(const std::vector<AnalyzedToken>& v) {
    std::vector<int> out;
    for (const auto& t : v) out.push_back(t.position);
    return out;
}

TEST(tokenizer_default_options_are_exact_match) {
    Tokenizer t;
    CHECK(!t.options().stem);
    CHECK(!t.options().stopwords);
    CHECK_EQ(t.tokenize("The Mutations of cells"), (std::vector<std::string>{"the", "mutations", "of", "cells"}));
}

TEST(tokenizer_stopwords_are_removed_and_positions_keep_their_gaps) {
    TokenizerOptions o;
    o.stopwords = true;
    Tokenizer t(o);
    auto toks = t.analyze("The mutations of stem cells");
    CHECK_EQ(terms_of(toks), (std::vector<std::string>{"mutations", "stem", "cells"}));
    CHECK_EQ(positions_of(toks), (std::vector<int>{1, 3, 4}));
}

TEST(tokenizer_all_stopword_text_gives_no_tokens) {
    TokenizerOptions o;
    o.stopwords = true;
    Tokenizer t(o);
    CHECK(t.tokenize("the of and to a").empty());
}

TEST(tokenizer_stopword_list_is_lucenes_33_words) {
    TokenizerOptions o;
    o.stopwords = true;
    Tokenizer t(o);
    const char* stop[] = {"a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
                          "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
                          "there", "these", "they", "this", "to", "was", "will", "with"};
    for (const char* w : stop) CHECK(t.tokenize(w).empty());
    // Near misses must survive.
    CHECK_EQ(t.tokenize("an ant, is islet, to tomato"), (std::vector<std::string>{"ant", "islet", "tomato"}));
    CHECK_EQ(t.tokenize("he she we you"), (std::vector<std::string>{"he", "she", "we", "you"}));
}

TEST(tokenizer_stopwords_are_case_insensitive) {
    TokenizerOptions o;
    o.stopwords = true;
    Tokenizer t(o);
    CHECK(t.tokenize("The THE tHe").empty());
}

TEST(porter_stem_matches_reference_stems) {
    // Expected values: NLTK PorterStemmer(mode=MARTIN_EXTENSIONS), i.e. Martin Porter's reference C code.
    const std::pair<const char*, const char*> cases[] = {
        {"caresses", "caress"}, {"ponies", "poni"}, {"ties", "ti"}, {"caress", "caress"}, {"cats", "cat"},
        {"feed", "feed"}, {"agreed", "agre"}, {"plastered", "plaster"}, {"bled", "bled"}, {"motoring", "motor"},
        {"sing", "sing"}, {"conflated", "conflat"}, {"troubled", "troubl"}, {"sized", "size"},
        {"hopping", "hop"}, {"tanned", "tan"}, {"falling", "fall"}, {"hissing", "hiss"}, {"fizzed", "fizz"},
        {"failing", "fail"}, {"filing", "file"}, {"happy", "happi"}, {"sky", "sky"}, {"relational", "relat"},
        {"conditional", "condit"}, {"rational", "ration"}, {"valenci", "valenc"}, {"digitizer", "digit"},
        {"conformabli", "conform"}, {"radicalli", "radic"}, {"differentli", "differ"}, {"vileli", "vile"},
        {"analogousli", "analog"}, {"vietnamization", "vietnam"}, {"predication", "predic"},
        {"operator", "oper"}, {"feudalism", "feudal"}, {"decisiveness", "decis"}, {"hopefulness", "hope"},
        {"callousness", "callous"}, {"formaliti", "formal"}, {"sensitiviti", "sensit"},
        {"sensibiliti", "sensibl"}, {"triplicate", "triplic"}, {"formative", "form"}, {"formalize", "formal"},
        {"electriciti", "electr"}, {"electrical", "electr"}, {"hopeful", "hope"}, {"goodness", "good"},
        {"revival", "reviv"}, {"allowance", "allow"}, {"inference", "infer"}, {"airliner", "airlin"},
        {"gyroscopic", "gyroscop"}, {"adjustable", "adjust"}, {"defensible", "defens"}, {"irritant", "irrit"},
        {"replacement", "replac"}, {"adjustment", "adjust"}, {"dependent", "depend"}, {"adoption", "adopt"},
        {"homologou", "homolog"}, {"communism", "commun"}, {"activate", "activ"}, {"angulariti", "angular"},
        {"homologous", "homolog"}, {"effective", "effect"}, {"bowdlerize", "bowdler"}, {"probate", "probat"},
        {"rate", "rate"}, {"cease", "ceas"}, {"controll", "control"}, {"roll", "roll"},
        {"mutation", "mutat"}, {"mutations", "mutat"}, {"mutating", "mutat"}, {"mutated", "mutat"},
        {"stemming", "stem"}, {"cells", "cell"}, {"cell", "cell"}, {"diseases", "diseas"},
        {"disease", "diseas"}, {"analyses", "analys"}};
    for (const auto& c : cases) CHECK_EQ(porter_stem(c.first), std::string(c.second));
}

TEST(porter_stem_leaves_very_short_words_and_digit_words_alone) {
    CHECK_EQ(porter_stem(""), std::string(""));
    CHECK_EQ(porter_stem("a"), std::string("a"));
    CHECK_EQ(porter_stem("is"), std::string("is"));
    CHECK_EQ(porter_stem("t"), std::string("t"));
    CHECK_EQ(porter_stem("il6"), std::string("il6"));
    CHECK_EQ(porter_stem("p53"), std::string("p53"));
    CHECK_EQ(porter_stem("cd4"), std::string("cd4"));
    CHECK_EQ(porter_stem("2024"), std::string("2024"));
}

TEST(tokenizer_stemming_option_stems_every_token) {
    TokenizerOptions o;
    o.stem = true;
    Tokenizer t(o);
    CHECK_EQ(t.tokenize("Mutations of Stemming Cells"), (std::vector<std::string>{"mutat", "of", "stem", "cell"}));
}

TEST(tokenizer_stopwords_are_removed_before_stemming) {
    // "thes" and "ins" are not stopwords, but stem to the stopwords "the" and "in".
    // Lucene removes stopwords first, so these words must survive as stems.
    TokenizerOptions o;
    o.stem = true;
    o.stopwords = true;
    Tokenizer t(o);
    CHECK_EQ(t.tokenize("thes ins the in"), (std::vector<std::string>{"the", "in"}));
}

TEST(tokenizer_stemming_and_stopwords_together) {
    TokenizerOptions o;
    o.stem = true;
    o.stopwords = true;
    Tokenizer t(o);
    auto toks = t.analyze("The mutations of stem cells cause diseases");
    CHECK_EQ(terms_of(toks), (std::vector<std::string>{"mutat", "stem", "cell", "caus", "diseas"}));
    CHECK_EQ(positions_of(toks), (std::vector<int>{1, 3, 4, 5, 6}));
}

// ---- indexes built with stemming and stopwords. Corpus G values come from the Python reference.

static InvertedIndex corpus_g(bool stem, bool stop) {
    TokenizerOptions o;
    o.stem = stem;
    o.stopwords = stop;
    InvertedIndex idx(o);
    idx.add_document(1, "The mutations of stem cells cause diseases");
    idx.add_document(2, "A mutation in the cell");
    idx.add_document(3, "Stemming cells are mutating");
    return idx;
}

TEST(stemmed_index_matches_all_word_forms) {
    InvertedIndex idx = corpus_g(true, false);
    std::vector<int> all{1, 2, 3};
    CHECK_EQ(S(idx, "mutation"), all);
    CHECK_EQ(S(idx, "mutations"), all);
    CHECK_EQ(S(idx, "mutating"), all);
    CHECK_EQ(S(idx, "MUTATED"), all);
    CHECK_EQ(S(idx, "cell"), all);
    CHECK_EQ(S(idx, "cells"), all);
}

TEST(exact_index_does_not_match_other_word_forms) {
    InvertedIndex idx = corpus_g(false, false);
    CHECK_EQ(S(idx, "mutation"), (std::vector<int>{2}));
    CHECK_EQ(S(idx, "mutations"), (std::vector<int>{1}));
    CHECK_EQ(S(idx, "cell"), (std::vector<int>{2}));
}

TEST(stemmed_phrase_matches_inflected_text) {
    InvertedIndex idx = corpus_g(true, true);
    CHECK_EQ(S(idx, "\"stem cells\""), (std::vector<int>{1, 3}));
    CHECK_EQ(S(idx, "\"stemming cell\""), (std::vector<int>{1, 3}));
}

TEST(stemmed_not_excludes_every_word_form) {
    InvertedIndex idx = corpus_g(true, true);
    CHECK_EQ(S(idx, "(- cell disease)"), (std::vector<int>{2, 3}));  // "diseases" in doc 1 is excluded
    InvertedIndex exact = corpus_g(false, false);
    CHECK_EQ(S(exact, "(- cells disease)"), (std::vector<int>{1, 3}));  // exact: "diseases" does not match "disease"
}

TEST(stopwords_are_not_indexed_and_do_not_count_toward_length) {
    InvertedIndex idx = corpus_g(true, true);
    CHECK_EQ(idx.get_doc_frequency("the"), 0);
    CHECK_EQ(idx.get_doc_frequency("of"), 0);
    CHECK_EQ(idx.get_doc_length(1), 5);
    CHECK_EQ(idx.get_doc_length(2), 2);
    CHECK_EQ(idx.get_doc_length(3), 3);
    CHECK_NEAR(idx.get_avg_doc_length(), 10.0 / 3.0, TOL);

    InvertedIndex plain = corpus_g(false, false);
    CHECK_EQ(plain.get_doc_frequency("the"), 2);
    CHECK_EQ(plain.get_doc_length(1), 7);
}

TEST(stopword_phrase_gap_semantics_follow_lucene) {
    // "mutation in the cell" becomes mutat ... cell at distance 3. Doc 1 has "mutations of stem cells"
    // (distance 3 as well), so it matches too; this mirrors Lucene's position-increment handling.
    InvertedIndex idx = corpus_g(true, true);
    CHECK_EQ(S(idx, "\"mutation in the cell\""), (std::vector<int>{1, 2}));
    CHECK(S(idx, "\"mutation cell\"").empty());  // adjacent words only: nobody has mutat directly before cell
}

TEST(query_made_only_of_stopwords_is_a_parse_error_when_stopwords_are_on) {
    InvertedIndex idx = corpus_g(true, true);
    CHECK_PARSE_ERROR(idx, "the", 0);
    CHECK_PARSE_ERROR(idx, "\"the of\"", 0);
    CHECK_PARSE_ERROR(idx, "(^ mutation the)", 12);
    InvertedIndex plain = corpus_g(false, false);
    CHECK_EQ(S(plain, "the"), (std::vector<int>{1, 2}));
}

TEST(bm25_stemmed_stopworded_values) {
    InvertedIndex idx = corpus_g(true, true);
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "mutation");
    CHECK_EQ(ids_of(r), (std::vector<int>{2, 3, 1}));
    CHECK_NEAR(score_of(r, 1), 0.11085625048073573, TOL);
    CHECK_NEAR(score_of(r, 2), 0.15965709987714657, TOL);
    CHECK_NEAR(score_of(r, 3), 0.13922704444263018, TOL);

    auto p = ranked(idx, sc, "\"stem cells\"");
    CHECK_EQ(ids_of(p), (std::vector<int>{3, 1}));
    CHECK_NEAR(score_of(p, 1), 0.5010479426847427, TOL);
    CHECK_NEAR(score_of(p, 3), 0.6292782218552456, TOL);
}

TEST(bm25_hyphenated_term_scores_as_a_phrase) {
    InvertedIndex idx = make_index({{1, "a T-cell response"}, {2, "the t cell receptor t cell"},
                                    {3, "t and cell separate"}, {4, "cell t"}});
    BM25Scorer sc(idx, 1.2, 0.75);
    auto r = ranked(idx, sc, "T-cell");
    CHECK_EQ(ids_of(r), (std::vector<int>{2, 1}));
    CHECK_NEAR(score_of(r, 1), 0.2107210313156527, TOL);
    CHECK_NEAR(score_of(r, 2), 0.25401987336681425, TOL);
}

TEST(phrase_frequency_counts_occurrences) {
    InvertedIndex idx = corpus_d();
    std::vector<AnalyzedToken> stem_cells = idx.analyze("stem cells");
    CHECK_EQ(idx.phrase_frequency(stem_cells, 1), 2);
    CHECK_EQ(idx.phrase_frequency(stem_cells, 2), 1);
    CHECK_EQ(idx.phrase_frequency(stem_cells, 3), 0);
    CHECK_EQ(idx.phrase_frequency(stem_cells, 99), 0);
    CHECK_EQ(idx.phrase_frequency(idx.analyze("zzz cells"), 1), 0);
    CHECK_EQ(idx.phrase_frequency({}, 1), 0);
}

TEST(parse_error_for_absurdly_deep_nesting) {
    InvertedIndex idx = corpus_a();
    std::string q = "stem";
    for (int i = 0; i < 5000; ++i) q = "(^ " + q + ")";
    CHECK_THROWS(idx.search(q), ParseError);
}

// ================================================================ save / load: options, determinism, damaged files

static std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

static void write_file(const std::string& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

TEST(saveload_keeps_tokenizer_options) {
    InvertedIndex idx = corpus_g(true, true);
    std::string path = temp_file("engine_test_options.idx");
    idx.save(path);

    InvertedIndex loaded;  // default (exact) options, replaced by the file's
    loaded.load(path);
    std::remove(path.c_str());

    CHECK(loaded.tokenizer_options().stem);
    CHECK(loaded.tokenizer_options().stopwords);
    CHECK_EQ(S(loaded, "mutations"), (std::vector<int>{1, 2, 3}));
    CHECK_EQ(S(loaded, "\"stem cells\""), (std::vector<int>{1, 3}));
    CHECK_EQ(loaded.get_doc_length(1), 5);
}

TEST(saveload_files_are_identical_when_saved_twice) {
    InvertedIndex idx = corpus_a();
    std::string p1 = temp_file("engine_test_det1.idx");
    std::string p2 = temp_file("engine_test_det2.idx");
    idx.save(p1);
    idx.save(p2);
    std::string a = read_file(p1), b = read_file(p2);
    std::remove(p1.c_str());
    std::remove(p2.c_str());
    CHECK(!a.empty());
    CHECK(a == b);
}

TEST(saveload_saved_loaded_saved_gives_identical_bytes) {
    InvertedIndex idx = corpus_g(true, true);
    std::string p1 = temp_file("engine_test_rt1.idx");
    std::string p2 = temp_file("engine_test_rt2.idx");
    idx.save(p1);
    InvertedIndex loaded;
    loaded.load(p1);
    loaded.save(p2);
    std::string a = read_file(p1), b = read_file(p2);
    std::remove(p1.c_str());
    std::remove(p2.c_str());
    CHECK(a == b);
}

TEST(saveload_every_truncation_throws_and_leaves_the_index_unchanged) {
    InvertedIndex good = corpus_a();
    std::string path = temp_file("engine_test_trunc_src.idx");
    good.save(path);
    std::string bytes = read_file(path);
    std::remove(path.c_str());
    CHECK(bytes.size() > 100);

    std::string cut_path = temp_file("engine_test_trunc.idx");
    for (size_t len = 0; len < bytes.size(); ++len) {
        write_file(cut_path, bytes.substr(0, len));
        InvertedIndex target = corpus_g(false, false);  // already holds data
        bool threw = false;
        try {
            target.load(cut_path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        ++g_checks;
        if (!threw) fail(__FILE__, __LINE__, "no error for file cut to " + show((long)len) + " bytes");
        // The failed load must not have touched the existing content.
        if (target.get_total_docs() != 3 || target.search("mutation") != std::vector<int>{2}) {
            fail(__FILE__, __LINE__, "index changed by a failed load (cut at " + show((long)len) + ")");
            break;
        }
    }
    std::remove(cut_path.c_str());
}

TEST(saveload_flipped_bytes_never_crash) {
    InvertedIndex good = corpus_a();
    std::string path = temp_file("engine_test_flip_src.idx");
    good.save(path);
    std::string bytes = read_file(path);
    std::remove(path.c_str());

    std::string bad_path = temp_file("engine_test_flip.idx");
    for (size_t i = 0; i < bytes.size(); ++i) {
        std::string damaged = bytes;
        damaged[i] = static_cast<char>(damaged[i] ^ 0xFF);
        write_file(bad_path, damaged);
        InvertedIndex target;
        ++g_checks;
        try {
            target.load(bad_path);  // either loads (the flip hit harmless data) or throws runtime_error
            (void)target.search("mutation");
        } catch (const std::runtime_error&) {
        } catch (const std::exception& e) {
            fail(__FILE__, __LINE__, "unexpected exception type after flipping byte " + show((long)i) + ": " + e.what());
        }
    }
    std::remove(bad_path.c_str());
}

TEST(saveload_a_reloaded_index_can_be_extended_and_saved_again) {
    InvertedIndex idx = make_index({{10, "alpha beta"}, {20, "beta gamma"}});
    std::string p1 = temp_file("engine_test_ext1.idx");
    idx.save(p1);
    InvertedIndex loaded;
    loaded.load(p1);
    std::remove(p1.c_str());
    loaded.add_document(30, "gamma delta");
    CHECK_EQ(S(loaded, "gamma"), (std::vector<int>{20, 30}));
    CHECK_EQ(loaded.get_total_docs(), 3);

    std::string p2 = temp_file("engine_test_ext2.idx");
    loaded.save(p2);
    InvertedIndex again;
    again.load(p2);
    std::remove(p2.c_str());
    CHECK_EQ(S(again, "gamma"), (std::vector<int>{20, 30}));
}

// ================================================================ runner

int main(int argc, char** argv) {
    std::string filter = argc > 1 ? argv[1] : "";
    int ran = 0, failed = 0;
    std::vector<std::string> failed_names;

    for (const auto& t : registry()) {
        if (!filter.empty() && t.name.find(filter) == std::string::npos) continue;
        ++ran;
        g_current_failed = false;
        try {
            t.fn();
        } catch (const std::exception& e) {
            g_current_failed = true;
            ++g_failed_checks;
            std::cerr << "    unexpected exception: " << e.what() << "\n";
        } catch (...) {
            g_current_failed = true;
            ++g_failed_checks;
            std::cerr << "    unexpected non-standard exception\n";
        }
        if (g_current_failed) {
            ++failed;
            failed_names.push_back(t.name);
            std::cout << "[FAIL] " << t.name << "\n";
        } else {
            std::cout << "[PASS] " << t.name << "\n";
        }
    }

    std::cout << "\n" << (ran - failed) << " / " << ran << " tests passed (" << g_checks << " checks, "
              << g_failed_checks << " failed)\n";
    if (!failed_names.empty()) {
        std::cout << "Failing tests:\n";
        for (const auto& n : failed_names) std::cout << "  - " << n << "\n";
    }
    return failed == 0 ? 0 : 1;
}
