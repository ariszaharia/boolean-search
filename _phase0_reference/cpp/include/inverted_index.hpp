#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "tokenizer.hpp"
#include "types.hpp"

// Thrown for any malformed query. `position` is the 0-based byte offset in the
// query string where the problem was detected (for unbalanced parentheses it is
// the position of the '(' that was never closed). The message is meant to be fed
// back to the LLM planner so it can repair the query.
class ParseError : public std::runtime_error {
public:
    size_t position;
    ParseError(const std::string& msg, size_t pos)
        : std::runtime_error(msg + " (at position " + std::to_string(pos) + ")"), position(pos) {}
};

// Query grammar (S-expressions):
//   query   := term | "phrase words" | ( op query+ )
//   op      := ^   AND (intersection)
//            | +   OR  (union)
//            | -   NOT (first operand minus the union of all other operands)
enum class QueryOperator { AND, OR, NOT, TERM, PHRASE };

struct QueryNode {
    QueryOperator op;
    std::string term;                // TERM / PHRASE text as written in the query
    std::vector<QueryNode> children; // AND / OR / NOT operands

    QueryNode(QueryOperator o) : op(o) {}
    QueryNode(QueryOperator o, const std::string& t) : op(o), term(t) {}
};

class InvertedIndex {
private:
    std::unordered_map<std::string, std::vector<Posting>> index;  // postings sorted by doc_id
    std::unordered_map<int, std::string> documents;

    Tokenizer tokenizer;

    std::unordered_map<int, int> doc_lengths;  // number of indexed terms (stopwords excluded)
    double total_length = 0.0;                 // sum of all document lengths
    int total_docs = 0;

    int last_doc_id = 0;
    bool has_documents = false;

    QueryNode parse_query(const std::string& query, size_t& pos, int depth) const;
    std::vector<int> evaluate_ast(const QueryNode& node) const;
    std::vector<int> evaluate_phrase(const std::vector<AnalyzedToken>& tokens) const;

    std::vector<int> unite(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<int> intersect(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<int> difference(const std::vector<int>& a, const std::vector<int>& b) const;

public:
    explicit InvertedIndex(TokenizerOptions options = {});

    // doc_id must be strictly increasing across calls.
    void add_document(int doc_id, const std::string& text);

    // Parses and evaluates a query; returns matching doc ids in ascending order.
    // Throws ParseError for malformed queries.
    std::vector<int> search(const std::string& query) const;

    // Parses a query into its syntax tree (throws ParseError). Used by the scorer.
    QueryNode parse(const std::string& query) const;

    // Normalizes text exactly like indexing does (lowercase, optional stopwords/stemming).
    std::vector<AnalyzedToken> analyze(const std::string& text) const;
    const TokenizerOptions& tokenizer_options() const;

    const std::string& get_document(int doc_id) const;

    int get_total_docs() const;
    double get_avg_doc_length() const;
    int get_doc_length(int doc_id) const;
    int get_doc_frequency(const std::string& term) const;                 // documents containing the term
    int get_term_frequency(const std::string& term, int doc_id) const;    // occurrences in a document

    // Number of occurrences of the phrase (terms at their relative positions) in a document.
    int phrase_frequency(const std::vector<AnalyzedToken>& tokens, int doc_id) const;

    // Binary format version 4. load() replaces the whole index and leaves the object
    // unchanged if it throws (missing file, wrong version, truncated or corrupt data).
    void save(const std::string& path) const;
    void load(const std::string& path);
};
