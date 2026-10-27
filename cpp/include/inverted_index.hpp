#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "tokenizer.hpp"
#include "types.hpp"

#include <stdexcept>

class ParseError : public std::runtime_error {
public:
    size_t position;
    ParseError(const std::string& msg, size_t pos)
        : std::runtime_error(msg + " (at position " + std::to_string(pos) + ")"),
          position(pos) {}
};


enum class QueryOperator { AND, OR, TERM, PHRASE, NOT};

struct QueryNode {
    QueryOperator op;
    std::string term; 
    std::vector<QueryNode> children; 

    QueryNode(QueryOperator o) : op(o) {}
    QueryNode(QueryOperator o, const std::string& t) : op(o), term(t) {}
};

// why-not: one part of a query, whether one document satisfies it, and the evidence.
// The children are the parts inside an operator, in the order they were written.
struct Explanation {
    std::string part;      // the part of the query, e.g.  tb   "vitamin d"   (^ ... 3 parts)
    bool holds = false;    // true = the document satisfies this part
    std::string reason;    // the evidence, e.g. "occurs 4 times in the document"
    std::vector<Explanation> children;
};

class InvertedIndex {
private:
    std::unordered_map<std::string, std::vector<Posting>> index;
    std::unordered_map<int, std::string> documents;
    
    Tokenizer tokenizer; 


    std::unordered_map<int, int> doc_lengths;
    double total_length = 0.0; //sum of all documents length
    int total_docs = 0; //number of documents

    int last_doc_id = 0;
    bool has_documents = false; 

    QueryNode parse_query(const std::string& query, size_t& pos, int depth = 0) const;
    std::vector<int> evaluate_ast(const QueryNode& node) const;
    std::vector<int> evaluate_phrase(const std::vector<std::string>& tokens) const;
    Explanation explain_node(const QueryNode& node, int doc_id) const;
    Explanation explain_phrase(const std::string& part, const std::vector<std::string>& tokens, int doc_id) const;


    std::vector<int> unite(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<int> intersect(const std::vector<int>& a, const std::vector<int>& b) const;
    std::vector<int> difference(const std::vector<int>& a, const std::vector<int>& b) const;
    

public:
    void add_document(int doc_id, const std::string& text);
    std::vector<int> search(const std::string& query) const;

    // Parses a query into its syntax tree (throws ParseError). The scorer uses it.
    QueryNode parse(const std::string& query) const;

    // How many times the words appear one after another in a document.
    int phrase_frequency(const std::vector<std::string>& tokens, int doc_id) const;

    // why-not(doc, query): evaluates the query for one document only, and records for every part of the query
    // whether the document satisfies it and why. The top's holds is true exactly when search(query) contains
    // the document. Throws ParseError for a bad query, std::out_of_range for a document not in the index.
    Explanation explain(const std::string& query, int doc_id) const;
    // The explanation as indented text, one line per part:  [yes] or [no], the part, the evidence.
    static std::string render(const Explanation& explanation);

    const std::string& get_document(int doc_id) const;

    int get_total_docs() const;
    double get_avg_doc_length() const;
    int get_doc_length(int doc_id) const;
    int get_doc_frequency(const std::string& term) const; //how many documents have this word
    int get_term_frequency(const std::string& term, int doc_id) const; //how many times does a term appear in a document


    void save(const std::string& path) const;
    void load(const std::string& path);



};