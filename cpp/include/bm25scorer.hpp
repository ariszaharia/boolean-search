#pragma once
#include <vector>
#include <string>
#include "inverted_index.hpp"
#include "tokenizer.hpp"

struct SearchResult {
    int doc_id;
    double score;
};

class BM25Scorer {
private:
    const InvertedIndex& index;
    Tokenizer tokenizer;
    
    double k1;
    double b;

public:
    BM25Scorer(const InvertedIndex& idx, double k1_val = 1.5, double b_val = 0.75);
    
    std::vector<SearchResult> score(const std::vector<int>& matched_docs, const std::string& query) const;
};