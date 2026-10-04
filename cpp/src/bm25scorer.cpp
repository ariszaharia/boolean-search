#include "bm25scorer.hpp"
#include <cmath>    
#include <algorithm>

BM25Scorer::BM25Scorer(const InvertedIndex& idx, double k1_val, double b_val): index(idx), k1(k1_val), b(b_val) {}

std::vector<SearchResult> BM25Scorer::score(const std::vector<int>& matched_docs, const std::string& query) const {
    std::vector<SearchResult> results;
    std::vector<std::string> query_tokens = tokenizer.tokenize(query);
    
    int N = index.get_total_docs();
    double avgdl = index.get_avg_doc_length();

    for (int doc_id : matched_docs) {
        double doc_score = 0.0;
        int doc_len = index.get_doc_length(doc_id);

        for (const std::string& term : query_tokens) {
            int df = index.get_doc_frequency(term);
            if (df == 0) continue; 

            int tf = index.get_term_frequency(term, doc_id);
            if (tf == 0) continue; 

            double idf = std::log(1.0 + (N - df + 0.5) / (df + 0.5));

            double numerator = tf * (k1 + 1.0);
            double denominator = tf + k1 * (1.0 - b + b * (static_cast<double>(doc_len) / avgdl));
            
            doc_score += idf * (numerator / denominator);
        }

        results.push_back({doc_id, doc_score});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.score > b.score; 
    });

    return results;
}