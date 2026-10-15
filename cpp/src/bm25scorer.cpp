#include "bm25scorer.hpp"
#include <cmath>
#include <algorithm>

BM25Scorer::BM25Scorer(const InvertedIndex& idx, double k1_val, double b_val)
    : index(idx), k1(k1_val), b(b_val) {}

double BM25Scorer::score_words(const std::vector<std::string>& words, int doc_id, int N, double avgdl) const {
    if (words.empty()) return 0.0;

    int tf = (words.size() == 1) ? index.get_term_frequency(words[0], doc_id)
                                 : index.phrase_frequency(words, doc_id);
    if (tf == 0) return 0.0;

    double idf = 0.0;
    for (const std::string& word : words) {
        int n_qi = index.get_doc_frequency(word);
        idf += std::log(1.0 + (N - n_qi + 0.5) / (n_qi + 0.5));
    }

    int doc_length = index.get_doc_length(doc_id);
    double numerator = tf * (k1 + 1.0);
    double denominator = tf + k1 * (1.0 - b + b * (static_cast<double>(doc_length) / avgdl));

    return idf * (numerator / denominator);
}

double BM25Scorer::score_node(const QueryNode& node, int doc_id, int N, double avgdl) const {
    if (node.op == QueryOperator::TERM || node.op == QueryOperator::PHRASE) {
        return score_words(tokenizer.tokenize(node.term), doc_id, N, avgdl);
    }
    if (node.op == QueryOperator::NOT) {
        if (node.children.empty()) return 0.0;
        return score_node(node.children[0], doc_id, N, avgdl);
    }
    double total = 0.0;
    for (const QueryNode& child : node.children) {
        total += score_node(child, doc_id, N, avgdl);
    }
    return total;
}

std::vector<SearchResult> BM25Scorer::score(const std::vector<int>& matched_docs, const std::string& query) const {
    std::vector<SearchResult> results;
    if (matched_docs.empty()) return results;

    int N = index.get_total_docs();
    if (N == 0) return results; 

    double avgdl = index.get_avg_doc_length();

    QueryNode ast = index.parse(query); 

    results.reserve(matched_docs.size());
    for (int doc_id : matched_docs) {
        results.push_back({doc_id, score_node(ast, doc_id, N, avgdl)});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.doc_id < b.doc_id;
    });

    return results;
}
