#pragma once

#include <string>
#include <vector>

#include "inverted_index.hpp"

struct SearchResult {
    int doc_id;
    double score;
};

// BM25 ranking of documents that already satisfy a Boolean query.
//
// The query is parsed again and scored from its syntax tree:
//   * every TERM leaf contributes idf * tf * (k1+1) / (tf + k1*(1-b+b*dl/avgdl));
//   * a PHRASE leaf (or a term that normalizes to several words, such as "T-cell")
//     is scored as one unit: tf = phrase frequency, idf = sum of the words' idfs;
//   * operands of NOT after the first are excluded, so they add nothing;
//   * AND and OR only decide which documents match; scores are summed over leaves,
//     so a document matching more leaves ranks higher.
// idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)) (the Lucene/Pyserini variant).
// Results are sorted by descending score, ties broken by ascending doc_id.
class BM25Scorer {
private:
    const InvertedIndex& index;
    double k1;
    double b;

public:
    BM25Scorer(const InvertedIndex& idx, double k1_val = 1.2, double b_val = 0.75);

    // Throws ParseError if `query` is malformed.
    std::vector<SearchResult> score(const std::vector<int>& matched_docs, const std::string& query) const;
};
