#pragma once
// Shortest substring ranking (SSR), as in Clarke & Smucker, "Boolean queries are all you need?" (arXiv 2607.11362,
// section 3) and Clarke & Cormack, "Shortest-substring retrieval and ranking" (TOIS 2000).
//
// Every solution of a query in a document is a shortest substring: a stretch of words [p, q] that satisfies the
// query and contains no shorter stretch that does. A document's score is the sum, over its shortest substrings, of
//     1 / (C + q - p),        C = 42 by default (the paper's value),
// so documents with more and shorter matches rank higher. No global statistics or term weights are used.
//
// Positions are word numbers inside one document (the engine's tokens), so a stretch never crosses documents.
// Per operator, the shortest substrings of a node are:
//   word        every position p, as [p, p]
//   phrase      every place the words follow each other, as [start, start + words - 1]
//   OR (+)      the stretches of all its parts, without those that contain another one
//   AND (^)     the shortest stretches that contain a stretch of every part
//   NOT (-)     the stretches of its first part. The engine's NOT works per document (a document that matches an
//               excluded part is not in the set at all), so the excluded parts only decide which documents are scored.
#include <utility>
#include <vector>
#include <string>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"   // SearchResult

using Interval = std::pair<int, int>;   // [first word, last word], both included

class SSRScorer {
private:
    const InvertedIndex& index;
    double C;

    std::vector<Interval> intervals(const QueryNode& node, int doc_id) const;
    std::vector<Interval> words_intervals(const std::vector<std::string>& words, int doc_id) const;

public:
    explicit SSRScorer(const InvertedIndex& idx, double c = 42.0);

    // The shortest substrings of a whole query in one document, sorted by start (and so by end).
    std::vector<Interval> shortest_substrings(const std::string& query, int doc_id) const;

    // Scores the matched documents (from index.search) and sorts them: higher score first, ties by lower doc_id.
    std::vector<SearchResult> score(const std::vector<int>& matched_docs, const std::string& query) const;

    // Keeps only the intervals that contain no other one (removing duplicates); the result is sorted by start,
    // and then also by end. Public for the tests.
    static std::vector<Interval> minimal(std::vector<Interval> candidates);
};
