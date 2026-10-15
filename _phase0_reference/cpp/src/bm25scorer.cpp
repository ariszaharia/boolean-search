#include "bm25scorer.hpp"

#include <algorithm>
#include <cmath>

BM25Scorer::BM25Scorer(const InvertedIndex& idx, double k1_val, double b_val)
    : index(idx), k1(k1_val), b(b_val) {}

namespace {

// Leaves that contribute to the score: everything except the excluded operands of NOT.
void collect_leaves(const QueryNode& n, std::vector<const QueryNode*>& out) {
    if (n.op == QueryOperator::TERM || n.op == QueryOperator::PHRASE) {
        out.push_back(&n);
        return;
    }
    if (n.op == QueryOperator::NOT) {
        if (!n.children.empty()) collect_leaves(n.children[0], out);
        return;
    }
    for (const auto& c : n.children) collect_leaves(c, out);
}

struct Clause {
    std::vector<AnalyzedToken> tokens;
    double idf = 0.0;
    bool usable = true;
};

}  // namespace

std::vector<SearchResult> BM25Scorer::score(const std::vector<int>& matched_docs, const std::string& query) const {
    std::vector<SearchResult> results;
    if (matched_docs.empty()) return results;

    const int N = index.get_total_docs();
    if (N == 0) return results;
    const double avgdl = index.get_avg_doc_length();

    QueryNode ast = index.parse(query);
    std::vector<const QueryNode*> leaves;
    collect_leaves(ast, leaves);

    std::vector<Clause> clauses;
    clauses.reserve(leaves.size());
    for (const QueryNode* leaf : leaves) {
        Clause c;
        c.tokens = index.analyze(leaf->term);
        if (c.tokens.empty()) c.usable = false;
        for (const AnalyzedToken& t : c.tokens) {
            int df = index.get_doc_frequency(t.term);
            if (df == 0) {
                c.usable = false;
                break;
            }
            c.idf += std::log(1.0 + (N - df + 0.5) / (df + 0.5));
        }
        clauses.push_back(std::move(c));
    }

    results.reserve(matched_docs.size());
    for (int doc_id : matched_docs) {
        const double dl = index.get_doc_length(doc_id);
        double doc_score = 0.0;
        for (const Clause& c : clauses) {
            if (!c.usable) continue;
            int tf = c.tokens.size() == 1 ? index.get_term_frequency(c.tokens[0].term, doc_id)
                                          : index.phrase_frequency(c.tokens, doc_id);
            if (tf == 0) continue;
            doc_score += c.idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (dl / avgdl)));
        }
        results.push_back({doc_id, doc_score});
    }

    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& o) {
        if (a.score != o.score) return a.score > o.score;
        return a.doc_id < o.doc_id;  // deterministic order for equal scores
    });
    return results;
}
