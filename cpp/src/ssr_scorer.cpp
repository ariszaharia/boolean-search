#include "ssr_scorer.hpp"
#include <algorithm>
#include <climits>
#include <iterator>

SSRScorer::SSRScorer(const InvertedIndex& idx, double c) : index(idx), C(c) {}

std::vector<Interval> SSRScorer::minimal(std::vector<Interval> candidates) {
    // An interval is kept if no other interval lies inside it. Going through them by end (and, for equal ends, by
    // later start first), every interval already seen ends no later; one of them lies inside the current interval
    // exactly when it starts no earlier. So the current one is kept when its start is later than every start seen.
    std::sort(candidates.begin(), candidates.end(), [](const Interval& a, const Interval& b) {
        if (a.second != b.second) return a.second < b.second;
        return a.first > b.first;
    });
    std::vector<Interval> kept;
    int latest_start = INT_MIN;
    for (const Interval& iv : candidates) {
        if (iv.first > latest_start) {
            kept.push_back(iv);
            latest_start = iv.first;
        }
    }
    // kept is sorted by end, and its starts increase too (no interval contains another).
    return kept;
}

std::vector<Interval> SSRScorer::words_intervals(const std::vector<std::string>& words, int doc_id) const {
    if (words.empty()) return {};
    std::vector<const std::vector<int>*> pos;
    for (const std::string& w : words) {
        const std::vector<int>* p = index.positions(w, doc_id);
        if (p == nullptr) return {};
        pos.push_back(p);
    }
    std::vector<Interval> out;
    const int n = static_cast<int>(words.size());
    for (int start : *pos[0]) {
        bool all = true;
        for (int i = 1; i < n && all; ++i) {
            all = std::binary_search(pos[i]->begin(), pos[i]->end(), start + i);
        }
        if (all) out.push_back({start, start + n - 1});
    }
    return out;   // already minimal: same length, increasing starts
}

std::vector<Interval> SSRScorer::intervals(const QueryNode& node, int doc_id) const {
    if (node.op == QueryOperator::TERM || node.op == QueryOperator::PHRASE) {
        // A bare term that the tokenizer splits (g-to-a) is matched as a phrase, as in the engine's search.
        return words_intervals(index.tokenize(node.term), doc_id);
    }
    if (node.op == QueryOperator::NOT) {
        if (node.children.empty()) return {};
        return intervals(node.children[0], doc_id);
    }
    std::vector<std::vector<Interval>> parts;
    for (const QueryNode& child : node.children) parts.push_back(intervals(child, doc_id));

    if (node.op == QueryOperator::OR) {
        std::vector<Interval> all;
        for (const auto& p : parts) all.insert(all.end(), p.begin(), p.end());
        return minimal(all);
    }
    // AND. Every shortest stretch [p, q] ends where some part's stretch ends. For each such end q, take from every
    // part its stretch that ends at or before q and starts latest (the last one, since a part's stretches are
    // sorted by end with increasing starts); the candidate is [earliest of those starts, q]. Every candidate
    // satisfies the AND, every shortest stretch is one of them, and minimal() drops the rest.
    if (parts.empty()) return {};
    for (const auto& p : parts) if (p.empty()) return {};
    std::vector<int> ends;
    for (const auto& p : parts) for (const Interval& iv : p) ends.push_back(iv.second);
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    std::vector<Interval> candidates;
    for (int q : ends) {
        int start = INT_MAX;
        bool ok = true;
        for (const auto& p : parts) {
            // Last stretch of this part with end <= q.
            auto it = std::upper_bound(p.begin(), p.end(), q, [](int value, const Interval& iv) {
                return value < iv.second;
            });
            if (it == p.begin()) { ok = false; break; }
            start = std::min(start, std::prev(it)->first);
        }
        if (ok) candidates.push_back({start, q});
    }
    return minimal(candidates);
}

std::vector<Interval> SSRScorer::shortest_substrings(const std::string& query, int doc_id) const {
    return intervals(index.parse(query), doc_id);
}

std::vector<SearchResult> SSRScorer::score(const std::vector<int>& matched_docs, const std::string& query) const {
    std::vector<SearchResult> results;
    if (matched_docs.empty()) return results;
    QueryNode ast = index.parse(query);
    results.reserve(matched_docs.size());
    for (int doc_id : matched_docs) {
        double s = 0.0;
        for (const Interval& iv : intervals(ast, doc_id)) s += 1.0 / (C + iv.second - iv.first);
        results.push_back({doc_id, s});
    }
    std::sort(results.begin(), results.end(), [](const SearchResult& a, const SearchResult& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.doc_id < b.doc_id;
    });
    return results;
}
