// Demo / smoke test: loads a TSV corpus (doc_id <TAB> text) and runs a few queries.
//
//   ./demo [--corpus path] [--stem] [--stopwords] [query ...]
//
// With no query arguments a few built-in queries are run. Run it from the cpp/
// folder so the default corpus path resolves.

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bm25scorer.hpp"
#include "inverted_index.hpp"

namespace {

void print_scored_results(const std::string& query, const InvertedIndex& index, const BM25Scorer& scorer,
                          int top_k = 3) {
    std::cout << "Query: " << query << "\n";
    try {
        std::vector<int> matched = index.search(query);
        if (matched.empty()) {
            std::cout << "Results: [No results found]\n--------------------------\n";
            return;
        }
        std::vector<SearchResult> ranked = scorer.score(matched, query);
        std::cout << "Found " << ranked.size() << " matching documents. Showing top " << top_k << ":\n";
        int count = 0;
        for (const auto& res : ranked) {
            if (count++ >= top_k) break;
            const std::string& text = index.get_document(res.doc_id);
            std::string snippet = text.size() > 100 ? text.substr(0, 100) + "..." : text;
            std::cout << " -> Doc ID: " << res.doc_id << " | Score: " << res.score << " | Snippet: " << snippet << "\n";
        }
    } catch (const ParseError& e) {
        std::cout << "Parse error: " << e.what() << "\n";
    }
    std::cout << "--------------------------\n";
}

int load_tsv_corpus(const std::string& filepath, InvertedIndex& index) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: could not open " << filepath << ". Run from the cpp/ folder or pass --corpus.\n";
        return 0;
    }
    std::string line;
    int count = 0;
    while (std::getline(file, line)) {
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        try {
            int doc_id = std::stoi(line.substr(0, tab));
            index.add_document(doc_id, line.substr(tab + 1));
            count++;
        } catch (const std::invalid_argument&) {
            continue;  // header line or bad id
        } catch (const std::out_of_range&) {
            continue;
        }
    }
    return count;
}

}  // namespace

int main(int argc, char** argv) {
    std::string corpus = "../data/scifact.tsv";
    TokenizerOptions options;
    std::vector<std::string> queries;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--stem") options.stem = true;
        else if (a == "--stopwords") options.stopwords = true;
        else if (a == "--corpus" && i + 1 < argc) corpus = argv[++i];
        else queries.push_back(a);
    }

    InvertedIndex index(options);
    std::cout << "Loading corpus (stem=" << options.stem << ", stopwords=" << options.stopwords << ")...\n";
    int loaded = load_tsv_corpus(corpus, index);
    std::cout << "Loaded " << loaded << " documents.\n\n";
    if (index.get_total_docs() == 0) return 1;

    BM25Scorer scorer(index);  // k1 = 1.2, b = 0.75

    if (queries.empty()) {
        queries = {"mutation", "(^ immune response)", "\"stem cells\"", "(- \"stem cells\" mutation)"};
    }
    for (const auto& q : queries) print_scored_results(q, index, scorer);
    return 0;
}
