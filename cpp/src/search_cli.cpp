// Runs the engine on a file of queries and writes the ranked results.
//
// Usage: search_cli <corpus.tsv> <queries_words.tsv> <out.tsv> [k = 100]
//   corpus.tsv         one document per line: id <TAB> text   (ids increasing)
//   queries_words.tsv  one query per line:    query_id <TAB> word word word ...
//   out.tsv            one result per line:   query_id <TAB> rank <TAB> doc_id <TAB> score
//
// Each query is run as (+ word1 word2 ...): a document matches if it contains any
// of the words, and BM25 adds up the score of every word (k1 = 1.2, b = 0.75).
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: search_cli <corpus.tsv> <queries_words.tsv> <out.tsv> [k]\n";
        return 1;
    }
    const std::string corpus_path = argv[1];
    const std::string queries_path = argv[2];
    const std::string out_path = argv[3];
    const size_t k = (argc > 4) ? static_cast<size_t>(std::stoul(argv[4])) : 100;

    InvertedIndex index;
    {
        std::ifstream in(corpus_path);
        if (!in) { std::cerr << "Cannot open " << corpus_path << "\n"; return 1; }
        std::string line;
        int loaded = 0;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            size_t tab = line.find('\t');
            if (tab == std::string::npos) continue;
            index.add_document(std::stoi(line.substr(0, tab)), line.substr(tab + 1));
            loaded++;
        }
        std::cerr << "Loaded " << loaded << " documents\n";
    }

    BM25Scorer scorer(index, 1.2, 0.75);

    std::ifstream queries(queries_path);
    if (!queries) { std::cerr << "Cannot open " << queries_path << "\n"; return 1; }
    std::ofstream out(out_path);
    if (!out) { std::cerr << "Cannot write " << out_path << "\n"; return 1; }

    out << std::setprecision(9);

    std::string line;
    int n_queries = 0;
    while (std::getline(queries, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string qid = line.substr(0, tab);

        std::istringstream words(line.substr(tab + 1));
        std::string word, query = "(+";
        int n_words = 0;
        while (words >> word) { query += " " + word; n_words++; }
        if (n_words == 0) continue;
        query += ")";

        std::vector<SearchResult> ranked = scorer.score(index.search(query), query);
        size_t shown = std::min(k, ranked.size());
        for (size_t i = 0; i < shown; i++) {
            out << qid << '\t' << (i + 1) << '\t' << ranked[i].doc_id << '\t' << ranked[i].score << '\n';
        }
        n_queries++;
    }
    std::cerr << "Wrote results for " << n_queries << " queries to " << out_path << "\n";
    return 0;
}
