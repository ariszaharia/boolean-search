// Runs Boolean queries (any query of the engine's language) on a corpus and writes, for each query, the
// ranked top k, the size of its whole Boolean set, or the reason it failed. Used by E0 now and E1/E2 later.
//
// Usage: boolean_cli <corpus.tsv> <queries.tsv> <run.tsv> <sizes.tsv> <failures.tsv> [k = 100]
//   corpus.tsv    one document per line:  id <TAB> text   (ids increasing)
//   queries.tsv   one query per line:     query_id <TAB> query   e.g.  q1 <TAB> (- (^ a b) c)
//   run.tsv       one result per line:    query_id <TAB> rank <TAB> doc_id <TAB> score   (top k, BM25)
//   sizes.tsv     one line per query that ran: query_id <TAB> set_size   (0 = the query matched nothing)
//   failures.tsv  one line per query that could not run: query_id <TAB> position <TAB> message
// A failed query has a line in failures.tsv only, never in sizes.tsv (hypotheses.md, H1 clarification).
// BM25 with k1 = 1.2, b = 0.75, scored from the parsed query; ties broken by doc id (BM25Scorer).
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: boolean_cli <corpus.tsv> <queries.tsv> <run.tsv> <sizes.tsv> <failures.tsv> [k]\n";
        return 1;
    }
    const std::string corpus_path = argv[1];
    const std::string queries_path = argv[2];
    const size_t k = (argc > 6) ? static_cast<size_t>(std::stoul(argv[6])) : 100;

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
    std::ofstream run(argv[3]), sizes(argv[4]), failures(argv[5]);
    if (!run || !sizes || !failures) { std::cerr << "Cannot write the output files\n"; return 1; }
    run << std::setprecision(9);

    std::string line;
    int n_ran = 0, n_empty = 0, n_failed = 0;
    while (std::getline(queries, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string qid = line.substr(0, tab);
        const std::string query = line.substr(tab + 1);

        std::vector<int> matched;
        try {
            matched = index.search(query);
        } catch (const ParseError& e) {
            failures << qid << '\t' << e.position << '\t' << e.what() << '\n';
            n_failed++;
            continue;
        }
        sizes << qid << '\t' << matched.size() << '\n';
        n_ran++;
        if (matched.empty()) { n_empty++; continue; }

        std::vector<SearchResult> ranked = scorer.score(matched, query);
        size_t shown = std::min(k, ranked.size());
        for (size_t i = 0; i < shown; i++) {
            run << qid << '\t' << (i + 1) << '\t' << ranked[i].doc_id << '\t' << ranked[i].score << '\n';
        }
    }
    std::cerr << "Ran " << n_ran << " queries (" << n_empty << " matched nothing), " << n_failed << " failed\n";
    return 0;
}
