// why-not(doc, query) for many requests at once: for each one, why a document is or is not in a query's result.
// Used for H5 (hypotheses.md): an annotator names the cause of a Boolean failure from the query and this output.
//
// Usage: why_not_cli <corpus.tsv> <requests.tsv> <explanations.txt> <summary.tsv> [k = 100]
//   corpus.tsv        one document per line:  id <TAB> text   (ids increasing), the same file the runs used
//   requests.tsv      one request per line:   request_id <TAB> doc_id <TAB> query
//   explanations.txt  one block per request: the verdict, then the query as a tree with [yes] or [no] at every
//                     part and the evidence (InvertedIndex::explain)
//   summary.tsv       one line per request:   request_id <TAB> doc_id <TAB> verdict <TAB> set_size <TAB> rank
//     verdict:  in_top_k     in the query's set and among its top k by BM25 (it reaches the reranker)
//               below_top_k  in the set, but ranked after k: cut before the reranker (a ranking failure)
//               not_in_set   the query's set does not contain it (a filtering failure; the tree says which part)
//               parse_error  the query does not parse;  unknown_doc  no such document in the corpus
//     set_size and rank are "-" when they do not apply. BM25 as in boolean_cli: k1 = 1.2, b = 0.75.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: why_not_cli <corpus.tsv> <requests.tsv> <explanations.txt> <summary.tsv> [k]\n";
        return 1;
    }
    const std::string corpus_path = argv[1];
    const std::string requests_path = argv[2];
    const size_t k = (argc > 5) ? static_cast<size_t>(std::stoul(argv[5])) : 100;

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

    std::ifstream requests(requests_path);
    if (!requests) { std::cerr << "Cannot open " << requests_path << "\n"; return 1; }
    std::ofstream explanations(argv[3]), summary(argv[4]);
    if (!explanations || !summary) { std::cerr << "Cannot write the output files\n"; return 1; }

    std::string line;
    int n_requests = 0;
    while (std::getline(requests, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t tab1 = line.find('\t');
        size_t tab2 = (tab1 == std::string::npos) ? std::string::npos : line.find('\t', tab1 + 1);
        if (tab2 == std::string::npos) continue;
        const std::string rid = line.substr(0, tab1);
        const std::string doc_text = line.substr(tab1 + 1, tab2 - tab1 - 1);
        const std::string query = line.substr(tab2 + 1);
        n_requests++;
        int doc_id = -1;   // no document has a negative id, so a bad id becomes unknown_doc below
        try { doc_id = std::stoi(doc_text); } catch (const std::exception&) {}

        explanations << "=== " << rid << "   document " << doc_id << "\n" << "query: " << query << "\n";

        Explanation why;
        try {
            why = index.explain(query, doc_id);
        } catch (const ParseError& e) {
            explanations << "verdict: the query does not parse: " << e.what() << "\n\n";
            summary << rid << '\t' << doc_id << "\tparse_error\t-\t-\n";
            continue;
        } catch (const std::out_of_range&) {
            explanations << "verdict: there is no document " << doc_id << " in the corpus\n\n";
            summary << rid << '\t' << doc_id << "\tunknown_doc\t-\t-\n";
            continue;
        }

        std::vector<int> matched = index.search(query);
        std::string verdict, rank_text = "-";
        if (!why.holds) {
            verdict = "not_in_set";
            explanations << "verdict: NOT in the query's set (the set has " << matched.size() << " documents)\n";
        } else {
            // Its BM25 rank inside the set, exactly as boolean_cli ranks the set.
            std::vector<SearchResult> ranked = scorer.score(matched, query);
            size_t rank = 0;
            for (size_t i = 0; i < ranked.size(); i++) {
                if (ranked[i].doc_id == doc_id) { rank = i + 1; break; }
            }
            rank_text = std::to_string(rank);
            verdict = (rank <= k) ? "in_top_k" : "below_top_k";
            explanations << "verdict: in the query's set (" << matched.size() << " documents), BM25 rank " << rank
                         << (rank <= k ? ": within the top " + std::to_string(k)
                                       : ": below the top " + std::to_string(k) + ", so it is cut before the reranker")
                         << "\n";
        }
        explanations << InvertedIndex::render(why) << "\n";
        summary << rid << '\t' << doc_id << '\t' << verdict << '\t' << matched.size() << '\t' << rank_text << '\n';
    }
    std::cerr << "Explained " << n_requests << " requests\n";
    return 0;
}
