#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

// Helper function that runs both retrieval and scoring, then prints the top K results
void print_scored_results(const std::string& query, const InvertedIndex& index, const BM25Scorer& scorer, int top_k = 3) {
    std::cout << "Query: " << query << "\n";
    
    // 1. Get exact boolean matches (The C++ AST Engine)
    std::vector<int> matched_docs = index.search(query);
    
    if (matched_docs.empty()) {
        std::cout << "Results: [No results found]\n--------------------------\n";
        return;
    }

    // 2. Score and rank the exact matches (BM25)
    std::vector<SearchResult> ranked_results = scorer.score(matched_docs, query);

    std::cout << "Found " << ranked_results.size() << " matching documents. Showing top " << top_k << ":\n";
    
    int count = 0;
    for (const auto& res : ranked_results) {
        if (count >= top_k) break;
        
        // Extract a short snippet of the text to keep the terminal readable
        std::string full_text = index.get_document(res.doc_id);
        std::string snippet = full_text.length() > 100 ? full_text.substr(0, 100) + "..." : full_text;
        
        std::cout << " -> Doc ID: " << res.doc_id 
                  << " | Score: " << res.score 
                  << " | Snippet: " << snippet << "\n";
        count++;
    }
    std::cout << "--------------------------\n";
}

// Helper function to load the TSV file exported by Python
void load_tsv_corpus(const std::string& filepath, InvertedIndex& index) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filepath << ". Are you running from the correct directory?\n";
        return;
    }

    std::string line;
    int count = 0;
    
    std::cout << "Loading SciFact corpus into C++ engine...\n";
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string id_str, text;
        
        if (std::getline(iss, id_str, '\t') && std::getline(iss, text)) {
            try {
                int doc_id = std::stoi(id_str);
                index.add_document(doc_id, text);
                count++;
            } catch (...) {
                continue; // Skip lines with invalid/missing IDs
            }
        }
    }
    std::cout << "Successfully loaded " << count << " documents.\n\n";
}

int main() {
    InvertedIndex index;

    // 1. Load the real SciFact data. 
    // Assuming you compile in the root 'licenta' folder and run as ./test_engine
    // or compile in 'src' and run. Adjust path if necessary.
    load_tsv_corpus("../data/scifact_5k.tsv", index);

    // If no documents loaded, we shouldn't continue
    if (index.get_total_docs() == 0) {
        std::cout << "Exiting due to empty corpus. Check path to TSV.\n";
        return 1;
    }

    // 2. Initialize the Scorer
    BM25Scorer scorer(index);

    std::cout << "=== SCIFACT EVALUATION ===\n\n";

    // 3. Test realistic scientific queries
    // Single term
    print_scored_results("mutation", index, scorer); 
    
    // Strict Boolean AND (Documents MUST contain both 'immune' and 'response')
    print_scored_results("(^ immune response)", index, scorer); 
    
    // Exact Phrase Match (Documents MUST contain "stem cells" side-by-side)
    print_scored_results("\"stem cells\"", index, scorer);

    return 0;
}