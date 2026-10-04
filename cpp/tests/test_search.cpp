#include <iostream>
#include <string>
#include <vector>
#include "inverted_index.hpp"

void print_results(const std::string& query, const std::vector<int>& results) {
    std::cout << "Query: \"" << query << "\"\n";
    if (results.empty()) {
        std::cout << " No matching documents found.\n";
    } else {
        std::cout << " Found in Document IDs: ";
        for (int doc_id : results) {
            std::cout << doc_id << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    InvertedIndex index;

    std::cout << " PHASE 1: INDEXING \n";
    std::cout << "Reading and processing documents...\n";
    
    index.add_document(1, "Hello world, this is a fast C++ search engine!");
    index.add_document(2, "Hello from the world of information retrieval.");
    index.add_document(3, "A fast search engine relies on an inverted index.");
    index.add_document(4, "This document only contains the word hello.");
    
    std::cout << "Indexing complete! The unordered_map is fully built.\n\n";

    std::cout << " PHASE 2: QUERYING \n";
    
    std::vector<int> res1 = index.search("hello world");
    print_results("hello world", res1);

    std::vector<int> res2 = index.search("fast search engine");
    print_results("fast search engine", res2);

    std::vector<int> res3 = index.search("hello");
    print_results("hello", res3);

    // Test 4: The AND logic test
    std::vector<int> res4 = index.search("hello elephant");
    print_results("hello elephant", res4);

    return 0;
}