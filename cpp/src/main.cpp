#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <filesystem>
#include "inverted_index.hpp"
#include "bm25scorer.hpp"

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream buf;
    buf << in.rdbuf();
    return buf.str();
}

// Loads every .txt file in `dir`, sorted by filename for reproducible doc_ids.
int build_index_from_directory(InvertedIndex& index, const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    int doc_id = 0;
    for (const fs::path& file : files) {
        index.add_document(doc_id, read_file(file));
        doc_id++;
    }
    return doc_id;
}

void run_query(const InvertedIndex& index, const std::string& query) {
    auto start = std::chrono::steady_clock::now();
    std::vector<int> matches = index.search(query);
    BM25Scorer scorer(index);
    std::vector<SearchResult> ranked = scorer.score(matches, query);
    auto elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

    std::cout << "Query \"" << query << "\": " << matches.size() << " matches in " << elapsed << " ms\n";
    for (size_t i = 0; i < ranked.size() && i < 3; i++) {
        std::string snippet = index.get_document(ranked[i].doc_id).substr(0, 80);
        std::cout << "  #" << i << " doc " << ranked[i].doc_id << " score=" << ranked[i].score
                   << " : " << snippet << "...\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    fs::path corpus_dir = argc > 1 ? fs::path(argv[1]) : fs::path("data/newsgroups");
    const std::string index_path = "newsgroups_index.bin";

    InvertedIndex index;

    auto build_start = std::chrono::steady_clock::now();
    int n_docs = build_index_from_directory(index, corpus_dir);
    auto build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count();

    std::cout << "Indexed " << n_docs << " documents in " << build_ms << " ms\n";
    std::cout << "avg_doc_length: " << index.get_avg_doc_length() << "\n";

    auto save_start = std::chrono::steady_clock::now();
    index.save(index_path);
    auto save_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - save_start).count();
    std::cout << "Saved to " << index_path << " in " << save_ms << " ms ("
              << fs::file_size(index_path) / 1024.0 / 1024.0 << " MB)\n";

    InvertedIndex loaded;
    auto load_start = std::chrono::steady_clock::now();
    loaded.load(index_path);
    auto load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    std::cout << "Loaded in " << load_ms << " ms\n\n";

    run_query(loaded, "hockey");
    run_query(loaded, "the");
    run_query(loaded, "graphics card driver");

    return 0;
}
