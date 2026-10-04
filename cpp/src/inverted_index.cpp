#include "inverted_index.hpp"
#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <sstream>

void InvertedIndex::add_document(int doc_id, const std::string& text){
    // intersect() relies on every term's postings vector staying sorted by doc_id.
    // Postings are appended in insertion order, so that only holds if callers add
    // documents with strictly increasing doc_id. Enforce that contract here.
    if (has_documents && doc_id <= last_doc_id) {
        throw std::invalid_argument("InvertedIndex::add_document: doc_id must be strictly increasing (got " + std::to_string(doc_id) + " after " + std::to_string(last_doc_id) + ")");
    }
    last_doc_id = doc_id;
    has_documents = true;

    documents[doc_id] = text;

    std::vector<std::string> tokens = tokenizer.tokenize(text);

    std::unordered_map<std::string, int> word_counts;

    for(const std::string& token : tokens){
        word_counts[token]++;
    }

    for(const auto& pair : word_counts){
        const std::string& word = pair.first;
        int freq = pair.second;

        index[word].push_back({doc_id, freq});
    }

    //for bm25 ranking

    doc_lengths[doc_id] = tokens.size();
    total_length += tokens.size();
    total_docs++;
}

std::vector<int> InvertedIndex::intersect(const std::vector<int>& doc_ids, const std::vector<Posting>& postings) const {
    std::vector<int> result;
    size_t i = 0, j = 0;

    while (i < doc_ids.size() && j < postings.size()) {
        int a = doc_ids[i];
        int b = postings[j].doc_id;

        if (a == b) {
            result.push_back(a);
            i++;
            j++;
        } else if (a < b) {
            i++;
        } else {
            j++;
        }
    }

    return result;
}

std::vector<int> InvertedIndex::search(const std::string& query) const {
    std::vector<std::string> query_tokens = tokenizer.tokenize(query);

    if(query_tokens.empty()){
        return {};
    }

    auto it = index.find(query_tokens[0]);
    if(it == index.end()){
        return {};
    }

    std::vector<int> doc_ids;
    doc_ids.reserve(it->second.size());
    for(const Posting& p : it->second){
        doc_ids.push_back(p.doc_id);
    }

    for(size_t t = 1 ; t < query_tokens.size() ; t++){
        auto term_it = index.find(query_tokens[t]);
        if(term_it == index.end()){
            return {};
        }

        doc_ids = intersect(doc_ids, term_it->second);

        if(doc_ids.empty()){
            return {};
        }
    }

    return doc_ids;
}

const std::string& InvertedIndex::get_document(int doc_id) const {
    auto it = documents.find(doc_id);
    if (it == documents.end()) {
        throw std::out_of_range("InvertedIndex::get_document: no document with id " + std::to_string(doc_id));
    }
    return it->second;
}

int InvertedIndex::get_total_docs() const{
    return total_docs;
}


double InvertedIndex::get_avg_doc_length() const{
    if (total_docs == 0) return 0.0;
    return total_length/total_docs;
}

int InvertedIndex::get_doc_length(int doc_id) const {
    auto it = doc_lengths.find(doc_id);
    if (it != doc_lengths.end()) {
        return it->second;
    }
    return 0;
}


int InvertedIndex::get_doc_frequency(const std::string& term) const{
    auto it = index.find(term);

    if(it == index.end()){
        return 0;
    }

    return it->second.size();
}


int InvertedIndex::get_term_frequency(const std::string& term, int doc_id) const{
    auto it = index.find(term);

    if(it == index.end()){
        return 0;
    }

    for(const Posting& p: it->second){
        if(p.doc_id == doc_id){
            return p.frequency;
        } 
    }

    return 0;
}



namespace {
    constexpr uint32_t INDEX_FORMAT_VERSION = 2;
}

void InvertedIndex::save(const std::string& path) const{
    std::ofstream out(path, std::ios::binary);
    if(!out.is_open()){
        throw std::runtime_error("File open");
    }

    out.write(reinterpret_cast<const char*>(&INDEX_FORMAT_VERSION), sizeof(INDEX_FORMAT_VERSION));

    out.write(reinterpret_cast<const char*>(&total_docs), sizeof(total_docs));
    out.write(reinterpret_cast<const char*>(&total_length), sizeof(total_length));

    size_t doc_lengths_count = doc_lengths.size();
    out.write(reinterpret_cast<const char*>(&doc_lengths_count), sizeof(doc_lengths_count));

    for(const auto& pair : doc_lengths){
        int doc_id = pair.first;
        int length = pair.second;

        out.write(reinterpret_cast<const char*>(&doc_id), sizeof(doc_id));
        out.write(reinterpret_cast<const char*>(&length), sizeof(length));
    }

    size_t index_count = index.size();
    out.write(reinterpret_cast<const char*>(&index_count), sizeof(index_count));

    for(const auto& pair : index){
        const std::string& word = pair.first;
        const std::vector<Posting>& postings = pair.second;

        size_t term_len = word.size();
        out.write(reinterpret_cast<const char*>(&term_len), sizeof(term_len));
        out.write(word.data(), term_len);

        size_t posting_size = postings.size();
        out.write(reinterpret_cast<const char*>(&posting_size), sizeof(posting_size));

        for(size_t i = 0; i < posting_size; ++i){
            const Posting& posting = postings[i];
            out.write(reinterpret_cast<const char*>(&posting.doc_id), sizeof(posting.doc_id));
            out.write(reinterpret_cast<const char*>(&posting.frequency), sizeof(posting.frequency));
        }
    }

    size_t documents_count = documents.size();
    out.write(reinterpret_cast<const char*>(&documents_count), sizeof(documents_count));

    for(const auto& pair : documents){
        int doc_id = pair.first;
        const std::string& text = pair.second;

        out.write(reinterpret_cast<const char*>(&doc_id), sizeof(doc_id));

        size_t text_len = text.size();
        out.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
        out.write(text.data(), text_len);
    }
}



void InvertedIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Could not open file for reading");
    }

    uint32_t format_version;
    in.read(reinterpret_cast<char*>(&format_version), sizeof(format_version));
    if (format_version != INDEX_FORMAT_VERSION) {
        throw std::runtime_error("InvertedIndex::load: unsupported index format version " +
            std::to_string(format_version) + " (expected " + std::to_string(INDEX_FORMAT_VERSION) +
            "). Rebuild the index.");
    }

    in.read(reinterpret_cast<char*>(&total_docs), sizeof(total_docs));
    in.read(reinterpret_cast<char*>(&total_length), sizeof(total_length));

    size_t doc_lengths_count;
    in.read(reinterpret_cast<char*>(&doc_lengths_count), sizeof(doc_lengths_count));

    for (size_t i = 0; i < doc_lengths_count; i++) {
        int doc_id, length;
        in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));
        in.read(reinterpret_cast<char*>(&length), sizeof(length));
        doc_lengths[doc_id] = length;
    }

    size_t index_count;
    in.read(reinterpret_cast<char*>(&index_count), sizeof(index_count));

    for (size_t i = 0; i < index_count; i++) {
        size_t term_len;
        in.read(reinterpret_cast<char*>(&term_len), sizeof(term_len));

        std::string word(term_len, '\0');
        in.read(&word[0], term_len);

        size_t posting_size;
        in.read(reinterpret_cast<char*>(&posting_size), sizeof(posting_size));

        std::vector<Posting> postings;
        postings.reserve(posting_size);

        for (size_t j = 0; j < posting_size; j++) {
            Posting p;
            in.read(reinterpret_cast<char*>(&p.doc_id), sizeof(p.doc_id));
            in.read(reinterpret_cast<char*>(&p.frequency), sizeof(p.frequency));
            postings.push_back(p);
        }

        index[word] = postings;
    }

    size_t documents_count;
    in.read(reinterpret_cast<char*>(&documents_count), sizeof(documents_count));

    for (size_t i = 0; i < documents_count; i++) {
        int doc_id;
        in.read(reinterpret_cast<char*>(&doc_id), sizeof(doc_id));

        size_t text_len;
        in.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));

        std::string text(text_len, '\0');
        in.read(&text[0], text_len);

        documents[doc_id] = std::move(text);
    }
}
