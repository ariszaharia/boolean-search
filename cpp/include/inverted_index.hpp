#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "tokenizer.hpp"
#include "types.hpp"


class InvertedIndex {
private:
    std::unordered_map<std::string, std::vector<Posting>> index;
    std::unordered_map<int, std::string> documents;
    
    Tokenizer tokenizer; 

    std::vector<int> intersect(const std::vector<int>& doc_ids, const std::vector<Posting>& postings) const;

    std::unordered_map<int, int> doc_lengths;
    double total_length = 0.0; //sum of all documents length
    int total_docs = 0; //number of documents

    int last_doc_id = 0;
    bool has_documents = false; 

public:
    void add_document(int doc_id, const std::string& text);
    std::vector<int> search(const std::string& query) const;

    const std::string& get_document(int doc_id) const;

    int get_total_docs() const;
    double get_avg_doc_length() const;
    int get_doc_length(int doc_id) const;
    int get_doc_frequency(const std::string& term) const; //how many documents have this word
    int get_term_frequency(const std::string& term, int doc_id) const; //how many times does a term appear in a document


    void save(const std::string& path) const;
    void load(const std::string& path);

};