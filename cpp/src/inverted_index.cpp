#include "inverted_index.hpp"
#include <cstdint>
#include <stdexcept>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <cctype>
#include <algorithm>

void InvertedIndex::add_document(int doc_id, const std::string& text) {
    if (has_documents && doc_id <= last_doc_id) {
        throw std::invalid_argument("InvertedIndex::add_document: doc_id must be strictly increasing (got " + std::to_string(doc_id) + " after " + std::to_string(last_doc_id) + ")");
    }
    last_doc_id = doc_id;
    has_documents = true;

    documents[doc_id] = text;

    std::vector<std::string> tokens = tokenizer.tokenize(text);

    std::unordered_map<std::string, std::vector<int>> word_positions;

    for (int pos = 0; pos < tokens.size(); ++pos) {
        word_positions[tokens[pos]].push_back(pos);
    }

    for (const auto& pair : word_positions) {
        const std::string& word = pair.first;
        const std::vector<int>& positions = pair.second;
        
        int freq = positions.size();

        index[word].push_back({doc_id, freq, positions});
    }

    doc_lengths[doc_id] = tokens.size();
    total_length += tokens.size();
    total_docs++;
}


// Set Union (OR) - Keeps all unique document IDs from both lists
std::vector<int> InvertedIndex::unite(const std::vector<int>& a, const std::vector<int>& b) const {
    std::vector<int> result;
    result.reserve(a.size() + b.size());
    size_t i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            result.push_back(a[i]);
            i++; j++;
        } else if (a[i] < b[j]) {
            result.push_back(a[i++]);
        } else {
            result.push_back(b[j++]);
        }
    }
    while (i < a.size()) result.push_back(a[i++]);
    while (j < b.size()) result.push_back(b[j++]);

    return result;
}

std::vector<int> InvertedIndex::intersect(const std::vector<int>& a, const std::vector<int>& b) const {
    std::vector<int> result;
    size_t i = 0, j = 0;

    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            result.push_back(a[i]);
            i++; j++;
        } else if (a[i] < b[j]) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}




QueryNode InvertedIndex::parse_query(const std::string& query, size_t& pos) const {
    // Skip leading whitespace
    while (pos < query.length() && std::isspace(query[pos])) pos++;
    if (pos >= query.length()) return QueryNode(QueryOperator::TERM, "");

    // Handle nested operators: (^ ...) or (+ ...)
    if (query[pos] == '(') {
        pos++; 
        while (pos < query.length() && std::isspace(query[pos])) pos++;
        
        QueryOperator op = QueryOperator::AND;
        if (query[pos] == '^') { op = QueryOperator::AND; pos++; }
        else if (query[pos] == '+') { op = QueryOperator::OR; pos++; }
        
        QueryNode node(op);
        
        // Parse children until closing parenthesis
        while (pos < query.length()) {
            while (pos < query.length() && std::isspace(query[pos])) pos++;
            if (pos < query.length() && query[pos] == ')') {
                pos++; 
                break;
            }
            node.children.push_back(parse_query(query, pos));
        }
        return node;
    } 
    // Handle exact phrases: "goth rock"
    else if (query[pos] == '"') {
        pos++; 
        size_t start = pos;
        while (pos < query.length() && query[pos] != '"') pos++;
        std::string phrase = query.substr(start, pos - start);
        if (pos < query.length()) pos++; 
        return QueryNode(QueryOperator::PHRASE, phrase);
    } 
    // Handle single terms: society
    else {
        size_t start = pos;
        while (pos < query.length() && !std::isspace(query[pos]) && query[pos] != ')') pos++;
        return QueryNode(QueryOperator::TERM, query.substr(start, pos - start));
    }
}

std::vector<int> InvertedIndex::evaluate_phrase(const std::vector<std::string>& tokens) const {
    if (tokens.empty()) return {};

    // 1. Luăm pozițiile pentru primul cuvânt din frază
    auto it = index.find(tokens[0]);
    if (it == index.end()) return {};

    // Mapăm doc_id -> listă de poziții valide pentru fraza "de până acum"
    std::unordered_map<int, std::vector<int>> candidate_docs;
    for (const Posting& p : it->second) {
        candidate_docs[p.doc_id] = p.positions;
    }

    // 2. Iterăm prin restul cuvintelor din frază ("rock", "music", etc.)
    for (size_t i = 1; i < tokens.size(); ++i) {
        auto next_it = index.find(tokens[i]);
        if (next_it == index.end()) return {}; // Un cuvânt lipsește cu totul

        std::unordered_map<int, std::vector<int>> next_candidates;

        for (const Posting& p : next_it->second) {
            auto cand_it = candidate_docs.find(p.doc_id);
            if (cand_it != candidate_docs.end()) {
                // Documentul are ambele cuvinte. Acum verificăm adiacența pozițiilor.
                const std::vector<int>& prev_positions = cand_it->second;
                const std::vector<int>& curr_positions = p.positions;
                
                std::vector<int> valid_positions;
                size_t p1 = 0, p2 = 0;

                // Căutare cu doi pointeri: prev_pos + 1 == curr_pos
                while (p1 < prev_positions.size() && p2 < curr_positions.size()) {
                    if (prev_positions[p1] + 1 == curr_positions[p2]) {
                        valid_positions.push_back(curr_positions[p2]);
                        p1++; p2++;
                    } else if (prev_positions[p1] + 1 < curr_positions[p2]) {
                        p1++;
                    } else {
                        p2++;
                    }
                }

                // Dacă s-a găsit măcar o secvență validă, documentul trece mai departe
                if (!valid_positions.empty()) {
                    next_candidates[p.doc_id] = valid_positions;
                }
            }
        }
        candidate_docs = std::move(next_candidates);
        if (candidate_docs.empty()) return {}; // Ieșire rapidă
    }

    // 3. Extragem doar ID-urile documentelor rămase și le sortăm (pentru unite/intersect)
    std::vector<int> result;
    result.reserve(candidate_docs.size());
    for (const auto& pair : candidate_docs) {
        result.push_back(pair.first);
    }
    std::sort(result.begin(), result.end());

    return result;
}

std::vector<int> InvertedIndex::evaluate_ast(const QueryNode& node) const {
    if (node.op == QueryOperator::TERM) {
        auto it = index.find(node.term);
        if (it == index.end()) return {};
        
        std::vector<int> doc_ids;
        doc_ids.reserve(it->second.size());
        for (const Posting& p : it->second) {
            doc_ids.push_back(p.doc_id);
        }
        return doc_ids;
    } 
    else if (node.op == QueryOperator::PHRASE) {
        std::vector<std::string> phrase_tokens = tokenizer.tokenize(node.term);
        return evaluate_phrase(phrase_tokens);
    }
    else if (node.op == QueryOperator::OR) {
        if (node.children.empty()) return {};
        std::vector<int> result = evaluate_ast(node.children[0]);
        for (size_t i = 1; i < node.children.size(); i++) {
            result = unite(result, evaluate_ast(node.children[i]));
        }
        return result;
    }
    else if (node.op == QueryOperator::AND) {
        if (node.children.empty()) return {};
        std::vector<int> result = evaluate_ast(node.children[0]);
        for (size_t i = 1; i < node.children.size(); i++) {
            if (result.empty()) break; // Early exit optimization
            result = intersect(result, evaluate_ast(node.children[i]));
        }
        return result;
    }
    return {};
}

std::vector<int> InvertedIndex::search(const std::string& query) const {
    size_t pos = 0;
    QueryNode ast = parse_query(query, pos);
    return evaluate_ast(ast);
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
    constexpr uint32_t INDEX_FORMAT_VERSION = 3;
}

void InvertedIndex::save(const std::string& path) const {
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
            
            size_t pos_size = posting.positions.size();
            out.write(reinterpret_cast<const char*>(&pos_size), sizeof(pos_size));
            if (pos_size > 0) {
                out.write(reinterpret_cast<const char*>(posting.positions.data()), pos_size * sizeof(int));
            }
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
            
            size_t pos_size;
            in.read(reinterpret_cast<char*>(&pos_size), sizeof(pos_size));
            p.positions.resize(pos_size);
            if (pos_size > 0) {
                in.read(reinterpret_cast<char*>(p.positions.data()), pos_size * sizeof(int));
            }

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


