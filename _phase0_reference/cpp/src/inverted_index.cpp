#include "inverted_index.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <utility>

namespace {

constexpr uint32_t INDEX_FORMAT_VERSION = 4;
constexpr int MAX_QUERY_DEPTH = 500;

bool is_space(char c) { return std::isspace(static_cast<unsigned char>(c)) != 0; }

// Postings are appended in increasing doc_id order, so each list is sorted.
const Posting* find_posting(const std::vector<Posting>& list, int doc_id) {
    auto it = std::lower_bound(list.begin(), list.end(), doc_id,
                               [](const Posting& p, int id) { return p.doc_id < id; });
    if (it == list.end() || it->doc_id != doc_id) return nullptr;
    return &*it;
}

// Occurrences of the phrase in one document. toks[i].position - toks[0].position is the
// required distance between term 0 and term i (1 for adjacent words, more when a
// stopword was removed in between).
int count_phrase(const std::vector<AnalyzedToken>& toks,
                 const std::vector<const std::vector<Posting>*>& lists, int doc_id) {
    std::vector<const Posting*> ps;
    ps.reserve(lists.size());
    for (const auto* list : lists) {
        const Posting* p = find_posting(*list, doc_id);
        if (!p) return 0;
        ps.push_back(p);
    }
    int count = 0;
    for (int start : ps[0]->positions) {
        bool ok = true;
        for (size_t i = 1; i < ps.size() && ok; ++i) {
            int want = start + (toks[i].position - toks[0].position);
            ok = std::binary_search(ps[i]->positions.begin(), ps[i]->positions.end(), want);
        }
        if (ok) ++count;
    }
    return count;
}

}  // namespace

InvertedIndex::InvertedIndex(TokenizerOptions options) : tokenizer(options) {}

std::vector<AnalyzedToken> InvertedIndex::analyze(const std::string& text) const {
    return tokenizer.analyze(text);
}

const TokenizerOptions& InvertedIndex::tokenizer_options() const { return tokenizer.options(); }

void InvertedIndex::add_document(int doc_id, const std::string& text) {
    if (has_documents && doc_id <= last_doc_id) {
        throw std::invalid_argument("InvertedIndex::add_document: doc_id must be strictly increasing (got " +
                                    std::to_string(doc_id) + " after " + std::to_string(last_doc_id) + ")");
    }

    std::vector<AnalyzedToken> tokens = tokenizer.analyze(text);
    std::unordered_map<std::string, std::vector<int>> word_positions;
    for (const AnalyzedToken& t : tokens) word_positions[t.term].push_back(t.position);

    last_doc_id = doc_id;
    has_documents = true;
    documents[doc_id] = text;

    for (auto& pair : word_positions) {
        int freq = static_cast<int>(pair.second.size());
        index[pair.first].push_back({doc_id, freq, std::move(pair.second)});
    }

    doc_lengths[doc_id] = static_cast<int>(tokens.size());
    total_length += static_cast<double>(tokens.size());
    total_docs++;
}

// ---------------------------------------------------------------- set operations on sorted id lists

// Set union (OR).
std::vector<int> InvertedIndex::unite(const std::vector<int>& a, const std::vector<int>& b) const {
    std::vector<int> result;
    result.reserve(a.size() + b.size());
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            result.push_back(a[i]);
            i++;
            j++;
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

// Set intersection (AND).
std::vector<int> InvertedIndex::intersect(const std::vector<int>& a, const std::vector<int>& b) const {
    std::vector<int> result;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            result.push_back(a[i]);
            i++;
            j++;
        } else if (a[i] < b[j]) {
            i++;
        } else {
            j++;
        }
    }
    return result;
}

// Set difference (NOT): elements of a that are not in b.
std::vector<int> InvertedIndex::difference(const std::vector<int>& a, const std::vector<int>& b) const {
    std::vector<int> result;
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size()) {
        if (a[i] == b[j]) {
            i++;
            j++;
        } else if (a[i] < b[j]) {
            result.push_back(a[i]);
            i++;
        } else {
            j++;
        }
    }
    while (i < a.size()) result.push_back(a[i++]);
    return result;
}

// ---------------------------------------------------------------- parsing

QueryNode InvertedIndex::parse_query(const std::string& query, size_t& pos, int depth) const {
    if (depth > MAX_QUERY_DEPTH) throw ParseError("query is nested too deeply", pos);

    while (pos < query.length() && is_space(query[pos])) pos++;
    if (pos >= query.length()) throw ParseError("unexpected end of query", pos);

    if (query[pos] == '(') {
        size_t open_pos = pos;
        pos++;
        while (pos < query.length() && is_space(query[pos])) pos++;
        if (pos >= query.length()) throw ParseError("expected an operator after '('", pos);

        QueryOperator op;
        if (query[pos] == '^') op = QueryOperator::AND;
        else if (query[pos] == '+') op = QueryOperator::OR;
        else if (query[pos] == '-') op = QueryOperator::NOT;
        else throw ParseError("expected operator ^, + or - after '('", pos);
        pos++;

        QueryNode node(op);
        while (true) {
            while (pos < query.length() && is_space(query[pos])) pos++;
            if (pos >= query.length()) throw ParseError("missing ')' for the '(' opened", open_pos);
            if (query[pos] == ')') {
                pos++;
                break;
            }
            node.children.push_back(parse_query(query, pos, depth + 1));
        }
        if (node.children.empty()) throw ParseError("operator with no operands", open_pos);
        return node;
    }

    if (query[pos] == ')') throw ParseError("unexpected ')'", pos);

    if (query[pos] == '"') {
        size_t open_pos = pos;
        pos++;
        size_t start = pos;
        while (pos < query.length() && query[pos] != '"') pos++;
        if (pos >= query.length()) throw ParseError("unterminated quote", open_pos);
        std::string phrase = query.substr(start, pos - start);
        pos++;
        if (tokenizer.analyze(phrase).empty()) throw ParseError("phrase has no searchable words", open_pos);
        return QueryNode(QueryOperator::PHRASE, phrase);
    }

    size_t start = pos;
    while (pos < query.length() && !is_space(query[pos]) && query[pos] != ')') pos++;
    std::string term = query.substr(start, pos - start);
    if (tokenizer.analyze(term).empty()) throw ParseError("term has no searchable characters", start);
    return QueryNode(QueryOperator::TERM, term);
}

QueryNode InvertedIndex::parse(const std::string& query) const {
    size_t pos = 0;
    QueryNode ast = parse_query(query, pos, 0);
    while (pos < query.length() && is_space(query[pos])) pos++;
    if (pos < query.length()) throw ParseError("unexpected text after the query", pos);
    return ast;
}

// ---------------------------------------------------------------- evaluation

std::vector<int> InvertedIndex::evaluate_phrase(const std::vector<AnalyzedToken>& tokens) const {
    if (tokens.empty()) return {};

    std::vector<const std::vector<Posting>*> lists;
    lists.reserve(tokens.size());
    size_t driver = 0;  // the rarest term drives the scan
    for (size_t i = 0; i < tokens.size(); ++i) {
        auto it = index.find(tokens[i].term);
        if (it == index.end()) return {};
        lists.push_back(&it->second);
        if (it->second.size() < lists[driver]->size()) driver = i;
    }

    std::vector<int> result;
    for (const Posting& p : *lists[driver]) {
        if (count_phrase(tokens, lists, p.doc_id) > 0) result.push_back(p.doc_id);
    }
    return result;  // ascending, because every posting list is sorted by doc_id
}

std::vector<int> InvertedIndex::evaluate_ast(const QueryNode& node) const {
    switch (node.op) {
        case QueryOperator::TERM:
        case QueryOperator::PHRASE: {
            std::vector<AnalyzedToken> toks = tokenizer.analyze(node.term);
            if (toks.empty()) return {};
            if (toks.size() == 1) {
                auto it = index.find(toks[0].term);
                if (it == index.end()) return {};
                std::vector<int> doc_ids;
                doc_ids.reserve(it->second.size());
                for (const Posting& p : it->second) doc_ids.push_back(p.doc_id);
                return doc_ids;
            }
            return evaluate_phrase(toks);  // "T-cell" or "stem cells": adjacent words
        }
        case QueryOperator::OR: {
            if (node.children.empty()) return {};
            std::vector<int> result = evaluate_ast(node.children[0]);
            for (size_t i = 1; i < node.children.size(); i++) {
                result = unite(result, evaluate_ast(node.children[i]));
            }
            return result;
        }
        case QueryOperator::AND: {
            if (node.children.empty()) return {};
            std::vector<int> result = evaluate_ast(node.children[0]);
            for (size_t i = 1; i < node.children.size(); i++) {
                if (result.empty()) break;  // early exit
                result = intersect(result, evaluate_ast(node.children[i]));
            }
            return result;
        }
        case QueryOperator::NOT: {
            if (node.children.empty()) return {};
            std::vector<int> result = evaluate_ast(node.children[0]);
            for (size_t i = 1; i < node.children.size(); i++) {
                if (result.empty()) break;
                result = difference(result, evaluate_ast(node.children[i]));
            }
            return result;
        }
    }
    return {};
}

std::vector<int> InvertedIndex::search(const std::string& query) const {
    return evaluate_ast(parse(query));
}

// ---------------------------------------------------------------- statistics

const std::string& InvertedIndex::get_document(int doc_id) const {
    auto it = documents.find(doc_id);
    if (it == documents.end()) {
        throw std::out_of_range("InvertedIndex::get_document: no document with id " + std::to_string(doc_id));
    }
    return it->second;
}

int InvertedIndex::get_total_docs() const { return total_docs; }

double InvertedIndex::get_avg_doc_length() const {
    if (total_docs == 0) return 0.0;
    return total_length / total_docs;
}

int InvertedIndex::get_doc_length(int doc_id) const {
    auto it = doc_lengths.find(doc_id);
    return it != doc_lengths.end() ? it->second : 0;
}

int InvertedIndex::get_doc_frequency(const std::string& term) const {
    auto it = index.find(term);
    return it == index.end() ? 0 : static_cast<int>(it->second.size());
}

int InvertedIndex::get_term_frequency(const std::string& term, int doc_id) const {
    auto it = index.find(term);
    if (it == index.end()) return 0;
    const Posting* p = find_posting(it->second, doc_id);
    return p ? p->frequency : 0;
}

int InvertedIndex::phrase_frequency(const std::vector<AnalyzedToken>& tokens, int doc_id) const {
    if (tokens.empty()) return 0;
    std::vector<const std::vector<Posting>*> lists;
    lists.reserve(tokens.size());
    for (const AnalyzedToken& t : tokens) {
        auto it = index.find(t.term);
        if (it == index.end()) return 0;
        lists.push_back(&it->second);
    }
    return count_phrase(tokens, lists, doc_id);
}

// ---------------------------------------------------------------- persistence

namespace {

template <typename T>
void write_pod(std::ostream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

[[noreturn]] void corrupt() {
    throw std::runtime_error("InvertedIndex::load: file is truncated or corrupt");
}

template <typename T>
T read_pod(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) corrupt();
    return value;
}

}  // namespace

void InvertedIndex::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("InvertedIndex::save: could not open " + path + " for writing");
    }

    write_pod(out, INDEX_FORMAT_VERSION);
    write_pod<uint8_t>(out, tokenizer.options().stem ? 1 : 0);
    write_pod<uint8_t>(out, tokenizer.options().stopwords ? 1 : 0);
    write_pod(out, total_docs);
    write_pod(out, total_length);

    // Sorted output so that saving the same index twice gives identical bytes.
    std::vector<int> doc_ids;
    doc_ids.reserve(doc_lengths.size());
    for (const auto& kv : doc_lengths) doc_ids.push_back(kv.first);
    std::sort(doc_ids.begin(), doc_ids.end());

    write_pod<size_t>(out, doc_ids.size());
    for (int id : doc_ids) {
        write_pod(out, id);
        write_pod(out, doc_lengths.at(id));
    }

    std::vector<const std::string*> words;
    words.reserve(index.size());
    for (const auto& kv : index) words.push_back(&kv.first);
    std::sort(words.begin(), words.end(), [](const std::string* a, const std::string* b) { return *a < *b; });

    write_pod<size_t>(out, words.size());
    for (const std::string* word : words) {
        const std::vector<Posting>& postings = index.at(*word);
        write_pod<size_t>(out, word->size());
        out.write(word->data(), static_cast<std::streamsize>(word->size()));
        write_pod<size_t>(out, postings.size());
        for (const Posting& p : postings) {
            write_pod(out, p.doc_id);
            write_pod(out, p.frequency);
            write_pod<size_t>(out, p.positions.size());
            if (!p.positions.empty()) {
                out.write(reinterpret_cast<const char*>(p.positions.data()),
                          static_cast<std::streamsize>(p.positions.size() * sizeof(int)));
            }
        }
    }

    write_pod<size_t>(out, doc_ids.size());
    for (int id : doc_ids) {
        const std::string& text = documents.at(id);
        write_pod(out, id);
        write_pod<size_t>(out, text.size());
        out.write(text.data(), static_cast<std::streamsize>(text.size()));
    }

    out.flush();
    if (!out) throw std::runtime_error("InvertedIndex::save: write to " + path + " failed");
}

void InvertedIndex::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("InvertedIndex::load: could not open " + path + " for reading");
    }
    in.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(in.tellg());
    in.seekg(0, std::ios::beg);

    // Everything is read into locals first; the index changes only if the whole file is valid.
    uint32_t format_version = read_pod<uint32_t>(in);
    if (format_version != INDEX_FORMAT_VERSION) {
        throw std::runtime_error("InvertedIndex::load: unsupported index format version " +
                                 std::to_string(format_version) + " (expected " +
                                 std::to_string(INDEX_FORMAT_VERSION) + "). Rebuild the index.");
    }

    TokenizerOptions options;
    options.stem = read_pod<uint8_t>(in) != 0;
    options.stopwords = read_pod<uint8_t>(in) != 0;

    int new_total_docs = read_pod<int>(in);
    double new_total_length = read_pod<double>(in);
    if (new_total_docs < 0 || new_total_length < 0.0) corrupt();

    size_t doc_lengths_count = read_pod<size_t>(in);
    if (doc_lengths_count > file_size || static_cast<int>(doc_lengths_count) != new_total_docs) corrupt();
    std::unordered_map<int, int> new_doc_lengths;
    new_doc_lengths.reserve(doc_lengths_count);
    for (size_t i = 0; i < doc_lengths_count; i++) {
        int doc_id = read_pod<int>(in);
        int length = read_pod<int>(in);
        if (length < 0) corrupt();
        new_doc_lengths[doc_id] = length;
    }
    if (new_doc_lengths.size() != doc_lengths_count) corrupt();

    size_t index_count = read_pod<size_t>(in);
    if (index_count > file_size) corrupt();
    std::unordered_map<std::string, std::vector<Posting>> new_index;
    new_index.reserve(index_count);
    for (size_t i = 0; i < index_count; i++) {
        size_t term_len = read_pod<size_t>(in);
        if (term_len > file_size) corrupt();
        std::string word(term_len, '\0');
        if (term_len > 0) {
            in.read(&word[0], static_cast<std::streamsize>(term_len));
            if (!in) corrupt();
        }

        size_t posting_size = read_pod<size_t>(in);
        if (posting_size > file_size) corrupt();
        std::vector<Posting> postings;
        postings.reserve(posting_size);
        int previous_doc = -1;
        bool first = true;
        for (size_t j = 0; j < posting_size; j++) {
            Posting p;
            p.doc_id = read_pod<int>(in);
            p.frequency = read_pod<int>(in);
            size_t pos_size = read_pod<size_t>(in);
            if (pos_size > file_size / sizeof(int) || static_cast<int>(pos_size) != p.frequency) corrupt();
            p.positions.resize(pos_size);
            if (pos_size > 0) {
                in.read(reinterpret_cast<char*>(p.positions.data()),
                        static_cast<std::streamsize>(pos_size * sizeof(int)));
                if (!in) corrupt();
            }
            if (!first && p.doc_id <= previous_doc) corrupt();  // lists must be sorted for binary search
            first = false;
            previous_doc = p.doc_id;
            postings.push_back(std::move(p));
        }
        new_index[std::move(word)] = std::move(postings);
    }

    size_t documents_count = read_pod<size_t>(in);
    if (documents_count != doc_lengths_count) corrupt();
    std::unordered_map<int, std::string> new_documents;
    new_documents.reserve(documents_count);
    int new_last_doc_id = 0;
    bool new_has_documents = false;
    for (size_t i = 0; i < documents_count; i++) {
        int doc_id = read_pod<int>(in);
        size_t text_len = read_pod<size_t>(in);
        if (text_len > file_size) corrupt();
        std::string text(text_len, '\0');
        if (text_len > 0) {
            in.read(&text[0], static_cast<std::streamsize>(text_len));
            if (!in) corrupt();
        }
        if (!new_doc_lengths.count(doc_id)) corrupt();
        if (!new_has_documents || doc_id > new_last_doc_id) new_last_doc_id = doc_id;
        new_has_documents = true;
        new_documents[doc_id] = std::move(text);
    }
    if (new_documents.size() != documents_count) corrupt();

    // Whole file is valid: replace the current state.
    tokenizer = Tokenizer(options);
    index = std::move(new_index);
    documents = std::move(new_documents);
    doc_lengths = std::move(new_doc_lengths);
    total_docs = new_total_docs;
    total_length = new_total_length;
    last_doc_id = new_last_doc_id;
    has_documents = new_has_documents;
}
