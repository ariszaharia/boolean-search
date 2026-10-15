#pragma once

#include <string>
#include <vector>

// Text normalization options. Both default to off, which is the exact-match
// tokenizer used in engine-v1 (lowercase, split on every non-alphanumeric byte).
struct TokenizerOptions {
    // Porter stemming (the original 1980 algorithm as in Martin Porter's reference
    // C code; this is the stemmer Lucene's PorterStemFilter uses).
    bool stem = false;
    // Remove the 33 words of Lucene's classic English stopword list. Removed words
    // still occupy a position, so phrases keep their gaps (same as Lucene).
    bool stopwords = false;
};

struct AnalyzedToken {
    std::string term;
    int position;  // index in the raw token stream; removed stopwords leave gaps
};

class Tokenizer {
public:
    explicit Tokenizer(TokenizerOptions options = {}) : options_(options) {}

    const TokenizerOptions& options() const { return options_; }

    // Terms with their positions. Order of operations: lowercase, split, stopword
    // removal, stemming (the order Lucene's EnglishAnalyzer uses).
    std::vector<AnalyzedToken> analyze(const std::string& text) const;

    // Terms only (positions dropped).
    std::vector<std::string> tokenize(const std::string& text) const;

private:
    TokenizerOptions options_;
};

// Porter stemmer on an already lowercased word. Exposed for tests.
std::string porter_stem(const std::string& word);
