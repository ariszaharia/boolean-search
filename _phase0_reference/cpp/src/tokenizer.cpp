#include "tokenizer.hpp"

#include <cctype>
#include <cstring>
#include <unordered_set>

namespace {

// ---------------------------------------------------------------------------
// Stopwords: Lucene's classic English set.
// ---------------------------------------------------------------------------
bool is_stopword(const std::string& w) {
    static const std::unordered_set<std::string> kStop = {
        "a",     "an",   "and",  "are",   "as",   "at",   "be",    "but",   "by",
        "for",   "if",   "in",   "into",  "is",   "it",   "no",    "not",   "of",
        "on",    "or",   "such", "that",  "the",  "their", "then", "there", "these",
        "they",  "this", "to",   "was",   "will", "with"};
    return kStop.count(w) != 0;
}

// ---------------------------------------------------------------------------
// Porter stemmer: a direct port of Martin Porter's reference ANSI C
// implementation (including its two documented departures from the paper:
// "bli" -> "ble" and "logi" -> "log", and words of length <= 2 are left alone).
// ---------------------------------------------------------------------------
class Porter {
public:
    std::string stem(const std::string& word) {
        if (word.size() <= 2) return word;
        b_ = word;
        k0_ = 0;
        k_ = static_cast<int>(b_.size()) - 1;
        j_ = 0;
        step1ab();
        if (k_ > k0_) {
            step1c();
            step2();
            step3();
            step4();
            step5();
        }
        return b_.substr(0, static_cast<size_t>(k_) + 1);
    }

private:
    std::string b_;
    int k_ = 0, k0_ = 0, j_ = 0;

    // True if b_[i] is a consonant.
    bool cons(int i) const {
        switch (b_[i]) {
            case 'a': case 'e': case 'i': case 'o': case 'u':
                return false;
            case 'y':
                return i == k0_ ? true : !cons(i - 1);
            default:
                return true;
        }
    }

    // Number of consonant sequences between k0_ and j_ (the "measure").
    int m() const {
        int n = 0;
        int i = k0_;
        while (true) {
            if (i > j_) return n;
            if (!cons(i)) break;
            i++;
        }
        i++;
        while (true) {
            while (true) {
                if (i > j_) return n;
                if (cons(i)) break;
                i++;
            }
            i++;
            n++;
            while (true) {
                if (i > j_) return n;
                if (!cons(i)) break;
                i++;
            }
            i++;
        }
    }

    bool vowel_in_stem() const {
        for (int i = k0_; i <= j_; i++)
            if (!cons(i)) return true;
        return false;
    }

    bool doublec(int j) const {
        if (j < k0_ + 1) return false;
        if (b_[j] != b_[j - 1]) return false;
        return cons(j);
    }

    // consonant-vowel-consonant ending, where the last consonant is not w, x or y.
    bool cvc(int i) const {
        if (i < k0_ + 2 || !cons(i) || cons(i - 1) || !cons(i - 2)) return false;
        char ch = b_[i];
        return !(ch == 'w' || ch == 'x' || ch == 'y');
    }

    bool ends(const char* s) {
        int length = static_cast<int>(std::strlen(s));
        if (length == 0) return true;
        if (s[length - 1] != b_[k_]) return false;
        if (length > k_ - k0_ + 1) return false;
        if (b_.compare(static_cast<size_t>(k_ - length + 1), static_cast<size_t>(length), s) != 0) return false;
        j_ = k_ - length;
        return true;
    }

    // Replace b_[j_+1 .. k_] by s and set k_ to the new last index.
    void setto(const char* s) {
        int length = static_cast<int>(std::strlen(s));
        if (static_cast<size_t>(j_ + 1 + length) > b_.size()) b_.resize(static_cast<size_t>(j_ + 1 + length));
        for (int i = 0; i < length; i++) b_[static_cast<size_t>(j_ + 1 + i)] = s[i];
        k_ = j_ + length;
    }

    void r(const char* s) {
        if (m() > 0) setto(s);
    }

    void step1ab() {
        if (b_[k_] == 's') {
            if (ends("sses")) k_ -= 2;
            else if (ends("ies")) setto("i");
            else if (b_[k_ - 1] != 's') k_--;
        }
        if (ends("eed")) {
            if (m() > 0) k_--;
        } else if ((ends("ed") || ends("ing")) && vowel_in_stem()) {
            k_ = j_;
            if (ends("at")) setto("ate");
            else if (ends("bl")) setto("ble");
            else if (ends("iz")) setto("ize");
            else if (doublec(k_)) {
                k_--;
                char ch = b_[k_];
                if (ch == 'l' || ch == 's' || ch == 'z') k_++;
            } else if (m() == 1 && cvc(k_)) setto("e");
        }
    }

    void step1c() {
        if (ends("y") && vowel_in_stem()) b_[k_] = 'i';
    }

    void step2() {
        if (k_ < 1) return;
        switch (b_[k_ - 1]) {
            case 'a':
                if (ends("ational")) { r("ate"); break; }
                if (ends("tional")) { r("tion"); break; }
                break;
            case 'c':
                if (ends("enci")) { r("ence"); break; }
                if (ends("anci")) { r("ance"); break; }
                break;
            case 'e':
                if (ends("izer")) { r("ize"); break; }
                break;
            case 'l':
                if (ends("bli")) { r("ble"); break; }
                if (ends("alli")) { r("al"); break; }
                if (ends("entli")) { r("ent"); break; }
                if (ends("eli")) { r("e"); break; }
                if (ends("ousli")) { r("ous"); break; }
                break;
            case 'o':
                if (ends("ization")) { r("ize"); break; }
                if (ends("ation")) { r("ate"); break; }
                if (ends("ator")) { r("ate"); break; }
                break;
            case 's':
                if (ends("alism")) { r("al"); break; }
                if (ends("iveness")) { r("ive"); break; }
                if (ends("fulness")) { r("ful"); break; }
                if (ends("ousness")) { r("ous"); break; }
                break;
            case 't':
                if (ends("aliti")) { r("al"); break; }
                if (ends("iviti")) { r("ive"); break; }
                if (ends("biliti")) { r("ble"); break; }
                break;
            case 'g':
                if (ends("logi")) { r("log"); break; }
                break;
            default:
                break;
        }
    }

    void step3() {
        switch (b_[k_]) {
            case 'e':
                if (ends("icate")) { r("ic"); break; }
                if (ends("ative")) { r(""); break; }
                if (ends("alize")) { r("al"); break; }
                break;
            case 'i':
                if (ends("iciti")) { r("ic"); break; }
                break;
            case 'l':
                if (ends("ical")) { r("ic"); break; }
                if (ends("ful")) { r(""); break; }
                break;
            case 's':
                if (ends("ness")) { r(""); break; }
                break;
            default:
                break;
        }
    }

    void step4() {
        if (k_ < 1) return;
        switch (b_[k_ - 1]) {
            case 'a':
                if (ends("al")) break;
                return;
            case 'c':
                if (ends("ance")) break;
                if (ends("ence")) break;
                return;
            case 'e':
                if (ends("er")) break;
                return;
            case 'i':
                if (ends("ic")) break;
                return;
            case 'l':
                if (ends("able")) break;
                if (ends("ible")) break;
                return;
            case 'n':
                if (ends("ant")) break;
                if (ends("ement")) break;
                if (ends("ment")) break;
                if (ends("ent")) break;
                return;
            case 'o':
                if (ends("ion") && j_ >= k0_ && (b_[j_] == 's' || b_[j_] == 't')) break;
                if (ends("ou")) break;
                return;
            case 's':
                if (ends("ism")) break;
                return;
            case 't':
                if (ends("ate")) break;
                if (ends("iti")) break;
                return;
            case 'u':
                if (ends("ous")) break;
                return;
            case 'v':
                if (ends("ive")) break;
                return;
            case 'z':
                if (ends("ize")) break;
                return;
            default:
                return;
        }
        if (m() > 1) k_ = j_;
    }

    void step5() {
        j_ = k_;
        if (b_[k_] == 'e') {
            int a = m();
            if (a > 1 || (a == 1 && !cvc(k_ - 1))) k_--;
        }
        if (b_[k_] == 'l' && doublec(k_) && m() > 1) k_--;
    }
};

}  // namespace

std::string porter_stem(const std::string& word) {
    Porter p;
    return p.stem(word);
}

std::vector<AnalyzedToken> Tokenizer::analyze(const std::string& text) const {
    std::vector<AnalyzedToken> out;
    int pos = 0;
    std::string cur;

    auto flush = [&]() {
        if (cur.empty()) return;
        if (!(options_.stopwords && is_stopword(cur))) {
            out.push_back({options_.stem ? porter_stem(cur) : cur, pos});
        }
        ++pos;  // a removed stopword still takes a position
        cur.clear();
    };

    for (unsigned char c : text) {
        if (std::isalnum(c)) {
            cur += static_cast<char>(std::tolower(c));
        } else {
            flush();
        }
    }
    flush();
    return out;
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> terms;
    for (auto& t : analyze(text)) terms.push_back(std::move(t.term));
    return terms;
}
