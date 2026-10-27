// Tests for non-ASCII text, terms glued to '(' or '"', and saving/loading an index.
// Each test prints PASS or FAIL; the program exits with 1 if any fail.
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "inverted_index.hpp"
#include "tokenizer.hpp"

static int failures = 0;

static void report(const std::string& name, bool ok) {
    if (!ok) failures++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "\n";
}

static bool parse_fails(const InvertedIndex& ix, const std::string& query) {
    try { ix.parse(query); } catch (const ParseError&) { return true; }
    return false;
}

int main() {
    // --- 1. Non-ASCII bytes (UTF-8) separate words, like punctuation ---
    {
        Tokenizer t;
        // "α-synuclein 5 µm": α and µ are two-byte UTF-8 letters the tokenizer does not keep.
        std::vector<std::string> got = t.tokenize("\xCE\xB1-synuclein 5 \xC2\xB5m");
        report("UTF-8 bytes are separators, ASCII words kept",
               got == std::vector<std::string>({"synuclein", "5", "m"}));

        InvertedIndex ix;
        ix.add_document(1, "Caf\xC3\xA9 culture");
        report("search in a document with UTF-8 text", ix.search("culture") == std::vector<int>({1}));
        report("query with UTF-8 bytes still parses", !parse_fails(ix, "(^ caf\xC3\xA9 culture)"));
    }

    // --- 2. A term glued to '(' or '"' is an error, not one strange term ---
    {
        InvertedIndex ix;
        ix.add_document(1, "foo bar");
        report("foo\"bar\" is rejected",   parse_fails(ix, "(^ foo\"bar\")"));
        report("a(b is rejected",          parse_fails(ix, "a(b"));
        report("(^ foo \"bar\") still works", ix.search("(^ foo \"bar\")") == std::vector<int>({1}));
        report("term right before ')' still works", ix.search("(+ foo)") == std::vector<int>({1}));
    }

    // --- 2b. NOT needs at least two operands: (- A) would silently return A ---
    {
        InvertedIndex ix;
        ix.add_document(1, "foo bar");
        ix.add_document(2, "foo baz");
        report("(- foo) is rejected",                 parse_fails(ix, "(- foo)"));
        report("(- (+ foo bar)) is rejected",         parse_fails(ix, "(- (+ foo bar))"));
        report("(- \"foo bar\") is rejected",         parse_fails(ix, "(- \"foo bar\")"));
        report("(- foo bar) still works",             ix.search("(- foo bar)") == std::vector<int>({2}));
        report("(- foo bar baz) still works",         ix.search("(- foo bar baz)").empty());
        bool at_paren = false;
        try { ix.parse("(^ foo (- bar))"); } catch (const ParseError& e) { at_paren = (e.position == 7); }
        report("the error points at the '(' of the NOT", at_paren);
    }

    // --- 3. Save and load ---
    const std::string path = (std::filesystem::temp_directory_path() / "licenta_test_index.idx").string();
    {
        InvertedIndex original;
        original.add_document(1, "stem cells");
        original.add_document(5, "immune response");
        original.save(path);

        InvertedIndex loaded;
        loaded.add_document(100, "old content that must disappear");
        loaded.load(path);
        report("load replaces what the index held before",
               loaded.search("old").empty() && loaded.get_total_docs() == 2);
        report("loaded index answers the same queries",
               loaded.search("\"stem cells\"") == std::vector<int>({1}) &&
               loaded.search("response") == std::vector<int>({5}));

        bool rejected = false;
        try { loaded.add_document(5, "same id as a loaded document"); }
        catch (const std::invalid_argument&) { rejected = true; }
        report("after load, an id not above the highest loaded id is rejected", rejected);

        loaded.add_document(6, "new document");
        report("after load, the next id can be added", loaded.search("new") == std::vector<int>({6}));
    }
    {
        // Cut the saved file in half: load must refuse it.
        std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
        InvertedIndex ix;
        bool threw = false;
        try { ix.load(path); } catch (const std::runtime_error&) { threw = true; }
        report("truncated index file is rejected", threw);
    }
    std::filesystem::remove(path);

    std::cout << (failures == 0 ? "\nALL INDEX I/O TESTS PASSED\n"
                                : "\n" + std::to_string(failures) + " INDEX I/O TEST(S) FAILED\n");
    return failures == 0 ? 0 : 1;
}
