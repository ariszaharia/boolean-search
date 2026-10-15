// Phrase matching tests. Each test prints PASS or FAIL; the program exits with 1 if any fail.
#include <iostream>
#include <string>
#include <vector>
#include "inverted_index.hpp"

static int failures = 0;

static std::string show(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? ", " : "") + std::to_string(v[i]);
    return s + "]";
}

static void check(const std::string& name, const InvertedIndex& ix,
                  const std::string& query, const std::vector<int>& expected) {
    std::vector<int> got = ix.search(query);
    bool ok = (got == expected);
    if (!ok) failures++;
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << "   query: " << query
              << "   expected " << show(expected) << ", got " << show(got) << "\n";
}

int main() {
    // --- basic adjacency and order ---
    InvertedIndex a;
    a.add_document(1, "stem cells are critical");
    a.add_document(2, "cells stem");                   // same words, reversed order
    a.add_document(3, "stem and cells");               // same words, not adjacent
    a.add_document(4, "the stem cells");               // phrase at the end
    a.add_document(5, "stem cells");                   // phrase is the whole document
    check("adjacent words match",            a, "\"stem cells\"", {1, 4, 5});
    check("reversed order does not match",   a, "\"cells stem\"", {2});
    check("word missing from the index",     a, "\"stem banana\"", {});
    check("single-word phrase",              a, "\"stem\"", {1, 2, 3, 4, 5});

    // --- phrase boundaries between documents ---
    InvertedIndex b;
    b.add_document(1, "we study stem");
    b.add_document(2, "cells are small");
    check("phrase cannot span two documents", b, "\"stem cells\"", {});

    // --- repeated phrase and repeated words ---
    InvertedIndex c;
    c.add_document(1, "stem cells and more stem cells");   // phrase twice
    c.add_document(2, "cell cell cell");                   // overlapping repeats
    c.add_document(3, "cell x cell");                      // repeat but not adjacent
    check("phrase appearing twice (doc returned once)", c, "\"stem cells\"", {1});
    check("phrase of a repeated word",       c, "\"cell cell\"", {2});
    check("repeated word, three in a row",   c, "\"cell cell cell\"", {2});
    check("repeated word, too many",         c, "\"cell cell cell cell\"", {});

    // --- longer phrases and interleaved positions ---
    InvertedIndex d;
    d.add_document(1, "a b a a b");
    d.add_document(2, "a a a b");
    d.add_document(3, "a b c a b d");
    check("interleaved positions",           d, "\"a b\"", {1, 2, 3});
    check("a a b inside a a a b",            d, "\"a a b\"", {1, 2});
    check("three-word phrase",               d, "\"a b c\"", {3});
    check("three-word phrase at the end",   d, "\"a b d\"", {3});
    check("three-word phrase, not present",  d, "\"b c d\"", {});

    // --- normalization: case and punctuation ---
    InvertedIndex e;
    e.add_document(1, "Stem-Cells, and T-cell response!");
    check("phrase is case-insensitive",      e, "\"STEM cells\"", {1});
    check("punctuation in the text",         e, "\"stem cells\"", {1});
    check("punctuation in the phrase",       e, "\"t-cell response\"", {1});
    check("bare hyphenated term as phrase",  e, "T-cell", {1});

    // --- phrases combined with operators ---
    InvertedIndex f;
    f.add_document(1, "stem cells mutation");
    f.add_document(2, "stem cells therapy");
    f.add_document(3, "stem therapy cells");
    check("phrase inside AND",               f, "(^ \"stem cells\" therapy)", {2});
    check("phrase inside OR",                f, "(+ \"stem cells\" \"stem therapy\")", {1, 2, 3});
    check("phrase inside NOT",               f, "(- \"stem cells\" mutation)", {2});
    check("phrase as the excluded part",     f, "(- stem \"stem cells\")", {3});

    std::cout << (failures == 0 ? "\nALL PHRASE TESTS PASSED\n"
                                : "\n" + std::to_string(failures) + " PHRASE TEST(S) FAILED\n");
    return failures == 0 ? 0 : 1;
}
