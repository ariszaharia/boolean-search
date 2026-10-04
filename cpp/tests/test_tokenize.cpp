#include "tokenizer.hpp"
#include <iostream>

int main() {
    Tokenizer tokenizer;

    std::vector<std::string> tokens = tokenizer.tokenize("The cat sat on the Mat!! 123 cats.");

    for (const auto& token : tokens) {
        std::cout << "[" << token << "]\n";
    }

    return 0;
}