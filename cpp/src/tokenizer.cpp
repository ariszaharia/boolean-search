#include "tokenizer.hpp"
#include <cctype>

std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current_token = "";

    // unsigned char: bytes of non-ASCII letters (UTF-8) are negative as char, and passing a
    // negative value to isalnum is undefined. As unsigned they are simply not letters here.
    for(unsigned char c : text){
        if(std::isalnum(c)){
            current_token += static_cast<char>(std::tolower(c));
        } else if (!current_token.empty()){
            tokens.push_back(current_token);
            current_token = "";
        }
    }
    
    if(!current_token.empty()){
        tokens.push_back(current_token);
    }

    return tokens;
}