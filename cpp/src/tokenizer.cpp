#include "tokenizer.hpp"
#include <cctype>

std::vector<std::string> Tokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current_token = "";

    for(char c : text){
        if(std::isalnum(c)){        
            current_token += std::tolower(c); 
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