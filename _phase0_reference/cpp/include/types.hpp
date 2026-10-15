#pragma once

#include <vector>

struct Posting {
    int doc_id;
    int frequency;
    std::vector<int> positions;  // ascending
};
