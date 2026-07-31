#pragma once

#include <string>
#include <vector>
#include <memory>
#include <string_view>
#include "robin_hood.h"

namespace mlx_qwen_tokenizer {

class TrieNode {
public:
    robin_hood::unordered_flat_map<char, std::unique_ptr<TrieNode>> children;
    bool is_end;
    std::string token;

    TrieNode() : is_end(false) {}
};

class SpecialTokenTrie {
public:
    SpecialTokenTrie() : root(std::make_unique<TrieNode>()) {}

    void add(const std::string& token);

    struct MatchResult {
        bool found;
        size_t length;
        std::string token;
    };

    // Find the longest special token starting at text[start_pos]
    MatchResult match_longest(std::string_view text, size_t start_pos) const;

    // Split text into special tokens and normal text pieces
    std::vector<std::string> split(std::string_view text) const;

private:
    std::unique_ptr<TrieNode> root;
};

} // namespace mlx_qwen_tokenizer
