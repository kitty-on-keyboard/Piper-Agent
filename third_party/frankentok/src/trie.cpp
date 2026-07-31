#include "mlx_qwen_tokenizer/trie.h"

namespace mlx_qwen_tokenizer {

void SpecialTokenTrie::add(const std::string& token) {
    if (token.empty()) return;

    TrieNode* curr = root.get();
    for (char c : token) {
        if (curr->children.find(c) == curr->children.end()) {
            curr->children[c] = std::make_unique<TrieNode>();
        }
        curr = curr->children[c].get();
    }
    curr->is_end = true;
    curr->token = token;
}

SpecialTokenTrie::MatchResult SpecialTokenTrie::match_longest(std::string_view text, size_t start_pos) const {
    MatchResult result{false, 0, ""};
    TrieNode* curr = root.get();

    size_t i = start_pos;
    while (i < text.length()) {
        char c = text[i];
        auto it = curr->children.find(c);
        if (it == curr->children.end()) {
            break;
        }
        curr = it->second.get();
        i++;
        if (curr->is_end) {
            result.found = true;
            result.length = i - start_pos;
            result.token = curr->token;
        }
    }

    return result;
}

std::vector<std::string> SpecialTokenTrie::split(std::string_view text) const {
    std::vector<std::string> parts;
    if (text.empty()) return parts;

    size_t start = 0;
    while (start < text.length()) {
        MatchResult match = match_longest(text, start);
        if (match.found) {
            parts.push_back(std::string(text.substr(start, match.length)));
            start += match.length;
        } else {
            // Find next match to jump to
            size_t next_start = start + 1;
            MatchResult next_match{false, 0, ""};
            while (next_start < text.length()) {
                next_match = match_longest(text, next_start);
                if (next_match.found) break;
                next_start++;
            }
            parts.push_back(std::string(text.substr(start, next_start - start)));
            start = next_start;
        }
    }
    return parts;
}

} // namespace mlx_qwen_tokenizer
