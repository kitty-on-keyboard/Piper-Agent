#pragma once

#include <string>
#include <vector>
#include <string_view>
#include <memory>

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace mlx_qwen_tokenizer {

class Pretokenizer {
public:
    // Compiles the given Split pattern (PCRE2 syntax, which covers the patterns HF ships).
    // The no-argument form keeps the classic Qwen pattern for callers with no tokenizer.json
    // in hand; real loads should pass Vocab::pretokenizer_regex() so the split always matches
    // the vocabulary's own declaration.
    Pretokenizer();
    explicit Pretokenizer(const std::string& pattern);
    ~Pretokenizer();

    // Prevent copying
    Pretokenizer(const Pretokenizer&) = delete;
    Pretokenizer& operator=(const Pretokenizer&) = delete;

    // Appends the pre-token boundaries of `text` to `out`, which is cleared first.
    //
    // The views borrow from `text` and stay valid only as long as it does. Every caller here splits
    // a piece that outlives the loop consuming it, and the pieces are short-lived enough that
    // owning them cost a std::string per pre-token — about half a million per megabyte.
    void split(std::string_view text, std::vector<std::string_view>& out) const;

    // Convenience overload for callers that want to own the pieces.
    std::vector<std::string> split(std::string_view text) const;

private:
    pcre2_code* re_;
};

} // namespace mlx_qwen_tokenizer
