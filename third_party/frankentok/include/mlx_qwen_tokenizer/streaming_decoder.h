#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include "mlx_qwen_tokenizer/vocab.h"

namespace mlx_qwen_tokenizer {

class StreamingDecoder {
public:
    StreamingDecoder(const Vocab& vocab) : vocab_(vocab) {}

    // Decode a sequence of ids, buffering incomplete utf-8 sequences.
    // Returns valid utf-8 decoded string so far.
    std::string decode(const std::vector<int32_t>& ids);

    // Decode a single id
    std::string decode(int32_t id);

    // Push raw bytes through the same buffer, returning whatever is now complete UTF-8.
    //
    // Exists so the split-sequence behaviour is reachable at all: `decode(id)` always appends a
    // whole token's worth of bytes, so no test written against it can land a codepoint boundary
    // mid-call. That is why every implementation reviewed claimed to handle split sequences while
    // only ever asserting a full round-trip.
    std::string decode_bytes(std::string_view bytes);

    // Flush remaining buffered bytes (even if invalid utf-8)
    std::string flush();

    // Clear buffer
    void clear() { buffer_.clear(); }

private:
    const Vocab& vocab_;
    std::string buffer_;

    bool is_valid_utf8(const std::string& str) const;
    size_t get_valid_utf8_prefix(const std::string& str) const;
    size_t get_valid_utf8_prefix_sv(std::string_view str) const;
};

} // namespace mlx_qwen_tokenizer
