#pragma once
//
// Closed markdown fences harvested from think text for `commit_think_block`.
//
// Exact-byte copies only. Unclosed, empty, or oversized fences are dropped, never
// repaired. No GPU, no tokenizer.
//
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::tools {

inline constexpr std::size_t kMaxThinkBlocksPerTurn = 3;
inline constexpr std::size_t kMaxThinkBlockChars = 200000;

struct ThinkBlock {
    int block_id = 0; // 0-based among kept blocks
    std::string language;
    std::string content; // exact body bytes; no trim
    std::string sha256;  // platform::content_sha256_hex(content)
};

struct FenceHarvest {
    std::vector<ThinkBlock> blocks;
    std::size_t dropped_overflow = 0;  // valid fences past kMaxThinkBlocksPerTurn
    std::size_t dropped_too_large = 0; // closed fences over kMaxThinkBlockChars
};

// Body = bytes after the opener's newline, not including the closing fence line.
// Inner triple-backticks on their own line close the fence. 4-backtick outer fences
// are unsupported.
[[nodiscard]] FenceHarvest extract_fenced_blocks(std::string_view text);

} // namespace lmp::tools
