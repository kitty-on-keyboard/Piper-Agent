#pragma once
//
// Grammar jump-forward (v0 stub) — skip per-token Metal forwards over contiguous
// zero-entropy ToolCallGuard spans (allowed_bytes cardinality == 1).
//
// Flag: LMP_GRAMMAR_JUMP_FORWARD=1 enables (exact `1` only). Default OFF.
// Never jumps FreeText / MaskClass::FreeText, card>1 forks, or spans shorter
// than kMinJumpBytes. Speculative decode leaves this path alone so guard
// checkpoint/rollback stays correct under MTP/SuffixProposer drafts.
//
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <parsephony/toolcall.hpp>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

// Minimum contiguous card==1 run (bytes) before a jump is allowed.
inline constexpr std::size_t kGrammarJumpMinBytes = 8;

// Exact `1` enables. Unset / `0` / anything else keeps the feature off.
[[nodiscard]] bool grammar_jump_forward_enabled() noexcept;

// Override for tests (null restores getenv). Not for product call sites.
void grammar_jump_forward_set_enabled_for_test(std::optional<bool> enabled) noexcept;

// Peek the unique next forced byte when card==1 and not FreeText.
[[nodiscard]] std::optional<unsigned char>
unique_forced_byte(const parsephony::ToolCallGuard& guard) noexcept;

// Accumulate a contiguous card==1 non-FreeText run by probing a mute copy.
// Does not mutate `guard`. Returns the forced bytes (may be shorter than the
// jump threshold — callers decide whether to jump).
[[nodiscard]] std::string collect_forced_span(const parsephony::ToolCallGuard& guard);

// True when `span` is long enough to skip per-token forwards.
[[nodiscard]] inline bool should_grammar_jump(std::string_view span) noexcept {
    return span.size() >= kGrammarJumpMinBytes;
}

// Walk a golden tool-XML string under the guard and record every maximal
// card==1 non-FreeText span (for museum-style assertions). Does not mutate
// beyond a local probe; feeds `raw` into a fresh walk starting from `guard`'s
// state copy.
struct ForcedSpanRecord {
    std::size_t offset = 0; // byte offset into `raw` where the span began
    std::string bytes;
};

[[nodiscard]] std::vector<ForcedSpanRecord>
enumerate_forced_spans(const parsephony::ToolCallGuard& start, std::string_view raw);

// Tokenise a forced byte span for one KV append-N. Empty on encode/round-trip
// failure (caller must fall back to ordinary decode).
[[nodiscard]] std::vector<TokenId>
tokens_for_forced_span(const QwenTokenizer& tok, std::string_view bytes);

} // namespace lmp::model
