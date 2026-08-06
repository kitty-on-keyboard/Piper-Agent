#pragma once
//
// QwenTokenizer -- L1's tokenizer seam, backed by frankentok (spec S5.1-S5.3).
//
// frankentok (third_party/frankentok, project name mlx_qwen_tokenizer) is the "one
// vocab, one owner" object the spec asks for, already proven against this machine's
// production checkpoint: HF-parity encode across 15,045 cases id-for-id, streaming
// decode byte-identical to batch decode (the split-codepoint property -- 944 of the
// 248k tokens are byte fragments), a binary vocab cache with ~6 ms warm loads, and a
// pretokenizer + NFC normalizer READ FROM tokenizer.json rather than assumed. A
// hand-rolled replacement existed here briefly and was deleted for it: its
// pretokenizer approximated the Split regex and skipped NFC, both of which were live
// encode bugs in frankentok's own history until checked against the reference.
//
// This adapter adds the two things the library does not decide for the caller:
//
//   FAMILY VERIFICATION (S5.2). Declared, then verified against the loaded vocab's
//   size band and structural id<->literal pairs. Refused on mismatch -- v1 accepted a
//   Gemma tokenizer by model-name sniffing and mis-tokenized silently.
//
//   SPECIAL-TOKEN POLICY (S5.4). Structural tokens are single ids minted only by the
//   chat template and the grammar. encode() here strips that power from ordinary
//   content: a user message containing the literal "<|im_end|>" must tokenize as text.
//
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace mlx_qwen_tokenizer {
class Tokenizer;
class StreamingDecoder;
} // namespace mlx_qwen_tokenizer

namespace lmp::model {

using TokenId = std::int32_t;
inline constexpr TokenId kInvalidToken = -1;

enum class Family : std::uint8_t { Qwen3 };

struct LoadStatus {
    bool ok = false;
    std::string error;
};

// Qwen3.6's structural tokens as SINGLE ids (measured from the checkpoint:
// im_start=248045, im_end=248046, tool_call=248058/248059, think=248068/248069,
// endoftext=248044). Filled at load, never hardcoded.
struct SpecialIds {
    TokenId im_start = kInvalidToken;
    TokenId im_end = kInvalidToken;
    TokenId endoftext = kInvalidToken;
    TokenId tool_call_open = kInvalidToken;
    TokenId tool_call_close = kInvalidToken;
    TokenId tool_response_open = kInvalidToken;
    TokenId tool_response_close = kInvalidToken;
    TokenId think_open = kInvalidToken;
    TokenId think_close = kInvalidToken;

    [[nodiscard]] bool complete() const noexcept;
};

class QwenTokenizer {
  public:
    QwenTokenizer();
    ~QwenTokenizer();
    QwenTokenizer(const QwenTokenizer&) = delete;
    QwenTokenizer& operator=(const QwenTokenizer&) = delete;

    // Loads tokenizer.json (binary cache consulted first), then verifies the declared
    // family. A mismatch unloads and refuses; there is no "load anyway".
    [[nodiscard]] LoadStatus load(const std::string& tokenizer_json_path, Family declared);

    [[nodiscard]] bool loaded() const noexcept { return loaded_; }
    [[nodiscard]] Family family() const noexcept { return family_; }
    [[nodiscard]] std::size_t vocab_size() const;
    [[nodiscard]] const SpecialIds& specials() const noexcept { return specials_; }

    // Content encoding: specials in the text are inert (encoded as ordinary bytes).
    [[nodiscard]] std::vector<TokenId> encode_content(std::string_view text) const;
    // Template encoding: the trusted path; special literals resolve to their ids.
    // Only chat_template.cpp calls this.
    [[nodiscard]] std::vector<TokenId> encode_template(std::string_view text) const;

    [[nodiscard]] std::string decode(const std::vector<TokenId>& ids) const;

    // The raw bytes a token stands for (decoded out of the byte-level alphabet;
    // literal content for specials). This is what the grammar feeds to ToolCallGuard.
    [[nodiscard]] std::string_view token_bytes(TokenId id) const;
    [[nodiscard]] bool is_special(TokenId id) const;
    [[nodiscard]] TokenId id_for(std::string_view literal) const;

    // The full token table for parsephony's mask engine: bytes per id, special flags.
    void export_mask_vocab(std::vector<std::string>& tokens,
                           std::vector<std::uint8_t>& special) const;

    // One per generation; wraps frankentok's StreamingDecoder (exact split handling).
    class Stream {
      public:
        explicit Stream(const QwenTokenizer& tok);
        ~Stream();
        Stream(const Stream&) = delete;
        Stream& operator=(const Stream&) = delete;
        [[nodiscard]] std::string push(TokenId id);
        [[nodiscard]] std::string flush();

      private:
        std::unique_ptr<mlx_qwen_tokenizer::StreamingDecoder> dec_;
    };

  private:
    std::unique_ptr<mlx_qwen_tokenizer::Tokenizer> tok_;
    SpecialIds specials_;
    Family family_ = Family::Qwen3;
    bool loaded_ = false;
};

} // namespace lmp::model
