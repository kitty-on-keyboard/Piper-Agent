#include "mlx_qwen_tokenizer/tokenizer.h"
#include "mlx_qwen_tokenizer/normalizer.h"
#include "mlx_qwen_tokenizer/streaming_decoder.h"
#include <iostream>

namespace mlx_qwen_tokenizer {

Tokenizer::Tokenizer() {}

Tokenizer::~Tokenizer() {}

bool Tokenizer::load(const std::string& vocab_path, const LoadOptions& options) {
    if (!vocab_.load_from_file(vocab_path, options)) {
        return false;
    }

    special_trie_ = std::make_unique<SpecialTokenTrie>();
    for (const auto& token : vocab_.get_special_tokens()) {
        special_trie_->add(token);
    }

    // Split with the pattern the vocabulary itself declares; the hardcoded default exists only
    // for callers with no tokenizer.json at all.
    pretokenizer_ = vocab_.pretokenizer_regex().empty()
                        ? std::make_unique<Pretokenizer>()
                        : std::make_unique<Pretokenizer>(vocab_.pretokenizer_regex());
    bpe_ = std::make_unique<BPE>(vocab_);

    return true;
}

void Tokenizer::append_ordinary(std::string_view text, BPE::Scratch& scratch,
                                std::vector<std::string_view>& pretokenized,
                                std::string& nfc_buf, std::vector<int32_t>& out) const {
    // NFC, then pretokenize, then BPE. Shared by encode() (per special-split piece, matching
    // the reference pipeline) and encode_ordinary() (the whole string, which is one piece
    // because that path never splits specials).
    std::string_view input = text;
    if (vocab_.wants_nfc() && normalize_nfc(text, nfc_buf)) input = nfc_buf;
    pretokenizer_->split(input, pretokenized);
    for (std::string_view chunk : pretokenized) {
        bpe_->encode_word(chunk, scratch, out);
    }
}

std::vector<int32_t> Tokenizer::encode(std::string_view text) const {
    std::vector<int32_t> result;
    if (!special_trie_ || !pretokenizer_ || !bpe_) return result;

    // 1. Split by special tokens
    std::vector<std::string> special_split = special_trie_->split(text);

    // Reused for every pre-token below. Local to the call rather than members, so two concurrent
    // encodes on the same Tokenizer stay independent — see BPE::Scratch.
    BPE::Scratch scratch;
    std::vector<std::string_view> pretokenized;
    std::string nfc_buf;

    for (const auto& piece : special_split) {
        if (piece.empty()) continue;

        // Check if this piece IS a special token
        auto it = vocab_.get_special_token_map().find(piece);
        if (it != vocab_.get_special_token_map().end()) {
            result.push_back(it->second);
        } else {
            append_ordinary(piece, scratch, pretokenized, nfc_buf, result);
        }
    }

    return result;
}

std::vector<int32_t> Tokenizer::encode_ordinary(std::string_view text) const {
    std::vector<int32_t> result;
    if (!pretokenizer_ || !bpe_) return result;
    BPE::Scratch scratch;
    std::vector<std::string_view> pretokenized;
    std::string nfc_buf;
    append_ordinary(text, scratch, pretokenized, nfc_buf, result);
    return result;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    // Batch decode is a pure concatenation of each token's raw bytes: streaming's buffer logic
    // only reorders *when* bytes are emitted, never which bytes, so running the ids through a
    // StreamingDecoder and flushing produces exactly this string the slow way.
    size_t total = 0;
    for (int32_t id : ids) total += vocab_.id_to_bytes_view(id).size();
    std::string res;
    res.reserve(total);
    for (int32_t id : ids) res.append(vocab_.id_to_bytes_view(id));
    return res;
}

} // namespace mlx_qwen_tokenizer
