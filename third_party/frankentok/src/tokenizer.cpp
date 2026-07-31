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

    const bool wants_nfc = vocab_.wants_nfc();

    for (const auto& piece : special_split) {
        if (piece.empty()) continue;

        // Check if this piece IS a special token
        auto it = vocab_.get_special_token_map().find(piece);
        if (it != vocab_.get_special_token_map().end()) {
            result.push_back(it->second);
        } else {
            // 2. Normalize, then pre-tokenize. Normalization is per piece, matching the reference
            // pipeline, which normalizes each segment between special tokens independently. The
            // views borrow from `piece` or `nfc_buf`, both of which outlive the loop consuming
            // them (nfc_buf is not rewritten until the next piece).
            std::string_view input = piece;
            if (wants_nfc && normalize_nfc(piece, nfc_buf)) input = nfc_buf;

            pretokenizer_->split(input, pretokenized);

            // 3. BPE encode each pre-tokenized chunk, appending straight into the result
            for (std::string_view chunk : pretokenized) {
                bpe_->encode_word(chunk, scratch, result);
            }
        }
    }

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
