#include "src/model/qwen_tokenizer.hpp"

#include <array>

#include "mlx_qwen_tokenizer/bpe.h"
#include "mlx_qwen_tokenizer/normalizer.h"
#include "mlx_qwen_tokenizer/pretokenizer.h"
#include "mlx_qwen_tokenizer/streaming_decoder.h"
#include "mlx_qwen_tokenizer/tokenizer.h"

namespace lmp::model {
namespace ft = mlx_qwen_tokenizer;

bool SpecialIds::complete() const noexcept {
    return im_start != kInvalidToken && im_end != kInvalidToken &&
           tool_call_open != kInvalidToken && tool_call_close != kInvalidToken &&
           tool_response_open != kInvalidToken && tool_response_close != kInvalidToken &&
           think_open != kInvalidToken && think_close != kInvalidToken;
}

QwenTokenizer::QwenTokenizer() = default;
QwenTokenizer::~QwenTokenizer() = default;

LoadStatus QwenTokenizer::load(const std::string& path, Family declared) {
    if (declared != Family::Qwen3) {
        return {false, "unsupported family"};
    }
    tok_ = std::make_unique<ft::Tokenizer>();
    if (!tok_->load(path)) {
        tok_.reset();
        return {false, path + ": frankentok failed to load (missing file or malformed "
                       "tokenizer.json)"};
    }

    // Family verification (S5.2): size band plus structural literals. Qwen3 dense ships
    // ~151,936 entries, Qwen3.6 MoE ships 248,077 (measured from this machine's
    // checkpoint). A Gemma tokenizer fails both probes below.
    const std::size_t n = tok_->get_vocab().size();
    if (n < 140000 || n > 260000) {
        tok_.reset();
        return {false, "vocab size " + std::to_string(n) +
                           " is outside the Qwen3 band [140000, 260000]. Refusing to "
                           "load: silent mis-tokenization looks like a stupid model, so "
                           "this is a refusal, not a warning."};
    }
    const auto lookup = [this](const char* lit) {
        return static_cast<TokenId>(tok_->get_vocab().token_to_id(lit));
    };
    specials_.im_start = lookup("<|im_start|>");
    specials_.im_end = lookup("<|im_end|>");
    specials_.endoftext = lookup("<|endoftext|>");
    specials_.tool_call_open = lookup("<tool_call>");
    specials_.tool_call_close = lookup("</tool_call>");
    specials_.tool_response_open = lookup("<tool_response>");
    specials_.tool_response_close = lookup("</tool_response>");
    specials_.think_open = lookup("<think>");
    specials_.think_close = lookup("</think>");
    if (!specials_.complete()) {
        tok_.reset();
        return {false, "declared family Qwen3 but the structural tokens are incomplete; "
                       "refusing to load"};
    }
    family_ = declared;
    loaded_ = true;
    return {true, {}};
}

std::size_t QwenTokenizer::vocab_size() const {
    return loaded_ ? tok_->get_vocab().size() : 0;
}

std::vector<TokenId> QwenTokenizer::encode_template(std::string_view text) const {
    if (!loaded_) {
        return {};
    }
    return tok_->encode(text);
}

std::vector<TokenId> QwenTokenizer::encode_content(std::string_view text) const {
    if (!loaded_) {
        return {};
    }
    // The same NFC -> pretokenize -> BPE pipeline frankentok's own encode runs, minus
    // the special-token trie. Content cannot mint control ids: "<|im_end|>" in a user
    // message tokenizes as ordinary text (S5.4). This is the token layer's entire
    // prompt-injection defence, so it is structural, not a flag someone can forget.
    const ft::Vocab& vocab = tok_->get_vocab();
    const ft::Pretokenizer pre = vocab.pretokenizer_regex().empty()
                                     ? ft::Pretokenizer()
                                     : ft::Pretokenizer(vocab.pretokenizer_regex());
    const ft::BPE bpe(vocab);

    std::string nfc;
    std::string_view input = text;
    if (vocab.wants_nfc() && ft::normalize_nfc(text, nfc)) {
        input = nfc;
    }

    std::vector<TokenId> out;
    std::vector<std::string_view> chunks;
    pre.split(input, chunks);
    ft::BPE::Scratch scratch;
    for (std::string_view chunk : chunks) {
        bpe.encode_word(chunk, scratch, out);
    }
    return out;
}

std::string QwenTokenizer::decode(const std::vector<TokenId>& ids) const {
    if (!loaded_) {
        return {};
    }
    return tok_->decode(ids);
}

std::string_view QwenTokenizer::token_bytes(TokenId id) const {
    if (!loaded_) {
        return {};
    }
    return tok_->get_vocab().id_to_bytes_view(id);
}

bool QwenTokenizer::is_special(TokenId id) const {
    return loaded_ && tok_->get_vocab().is_special_id(id);
}

TokenId QwenTokenizer::id_for(std::string_view literal) const {
    if (!loaded_) {
        return kInvalidToken;
    }
    return static_cast<TokenId>(tok_->get_vocab().token_to_id(literal));
}

void QwenTokenizer::export_mask_vocab(std::vector<std::string>& tokens,
                                      std::vector<std::uint8_t>& special) const {
    tokens.clear();
    special.clear();
    if (!loaded_) {
        return;
    }
    const std::size_t n = tok_->get_vocab().size();
    tokens.reserve(n);
    special.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto id = static_cast<TokenId>(i);
        tokens.emplace_back(token_bytes(id));
        special.push_back(is_special(id) ? 1U : 0U);
    }
}

QwenTokenizer::Stream::Stream(const QwenTokenizer& tok)
    : dec_(std::make_unique<ft::StreamingDecoder>(tok.tok_->get_vocab())) {}

QwenTokenizer::Stream::~Stream() = default;

std::string QwenTokenizer::Stream::push(TokenId id) { return dec_->decode(id); }

std::string QwenTokenizer::Stream::flush() { return dec_->flush(); }

} // namespace lmp::model
