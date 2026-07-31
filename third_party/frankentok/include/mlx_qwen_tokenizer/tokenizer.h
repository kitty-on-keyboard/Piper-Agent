#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <string_view>

#include "mlx_qwen_tokenizer/vocab.h"
#include "mlx_qwen_tokenizer/trie.h"
#include "mlx_qwen_tokenizer/pretokenizer.h"
#include "mlx_qwen_tokenizer/bpe.h"

namespace mlx_qwen_tokenizer {

class Tokenizer {
public:
    Tokenizer();
    ~Tokenizer();

    // Prevent copying to manage unique_ptrs safely
    Tokenizer(const Tokenizer&) = delete;
    Tokenizer& operator=(const Tokenizer&) = delete;

    bool load(const std::string& vocab_path, const LoadOptions& options = {});

    std::vector<int32_t> encode(std::string_view text) const;
    std::string decode(const std::vector<int32_t>& ids) const;

    const Vocab& get_vocab() const { return vocab_; }

private:
    Vocab vocab_;
    std::unique_ptr<SpecialTokenTrie> special_trie_;
    std::unique_ptr<Pretokenizer> pretokenizer_;
    std::unique_ptr<BPE> bpe_;
};

} // namespace mlx_qwen_tokenizer
