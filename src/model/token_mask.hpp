#pragma once
//
// TokenMask -- the legal-token set for one decode step, as a bitset (spec S5.6).
//
// This type exists because the obvious interface was a 3.4x throughput bug. The
// sampler used to take `std::function<bool(TokenId)>` and call it once per id, so a
// mask that is conceptually "everything except eight structural tokens" cost 248,077
// virtual calls per token -- measured at 22.8 ms/token on this machine, a 43.8 tok/s
// ceiling before the model did any work at all. Answering for the whole vocabulary at
// once lets the grammar say that in constant time, and lets the sampler skip 64 ids
// per instruction where the mask is dense.
//
// Bit i set means id i may be emitted next. Ids at or beyond size() are not
// representable and are therefore denied: the model's logits row is wider than the
// tokenizer's vocabulary (248,320 vs 248,077 on this checkpoint) and the trailing rows
// decode to nothing.
//
#include <bit>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

class TokenMask {
  public:
    TokenMask() = default;
    explicit TokenMask(std::size_t vocab_size)
        : n_(vocab_size), w_((vocab_size + 63) / 64, 0) {}

    void reset(std::size_t vocab_size) {
        n_ = vocab_size;
        w_.assign((vocab_size + 63) / 64, 0);
    }

    // Every id legal. The tail bits past n_ stay clear so allows() and count() do not
    // have to special-case the last word.
    void allow_all() {
        w_.assign(w_.size(), ~std::uint64_t{0});
        const std::size_t tail = n_ % 64;
        if (tail != 0 && !w_.empty()) {
            w_.back() = (std::uint64_t{1} << tail) - 1;
        }
    }

    void allow(TokenId id) {
        if (in_range(id)) {
            w_[static_cast<std::size_t>(id) >> 6U] |= bit(id);
        }
    }

    void deny(TokenId id) {
        if (in_range(id)) {
            w_[static_cast<std::size_t>(id) >> 6U] &= ~bit(id);
        }
    }

    [[nodiscard]] bool allows(TokenId id) const noexcept {
        return in_range(id) && (w_[static_cast<std::size_t>(id) >> 6U] & bit(id)) != 0;
    }

    [[nodiscard]] std::size_t size() const noexcept { return n_; }

    [[nodiscard]] std::size_t count() const noexcept {
        std::size_t c = 0;
        for (std::uint64_t word : w_) {
            c += static_cast<std::size_t>(std::popcount(word));
        }
        return c;
    }

    [[nodiscard]] const std::vector<std::uint64_t>& words() const noexcept { return w_; }

    // Adopts a bitset computed elsewhere (parsephony's TokenMaskT, whose layout is the
    // same little-endian word order). `vocab_size` is the caller's, not the source's.
    void adopt(const std::vector<std::uint64_t>& words, std::size_t vocab_size) {
        n_ = vocab_size;
        w_ = words;
        w_.resize((vocab_size + 63) / 64, 0);
    }

  private:
    [[nodiscard]] bool in_range(TokenId id) const noexcept {
        return id >= 0 && static_cast<std::size_t>(id) < n_;
    }
    [[nodiscard]] static std::uint64_t bit(TokenId id) noexcept {
        return std::uint64_t{1} << (static_cast<std::size_t>(id) & 63U);
    }

    std::size_t n_ = 0;
    std::vector<std::uint64_t> w_;
};

// The constrained-decoding seam (S5.6). The backend holds one of these and asks it for
// the legal set once per step; the implementation is stateful -- the sink's on_token
// advances it, so by the time the next step asks, it answers for the new state.
class MaskSource {
  public:
    MaskSource() = default;
    MaskSource(const MaskSource&) = default;
    MaskSource& operator=(const MaskSource&) = default;
    MaskSource(MaskSource&&) = default;
    MaskSource& operator=(MaskSource&&) = default;
    virtual ~MaskSource() = default;

    [[nodiscard]] virtual const TokenMask& mask() const = 0;
};

} // namespace lmp::model
