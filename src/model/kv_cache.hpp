#pragma once
//
// KvCacheLedger -- the bookkeeping half of the KV cache (spec S5.10).
//
// The tensors live in the MLX backend; this class owns the INVARIANTS, which is where
// v1's bugs were:
//
//   * Append-only. The ledger holds exactly the token ids the model actually consumed.
//   * Prefix reuse is VERIFIED, never assumed: reuse is keyed on a hash of the exact
//     token-id prefix and confirmed id-by-id before use. Reusing by index without
//     verification produces silent stale context -- the run doesn't error, it just
//     gets worse.
//   * Trimming pays one honest full re-prefill. There is no partial-trim API on
//     purpose; a fresh window is the natural restart under append-only KV (S8.3).
//
// Separating the ledger from the tensors also makes the invariants testable in the
// gate, with no GPU -- which is the only place a "silent stale context" bug can be made
// loud cheaply.
//
#include <cstdint>
#include <vector>

#include "src/model/qwen_tokenizer.hpp"

namespace lmp::model {

[[nodiscard]] std::uint64_t hash_ids(const std::vector<TokenId>& ids) noexcept;

struct ReuseDecision {
    // Number of leading tokens of `prompt` that are already in cache and VERIFIED
    // identical. The backend prefills from this offset.
    std::size_t reusable = 0;
    // True when the cache held something the prompt does not start with; the caller
    // must reset before prefilling. Never silently ignored.
    bool divergent = false;
};

class KvCacheLedger {
  public:
    // Called after the backend actually consumed these ids (prefill or decode).
    void append(const std::vector<TokenId>& ids);
    void append(TokenId id);

    // Compares the cached ids against `prompt`, id by id. The hash is a fast reject;
    // equality is the proof. A hash match alone is never trusted (S5.10: verified,
    // never assumed).
    [[nodiscard]] ReuseDecision plan_reuse(const std::vector<TokenId>& prompt) const;

    // Full reset -- the only trim there is.
    void clear() noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }
    [[nodiscard]] const std::vector<TokenId>& ids() const noexcept { return ids_; }
    [[nodiscard]] std::uint64_t content_hash() const noexcept { return hash_; }

  private:
    std::vector<TokenId> ids_;
    std::uint64_t hash_ = 14695981039346656037ULL; // FNV offset basis
};

} // namespace lmp::model
