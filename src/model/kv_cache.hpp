#pragma once
//
// KvCacheLedger -- the bookkeeping half of the KV cache (spec S5.10).
//
// The tensors live in the MLX backend; this class owns the INVARIANTS, which is where
// v1's bugs were:
//
//   * The ledger holds exactly the token ids the model actually consumed.
//   * Prefix reuse is VERIFIED, never assumed: reuse is keyed on a hash of the exact
//     token-id prefix and confirmed id-by-id before use. Reusing by index without
//     verification produces silent stale context -- the run doesn't error, it just
//     gets worse.
//   * Truncation is exact: a ledger rolled back to n is indistinguishable from one that
//     only ever saw n tokens, fingerprint included.
//
// Separating the ledger from the tensors also makes the invariants testable in the
// gate, with no GPU -- which is the only place a "silent stale context" bug can be made
// loud cheaply.
//
// TRUNCATION AND THE FINGERPRINT (adopted 2026-08-01 from the Brief D cook-off; the
// amalgamation and its scoreboard are in bakeoff/prefix_ledger/). This used to be
// append-only, with "one honest full re-prefill" as the only trim, because under
// append-only KV a fresh window was the natural restart. Speculative decoding needs the
// thing in between: it forwards k guessed tokens and keeps only the ones that survive.
//
// All three cook-off entrants implemented that with a polynomial rolling hash mod 2^64,
// un-rolled on truncation via the multiplier's modular inverse. That design is forced --
// to subtract a token from a single scalar of state you need an invertible fold, and
// invertible-and-cheap means linear -- and it is constructively breakable: for a
// Thue-Morse word and its complement the hash difference is divisible by
// prod_j (1 - P^(2^j)), whose 2-adic valuation grows quadratically, so it exceeds 64 at a
// short length for ANY odd multiplier. Measured against all three entrants' actual
// constants: collision at 1024 tokens, which is a short prompt.
//
// So the running hash is kept PER POSITION instead. hashes_[i] is the fingerprint of the
// first i tokens. Truncation stops un-rolling anything (a resize and a read), the fold no
// longer has to be invertible -- so it is splitmix64's finaliser, which the Thue-Morse
// identity does not touch -- and fingerprint_at() becomes O(1), which is what the caller
// actually wants after plan_reuse hands it a reusable prefix length. Costs 8 bytes per
// token against a 19 GB model.
//
#include <cstdint>
#include <span>
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
    KvCacheLedger();

    // Called after the backend actually consumed these ids (prefill or decode).
    void append(const std::vector<TokenId>& ids);
    void append(std::span<const TokenId> ids);
    void append(TokenId id);

    // Compares the cached ids against `prompt`, id by id. The hash is a fast reject;
    // equality is the proof. A hash match alone is never trusted (S5.10: verified,
    // never assumed).
    [[nodiscard]] ReuseDecision plan_reuse(const std::vector<TokenId>& prompt) const;

    // Drop the last `n` tokens; more than is held clears it. THE SPECULATIVE ROLLBACK.
    void truncate_last(std::size_t n);
    // Keep exactly the first `n`. O(1), and exact: the surviving prefix's fingerprint was
    // computed when those tokens were appended and has not moved since.
    void truncate_to(std::size_t n);

    // Full reset.
    void clear() noexcept;

    [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }
    [[nodiscard]] const std::vector<TokenId>& ids() const noexcept { return ids_; }
    [[nodiscard]] std::uint64_t content_hash() const noexcept { return hashes_.back(); }

    // The fingerprint this ledger WOULD have if truncated to its first `k` tokens,
    // without truncating it. O(1); `k > size()` clamps.
    [[nodiscard]] std::uint64_t fingerprint_at(std::size_t k) const noexcept;

  private:
    std::vector<TokenId> ids_;
    // hashes_[i] is the fingerprint of ids_[0..i). Always size() + 1 entries, so
    // hashes_[0] is the empty-ledger seed and "a ledger equals the prefix it was
    // truncated to" holds by construction rather than by arithmetic.
    std::vector<std::uint64_t> hashes_;
};

} // namespace lmp::model
