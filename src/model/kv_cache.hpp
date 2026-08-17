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

// What a turn should do with the cache it inherited from the previous one.
//
// WHY THIS IS NEEDED AT ALL. plan_reuse() answers "is the cache a prefix of this prompt",
// which between turns is always NO: after turn N the ledger holds [prompt_N][generated_N],
// and turn N+1's prompt inserts a new turn record BEFORE the live-state block, so the two
// diverge mid-ledger. The backend's response was a full reset, so every turn re-prefilled
// the entire context -- which made context.cpp's most-stable-first layout, and the
// measurement that motivated it, buy nothing.
//
// The way out is a checkpoint taken in advance at the end of the stable prefix.
// qwen35_moe_model.hpp says why it must be a checkpoint rather than a rollback_to(n): 30
// of this model's 40 layers are gated-delta, a recurrence with no per-token history, so
// only positions snapshotted ahead of time are reachable at all.
enum class ReuseMode : std::uint8_t {
    Extend,   // the cache is a verified prefix of the prompt; prefill the tail
    Restore,  // roll back to the saved checkpoint, then prefill from there
    Reset,    // nothing usable; one honest full re-prefill
};

struct TurnReuse {
    ReuseMode mode = ReuseMode::Reset;
    std::size_t prefill_from = 0;
};

// WHAT A POSITION HOLDS THAT ITS TOKEN ID DOES NOT.
//
// The ledger's whole claim is "the KV at these positions is what this prompt prefix would
// produce", and it proves it by comparing token ids. That is sound only while the ids
// determine the embeddings -- which images break: every image in the prompt is a run of
// the SAME `<|image_pad|>` id, and the rows spliced over them come from the picture. Two
// different screenshots produce byte-identical id runs, so an id-only comparison reports
// a verified reuse of a prefix whose KV encodes the other one, and the model describes a
// picture it was never shown. Fluently, and with nothing in the log to say so.
//
// So a position may carry a tag: zero for an ordinary token, and the content hash of the
// image covering it otherwise. Comparison is over the PAIR. This strengthens the existing
// contract rather than adding an exception to it -- ids alone were never the thing that
// determined the cache, they were only a proxy that happened to be exact until images.
using ContentTag = std::uint64_t;

class KvCacheLedger {
  public:
    KvCacheLedger();

    // Called after the backend actually consumed these ids (prefill or decode). The
    // overloads without tags mean "all ordinary", which is every generated token and
    // every text prompt.
    void append(const std::vector<TokenId>& ids);
    void append(std::span<const TokenId> ids);
    void append(TokenId id);
    void append(TokenId id, ContentTag tag);
    void append(std::span<const TokenId> ids, std::span<const ContentTag> tags);

    // Compares the cached ids against `prompt`, id by id AND tag by tag. The hash is a
    // fast reject; equality is the proof. A hash match alone is never trusted (S5.10:
    // verified, never assumed).
    //
    // `prompt_tags` empty means the prompt carries no images. It is NOT required to be
    // the same length as `prompt`; positions past its end are ordinary.
    [[nodiscard]] ReuseDecision plan_reuse(
        const std::vector<TokenId>& prompt,
        std::span<const ContentTag> prompt_tags = {}) const;

    [[nodiscard]] const std::vector<ContentTag>& tags() const noexcept { return tags_; }

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
    // Parallel to ids_, and zero for every ordinary token -- which is nearly all of them,
    // so this is 8 bytes a position that is almost always the same 8 bytes. Kept dense
    // rather than as a sparse map so that truncate_to stays the O(1) resize that the
    // speculative rollback depends on.
    std::vector<ContentTag> tags_;
    // hashes_[i] is the fingerprint of ids_[0..i). Always size() + 1 entries, so
    // hashes_[0] is the empty-ledger seed and "a ledger equals the prefix it was
    // truncated to" holds by construction rather than by arithmetic.
    std::vector<std::uint64_t> hashes_;
};

// Pure, so the whole decision is testable in the gate with no GPU and no 19 GB
// checkpoint -- the same argument speculative.hpp makes for keeping the block algebra
// model-free, and for the same reason: an off-by-one here does not crash, it reuses a
// cache against the wrong prefix and the text stays fluent.
//
// Invalidation is NOT enumerated. Compaction, a steering message, a persona change and a
// tools-block change all rewrite the stable prefix; none of them is special-cased,
// because rule 2's id-by-id comparison fails and the answer falls through to Reset. An
// enumeration is a list someone forgets to extend.
[[nodiscard]] TurnReuse plan_turn_reuse(const KvCacheLedger& ledger,
                                        const std::vector<TokenId>& prompt,
                                        std::size_t checkpoint_len, bool checkpoint_valid,
                                        std::span<const ContentTag> prompt_tags = {});

} // namespace lmp::model
