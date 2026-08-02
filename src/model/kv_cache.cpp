#include "src/model/kv_cache.hpp"

#include <algorithm>

namespace lmp::model {
namespace {

constexpr std::uint64_t kSeed = 14695981039346656037ULL; // FNV offset basis, as a seed only

// splitmix64's finaliser. Chosen for avalanche, NOT for invertibility: because the ledger
// stores a hash per position it never needs to subtract a token, which is exactly the
// constraint that pushes a rolling fingerprint onto a linear polynomial -- and onto the
// Thue-Morse collision that comes with one. See the header.
constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
    z ^= z >> 30U;
    z *= 0xBF58476D1CE4E5B9ULL;
    z ^= z >> 27U;
    z *= 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

// The +1 matters: without it a token id of 0 folds as an identity and appending one is
// invisible to the fingerprint.
constexpr std::uint64_t fold(std::uint64_t prev, TokenId id) noexcept {
    const auto widened = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id));
    return mix64(prev ^ (0x9E3779B97F4A7C15ULL * (widened + 1ULL)));
}

} // namespace

std::uint64_t hash_ids(const std::vector<TokenId>& ids) noexcept {
    std::uint64_t h = kSeed;
    for (TokenId id : ids) {
        h = fold(h, id);
    }
    return h;
}

namespace {

// The fingerprint the ledger WOULD hold for the first `n` ids of `v`. Folded the same way
// hashes_ is, so the two are comparable by construction rather than by remembering to.
std::uint64_t hash_prefix(const std::vector<TokenId>& v, std::size_t n) noexcept {
    std::uint64_t h = kSeed;
    for (std::size_t i = 0; i < n && i < v.size(); ++i) {
        h = fold(h, v[i]);
    }
    return h;
}

} // namespace

KvCacheLedger::KvCacheLedger() { hashes_.push_back(kSeed); }

void KvCacheLedger::append(TokenId id) {
    ids_.push_back(id);
    hashes_.push_back(fold(hashes_.back(), id));
}

void KvCacheLedger::append(const std::vector<TokenId>& ids) {
    append(std::span<const TokenId>(ids));
}

void KvCacheLedger::append(std::span<const TokenId> ids) {
    ids_.reserve(ids_.size() + ids.size());
    hashes_.reserve(hashes_.size() + ids.size());
    for (TokenId id : ids) {
        append(id);
    }
}

ReuseDecision KvCacheLedger::plan_reuse(const std::vector<TokenId>& prompt) const {
    ReuseDecision d;
    // Id-by-id, not by hash: the hash exists to key lookups, equality is the proof.
    // std::mismatch rather than a hand-rolled loop -- it vectorises, and it measured
    // fastest of the three cook-off entrants, beating a blocked memcmp and a binary
    // search over memcmp.
    const auto [led_it, prompt_it] =
        std::mismatch(ids_.begin(), ids_.end(), prompt.begin(), prompt.end());
    (void)prompt_it;
    d.reusable = static_cast<std::size_t>(std::distance(ids_.begin(), led_it));
    // Divergent if the cache holds anything beyond the verified-identical prefix --
    // whether a mid-prompt mismatch or a cache LONGER than the prompt. Either way those
    // entries are stale context and must be dropped, not decoded past.
    d.divergent = (led_it != ids_.end());
    return d;
}

void KvCacheLedger::truncate_last(std::size_t n) {
    truncate_to(n >= ids_.size() ? 0 : ids_.size() - n);
}

void KvCacheLedger::truncate_to(std::size_t n) {
    if (n >= ids_.size()) {
        return;
    }
    ids_.resize(n);
    hashes_.resize(n + 1);
}

void KvCacheLedger::clear() noexcept {
    ids_.clear();
    hashes_.clear();
    hashes_.push_back(kSeed);
}

std::uint64_t KvCacheLedger::fingerprint_at(std::size_t k) const noexcept {
    return hashes_[std::min(k, ids_.size())];
}

TurnReuse plan_turn_reuse(const KvCacheLedger& ledger, const std::vector<TokenId>& prompt,
                          std::size_t checkpoint_len, bool checkpoint_valid) {
    const ReuseDecision d = ledger.plan_reuse(prompt);
    if (!d.divergent) {
        // The cache is a verified prefix of the prompt: prefill only the tail. Unchanged
        // from before this function existed, and still the fast path within a turn.
        return {ReuseMode::Extend, d.reusable};
    }
    if (!checkpoint_valid || checkpoint_len == 0 || checkpoint_len > prompt.size() ||
        checkpoint_len > ledger.size()) {
        return {ReuseMode::Reset, 0};
    }
    // VERIFIED, NEVER ASSUMED (S5.10). fingerprint_at() is O(1) and is the right fast
    // reject, but the hash keys the lookup and equality is the proof -- a few thousand
    // int32 compares against a 19 GB model is not a cost worth reasoning about, and the
    // failure this guards is silent.
    if (ledger.fingerprint_at(checkpoint_len) != hash_prefix(prompt, checkpoint_len)) {
        return {ReuseMode::Reset, 0};
    }
    const std::vector<TokenId>& cached = ledger.ids();
    for (std::size_t i = 0; i < checkpoint_len; ++i) {
        if (cached[i] != prompt[i]) {
            return {ReuseMode::Reset, 0};
        }
    }
    return {ReuseMode::Restore, checkpoint_len};
}

} // namespace lmp::model
