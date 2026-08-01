#include "prefix_ledger.hpp"

#include <algorithm>

namespace kv {

namespace {

constexpr std::uint64_t kSeed = 0xcbf29ce484222325ULL; // FNV offset basis, as a seed only

// splitmix64's finaliser. Chosen for avalanche, not for invertibility: because the ledger
// stores a hash per position it never needs to subtract a token, which is exactly the
// constraint that pushed all three entrants onto a linear polynomial and onto the
// Thue-Morse collision with it.
constexpr std::uint64_t mix64(std::uint64_t z) noexcept {
    z ^= z >> 30;
    z *= 0xbf58476d1ce4e5b9ULL;
    z ^= z >> 27;
    z *= 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// The +1 and the odd multiplier are e3's point, kept: without them a token id of 0 folds as
// an identity and "append a zero token" becomes invisible to the fingerprint.
constexpr std::uint64_t fold(std::uint64_t prev, TokenId id) noexcept {
    const auto widened = static_cast<std::uint64_t>(static_cast<std::uint32_t>(id));
    return mix64(prev ^ (0x9E3779B97F4A7C15ULL * (widened + 1ULL)));
}

} // namespace

PrefixLedger::PrefixLedger() { hashes_.push_back(kSeed); }

void PrefixLedger::append(TokenId id) {
    ids_.push_back(id);
    hashes_.push_back(fold(hashes_.back(), id));
}

void PrefixLedger::append(std::span<const TokenId> ids) {
    ids_.reserve(ids_.size() + ids.size());
    hashes_.reserve(hashes_.size() + ids.size());
    for (TokenId id : ids) {
        append(id);
    }
}

ReuseDecision PrefixLedger::plan_reuse(std::span<const TokenId> candidate) const {
    // e3's formulation: one std::mismatch, vectorised by the compiler, and measured faster
    // than e1's hand-rolled 256-token memcmp blocks and e2's binary search over memcmp.
    const auto [ledger_it, cand_it] =
        std::mismatch(ids_.begin(), ids_.end(), candidate.begin(), candidate.end());
    (void)cand_it;
    ReuseDecision decision;
    decision.reusable = static_cast<std::size_t>(std::distance(ids_.begin(), ledger_it));
    // Divergent means the candidate left the ledger BEFORE the ledger ended, so cached state
    // past `reusable` is stale. A strict prefix IS divergent; a pure extension is not.
    decision.divergent = (ledger_it != ids_.end());
    return decision;
}

void PrefixLedger::truncate_last(std::size_t n) {
    truncate_to(n >= ids_.size() ? 0 : ids_.size() - n);
}

void PrefixLedger::truncate_to(std::size_t n) {
    if (n >= ids_.size()) {
        return;
    }
    // The whole payoff: no un-rolling, no rehash. The surviving prefix's fingerprint was
    // already computed when those tokens were appended and has not moved since.
    ids_.resize(n);
    hashes_.resize(n + 1);
}

std::size_t PrefixLedger::size() const noexcept { return ids_.size(); }

std::span<const TokenId> PrefixLedger::tokens() const noexcept { return ids_; }

void PrefixLedger::clear() noexcept {
    ids_.clear();
    hashes_.clear();
    hashes_.push_back(kSeed);
}

std::uint64_t PrefixLedger::fingerprint() const noexcept { return hashes_.back(); }

std::uint64_t PrefixLedger::fingerprint_at(std::size_t k) const noexcept {
    return hashes_[std::min(k, ids_.size())];
}

} // namespace kv
