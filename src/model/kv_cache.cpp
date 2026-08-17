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

KvCacheLedger::KvCacheLedger() { hashes_.push_back(kSeed); }

void KvCacheLedger::append(TokenId id) { append(id, ContentTag{0}); }

void KvCacheLedger::append(TokenId id, ContentTag tag) {
    ids_.push_back(id);
    tags_.push_back(tag);
    // The tag folds into the fingerprint too, so the cheap reject stays as strong as the
    // proof. Folded as a second mix rather than xor'd into the id: an image tag is a
    // full 64-bit hash and xor'ing it against a token id would let a tag change cancel
    // an id change.
    std::uint64_t h = fold(hashes_.back(), id);
    if (tag != 0) {
        h = mix64(h ^ tag);
    }
    hashes_.push_back(h);
}

void KvCacheLedger::append(const std::vector<TokenId>& ids) {
    append(std::span<const TokenId>(ids));
}

void KvCacheLedger::append(std::span<const TokenId> ids,
                           std::span<const ContentTag> tags) {
    ids_.reserve(ids_.size() + ids.size());
    tags_.reserve(tags_.size() + ids.size());
    hashes_.reserve(hashes_.size() + ids.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
        append(ids[i], i < tags.size() ? tags[i] : ContentTag{0});
    }
}

void KvCacheLedger::append(std::span<const TokenId> ids) {
    ids_.reserve(ids_.size() + ids.size());
    tags_.reserve(tags_.size() + ids.size());
    hashes_.reserve(hashes_.size() + ids.size());
    for (TokenId id : ids) {
        append(id);
    }
}

ReuseDecision KvCacheLedger::plan_reuse(const std::vector<TokenId>& prompt,
                                        std::span<const ContentTag> prompt_tags) const {
    ReuseDecision d;
    // Id-by-id, not by hash: the hash exists to key lookups, equality is the proof.
    // std::mismatch rather than a hand-rolled loop -- it vectorises, and it measured
    // fastest of the three cook-off entrants, beating a blocked memcmp and a binary
    // search over memcmp.
    const auto [led_it, prompt_it] =
        std::mismatch(ids_.begin(), ids_.end(), prompt.begin(), prompt.end());
    (void)prompt_it;
    d.reusable = static_cast<std::size_t>(std::distance(ids_.begin(), led_it));

    // ...then the same walk over the tags, which is what stops two different images
    // behind identical `<|image_pad|>` runs from reading as the same prefix. Kept as a
    // separate pass so the id comparison stays the vectorised one it was measured to be:
    // the tag array is all zeros in every run without an image, and this loop exits at
    // the first difference.
    if (!tags_.empty() || !prompt_tags.empty()) {
        for (std::size_t i = 0; i < d.reusable; ++i) {
            const ContentTag mine = i < tags_.size() ? tags_[i] : ContentTag{0};
            const ContentTag theirs =
                i < prompt_tags.size() ? prompt_tags[i] : ContentTag{0};
            if (mine != theirs) {
                d.reusable = i;
                break;
            }
        }
    }
    // Divergent if the cache holds anything beyond the verified-identical prefix --
    // whether a mid-prompt mismatch or a cache LONGER than the prompt. Either way those
    // entries are stale context and must be dropped, not decoded past.
    //
    // Read off `reusable` rather than the id iterator, because the tag walk above may
    // have shortened it: a run whose ids match and whose images do not is divergent, and
    // testing the iterator alone would have called it clean.
    d.divergent = d.reusable < ids_.size();
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
    tags_.resize(n);
    hashes_.resize(n + 1);
}

void KvCacheLedger::clear() noexcept {
    ids_.clear();
    tags_.clear();
    hashes_.clear();
    hashes_.push_back(kSeed);
}

std::uint64_t KvCacheLedger::fingerprint_at(std::size_t k) const noexcept {
    return hashes_[std::min(k, ids_.size())];
}

TurnReuse plan_turn_reuse(const KvCacheLedger& ledger, const std::vector<TokenId>& prompt,
                          std::size_t checkpoint_len, bool checkpoint_valid,
                          std::span<const ContentTag> prompt_tags) {
    const ReuseDecision d = ledger.plan_reuse(prompt, prompt_tags);
    if (!d.divergent) {
        // The cache is a verified prefix of the prompt: prefill only the tail. Unchanged
        // from before this function existed, and still the fast path within a turn.
        return {ReuseMode::Extend, d.reusable};
    }
    if (!checkpoint_valid || checkpoint_len == 0 || checkpoint_len > prompt.size() ||
        checkpoint_len > ledger.size()) {
        return {ReuseMode::Reset, 0};
    }
    if (checkpoint_len <= d.reusable) {
        return {ReuseMode::Restore, checkpoint_len};
    }
    return {ReuseMode::Reset, 0};
}

} // namespace lmp::model
