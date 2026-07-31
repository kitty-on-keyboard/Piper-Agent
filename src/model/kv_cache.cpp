#include "src/model/kv_cache.hpp"

namespace lmp::model {
namespace {

constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kFnvBasis = 14695981039346656037ULL;

std::uint64_t fnv_step(std::uint64_t h, TokenId id) noexcept {
    const auto v = static_cast<std::uint32_t>(id);
    for (int shift = 0; shift < 32; shift += 8) {
        h ^= (v >> static_cast<unsigned>(shift)) & 0xFFU;
        h *= kFnvPrime;
    }
    return h;
}

} // namespace

std::uint64_t hash_ids(const std::vector<TokenId>& ids) noexcept {
    std::uint64_t h = kFnvBasis;
    for (TokenId id : ids) {
        h = fnv_step(h, id);
    }
    return h;
}

void KvCacheLedger::append(const std::vector<TokenId>& ids) {
    for (TokenId id : ids) {
        append(id);
    }
}

void KvCacheLedger::append(TokenId id) {
    ids_.push_back(id);
    hash_ = fnv_step(hash_, id);
}

ReuseDecision KvCacheLedger::plan_reuse(const std::vector<TokenId>& prompt) const {
    ReuseDecision d;
    const std::size_t overlap = std::min(ids_.size(), prompt.size());
    // Id-by-id, not by hash: the hash exists to key lookups, equality is the proof.
    std::size_t same = 0;
    while (same < overlap && ids_[same] == prompt[same]) {
        ++same;
    }
    d.reusable = same;
    // Divergent if the cache holds anything beyond the verified-identical prefix --
    // whether a mid-prompt mismatch or a cache LONGER than the prompt. Either way those
    // entries are stale context and must be dropped, not decoded past.
    d.divergent = same < ids_.size();
    return d;
}

void KvCacheLedger::clear() noexcept {
    ids_.clear();
    hash_ = kFnvBasis;
}

} // namespace lmp::model
