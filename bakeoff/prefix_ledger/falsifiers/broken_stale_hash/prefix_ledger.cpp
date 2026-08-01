// One planted defect: the running fingerprint is updated on append and NOT on truncation,
// so a rolled-back ledger keeps the hash of tokens it no longer holds. This is the classic
// version of the bug and it is invisible to every functional test in the brief.
#include <prefix_ledger.hpp>
namespace kv {
static std::uint64_t fold(std::uint64_t h, TokenId t) {
    h ^= static_cast<std::uint64_t>(static_cast<std::uint32_t>(t));
    return h * 1099511628211ULL;
}
void PrefixLedger::append(TokenId id) { ids_.push_back(id); hash_ = fold(hash_, id); }
void PrefixLedger::append(std::span<const TokenId> ids) { for (TokenId t : ids) append(t); }
ReuseDecision PrefixLedger::plan_reuse(std::span<const TokenId> c) const {
    std::size_t i = 0;
    while (i < c.size() && i < ids_.size() && c[i] == ids_[i]) ++i;
    return ReuseDecision{i, i < ids_.size()};
}
void PrefixLedger::truncate_last(std::size_t n) { ids_.resize(n >= ids_.size() ? 0 : ids_.size() - n); }
void PrefixLedger::truncate_to(std::size_t n) { if (n < ids_.size()) ids_.resize(n); }
std::size_t PrefixLedger::size() const noexcept { return ids_.size(); }
std::span<const TokenId> PrefixLedger::tokens() const noexcept { return ids_; }
void PrefixLedger::clear() noexcept { ids_.clear(); hash_ = 1469598103934665603ULL; }
std::uint64_t PrefixLedger::fingerprint() const noexcept { return hash_; }
} // namespace kv
