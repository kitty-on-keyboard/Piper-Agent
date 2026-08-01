// Four planted defects, one per figure of merit the scoreboard claims to detect:
//  (a) divergent = (reusable != candidate.size())  -- inverts prefix vs extension
//  (b) fingerprint = sum of tokens                 -- collides on any transposition
//  (c) fingerprint folds in size()                 -- breaks path-equality? (no: size is
//      equal on equal contents, so this one is a control that should NOT fire)
//  (d) fingerprint ignores position                -- covered by (b)
#include <prefix_ledger.hpp>
namespace kv {
void PrefixLedger::append(TokenId id) { ids_.push_back(id); }
void PrefixLedger::append(std::span<const TokenId> ids) { ids_.insert(ids_.end(), ids.begin(), ids.end()); }
ReuseDecision PrefixLedger::plan_reuse(std::span<const TokenId> c) const {
    std::size_t i = 0;
    while (i < c.size() && i < ids_.size() && c[i] == ids_[i]) ++i;
    return ReuseDecision{i, i != c.size()};   // (a)
}
void PrefixLedger::truncate_last(std::size_t n) { ids_.resize(n >= ids_.size() ? 0 : ids_.size() - n); }
void PrefixLedger::truncate_to(std::size_t n) { if (n < ids_.size()) ids_.resize(n); }
std::size_t PrefixLedger::size() const noexcept { return ids_.size(); }
std::span<const TokenId> PrefixLedger::tokens() const noexcept { return ids_; }
void PrefixLedger::clear() noexcept { ids_.clear(); }
std::uint64_t PrefixLedger::fingerprint() const noexcept {
    std::uint64_t h = 0;
    for (TokenId t : ids_) h += static_cast<std::uint64_t>(t);   // (b)
    return h;
}
} // namespace kv
