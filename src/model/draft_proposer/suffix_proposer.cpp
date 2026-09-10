#include "suffix_proposer.hpp"

#include <algorithm>
#include <deque>
#include <unordered_map>

namespace draft {

namespace {
constexpr std::uint32_t kNil = 0;
}

struct SuffixProposer::Impl {
    // 16 bytes. Count is capped rather than bit-packed: A1 packed it into 24 bits, which
    // saves 4 bytes per node and costs a mask on every read in the hot loop. At the sizes
    // this index reaches, the allocation is not the constraint.
    struct Node {
        TokenId token = 0;
        std::uint32_t count = 0;
        std::uint32_t first_child = kNil;
        std::uint32_t next_sibling = kNil;
    };

    explicit Impl(Config c) : config_(c) {
        nodes_.push_back({}); // index 0 is the nil sentinel
        if (config_.min_match_len == 0) {
            config_.min_match_len = 1;
        }
        if (config_.max_match_len < config_.min_match_len) {
            config_.max_match_len = config_.min_match_len;
        }
    }

    // --- build ---------------------------------------------------------------

    std::uint32_t child_of(std::uint32_t parent, TokenId tok) const {
        std::uint32_t c = nodes_[parent].first_child;
        while (c != kNil) {
            if (nodes_[c].token == tok) {
                return c;
            }
            c = nodes_[c].next_sibling;
        }
        return kNil;
    }

    std::uint32_t child_or_create(std::uint32_t parent, TokenId tok) {
        const std::uint32_t existing = child_of(parent, tok);
        if (existing != kNil) {
            return existing;
        }
        nodes_.push_back({tok, 0, kNil, nodes_[parent].first_child});
        const auto idx = static_cast<std::uint32_t>(nodes_.size() - 1);
        nodes_[parent].first_child = idx;
        return idx;
    }

    void insert_suffix(std::span<const TokenId> s) {
        if (s.empty()) {
            return;
        }
        auto it = roots_.find(s[0]);
        if (it == roots_.end()) {
            nodes_.push_back({s[0], 0, kNil, kNil});
            it = roots_.emplace(s[0], static_cast<std::uint32_t>(nodes_.size() - 1)).first;
        }
        std::uint32_t cur = it->second;
        ++nodes_[cur].count;
        for (std::size_t i = 1; i < s.size(); ++i) {
            cur = child_or_create(cur, s[i]);
            ++nodes_[cur].count;
        }
    }

    void remove_suffix(std::span<const TokenId> s) {
        if (s.empty()) {
            return;
        }
        const auto it = roots_.find(s[0]);
        if (it == roots_.end()) {
            return;
        }
        std::uint32_t cur = it->second;
        if (nodes_[cur].count > 0) {
            --nodes_[cur].count;
        }
        for (std::size_t i = 1; i < s.size(); ++i) {
            cur = child_of(cur, s[i]);
            if (cur == kNil) {
                return;
            }
            if (nodes_[cur].count > 0) {
                --nodes_[cur].count;
            }
        }
    }

    // Every suffix start, capped at max_match_len + 1 so a proposal can always be read off
    // the end of the deepest match.
    std::size_t window() const { return config_.max_match_len + 1; }

    void ingest(std::span<const TokenId> seq) {
        if (seq.empty()) {
            return;
        }
        history_.emplace_back(seq.begin(), seq.end());
        total_ += seq.size();
        const std::vector<TokenId>& s = history_.back();
        for (std::size_t i = 0; i < s.size(); ++i) {
            const std::size_t len = std::min(window(), s.size() - i);
            insert_suffix(std::span<const TokenId>(s.data() + i, len));
        }
        evict();
    }

    void evict() {
        if (config_.max_indexed_tokens == 0) {
            return;
        }
        while (total_ > config_.max_indexed_tokens && !history_.empty()) {
            const std::vector<TokenId>& s = history_.front();
            for (std::size_t i = 0; i < s.size(); ++i) {
                const std::size_t len = std::min(window(), s.size() - i);
                remove_suffix(std::span<const TokenId>(s.data() + i, len));
            }
            total_ -= s.size();
            history_.pop_front();
        }
    }

    // --- propose -------------------------------------------------------------

    // The node for the length-`len` suffix of `ctx`, or kNil. Count is 0 when absent, so
    // callers can treat "missing" and "unsupported" as one predicate.
    std::uint32_t descend(std::span<const TokenId> ctx, std::size_t len) const {
        const TokenId* tail = ctx.data() + (ctx.size() - len);
        const auto it = roots_.find(tail[0]);
        if (it == roots_.end()) {
            return kNil;
        }
        std::uint32_t cur = it->second;
        for (std::size_t i = 1; i < len; ++i) {
            cur = child_of(cur, tail[i]);
            if (cur == kNil) {
                return kNil;
            }
        }
        return cur;
    }

    // Longest match that ALSO has support, found by BINARY SEARCH on the length.
    //
    // Every occurrence of a length-(L+1) suffix contains an occurrence of the length-L one
    // ending at the same position, so support is monotonically non-increasing in L -- and
    // absence is monotone too, since if length L is missing so is L+1. That makes
    // "count(L) >= min_support" a monotone predicate and the longest satisfying length a
    // binary search: ~5 descents instead of up to 30. Every entrant scanned linearly from
    // the longest, which is what made the support check expensive enough to skip.
    std::uint32_t find_match(std::span<const TokenId> ctx, std::size_t& matched_len) const {
        matched_len = 0;
        const std::size_t longest = std::min(ctx.size(), config_.max_match_len);
        if (longest < config_.min_match_len) {
            return kNil;
        }
        std::size_t lo = config_.min_match_len; // candidate, not yet proven
        std::size_t hi = longest;
        std::uint32_t best = kNil;
        std::size_t best_len = 0;
        while (lo <= hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            const std::uint32_t node = descend(ctx, mid);
            if (node != kNil && nodes_[node].count >= config_.min_support) {
                best = node;
                best_len = mid;
                lo = mid + 1; // supported here, so look for a longer one
            } else {
                if (mid == config_.min_match_len) {
                    break;
                }
                hi = mid - 1;
            }
        }
        // An unsupported match is not evidence; report nothing rather than guess from it.
        matched_len = best_len;
        return best;
    }

    Proposal propose(std::span<const TokenId> ctx, std::size_t max_tokens) const {
        Proposal out;
        if (ctx.empty() || max_tokens == 0) {
            return out;
        }
        std::size_t matched = 0;
        std::uint32_t cur = find_match(ctx, matched);
        if (cur == kNil) {
            return out;
        }
        out.matched_len = matched;

        float cumulative = 1.0F;
        out.tokens.reserve(max_tokens);
        for (std::size_t i = 0; i < max_tokens; ++i) {
            const std::uint32_t parent_count = nodes_[cur].count;
            if (parent_count == 0) {
                break;
            }
            // Most frequent continuation; ties broken by smaller token id so the result is
            // independent of insertion order.
            std::uint32_t best = kNil;
            std::uint32_t best_count = 0;
            for (std::uint32_t c = nodes_[cur].first_child; c != kNil;
                 c = nodes_[c].next_sibling) {
                if (nodes_[c].count > best_count ||
                    (nodes_[c].count == best_count && best != kNil &&
                     nodes_[c].token < nodes_[best].token)) {
                    best_count = nodes_[c].count;
                    best = c;
                }
            }
            if (best == kNil || best_count == 0) {
                break;
            }
            const float step = static_cast<float>(best_count) / static_cast<float>(parent_count);
            const float next_cumulative = cumulative * step;
            // THE STOP RULE. One more verified position costs draft_cost_ratio and returns
            // a token with probability next_cumulative, so stop the moment that trade turns
            // negative. Gating on `step` instead (as the cook-off did) ignores that a token
            // after a wrong one is wasted no matter how likely it looks in isolation.
            if (next_cumulative < config_.draft_cost_ratio) {
                break;
            }
            cumulative = next_cumulative;
            out.tokens.push_back(nodes_[best].token);
            cur = best;
        }
        out.confidence = out.tokens.empty() ? 0.0F : cumulative;
        return out;
    }

    Config config_;
    std::vector<Node> nodes_;
    std::unordered_map<TokenId, std::uint32_t> roots_;
    std::deque<std::vector<TokenId>> history_;
    std::size_t total_ = 0;
};

SuffixProposer::SuffixProposer(Config config)
    : impl_(std::make_unique<Impl>(config)) {}
SuffixProposer::~SuffixProposer() = default;

void SuffixProposer::ingest(std::span<const TokenId> sequence) { impl_->ingest(sequence); }

Proposal SuffixProposer::propose(std::span<const TokenId> context,
                                 std::size_t max_tokens) const {
    return impl_->propose(context, max_tokens);
}

std::size_t SuffixProposer::indexed_tokens() const noexcept { return impl_->total_; }
std::size_t SuffixProposer::sequence_count() const noexcept { return impl_->history_.size(); }

void SuffixProposer::clear() noexcept {
    impl_->nodes_.clear();
    impl_->nodes_.push_back({});
    impl_->roots_.clear();
    impl_->history_.clear();
    impl_->total_ = 0;
}

} // namespace draft
