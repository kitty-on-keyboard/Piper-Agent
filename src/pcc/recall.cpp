#include "src/pcc/recall.hpp"

#include <algorithm>
#include <ctime>
#include <unordered_map>

namespace lmp::pcc {
namespace {

std::string format_time(TimeUs us) {
    const auto secs = static_cast<std::time_t>(us / 1000000);
    std::tm tm{};
    gmtime_r(&secs, &tm);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%SZ", &tm);
    return buf;
}

// One entry's prompt-facing form. The header is not decoration: the URI is how a caller
// asks for the rest, and the timestamp is what lets a model notice that two recalled
// facts disagree because one of them is old.
std::string render_entry(const Item& item, const std::string& uri) {
    std::string out = "## " + uri;
    if (!item.title.empty()) {
        out += "  " + item.title;
    }
    out += "\n[" + item.kind + " | " + format_time(item.valid_from);
    if (item.valid_to != kOpenEnded) {
        out += " until " + format_time(item.valid_to) + " -- SUPERSEDED";
    }
    out += "]\n";
    out += item.body;
    if (!item.body.empty() && item.body.back() != '\n') {
        out += '\n';
    }
    return out;
}

// Fuses BM25 order with recency order over the same candidates.
std::vector<RecallEntry> fuse(const std::vector<Item>& by_relevance) {
    std::vector<std::size_t> by_recency(by_relevance.size());
    for (std::size_t i = 0; i < by_recency.size(); ++i) {
        by_recency[i] = i;
    }
    std::stable_sort(by_recency.begin(), by_recency.end(),
                     [&by_relevance](std::size_t a, std::size_t b) {
                         return by_relevance[a].valid_from > by_relevance[b].valid_from;
                     });

    std::vector<double> score(by_relevance.size(), 0.0);
    for (std::size_t rank = 0; rank < by_relevance.size(); ++rank) {
        score[rank] += 1.0 / (kRrfK + static_cast<double>(rank) + 1.0);
    }
    for (std::size_t rank = 0; rank < by_recency.size(); ++rank) {
        score[by_recency[rank]] += 1.0 / (kRrfK + static_cast<double>(rank) + 1.0);
    }

    std::vector<RecallEntry> entries;
    entries.reserve(by_relevance.size());
    for (std::size_t i = 0; i < by_relevance.size(); ++i) {
        entries.push_back({by_relevance[i], score[i], item_uri(by_relevance[i]), false});
    }
    // Freshness breaks ties, and ties are the COMMON case here rather than a corner: two
    // lists over the same candidate set produce identical fused scores whenever the two
    // orderings are reverses of each other, which is exactly what happens when BM25
    // cannot separate two equally relevant rows. Without this the stable sort falls back
    // on BM25 order and the older of two identical facts wins -- silently, and precisely
    // when the store is doing the job it exists for.
    std::stable_sort(entries.begin(), entries.end(),
                     [](const RecallEntry& a, const RecallEntry& b) {
                         if (a.score != b.score) {
                             return a.score > b.score;
                         }
                         return a.item.valid_from > b.item.valid_from;
                     });
    return entries;
}

constexpr const char* kTailHeader = "## Over budget, not included -- read by URI:\n";

std::string pointer_line(const RecallEntry& entry) {
    return "- " + entry.uri + (entry.item.title.empty() ? "" : "  " + entry.item.title) +
           "\n";
}

// What each entry costs in each of its two forms.
struct Costs {
    std::vector<std::string> full;
    std::vector<std::string> pointer;
    std::vector<std::size_t> full_cost;
    std::vector<std::size_t> pointer_cost;
};

Costs measure(const std::vector<RecallEntry>& entries, const TokenCounter& count_tokens) {
    Costs c;
    const std::size_t n = entries.size();
    c.full.reserve(n);
    c.pointer.reserve(n);
    for (const RecallEntry& entry : entries) {
        c.full.push_back(render_entry(entry.item, entry.uri) + "\n");
        c.pointer.push_back(pointer_line(entry));
        c.full_cost.push_back(count_tokens(c.full.back()));
        c.pointer_cost.push_back(count_tokens(c.pointer.back()));
    }
    return c;
}

// Best-first, skipping what does not fit and CONTINUING. One 40 KB tool result in the
// middle of the ranking must not evict the eight small facts behind it, which is what
// stopping at the first overflow would do.
std::vector<bool> greedy(const Costs& costs, std::size_t budget) {
    std::vector<bool> included(costs.full.size(), false);
    std::size_t used = 0;
    for (std::size_t i = 0; i < costs.full.size(); ++i) {
        if (used + costs.full_cost[i] > budget) {
            continue;
        }
        included[i] = true;
        used += costs.full_cost[i];
    }
    return included;
}

// The pointer list gets at most this share of the budget. Without a cap a tight budget
// spends the whole allowance listing what it could not afford to include -- thirty URIs
// and no content, which is strictly worse than two facts and a truncated list. Reserved
// only when there IS something to list, so a recall that fits entirely pays nothing.
constexpr std::size_t kTailShare = 4;

std::string build_tail(const Costs& costs, const std::vector<bool>& included,
                       const TokenCounter& count_tokens, std::size_t reserve,
                       std::size_t& listed, std::size_t& withheld) {
    std::string tail = kTailHeader;
    std::size_t used = count_tokens(kTailHeader);
    for (std::size_t i = 0; i < included.size(); ++i) {
        if (included[i]) {
            continue;
        }
        ++withheld;
        if (used + costs.pointer_cost[i] > reserve) {
            continue;
        }
        tail += costs.pointer[i];
        used += costs.pointer_cost[i];
        ++listed;
    }
    if (listed < withheld) {
        // Saying how many were dropped costs one line and is the difference between a
        // truncated list and a misleading one.
        tail += "- (" + std::to_string(withheld - listed) +
                " further matches not listed; narrow the query or raise the budget)\n";
    }
    return tail;
}

void pack(Recall& out, const TokenCounter& count_tokens, std::size_t budget) {
    const Costs costs = measure(out.entries, count_tokens);

    // First fit everything into the whole budget. When it all fits there is no tail, so
    // reserving space for one would waste a quarter of the allowance on nothing.
    std::vector<bool> included = greedy(costs, budget);
    const bool complete =
        std::all_of(included.begin(), included.end(), [](bool b) { return b; });

    std::string tail;
    if (!complete) {
        const std::size_t reserve = budget / kTailShare;
        included = greedy(costs, budget - reserve);
        std::size_t listed = 0;
        std::size_t withheld = 0;
        tail = build_tail(costs, included, count_tokens, reserve, listed, withheld);
        out.pointers_only = withheld;
    }

    std::string text;
    for (std::size_t i = 0; i < out.entries.size(); ++i) {
        if (!included[i]) {
            continue;
        }
        out.entries[i].included = true;
        ++out.included;
        text += costs.full[i];
    }
    text += tail;

    out.text = std::move(text);
    // Counted from the final string rather than accumulated, so the number reported is
    // the number a caller would measure. A budget this still overruns is a bug that
    // shows up here rather than in someone's prompt.
    out.tokens_used = count_tokens(out.text);
}

} // namespace

std::size_t estimate_tokens(std::string_view text) {
    return (text.size() + 3) / 4;
}

std::string item_uri(const Item& item) {
    return "pcc://item/" + std::to_string(item.id);
}

Recall recall(const Store& store, const RecallRequest& request,
              const TokenCounter& count_tokens) {
    Recall out;
    const std::vector<Item> candidates =
        store.search(request.query, request.as_of, request.session, request.candidates);
    if (candidates.empty()) {
        return out;
    }
    out.entries = fuse(candidates);
    pack(out, count_tokens, request.token_budget);
    return out;
}

Recall rehydrate(const Store& store, std::uint64_t first_event, std::uint64_t last_event,
                 std::size_t token_budget, std::string_view session,
                 const TokenCounter& count_tokens) {
    Recall out;
    std::vector<Item> turns = store.events_between(first_event, last_event, session);
    if (turns.empty()) {
        return out;
    }
    // Newest first, so a budget too small for the whole span keeps the end of it. The
    // turns nearest the compaction boundary are the ones the current turn most likely
    // depends on; the oldest are the ones the summary already covers adequately.
    std::reverse(turns.begin(), turns.end());
    out.entries.reserve(turns.size());
    for (const Item& turn : turns) {
        out.entries.push_back({turn, 0.0, item_uri(turn), false});
    }
    pack(out, count_tokens, token_budget);
    return out;
}

} // namespace lmp::pcc
