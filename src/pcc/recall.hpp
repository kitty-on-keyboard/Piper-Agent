#pragma once
//
// Budgeted recall: the retrieval side of PCC.
//
// THE POINT
//   A search that returns twenty rows has not helped an agent -- it has moved the
//   problem, because those rows now have to fit somewhere and nothing decided what to
//   drop. The scarce resource is prompt tokens, so the budget is the argument. recall()
//   takes a token budget and returns text that FITS IT, plus a URI for everything that
//   did not make it. The agent gets an answer it can paste and a pointer to the rest.
//
// FUSING RANKS, NOT SCORES
//   Relevance and freshness are both real signals and they are not on the same scale.
//   Multiplying them -- which is what the cook-off entrant that tried this did -- gets
//   the sign wrong as easily as right; that entrant multiplied a DISTANCE by a freshness
//   factor and then sorted ascending, so its fresher memories ranked strictly worse.
//   Reciprocal rank fusion sidesteps the whole problem: it only ever compares positions,
//   so no normalisation constant can be wrong.
//
//   Both lists here rank the SAME candidate set -- BM25's top-N, reordered by recency --
//   so freshness re-ranks what is already relevant instead of injecting recent noise.
//   A third list (a real embedder's neighbours) drops into the same fusion untouched,
//   which is the reason for doing it this way now rather than when there is an embedder.
//
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

#include "src/pcc/store.hpp"

namespace lmp::pcc {

// The RRF constant from Cormack et al. 2009, unchanged. It damps the top of each list so
// one retriever's confident first place cannot outvote broad agreement further down.
inline constexpr double kRrfK = 60.0;

struct RecallRequest {
    std::string query;
    std::string session; // empty searches every session
    // In tokens, using the counter below. The default suits a mid-run lookup; a
    // session-opening rehydrate wants considerably more.
    std::size_t token_budget = 1500;
    AsOf as_of;
    // How deep to go before re-ranking. Larger costs a bigger BM25 scan and gives
    // recency more room to promote; 60 is where the promotion stops changing on the
    // corpus in tests/pcc.
    int candidates = 60;
    // The session whose MISSION row the caller is already looking at, or empty.
    //
    // The mission is T0 of the prompt: always present, never compacted, verbatim. Handing
    // it back spends a slice of the recall budget on text already on screen -- and it
    // ranks HIGH, because a query drawn from the mission is a near-perfect BM25 match for
    // the mission. MEASURED on the first run with a populated store: ~60 tokens of a 1500
    // token budget went on the run's own opening instruction.
    //
    // Narrow on purpose. Only the LIVE session's, and only the mission row: an earlier
    // mission in this workspace is genuinely new information, and every other row of this
    // session is too -- including the turns this run has since compacted away, which are
    // the whole reason the store exists.
    std::string suppress_mission_of;
};

struct RecallEntry {
    Item item;
    double score = 0.0;
    std::string uri;
    // False when the budget ran out and only the pointer was emitted. The distinction
    // matters to a caller deciding whether to spend a second call fetching the body.
    bool included = false;
};

struct Recall {
    std::string text; // ready to hand to a model
    std::vector<RecallEntry> entries;
    std::size_t tokens_used = 0;
    std::size_t included = 0;
    std::size_t pointers_only = 0;
};

// Counts tokens in a string. The default is bytes/4 -- an ESTIMATE, and named as one at
// every call site, because the real tokenizer lives in src/model and this component does
// not depend on it. The sidecar injects the real counter; nothing here pretends the
// default is exact.
using TokenCounter = std::function<std::size_t(std::string_view)>;

[[nodiscard]] std::size_t estimate_tokens(std::string_view text);

// The stable identifier for an item, usable as an MCP resource URI.
[[nodiscard]] std::string item_uri(const Item& item);

// Ranks, fuses, and packs to the budget.
[[nodiscard]] Recall recall(const Store& store, const RecallRequest& request,
                            const TokenCounter& count_tokens = estimate_tokens);

// Rebuilds the full text of a compacted span from the turns it was made from.
//
// This is the call that makes src/context's compaction non-destructive. compact_oldest()
// prints "events N-M" into the summary it leaves in the prompt; this turns that pointer
// back into the turns, newest-first, packed to a budget. Without it that range is a
// citation to a document nobody kept.
[[nodiscard]] Recall rehydrate(const Store& store, std::uint64_t first_event,
                               std::uint64_t last_event, std::size_t token_budget,
                               std::string_view session = {},
                               const TokenCounter& count_tokens = estimate_tokens);

} // namespace lmp::pcc
