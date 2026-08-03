#include "src/tools/registry.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "src/pcc/recall.hpp"
#include "src/pcc/store.hpp"

// The two tools that let a run READ what it has stored.
//
// WHY THIS FILE EXISTS SEPARATELY. src/pcc has been writing a durable, bi-temporal,
// budget-aware context store for some time, and until this file there was no way for the
// agent to get a single byte back out of it. The same two tools existed in the
// out-of-process pcc_mcp_server, which is not staged into the VSIX and needs a
// hand-written server entry with an absolute database path -- so out of the box the whole
// component was a write-only log. See Registry::declare_context_tools for why the native
// path is the right one and not merely the convenient one.
//
// Split out of registry.cpp rather than added to it for the same reason memory_file.cpp
// is split out: these are the tools whose state OUTLIVES the run. Every handler in
// registry.cpp acts on the workspace within one mission; these two read what previous
// missions left behind, which is a different lifetime and a different failure mode.

namespace lmp::tools {
namespace {

using parsephony::ParamType;

// What a recall may cost the prompt.
//
// The default matches pcc::RecallRequest, which is sized for a mid-run lookup. The cap is
// the part that matters: without one, a model that asks for a 100,000-token budget gets
// roughly 400 KB of text handed to ContextStore::add_turn, which asserts against the
// observation budget in every configuration this repo builds and clamps in a real run.
// A tool that can trip the context store's own door is a tool that forgot to bound
// itself, and tool_result.hpp says that is the tool's job.
constexpr std::size_t kDefaultRecallTokens = 1500;
constexpr std::size_t kDefaultRehydrateTokens = 4000;
constexpr std::size_t kMaxRecallTokens = 6000;

std::size_t number_arg(const std::vector<ToolParamValue>& p, const char* name,
                       std::size_t fallback) {
    const std::string* raw = get(p, name);
    if (raw == nullptr || raw->empty()) {
        return fallback;
    }
    const long long v = std::strtoll(raw->c_str(), nullptr, 10);
    return v <= 0 ? fallback : static_cast<std::size_t>(v);
}

// The grammar emits the literal true / false for a Boolean parameter, so this is a
// comparison rather than a parse.
bool flag_arg(const std::vector<ToolParamValue>& p, const char* name) {
    const std::string* raw = get(p, name);
    return raw != nullptr && *raw == "true";
}

ToolResult no_store() {
    // NotFound rather than a refusal: a refusal means policy said no and the tool never
    // ran, which would send the model looking for permission it cannot obtain. This is
    // an absent capability, and saying so plainly stops it retrying.
    return ToolResult::error(ErrorClass::NotFound, false,
                             "no durable context store is open for this run, so there is "
                             "nothing to recall; work from what is in front of you");
}

// Packs a Recall into the model-facing result, and holds the last line of defence on
// size: the budget is denominated in tokens and this is denominated in bytes, so a
// pathological corpus (one enormous token-dense row) can satisfy the first and still
// overrun the second.
ToolResult recall_result(const pcc::Recall& r, const char* nothing_found,
                         std::size_t max_bytes) {
    if (r.entries.empty()) {
        return ToolResult::okay(nothing_found);
    }
    std::string text = r.text;
    if (text.size() > max_bytes) {
        text.resize(max_bytes);
        text += "\n[recall truncated at the prompt budget; ask again with a narrower "
                "query or a smaller token budget]\n";
    }
    return ToolResult::okay(std::move(text));
}

} // namespace

bool Registry::declare_context_tools(ContextSourceFn source, TokenCounter count_tokens) {
    // The reused-Registry case. ensure_registry() keeps a Registry across missions, and
    // declare() does not check for duplicates -- handlers_.emplace would silently keep
    // the first handler while decls_ and specs_ each grew a second copy, advertising both
    // tools twice in the system prompt and adding a redundant grammar spec.
    if (find("context_recall") != nullptr) {
        return false;
    }
    context_source_ = std::move(source);

    // ONE LOCK FOR BOTH TOOLS, and it is load-bearing rather than defensive.
    //
    // Both tools are read-only, execute nothing and mutate nothing, which is exactly the
    // eligibility test in Agent::can_run_in_parallel -- so two recalls batched into one
    // turn run on two threads against one sqlite3 handle. macOS ships libsqlite3 built
    // with SQLITE_THREADSAFE=2 (multi-thread), measured on this host: mutexes exist, but
    // a single CONNECTION may not be used by two threads at once, and pcc opens with
    // plain READWRITE|CREATE rather than FULLMUTEX. Serialising here keeps that promise
    // local to the caller that broke it, instead of changing pcc's threading contract for
    // every other user of the store.
    //
    // Shared by pointer because both handlers need the same one and both outlive this
    // function.
    auto lock = std::make_shared<std::mutex>();

    // --- context_recall -----------------------------------------------------
    {
        ToolDecl d;
        d.name = "context_recall";
        // Reused almost verbatim from the MCP server's version, which already says the
        // one thing a description has to say: WHEN to reach for it.
        d.description =
            "Search everything this agent has stored in this workspace -- past turns, "
            "durable notes, earlier missions -- and get back as much as fits in a token "
            "budget, plus a pointer for anything that did not fit. Ranked by relevance "
            "fused with recency. Use it before re-reading a file you have read before, "
            "and when a question sounds like something this project has already answered. "
            "It searches every past session by default; set this_session_only to confine "
            "it to the run you are in.";
        d.spec.name = d.name;
        d.spec.params = {param("query", ParamType::Text, true),
                         param("token_budget", ParamType::Number, false),
                         param("this_session_only", ParamType::Boolean, false)};
        // Reads a database and runs no command, so it raises no approval card and stays
        // on the parallel read-only path. It is readable at T1 like search and
        // locate_symbol: `approved_tier` is deliberately unused because nothing here
        // reaches a sandbox to spend it on.
        declare(d, [this, lock, count_tokens](const std::vector<ToolParamValue>& p, int) {
            const ContextSource src = context_source_();
            if (src.store == nullptr) {
                return no_store();
            }
            const std::string* q = get(p, "query");
            if (q == nullptr || q->find_first_not_of(" \t\n") == std::string::npos) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "query must not be blank");
            }
            pcc::RecallRequest req;
            req.query = *q;
            // EMPTY MEANS EVERY SESSION, and that is the default on purpose: the value of
            // a durable store is what it knows about this repo from last week, not what
            // it knows about the last forty minutes. Recency is already the second of the
            // two fused rank lists, so the current run is favoured without being the only
            // thing visible.
            if (flag_arg(p, "this_session_only")) {
                req.session = src.session;
            }
            req.token_budget = std::min(
                number_arg(p, "token_budget", kDefaultRecallTokens), kMaxRecallTokens);
            // This run's own mission is the first thing in its own prompt. Recalling it is
            // the store spending the model's budget to quote the model's instructions back
            // at it.
            req.suppress_mission_of = src.session;
            const std::lock_guard<std::mutex> guard(*lock);
            // The empty answer NAMES THE WAY OUT, and that is not politeness.
            //
            // MEASURED on the first control run: against an empty store this tool
            // answered a bare "nothing stored matches that", and the model called it
            // four times in twelve turns -- the repeat-breaker had to suppress it --
            // before trying the workspace instead. The run then exhausted its budget
            // with no answer. An honest negative that suggests nothing reads to a model
            // like a transient failure worth retrying.
            return recall_result(
                pcc::recall(*src.store, req, count_tokens),
                "nothing stored matches that. This workspace has no remembered history "
                "of it, and asking again will not change that -- look in the files "
                "instead (search, then read_file).",
                ctx_.max_model_read_bytes);
        });
    }

    // --- context_rehydrate --------------------------------------------------
    {
        ToolDecl d;
        d.name = "context_rehydrate";
        d.description =
            "Given the event range printed in a compacted summary (a line reading "
            "'events 40-91'), return the full text of the turns it was made from, newest "
            "first, packed to a token budget. Use this when a summary of earlier work "
            "mentions something you now need the detail of.";
        d.spec.name = d.name;
        d.spec.params = {param("first_event", ParamType::Number, true),
                         param("last_event", ParamType::Number, true),
                         param("token_budget", ParamType::Number, false)};
        declare(d, [this, lock, count_tokens](const std::vector<ToolParamValue>& p, int) {
            const ContextSource src = context_source_();
            if (src.store == nullptr) {
                return no_store();
            }
            const auto first = static_cast<std::uint64_t>(number_arg(p, "first_event", 0));
            const auto last = static_cast<std::uint64_t>(number_arg(p, "last_event", 0));
            if (first == 0 || last < first) {
                return ToolResult::error(
                    ErrorClass::Malformed, false,
                    "first_event must be 1 or more and last_event must not be smaller; "
                    "copy both numbers from the summary line that mentions them");
            }
            const std::size_t budget = std::min(
                number_arg(p, "token_budget", kDefaultRehydrateTokens), kMaxRecallTokens);
            // SCOPED TO THIS SESSION, unlike recall above, and the asymmetry is not an
            // oversight. Event sequence numbers come from the event log writer, which
            // starts at 1 in every sidecar PROCESS -- so two missions started from two
            // launches of the editor both contain an event 40. An unscoped rehydrate
            // would interleave turns from unrelated runs under the range the current
            // prompt printed, which is worse than returning nothing. And the range only
            // ever appears in the current run's own summary, so there is nothing to lose.
            const std::lock_guard<std::mutex> guard(*lock);
            return recall_result(
                pcc::rehydrate(*src.store, first, last, budget, src.session, count_tokens),
                "no turns were journalled in that event range. Re-check the two numbers "
                "against the summary line you took them from; if they are right, that "
                "span predates this store and the detail is gone -- look in the files "
                "instead (search, then read_file).",
                ctx_.max_model_read_bytes);
        });
    }
    return true;
}

} // namespace lmp::tools
