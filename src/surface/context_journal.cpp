#include "src/surface/context_journal.hpp"

#include <cstdio>
#include <utility>
#include <vector>

namespace lmp::surface {
namespace {

// The prompt-facing summary keeps one truncated anchor line per turn. This keeps the
// whole thing -- what was said, what ran, and the full observation -- because the detail a
// later question needs is exactly the part the anchor had to cut.
std::string turn_body(const context::TurnRecord& turn) {
    std::string body;
    if (!turn.user_text.empty()) {
        body += "you were told: " + turn.user_text + "\n";
    }
    if (!turn.assistant_text.empty()) {
        body += turn.assistant_text + "\n";
    }
    if (!turn.tool_name.empty()) {
        body += turn.tool_name;
        if (!turn.tool_args_summary.empty()) {
            body += "(" + turn.tool_args_summary + ")";
        }
        body += turn.observation_is_error ? " FAILED\n" : "\n";
    }
    if (!turn.observation.empty()) {
        body += turn.observation;
    }
    return body;
}

} // namespace

std::unique_ptr<ContextJournal> ContextJournal::open(const std::string& workspace_root,
                                                     const std::string& session_id,
                                                     context::ContextStore& ctx) {
    std::unique_ptr<ContextJournal> journal;
    try {
        // Not make_unique: the constructor is private, deliberately, so the only way to
        // get one is through the path that also attaches the sink.
        journal.reset(new ContextJournal(workspace_root + "/" + kContextDbName, session_id));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lmp: context journal unavailable (%s); this run's "
                             "compacted turns will not be recoverable\n",
                     e.what());
        return nullptr;
    }

    // Captures the raw store, not the unique_ptr: the journal owns the store and outlives
    // the ContextStore it is attached to, because the session destroys the context first.
    pcc::Store* store = &journal->store_;
    const std::string session = journal->session_;
    ctx.set_compaction_sink(
        [store, session](const std::vector<context::TurnRecord>& dropped, std::size_t) {
            for (const context::TurnRecord& turn : dropped) {
                pcc::Record rec;
                rec.session = session;
                rec.kind = pcc::kind::kTurn;
                rec.title = turn.tool_name;
                rec.body = turn_body(turn);
                rec.first_event = turn.first_event_seq;
                rec.last_event = turn.last_event_seq;
                // A throw here would propagate out of compact_oldest() and abort a turn
                // that was otherwise fine. Journalling is an enhancement; losing it is
                // strictly better than losing the run.
                try {
                    store->append(std::move(rec));
                } catch (const std::exception&) {
                    return;
                }
            }
        });
    return journal;
}

} // namespace lmp::surface
