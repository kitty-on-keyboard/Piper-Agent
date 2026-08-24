#include "src/surface/context_journal.hpp"

#include <sys/stat.h>

#include <utility>
#include <vector>

#include "src/platform/fs.hpp"

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
        // The two producers of tool_args_summary do not agree on format. A turn's own
        // call gets preview_of(), which ALREADY names the tool -- "read_file(path=x)" --
        // while the extra calls batched behind it get a bare path or command. Prepending
        // the name unconditionally is right for the second and produced
        // "read_file(read_file(path=x))" for the first, in every row of the first real
        // run through this store. Cosmetic in a log; not cosmetic here, where this text
        // is what BM25 ranks on and what a recall spends its budget carrying.
        if (turn.tool_args_summary.rfind(turn.tool_name + "(", 0) == 0) {
            body += turn.tool_args_summary;
        } else {
            body += turn.tool_name;
            if (!turn.tool_args_summary.empty()) {
                body += "(" + turn.tool_args_summary + ")";
            }
        }
        body += turn.observation_is_error ? " FAILED\n" : "\n";
    }
    if (!turn.observation.empty()) {
        body += turn.observation;
    }
    return body;
}

// Every write from here is best-effort. A throw would propagate out of add_turn() or
// compact_oldest() and abort a turn that was otherwise fine; journalling is an
// enhancement, and losing it is strictly better than losing the run.
void try_append(pcc::Store& store, pcc::Record rec) {
    try {
        store.append(std::move(rec));
    } catch (const std::exception&) {
        // Deliberately swallowed. The failure that matters -- the store not opening at
        // all -- is reported by open() and logged by the caller; a single row lost to a
        // transient SQLITE_BUSY is not worth a dead run.
    }
}

// Keeps the store out of the user's `git status` without touching a tracked file.
//
// WHY .git/info/exclude AND NOT .gitignore. `.gitignore` is the user's file: it is
// tracked, it shows up in their next diff, and it would travel to their colleagues in a
// commit they did not write. `.git/info/exclude` is the per-clone, never-tracked
// equivalent that exists for exactly this -- a local tool wanting a local file ignored.
// Deleting the line undoes it completely.
//
// Best effort throughout. A workspace that is not a git repository, a read-only .git, a
// worktree whose .git is a FILE rather than a directory (its exclude file lives in the
// common directory, which is more indirection than a courtesy write deserves) -- all of
// them simply do nothing. Failing to write an ignore rule is never a reason to fail a run.
bool ensure_git_excludes_store(const std::string& workspace_root) {
    const std::string git_dir = workspace_root + "/.git";
    struct ::stat st {};
    if (::stat(git_dir.c_str(), &st) != 0 || !S_ISDIR(st.st_mode)) {
        return false;
    }
    const std::string info_dir = git_dir + "/info";
    // EEXIST is the expected case and is not an error; any other failure falls through to
    // the write, which will fail on its own and be ignored.
    (void)::mkdir(info_dir.c_str(), 0755);

    const std::string exclude_path = info_dir + "/exclude";
    const platform::FileContents current =
        platform::read_file_whole(exclude_path, 1U << 20);
    std::string body = current.ok() ? current.bytes : std::string();

    // Per PATTERN rather than per block. This runs once a mission, so an unconditional
    // append would grow the file every time the user starts a task; and checking each
    // pattern separately means a later version that adds one can add just that one to a
    // repository an earlier version already wrote to.
    std::vector<std::string> missing;
    for (const char* pattern : kGitExcludePatterns) {
        if (body.find(pattern) == std::string::npos) {
            missing.emplace_back(pattern);
        }
    }
    if (missing.empty()) {
        return false;
    }
    if (!body.empty() && body.back() != '\n') {
        body += '\n';
    }
    body += "\n# LM_Pipe's own scratch: a SQLite context store and two working\n";
    body += "# directories. Local-only -- your tracked .gitignore is untouched, and\n";
    body += "# deleting these lines undoes this completely.\n";
    for (const std::string& pattern : missing) {
        body += pattern;
        body += "\n";
    }
    return platform::write_file_atomic(exclude_path, body).ok();
}

} // namespace

ContextJournal::Result ContextJournal::open(const std::string& workspace_root,
                                            const std::string& session_id,
                                            context::ContextStore& ctx) {
    Result out;
    const platform::WorkspaceFs workspace(workspace_root);
    if (!workspace.valid()) {
        out.error = workspace.error();
        return out;
    }
    const platform::ContainedPath database =
        workspace.contained_path(kContextDbName);
    if (!database.ok()) {
        out.error = database.error;
        return out;
    }
    try {
        // Not make_unique: the constructor is private, deliberately, so the only way to
        // get one is through the path that also attaches the sinks.
        out.journal.reset(
            new ContextJournal(database.absolute, session_id));
    } catch (const std::exception& e) {
        out.error = e.what();
        return out;
    }

    // Only once the store is real. Writing an ignore rule for a database that failed to
    // open would leave a line in the user's repository referring to a file that does not
    // exist.
    out.git_exclude_written = ensure_git_excludes_store(workspace.root());

    // Captures the raw store, not the unique_ptr: the journal owns the store and outlives
    // the ContextStore it is attached to, because the session destroys the context first.
    pcc::Store* store = &out.journal->store_;
    const std::string session = out.journal->session_;

    // EVERY TURN, AS IT HAPPENS -- not only the ones a trim is about to destroy.
    //
    // This used to be attached as the compaction sink alone, which meant nothing was
    // persisted until the prompt crossed the compaction threshold. Long runs add little
    // per turn and often never cross it, so the store stayed empty through complete
    // 80-turn runs (measured 2026-08-03; see ContextStore::set_turn_sink). Writing here
    // makes the store a record of the run rather than a record of its trims.
    ctx.set_turn_sink([store, session](const context::TurnRecord& turn) {
        std::string body = turn_body(turn);
        // A turn with nothing in it is FTS noise: it can never match a query and it
        // still costs a row and an index entry. The loop guarantees a non-empty
        // observation for anything it executed, so this only ever drops genuinely
        // contentless records.
        if (body.empty()) {
            return;
        }
        pcc::Record rec;
        rec.session = session;
        rec.kind = pcc::kind::kTurn;
        rec.title = turn.tool_name;
        rec.body = std::move(body);
        // rehydrate() keys on this range, and it is what makes the "events 40-91"
        // pointer in a compacted summary resolvable.
        rec.first_event = turn.first_event_seq;
        rec.last_event = turn.last_event_seq;
        try_append(*store, std::move(rec));
    });

    // Compaction now writes the SUMMARY, and only the summary: the turns behind it are
    // already rows by the time a trim sees them, and appending them again here would
    // double every compacted turn in the store and in every recall that matched one.
    //
    // The span is worth its own row rather than being derivable: it is the cheap,
    // already-condensed answer to "what happened earlier in that run", where the turns it
    // covers can be thousands of tokens. A budgeted recall can afford the span when it
    // cannot afford the span's contents.
    ctx.set_compaction_sink([store, session](const std::vector<context::TurnRecord>& dropped,
                                             std::size_t span_index,
                                             std::string_view summary) {
        if (dropped.empty() || summary.empty()) {
            return;
        }
        pcc::Record rec;
        rec.session = session;
        rec.kind = pcc::kind::kSpan;
        rec.title = "compacted span " + std::to_string(span_index + 1);
        rec.body = std::string(summary);
        // The range the span itself prints, so events_between() reaches the turns it was
        // made from through exactly the numbers the model reads in the prompt.
        rec.first_event = dropped.front().first_event_seq;
        rec.last_event = dropped.back().last_event_seq;
        try_append(*store, std::move(rec));
    });

    // THE MISSION, which no sink can see.
    //
    // ContextStore takes the opening mission through its CONSTRUCTOR, so it never passes
    // through add_turn() and the turn sink above will never be handed it. That leaves the
    // one row a later session most wants -- "what was this run asked to do" -- as the only
    // part of the conversation not in the store.
    pcc::Record mission;
    mission.session = session;
    mission.kind = pcc::kind::kTurn;
    // The shared constant, because recall() recognises this row too -- it suppresses the
    // LIVE session's mission, which is already T0 of the prompt. Two literals would drift.
    mission.title = pcc::title::kMission;
    mission.body = "you were told: " + ctx.mission();
    // Event range left at 0, which events_between() excludes explicitly (`first_event >
    // 0`). The mission predates every logged event, so a rehydrate of an early range
    // must not pick it up as though it were a turn inside that range. Search still finds
    // it -- that is the path that matters for "what was I asked to do last time".
    try_append(*store, std::move(mission));

    return out;
}

} // namespace lmp::surface
