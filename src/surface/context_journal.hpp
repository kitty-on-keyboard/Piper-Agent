#pragma once
//
// Wires src/context's compaction to src/pcc's store, so a trim stops destroying turns.
//
// WHY THIS FILE EXISTS AT ALL
//   The two halves cannot know about each other. ContextStore is L3 and PCC is L1, and
//   the adapter needs both types -- so it lives at L5, where everything is already
//   visible, and the dependency still points downward. It is also the only place that
//   knows a session has a name and a workspace has a path, neither of which is any
//   business of a context tier or a database.
//
// DEGRADES RATHER THAN FAILS
//   open() yields a null journal if the database cannot be opened, and the run proceeds
//   exactly as it did before this component existed: the prompt still gets its summary,
//   the full text just is not kept. Journalling makes a run better, and a run that cannot
//   journal is still a run -- wedging the agent because a disk is full would be a worse
//   trade. It degrades QUIETLY, not SILENTLY: the reason comes back with the null.
//
#include <memory>
#include <string>

#include "src/context/context.hpp"
#include "src/pcc/store.hpp"

namespace lmp::surface {

class ContextJournal {
  public:
    struct Result {
        // Null when the store could not be opened.
        std::unique_ptr<ContextJournal> journal;
        // Verbatim from SQLite; empty if and only if `journal` is non-null. RETURNED
        // rather than printed. The previous version wrote it to stderr and dropped it,
        // which put the one actionable sentence about a broken journal in the only
        // channel nothing collects: stdout is the framed protocol stream, and the
        // sidecar's stderr is a subprocess pipe no client reads. The caller has an
        // event log, which is the record that outlives the run (S6).
        std::string error;
    };

    // Opens (or creates) the store under `workspace_root` and attaches it to `ctx` as the
    // compaction sink.
    [[nodiscard]] static Result open(const std::string& workspace_root,
                                     const std::string& session_id,
                                     context::ContextStore& ctx);

    [[nodiscard]] pcc::Store& store() noexcept { return store_; }

    // Which partition this mission's rows are written under. Exposed because the recall
    // tools need it to offer "this session only" against a store that deliberately holds
    // every session this workspace has ever had.
    [[nodiscard]] const std::string& session_id() const noexcept { return session_; }

  private:
    ContextJournal(const std::string& path, std::string session_id)
        : store_(path), session_(std::move(session_id)) {}

    pcc::Store store_;
    std::string session_;
};

// A single dotfile beside .lmp-memory.md, for the same reason that one is: nothing has to
// create a directory. SQLite adds -wal and -shm siblings in WAL mode, so .gitignore covers
// the stem with a wildcard.
inline constexpr const char* kContextDbName = ".lmp-context.db";

} // namespace lmp::surface
