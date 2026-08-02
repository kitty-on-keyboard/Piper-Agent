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
//   open() returns null if the database cannot be opened, and the run proceeds exactly as
//   it did before this component existed: the prompt still gets its summary, the full text
//   just is not kept. Journalling makes a run better, and a run that cannot journal is
//   still a run -- wedging the agent because a disk is full would be a worse trade.
//
#include <memory>
#include <string>

#include "src/context/context.hpp"
#include "src/pcc/store.hpp"

namespace lmp::surface {

class ContextJournal {
  public:
    // Opens (or creates) the store under `workspace_root` and attaches it to `ctx` as the
    // compaction sink. Null on failure, having written the reason to stderr -- stdout is
    // the framed protocol channel and a diagnostic there would desync the client.
    //
    // TODO: return the reason. It is dropped today, and the justification written here
    // was that no caller could use it and that sidecar.cpp had no line to spare. Both
    // halves are now false -- the sidebar has an error channel into the transcript, and
    // the size limit that made a spare line a currency is gone -- so a run whose
    // compacted turns are silently unrecoverable should say so.
    [[nodiscard]] static std::unique_ptr<ContextJournal> open(const std::string& workspace_root,
                                                              const std::string& session_id,
                                                              context::ContextStore& ctx);

    [[nodiscard]] pcc::Store& store() noexcept { return store_; }

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
