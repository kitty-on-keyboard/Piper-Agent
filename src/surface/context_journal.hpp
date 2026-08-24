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

        // True when this call added the store to the repository's local exclude file.
        // Reported so the caller can record a write it made to the user's repository
        // rather than performing it silently; see ensure_git_excludes_store.
        bool git_exclude_written = false;
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
        : store_(path, pcc::LinkPolicy::NoFollow),
          session_(std::move(session_id)) {}

    pcc::Store store_;
    std::string session_;
};

// A single dotfile at the workspace root. SQLite adds -wal and -shm siblings in WAL mode.
//
// THIS IS NOT THE SAME CASE AS .lmp-memory.md, though it sat next to it and was justified
// by the comparison. The memory file is 16 KiB of text the user can read, review and
// reasonably choose to commit. This is a SQLite database that reached 1.8 MB after a
// single 26-turn task, and it is binary, unreviewable and rebuildable. Left to itself it
// turns up as an untracked file in the user's own `git status` -- and, measured while
// building the SWE-bench harness, went straight into a `git add -A` and out into every
// generated patch.
//
// The header used to say ".gitignore covers the stem with a wildcard", which was an
// assumption about the user's file rather than anything this program did. Nothing wrote
// it. See ensure_git_excludes_store for what does now.
inline constexpr const char* kContextDbName = ".lmp-context.db";

// Everything this program leaves in a workspace that the user did not ask for, written
// into .git/info/exclude. All anchored with a leading slash: these live at the workspace
// root, and an unanchored pattern would also hide a file of the same name anywhere in the
// user's tree. The trailing wildcard on the store covers SQLite's -wal and -shm siblings.
//
// .lmp-memory.md is DELIBERATELY ABSENT. It is 16 KiB of readable text the user can
// review, and a team may reasonably want it committed; hiding it would be this program
// deciding that for them. The three below are a binary database and two scratch
// directories -- src/tools/ignore_dirs.hpp records one workspace that accumulated a
// 6.9 MB store and 824 leaked temp directories at its root.
inline constexpr const char* kGitExcludePatterns[] = {
    "/.lmp-context.db*",
    "/.lmp_tmp/",
    "/.lmp_spool/",
};

} // namespace lmp::surface
