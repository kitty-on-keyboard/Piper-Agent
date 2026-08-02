#pragma once
//
// SyntaxChecker -- non-model feedback on an edit, immediately after it lands (spec S10).
//
// The gap this closes: after a successful write_file / replace_in_file / append_file,
// NOTHING ran. The only non-model signal in the whole loop was `shell`, and only when the
// model chose to call it. So a broken edit stayed invisible until the model happened to
// run the tests, which is exactly the turn it was least likely to spend.
//
// WHY NOT AN LSP CLIENT, YET. LSP diagnostics are a textDocument/publishDiagnostics PUSH
// with no completion signal: a client needs server lifecycle management, didOpen/didChange
// version tracking, and a settle timeout that is a heuristic sitting in the one place this
// codebase refuses to put heuristics. A per-language check through the existing sandbox
// gets most of the signal with none of the daemon. The seam here is the one an LSP client
// would implement later.
//
// THREE RULES, each of which exists because breaking it is worse than having no check.
//
//   1. SILENCE WHEN THERE IS NO CONTRACT. An unrecognised extension produces nothing --
//      not "no checker available for .md", which would recur every turn for the whole run.
//
//   2. C++ ONLY WITH A COMPILE DATABASE. A bare `c++ -fsyntax-only` on a project header
//      emits a cascade of missing-include errors that are not about the edit, and a false
//      diagnostic handed to the model is worse than no check -- it sends the run off
//      fixing something that was never broken. `needs_compile_db` makes that a property of
//      the contract rather than a special case in the caller.
//
//   3. IT IS NEVER EVIDENCE. This class has no access to a Verifier and must never be
//      given one. A syntax check is not the contract the run declared, and a green from it
//      must never help a run complete (S10.1: one choke point writes the ledger). The
//      tempting tidy implementation -- route it through Verifier::run_and_record -- would
//      quietly make S10.4 completion cheaper, which is why the test asserts the
//      verification ledger is unchanged across a checked edit.
//
#include <cstddef>
#include <string>
#include <vector>

namespace lmp::tools {

struct SyntaxContract {
    std::string language;
    std::vector<std::string> extensions; // including the dot
    // "{file}" is replaced with the shell-quoted absolute path.
    std::string command;
    // True when the check is meaningless without project context, and must therefore stay
    // silent rather than guess at it.
    bool needs_compile_db = false;
};

struct SyntaxVerdict {
    // False means no contract matched, or the contract could not be run. NOT a failure --
    // the caller says nothing at all.
    bool ran = false;
    bool clean = false;
    std::string language;
    std::string diagnostics; // already compacted; empty when clean
};

// The contracts this build ships, in match order. Exposed so the test can assert the table
// rather than the behaviour of one entry.
[[nodiscard]] const std::vector<SyntaxContract>& syntax_contracts();

class SyntaxChecker {
  public:
    SyntaxChecker(std::string root, std::size_t budget_bytes)
        : root_(std::move(root)), budget_(budget_bytes) {}

    // Runs the contract for `rel_path` against what is on disk right now. `approved_tier`
    // is the sandbox tier the loop already granted; tier 0 cannot execute, so it returns
    // `ran = false` rather than refusing loudly -- a Plan-mode run has nothing to check.
    [[nodiscard]] SyntaxVerdict check(const std::string& rel_path, int approved_tier) const;

  private:
    std::string root_;
    std::size_t budget_;
};

// The compile command for `abs_path` from `<root>/build/compile_commands.json`, with
// -fsyntax-only appended and any -o dropped. Empty when there is no database or no entry.
// Exposed for the test; the file is JSON and is parsed as JSON, not scanned for substrings.
[[nodiscard]] std::string compile_db_syntax_command(const std::string& root,
                                                    const std::string& abs_path);

} // namespace lmp::tools
