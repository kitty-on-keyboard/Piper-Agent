#pragma once
//
// Directories a workspace walk descends into, and the ones it does not.
//
// Its own header rather than a table inside registry.cpp for a reason that is not style:
// the tool_honesty ratchet reads EVERY string literal in a tool-declaring file and requires
// any snake_case token to name a registered tool, because v1 shipped descriptions that
// taught the model to call tools that did not exist. `node_modules` and `__pycache__` are
// not tools and are not model-facing; keeping them here keeps registry.cpp's literals
// genuinely model-facing, so the gate stays strict instead of growing a stoplist of
// directory names that would blunt it for everyone after us.
//
#include <string>
#include <string_view>

namespace lmp::tools {

// Version-control internals, dependency and build trees, and our own run artifacts. Every
// one is bytes the model did not write and cannot usefully edit, and they are where the
// volume is: one observed workspace had a 6.9 MB context DB and 824 leaked temp
// directories sitting directly under the root, all of which `search` walked.
//
// Consulted only when a walk DESCENDS. Naming one of these as a search's `subdir` still
// searches it, because then it is what was asked for rather than something wandered into.
[[nodiscard]] inline bool skip_during_descent(const std::string& name) {
    static constexpr std::string_view kSkip[] = {
        ".git",         ".hg",           ".svn",         ".jj",
        ".build",       "build",         "DerivedData",  ".gradle",
        "target",       "dist",          ".next",        ".turbo",
        "node_modules", "__pycache__",   ".pytest_cache",
        ".venv",        "venv",          ".mypy_cache",  ".ruff_cache",
        ".cache",       ".ccache",       ".swiftpm",
        // Ours: the sandbox's TMPDIR and the shell spool.
        ".lmp_tmp",     ".lmp_spool",
    };
    for (std::string_view s : kSkip) {
        if (name == s) {
            return true;
        }
    }
    return false;
}

} // namespace lmp::tools
