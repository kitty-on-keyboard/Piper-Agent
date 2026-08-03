#include "src/tools/registry.hpp"

#include <filesystem>
#include <system_error>

#include "src/platform/fs.hpp"

// The cross-session memory file (spec S8). Split out of registry.cpp because it is the one
// tool whose state outlives the run: every other handler there acts on the workspace
// within a single mission, and this one writes a file the NEXT session reads. That is a
// different lifetime and a different failure mode, and it wants its own tests.

namespace lmp::tools {

namespace fsx = lmp::platform;

bool would_overwrite_existing(const std::string& root, const std::string& rel) {
    const std::string abs = resolve_contained(root, rel);
    if (abs.empty()) {
        return false; // outside the root: refused for a different reason, upstream
    }
    // Read-based rather than stat-based, so it uses the same door every other path check
    // uses. A file too large to read is certainly not empty, so it counts as content.
    const fsx::FileContents f = fsx::read_file_whole(abs, 1);
    if (f.status == fsx::FsStatus::NotFound) {
        return false;
    }
    return !(f.ok() && f.bytes.empty());
}

platform::WriteResult Registry::commit_write(const std::string& abs_path,
                                             std::string_view content) {
    if (!edit_sink_) {
        // The parent directory is MADE, not required to exist. Writing `src/store.py`
        // into an empty workspace is the most ordinary thing an agent does, and without
        // this it failed -- on the temp file, so the error named
        // `src/store.py.tmp.8276: No such file or directory`, a path the model never
        // asked for and could not act on.
        //
        // MEASURED, and it is the whole reason this file exists in this form: a real run
        // regenerated 8 KB of correct code SIX times against that error (turns 15-21),
        // then degenerated into sending empty content, and spent 20 of its 40 turns there
        // before stumbling onto `mkdir -p src` in a shell call. Nothing was wrong with
        // the model's work; it could not create a directory.
        //
        // Containment is already settled: resolve_contained() returned abs_path, so it is
        // inside the workspace root and so is every parent this creates.
        const std::filesystem::path parent =
            std::filesystem::path(abs_path).parent_path();
        if (!parent.empty()) {
            std::error_code ec;
            std::filesystem::create_directories(parent, ec);
            // Not reported here: if the directory is genuinely unusable the write below
            // fails and says so about the path the model NAMED, which is the better
            // sentence. create_directories also sets ec for "already exists" on some
            // implementations, and that is not an error.
        }
        return fsx::write_file_atomic(abs_path, content);
    }
    const EditOutcome o = edit_sink_(abs_path, std::string(content));
    platform::WriteResult w;
    w.status = o.applied ? platform::FsStatus::Ok : platform::FsStatus::IoError;
    w.error = o.applied ? std::string() : ("the editor did not apply the edit: " + o.error);
    return w;
}

ToolResult Registry::remember_fact(const std::string& raw) {
    // Folded to ONE LINE, not refused. Dedupe and trimming both work line-wise, so a
    // multi-line note would silently defeat both; folding keeps every character the
    // model wrote and keeps the file's one invariant.
    std::string fact;
    fact.reserve(raw.size());
    for (const char c : raw) {
        fact += (c == '\n' || c == '\r' || c == '\t') ? ' ' : c;
    }
    const std::size_t first = fact.find_first_not_of(' ');
    const std::size_t last = fact.find_last_not_of(' ');
    if (first == std::string::npos) {
        return ToolResult::error(ErrorClass::Malformed, false,
                                 "a note cannot be blank");
    }
    fact = fact.substr(first, last - first + 1);

    const std::string path = ctx_.root + "/" + kMemoryFileName;
    const std::string line = "- " + fact + "\n";

    // Read wider than the cap so a file that is already oversized can be trimmed back
    // rather than refused -- otherwise one oversized write would wedge the tool forever.
    const fsx::FileContents cur = fsx::read_file_whole(path, kMemoryMaxBytes * 4);
    std::string body = cur.ok() ? cur.bytes : std::string();

    // Exact repeats are the common case: a model that re-reads its own notes re-derives
    // the same conclusion. An unbounded pile of duplicates is how this section stops
    // being worth the tokens it costs.
    if (body.find(line) != std::string::npos) {
        return ToolResult::okay("already noted: " + fact);
    }
    body += line;

    // Trimmed from the FRONT, so a full file drops its OLDEST notes. Truncating the tail
    // instead would freeze the memory at whatever the project learned first and silently
    // discard everything it learned since.
    while (body.size() > kMemoryMaxBytes) {
        const std::size_t nl = body.find('\n');
        if (nl == std::string::npos) {
            break;
        }
        body.erase(0, nl + 1);
    }

    const fsx::WriteResult w = fsx::write_file_atomic(path, body);
    if (!w.ok()) {
        return ToolResult::error(ErrorClass::Transient, true, w.error);
    }
    return ToolResult::okay("noted for later sessions: " + fact);
}

} // namespace lmp::tools
