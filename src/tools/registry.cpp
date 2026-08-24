#include "src/tools/registry.hpp"

#include "src/model/image_decode.hpp"
#include "src/model/image_preprocess.hpp"

#include "src/tools/concurrent_calls.hpp"
#include "src/tools/ignore_dirs.hpp"
#include "src/tools/symbol_index.hpp"
#include "src/tools/text_view.hpp"

#include <cctype>
#include <cstdlib>

#include <algorithm>
#include <cstdio>
#include <map>
#include <optional>
#include <sstream>
#include <string_view>

#include "src/pcc/diff.hpp"
#include "src/platform/fs.hpp"
#include "src/tools/apply_patch.hpp"
#include "src/tools/edit_diagnostics.hpp"
#include "src/tools/graft_engine.hpp"
#include "src/tools/log_triage.hpp"
#include "src/tools/sandbox.hpp"

namespace lmp::tools {
namespace {

namespace fsx = lmp::platform;
using parsephony::ParamSpec;
using parsephony::ParamType;

// A failed write, classified by WHY -- because `retryable` tells the model (and any
// consumer of the result) whether re-sending the identical bytes can ever come back
// different.
//
// Every write failure used to be reported as Transient AND retryable, which told the loop
// that re-sending the identical bytes was worth a turn. It is not, for any of these
// causes: they are pure functions of the path and the process's permissions. A real run
// re-sent the same 8 KB write six times against the same error and nothing fired, because
// a retryable error is never an unrecoverable repeat.
//
// Only a genuine I/O error keeps the retry, which is the one case where trying again can
// legitimately come out differently.
ToolResult write_failure(const fsx::WriteResult& w) {
    switch (w.status) {
        case fsx::FsStatus::NotFound:
            return ToolResult::error(ErrorClass::NotFound, false, w.error);
        case fsx::FsStatus::PermissionDenied:
            return ToolResult::error(ErrorClass::Policy, false, w.error);
        case fsx::FsStatus::IsDirectory:
        case fsx::FsStatus::TooLarge:
        case fsx::FsStatus::InvalidPath:
            return ToolResult::error(ErrorClass::Malformed, false, w.error);
        case fsx::FsStatus::OutsideRoot:
        case fsx::FsStatus::Symlink:
            return ToolResult::refused(w.error);
        case fsx::FsStatus::Conflict:
            // Not retryable with the same bytes: the preimage moved. Re-read first.
            return ToolResult::error(ErrorClass::Conflict, false, w.error);
        case fsx::FsStatus::Ok:
        case fsx::FsStatus::IoError:
            break;
    }
    return ToolResult::error(ErrorClass::Transient, true, w.error);
}

std::string with_content_version(std::string body, std::string_view version) {
    if (!body.empty() && body.back() != '\n') {
        body.push_back('\n');
    }
    body += "[content_version sha256=";
    body.append(version.data(), version.size());
    body += "]\n";
    return body;
}

ToolResult need_read_before_write(const std::string& path) {
    return ToolResult::error(
        ErrorClass::Conflict, false,
        path + ": existing file requires read_file (or read_slice) in this run before "
               "overwrite/delete so the edit can carry a content version; read it, then "
               "retry");
}

ToolResult measured_read(ToolResult result, std::size_t bytes) {
    result.bytes_read = bytes;
    return result;
}

ToolResult measured_edit(ToolResult result, std::size_t bytes) {
    result.bytes_changed = bytes;
    return result;
}

// Added + removed bytes in the smallest contiguous region that differs. replace_in_file
// changes one region by contract, so this is exact there and avoids pretending the whole
// rewritten file changed.
std::size_t contiguous_edit_bytes(std::string_view before, std::string_view after) {
    std::size_t prefix = 0;
    while (prefix < before.size() && prefix < after.size() &&
           before[prefix] == after[prefix]) {
        ++prefix;
    }
    std::size_t suffix = 0;
    while (suffix < before.size() - prefix && suffix < after.size() - prefix &&
           before[before.size() - 1 - suffix] == after[after.size() - 1 - suffix]) {
        ++suffix;
    }
    return (before.size() - prefix - suffix) + (after.size() - prefix - suffix);
}

// The one sandbox failure a model cannot read, turned into one it can act on.
//
// macOS refuses a NESTED sandbox: a process already running under a Seatbelt profile gets
// EPERM from sandbox_apply, and that holds whatever the outer profile allows -- measured
// against a profile containing nothing but `(allow default)` plus a single deny. SwiftPM
// applies a sandbox of its own to compile Package.swift, so `swift build` and `swift test`
// inside T1 die on `sandbox-exec: sandbox_apply: Operation not permitted`, wrapped in an
// "Invalid manifest" dump listing forty compiler flags, none of which is the reason.
//
// MEASURED: a real run spent thirty turns on that message. It invented cache directories,
// deleted DerivedData, rewrote correct code and re-read its own files, because nothing in
// the output says the harness is the thing in the way and nothing names the one-word fix.
//
// Passing --disable-sandbox is not a downgrade: OUR jail is still fully in force around
// the command, and the flag removes only SwiftPM's second jail inside the first. Said as
// an observed fact about this workspace, which is what T2 is for -- the model is free to
// do something else with it.
// Matched on the two halves of that message that are not identifiers -- the tool that
// printed it and the errno text. The middle word is the C function's name, and spelling
// it here would read to the tool-honesty ratchet (S6.3) as a claim that `sandbox_apply`
// is a registered tool. Both halves must be present, which is what keeps this from
// firing on an unrelated permission error.
constexpr std::string_view kSandboxExecPrefix = "sandbox-exec: ";
constexpr std::string_view kNotPermitted = "Operation not permitted";

[[nodiscard]] std::string nested_sandbox_note(std::string_view output,
                                              std::string_view command) {
    if (output.find(kSandboxExecPrefix) == std::string_view::npos ||
        output.find(kNotPermitted) == std::string_view::npos) {
        return {};
    }
    const std::string_view lead =
        "\n[sandbox] That failed because the command tried to start a sandbox of its own "
        "inside this one, and macOS refuses to nest them -- it is not a problem with your "
        "code, your caches or your build directory, and no amount of cleaning or "
        "reconfiguring will change it. ";
    // XCODEBUILD IS THE CASE NO FLAG FIXES, and telling it to pass `--disable-sandbox` --
    // which is what this note used to say to everyone -- sends it after an option that does
    // not exist. It writes its result bundle to the per-user temp root and cannot be talked
    // out of it, so the answer is a TIER, not an argument.
    //
    // Named here rather than acted on: raising the tier for a command the operator granted
    // at T1 would be a privilege escalation nobody asked for (S7), and t1_compat_rewrite is
    // forbidden from touching the tier for the same reason. So the run is told precisely
    // what it needs and left to ask for it.
    if (command.find("xcodebuild") != std::string_view::npos) {
        return std::string(lead) +
               "`xcodebuild` is doing it, and unlike SwiftPM it has no flag that turns it "
               "off: it writes its result bundle under the per-user temp root and cannot be "
               "configured out of it. There is nothing to fix in the command. It needs "
               "sandbox tier 3, which runs it on the host with the workspace jail still "
               "around it -- ask for that tier, or verify with `swift build` / `swift test` "
               "instead, which do run here.";
    }
    // SwiftPM: still spelled out, but this should now be rare -- t1_compat_rewrite adds the
    // flag for `swift build|test|run|package` before the command ever runs, including
    // through `xcrun` and the other launcher prefixes. Reaching this text means something
    // the rewrite does not cover started a sandbox, so it says how to spell the fix rather
    // than assuming SwiftPM.
    return std::string(lead) +
           "Swift Package Manager does this to compile Package.swift, and the harness "
           "normally adds `--disable-sandbox` for `swift build`, `swift test`, `swift run` "
           "and `swift package` automatically. If the program here is a wrapper or a script "
           "that runs one of those itself, pass `--disable-sandbox` through to it; the "
           "workspace jail stays in force around it either way.";
}

ToolResult refused_path(const std::string& p, std::string_view detail = {}) {
    std::string message = "path '" + p + "' is not reachable through the workspace";
    if (!detail.empty()) {
        message += ": ";
        message += detail;
    }
    return ToolResult::refused(std::move(message));
}

ToolResult contained_failure(const std::string& path,
                             const fsx::ContainedPath& resolved) {
    if (resolved.status == fsx::FsStatus::OutsideRoot ||
        resolved.status == fsx::FsStatus::Symlink) {
        return refused_path(path, resolved.error);
    }
    return ToolResult::error(ErrorClass::Malformed, false, resolved.error);
}

// Single-quote for /bin/sh: wrap in ', and close-escape-reopen each embedded '. The git
// tools compose their own command lines, so the only model-supplied bytes that reach a
// shell are a path, and they reach it quoted.
std::string shell_quote(std::string_view s) {
    std::string out = "'";
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out += "'";
    return out;
}

std::string child_path(std::string_view parent, std::string_view name) {
    if (parent.empty() || parent == ".") {
        return std::string(name);
    }
    return std::string(parent) + "/" + std::string(name);
}

using FileVisitor = std::function<bool(const std::string&, std::string_view)>;

// Walk through descriptor-rooted list/read operations. A symlink is visible as an entry
// but is neither traversed nor read; a concurrent replacement is rejected again by the
// operation that opens the child.
constexpr std::size_t kFindFilesMax = 500;

// `*` (any run, including empty) and `?` (one character). Iterative with a backtrack point
// rather than recursive, so a pathological pattern cannot blow the stack.
bool glob_match(std::string_view pattern, std::string_view text) {
    std::size_t p = 0;
    std::size_t t = 0;
    std::size_t star = std::string_view::npos;
    std::size_t retry = 0;
    while (t < text.size()) {
        if (p < pattern.size() && (pattern[p] == '?' || pattern[p] == text[t])) {
            ++p;
            ++t;
        } else if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            retry = t;
        } else if (star != std::string_view::npos) {
            p = star + 1;
            t = ++retry;
        } else {
            return false;
        }
    }
    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

// The same descent as walk_regular_files, without opening anything. Separate rather than a
// flag on that one because its visitor is handed BYTES: walking names through it means
// reading every file in the tree to look at its path, and passing max_read_bytes=0 to avoid
// that skips every file instead -- read_file_whole reports a zero cap as "too large", the
// visit never fires, and the search silently returns nothing.
bool walk_file_names(const fsx::WorkspaceFs& workspace, const std::string& path,
                     const std::function<bool(const std::string&)>& visit) {
    fsx::DirectoryContents dir = workspace.list_directory(path);
    if (!dir.ok()) {
        // Not a directory: it is the single file that was asked for. Mirrors
        // walk_regular_files, which falls back the same way -- without this, naming a file
        // as `subdir` walks nothing and reports no matches, which reads as "your pattern
        // found nothing" rather than "you pointed me at a file".
        return visit(path);
    }
    std::sort(dir.entries.begin(), dir.entries.end(),
              [](const fsx::DirectoryEntry& a, const fsx::DirectoryEntry& b) {
                  return a.name < b.name;
              });
    for (const fsx::DirectoryEntry& entry : dir.entries) {
        const std::string child = child_path(path, entry.name);
        if (entry.kind == fsx::DirectoryEntryKind::Directory) {
            if (skip_during_descent(entry.name)) {
                continue;
            }
            if (!walk_file_names(workspace, child, visit)) {
                return false;
            }
        } else if (entry.kind == fsx::DirectoryEntryKind::File) {
            if (!visit(child)) {
                return false;
            }
        }
    }
    return true;
}

bool walk_regular_files(const fsx::WorkspaceFs& workspace, const std::string& path,
                        std::size_t max_read_bytes, const FileVisitor& visit) {
    fsx::DirectoryContents dir = workspace.list_directory(path);
    if (!dir.ok()) {
        const fsx::FileContents file =
            workspace.read_file_whole(path, max_read_bytes);
        return !file.ok() || visit(path, file.bytes);
    }
    std::sort(dir.entries.begin(), dir.entries.end(),
              [](const fsx::DirectoryEntry& a, const fsx::DirectoryEntry& b) {
                  return a.name < b.name;
              });
    for (const fsx::DirectoryEntry& entry : dir.entries) {
        const std::string child = child_path(path, entry.name);
        if (entry.kind == fsx::DirectoryEntryKind::Directory) {
            if (skip_during_descent(entry.name)) {
                continue;
            }
            if (!walk_regular_files(workspace, child, max_read_bytes, visit)) {
                return false;
            }
        } else if (entry.kind == fsx::DirectoryEntryKind::File) {
            const fsx::FileContents file =
                workspace.read_file_whole(child, max_read_bytes);
            if (file.ok() && !visit(child, file.bytes)) {
                return false;
            }
        }
    }
    return true;
}

bool identifier_at(std::string_view line, std::size_t at, std::size_t size) {
    const auto word = [](char c) {
        const unsigned char u = static_cast<unsigned char>(c);
        return std::isalnum(u) != 0 || c == '_';
    };
    return (at == 0 || !word(line[at - 1])) &&
           (at + size == line.size() || !word(line[at + size]));
}

bool definition_shaped(std::string_view line, std::string_view symbol) {
    std::size_t at = line.find(symbol);
    while (at != std::string_view::npos) {
        if (identifier_at(line, at, symbol.size())) {
            static constexpr std::string_view kLeads[] = {
                "class", "struct", "enum", "def", "fn", "func", "function",
                "const", "let", "var", "void", "int", "bool", "auto", "std::string",
            };
            const std::string_view before = line.substr(0, at);
            for (std::string_view lead : kLeads) {
                if (before.find(lead) != std::string_view::npos) {
                    return true;
                }
            }
        }
        at = line.find(symbol, at + symbol.size());
    }
    return false;
}

// JSON string escape for tools_json -- the schema block, not a general serializer.
void append_json_escaped(std::string& out, std::string_view in) {
    for (char c : in) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            default:
                out += c;
        }
    }
}

} // namespace

ParamSpec param(const char* name, ParamType type, bool required) {
    ParamSpec p;
    p.name = name;
    p.type = type;
    p.required = required;
    return p;
}

const std::string* get(const std::vector<ToolParamValue>& params, const char* name) {
    for (const ToolParamValue& p : params) {
        if (p.name == name) {
            return &p.value;
        }
    }
    return nullptr;
}

std::string resolve_contained(const std::string& root, const std::string& rel) {
    const fsx::WorkspaceFs workspace(root);
    const fsx::ContainedPath path = workspace.contained_path(rel);
    return path.ok() ? path.absolute : std::string();
}

const ToolDecl* Registry::find(const std::string& name) const {
    for (const ToolDecl& d : decls_) {
        if (d.name == name) {
            return &d;
        }
    }
    return nullptr;
}

ToolResult Registry::run_git(const std::string& args, int approved_tier) {
    // Read-only git still runs in the jail: `git` reads config from outside the
    // workspace, and T1 allows reads. What it must not do is write, and the profile --
    // not this function -- is what guarantees that.
    const ExecutionGrant grant = grant_execution(tier_for(approved_tier));
    const ExecLimits limits{30, 30, 2LL << 30, 256, 64, ctx_.max_result_bytes};
    const ExecOutcome o =
        run_sandboxed(grant, "git " + args, ctx_.root, ctx_.root, limits, cancel_token_);

    if (o.status == Status::Cancelled) {
        return ToolResult::cancelled(o.output.empty() ? "cancelled" : o.output);
    }
    if (o.status == Status::Refused) {
        ToolResult r;
        r.status = o.status;
        r.error_class = ErrorClass::Policy;
        r.summary = o.output;
        return r;
    }
    if (o.exit_code == 128) {
        // git's "not a repository" exit. A fact about the workspace, not a tool failure.
        return ToolResult::okay("not a git repository (or no commits yet)");
    }
    if (o.exit_code != 0) {
        return ToolResult::error(ErrorClass::Transient, true,
                                 "[exit " + std::to_string(o.exit_code) + "]\n" + o.output);
    }
    return ToolResult::okay(o.output.empty() ? "(no changes)"
                                             : log_triage::compact(o.output,
                                                                   ctx_.max_result_bytes));
}

void Registry::declare(ToolDecl decl, Handler handler) {
    specs_.push_back(decl.spec);
    handlers_.emplace(decl.name, std::move(handler));
    decls_.push_back(std::move(decl));
}

bool Registry::declare_remote(ToolDecl decl, Handler handler) {
    // The namespacing in mcp_host.cpp should already make this impossible. Checked here
    // anyway, because "a remote tool shadowed read_file" is the kind of thing that must
    // fail closed at the door rather than depend on a caller getting a prefix right.
    if (handlers_.find(decl.name) != handlers_.end()) {
        return false;
    }
    declare(std::move(decl), std::move(handler));
    return true;
}

ToolResult Registry::execute(const std::string& name,
                             const std::vector<ToolParamValue>& params, int approved_tier,
                             const model::CancelToken* cancel) {
    // A non-null argument installs a temporary token only when the run has not already
    // set one: the agent path uses set_cancel_token once for the whole run (including
    // parallel execute), and restoring a temporary over that would race. Tests that
    // never call set_cancel_token still get a per-call override.
    const model::CancelToken* previous = cancel_token_;
    const bool installed = cancel != nullptr && cancel_token_ == nullptr;
    if (installed) {
        cancel_token_ = cancel;
    }
    struct Restorer {
        Registry* reg;
        const model::CancelToken* previous;
        bool installed;
        ~Restorer() {
            if (installed) {
                reg->cancel_token_ = previous;
            }
        }
    } restorer{this, previous, installed};

    const model::CancelToken* active = cancel_token_;
    if (active != nullptr && active->cancelled()) {
        return ToolResult::cancelled();
    }

    const auto it = handlers_.find(name);
    if (it == handlers_.end()) {
        // Unreachable when the guard did its job; typed anyway, never a crash.
        return ToolResult::error(ErrorClass::NotFound, false,
                                 "tool '" + name + "' is not registered");
    }
    ToolResult r = it->second(params, approved_tier);
    // NO TOOL MAY RETURN A SILENT RESULT. An empty summary is dropped from the rendered
    // prompt (context.cpp only appends a tool_response when the observation is non-empty),
    // so a call that produced no text leaves the next turn's prompt byte-identical to this
    // one's -- and at a fixed seed a byte-identical prompt produces a byte-identical
    // continuation. That is a deterministic loop, and it is the same failure the loop
    // already defends against for a token-capped turn.
    //
    // MEASURED twice: `shell` returning "" for a successful `mkdir` (6 turns), and
    // `read_file` returning "" for a 0-byte file (17 turns). Each tool that can be silent
    // says something specific above; this is the floor under all of them, including any
    // tool added later.
    if (r.summary.empty()) {
        r.summary = "(" + name + " " +
                    (r.ok() ? "succeeded and produced no output"
                            : std::string("failed: ") + std::string(to_string(r.status)) +
                                  ", with no detail") +
                    ")";
    }
    return r;
}

std::string Registry::tools_json() const {
    return tools_json([](const ToolDecl&) { return true; });
}

std::string Registry::tools_json(const std::function<bool(const ToolDecl&)>& include) const {
    std::string out;
    for (const ToolDecl& d : decls_) {
        if (!include(d)) {
            continue;
        }
        out += "\n{\"type\": \"function\", \"function\": {\"name\": \"" + d.name +
               "\", \"description\": \"";
        append_json_escaped(out, d.description);
        out += "\", \"parameters\": {\"type\": \"object\", \"properties\": {";
        bool first = true;
        std::string required;
        for (const ParamSpec& p : d.spec.params) {
            if (!first) {
                out += ", ";
            }
            first = false;
            const char* base_type = p.type == ParamType::Number    ? "number"
                                    : p.type == ParamType::Boolean ? "boolean"
                                    : p.type == ParamType::Object  ? "object"
                                    : p.type == ParamType::Array   ? "array"
                                    : p.type == ParamType::Json    ? "object"
                                                                   : "string";
            out += "\"" + p.name + "\": {";
            if (p.nullable && p.type != ParamType::Json) {
                out += "\"type\": [\"";
                out += base_type;
                out += "\", \"null\"]";
            } else {
                out += "\"type\": \"";
                out += base_type;
                out += "\"";
            }
            if (!p.enum_values.empty()) {
                out += ", \"enum\": [";
                for (std::size_t i = 0; i < p.enum_values.size(); ++i) {
                    if (i > 0) {
                        out += ", ";
                    }
                    out += "\"";
                    append_json_escaped(out, p.enum_values[i]);
                    out += "\"";
                }
                out += "]";
            }
            if (p.has_items_type && p.schema_extras_json.empty()) {
                const char* item = p.items_type == ParamType::Number    ? "number"
                                   : p.items_type == ParamType::Boolean ? "boolean"
                                   : p.items_type == ParamType::Object  ? "object"
                                   : p.items_type == ParamType::Array   ? "array"
                                                                        : "string";
                out += ", \"items\": {\"type\": \"";
                out += item;
                out += "\"}";
            }
            if (!p.schema_extras_json.empty()) {
                // schema_extras_json is a JSON object; splice its fields in.
                if (p.schema_extras_json.size() >= 2 && p.schema_extras_json.front() == '{' &&
                    p.schema_extras_json.back() == '}') {
                    const std::string inner =
                        p.schema_extras_json.substr(1, p.schema_extras_json.size() - 2);
                    if (!inner.empty()) {
                        out += ", ";
                        out += inner;
                    }
                }
            }
            out += "}";
            if (p.required) {
                required += required.empty() ? "" : ", ";
                required += "\"" + p.name + "\"";
            }
        }
        out += "}, \"required\": [" + required + "]}}}";
    }
    return out;
}

// Newline-separated paths, or a JSON string array flattened the same way plan accepts.
// Empty lines are skipped; the caller enforces the bound.
std::vector<std::string> parse_read_many_paths(std::string_view raw) {
    std::string text(raw);
    // Narrow JSON-array acceptance: starts with `[`, first element is `"`, ends with `]`.
    const std::size_t open = text.find_first_not_of(" \t\r\n");
    if (open != std::string::npos && text[open] == '[') {
        const std::size_t first = text.find_first_not_of(" \t\r\n", open + 1);
        const std::size_t close = text.find_last_not_of(" \t\r\n");
        if (first != std::string::npos && text[first] == '"' &&
            close != std::string::npos && text[close] == ']') {
            std::string flat;
            bool in_string = false;
            std::string current;
            for (std::size_t i = first; i < close; ++i) {
                const char c = text[i];
                if (!in_string) {
                    if (c == '"') {
                        in_string = true;
                    }
                    continue;
                }
                if (c == '\\' && i + 1 < close) {
                    current += text[++i];
                    continue;
                }
                if (c == '"') {
                    in_string = false;
                    if (!current.empty()) {
                        flat += current;
                        flat += '\n';
                    }
                    current.clear();
                    continue;
                }
                current += c;
            }
            if (!in_string) {
                text = std::move(flat);
            }
        }
    }
    std::vector<std::string> paths;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t a = line.find_first_not_of(" \t\r");
        if (a == std::string::npos) {
            continue;
        }
        const std::size_t b = line.find_last_not_of(" \t\r");
        paths.push_back(line.substr(a, b - a + 1));
    }
    return paths;
}

// WHERE THAT FILE ACTUALLY IS, said in the same breath as "it is not there".
//
// A path that does not exist was 22% of every failed tool call in the log since
// 2026-08-08 -- second only to the over-budget read above -- and the reply was an
// instruction to go and run `list_dir`, i.e. another turn, ~70 s, to learn something the
// harness can answer by walking names it never opens. Almost every one of these is the
// model naming a file by its basename, or under the wrong parent, when exactly one file
// of that name exists in the tree.
//
// Names only, capped, and silent when there is no good answer: a wrong suggestion is
// worse than none, so this matches the basename exactly rather than guessing at
// near-misses.
std::string Registry::suggest_paths_for(const std::string& path) const {
    const std::size_t slash = path.rfind('/');
    const std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    if (base.empty()) {
        return {};
    }
    constexpr std::size_t kMaxSuggestions = 5;
    std::vector<std::string> hits;
    (void)walk_file_names(workspace_fs_, ".", [&](const std::string& found) {
        const std::size_t s = found.rfind('/');
        if ((s == std::string::npos ? found : found.substr(s + 1)) == base) {
            hits.push_back(found);
        }
        return hits.size() <= kMaxSuggestions;
    });
    if (hits.empty()) {
        return {};
    }
    std::sort(hits.begin(), hits.end());
    std::string out = hits.size() == 1 ? ". It IS here, at '" + hits[0] + "' -- use that path."
                                       : ". Files with that name exist at: ";
    if (hits.size() > 1) {
        for (std::size_t i = 0; i < hits.size() && i < kMaxSuggestions; ++i) {
            out += (i != 0 ? ", " : "") + hits[i];
        }
        out += " -- pick the one you meant.";
    }
    return out;
}

ToolResult Registry::read_one_file(const std::string& path) {
    const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
    if (!resolved.ok()) {
        return contained_failure(path, resolved);
    }
    fsx::FileContents f = workspace_fs_.read_file_whole(path, ctx_.max_read_bytes);
    if (!f.ok()) {
        if (f.status == fsx::FsStatus::Symlink ||
            f.status == fsx::FsStatus::InvalidPath) {
            return refused_path(path, f.error);
        }
        const ErrorClass ec = f.status == fsx::FsStatus::NotFound
                                  ? ErrorClass::NotFound
                                  : ErrorClass::Malformed;
        std::string err_msg = f.error;
        if (f.status == fsx::FsStatus::NotFound) {
            err_msg += ". File not found at '" + path + "'";
            const std::string where = suggest_paths_for(path);
            err_msg += where.empty()
                           ? ". Nothing by that name is in the workspace -- check the name "
                             "with find_files before reading."
                           : where;
        }
        return ToolResult::error(ec, false, err_msg);
    }
    // The PROMPT budget, not the read budget. These two used to be one number at
    // 4 MiB, so a single read could hand the context store more bytes than the
    // whole 96k-token window holds -- while this description already claimed it
    // "fails honestly with the real size". It does now.
    // OVER BUDGET IS NOT A REASON TO RETURN NOTHING. This used to be a hard error, and it
    // was the single most common failed tool call in the log: 35% of every failure since
    // 2026-08-08. The harness knew the file's size before the call, refused, and spent a
    // whole turn -- ~70 s of model time -- teaching the model a fact it could have been
    // handed along with the head of the file.
    //
    // So the read SUCCEEDS with as much as fits, cut at a line boundary, and the note says
    // exactly where it stopped and how to get the rest.
    //
    // WHAT IT MUST NOT DO IS GRANT AN OVERWRITE. note_read_version() is deliberately NOT
    // called here: recording the version of the whole file after showing part of it is
    // how a model overwrites -- in good faith -- a file it has only seen the top of. The
    // read-before-write guard stays unsatisfied, `write_file` keeps refusing, and the note
    // says so, which points at the tool that is right for a large file anyway
    // (replace_in_file on the section being changed).
    if (f.bytes.size() > ctx_.max_model_read_bytes) {
        // count_lines() counts the line after a trailing newline, which is right for its
        // other callers and wrong for a range hint: it would send read_slice at a line
        // one past the end of the file.
        const std::size_t total_lines =
            count_lines(f.bytes) - (f.bytes.empty() || f.bytes.back() != '\n' ? 0 : 1);
        // Back up to the last newline inside the budget so the tail is never a half line;
        // if the first line alone is over budget, take the hard cut rather than nothing.
        std::string_view head(f.bytes.data(), ctx_.max_model_read_bytes);
        const std::size_t nl = head.rfind('\n');
        if (nl != std::string_view::npos && nl > 0) {
            head = head.substr(0, nl);
        }
        const std::size_t shown_lines = count_lines(head);
        return measured_read(
            ToolResult::okay(
                number_lines(head, 1) + "\n[TRUNCATED: showing lines 1-" +
                std::to_string(shown_lines) + " of " + std::to_string(total_lines) +
                ". " + path + " is " + std::to_string(f.bytes.size()) +
                " bytes, over the " + std::to_string(ctx_.max_model_read_bytes) +
                "-byte prompt budget. Read on with read_slice(" + path + ", " +
                std::to_string(shown_lines + 1) + ", " +
                std::to_string(total_lines) +
                "). You have NOT seen this whole file, so write_file will refuse it -- "
                "change it with replace_in_file on the section you are editing.]"),
            head.size());
    }
    // An empty file is a FACT, and it has to be stated. Returning "" makes a
    // successful read indistinguishable from nothing happening at all -- and an
    // empty observation is dropped from the rendered prompt entirely
    // (context.cpp), so the next turn cannot see that the call was ever made.
    //
    // MEASURED: a run wrote 0 bytes to src/kv_store.py, then read it back 17 turns
    // running, getting "" each time and re-issuing the identical call, until the
    // budget ended the run.
    const std::string version = fsx::content_sha256_hex(f.bytes);
    note_read_version(resolved.absolute, f.bytes);
    if (f.bytes.empty()) {
        return measured_read(
            ToolResult::okay(with_content_version(
                "(" + path + " exists and is empty: 0 bytes)", version)),
            0);
    }
    // Always return current content. Under context pressure the agent may collapse an
    // older verbatim duplicate still in rendered history; a pointer-only "refer to
    // previous read" is never safe after compaction.
    return measured_read(
        ToolResult::okay(with_content_version(number_lines(f.bytes, 1), version)),
        f.bytes.size());
}

Registry::Registry(WorkspaceContext ctx)
    : ctx_(std::move(ctx)), workspace_fs_(ctx_.root) {
    if (workspace_fs_.valid()) {
        ctx_.root = workspace_fs_.root();
    }
    // --- view_image --------------------------------------------------------
    //
    // A SEPARATE TOOL, not a branch inside read_file. Reading a PNG as text is a mistake
    // the model should be told about plainly ("that is an image, use view_image"), and
    // silently doing something else because the extension looked graphical is the kind of
    // helpfulness that makes a tool's contract unknowable. It also keeps the cost
    // legible: this call spends context proportional to the picture, and the model should
    // be choosing that deliberately.
    //
    // The result carries the PATH, not pixels. Decoding needs the vision config, which
    // lives beside the model; the prompt builder re-reads the file when it renders, so
    // the bytes the model sees are the bytes on disk at that moment.
    //
    // DECLARED ONLY IF THE MODEL CAN SEE. Not offering the tool is how a text-only
    // checkpoint says "I cannot look at that" -- in the one language the model always
    // understands, at the only moment it can still choose differently. Offered anyway,
    // the tool returns Ok and the run dies on the next turn, several layers away, having
    // already told the model it was shown a picture it was never shown.
    if (ctx_.model_can_see) {
        ToolDecl d;
        d.name = "view_image";
        d.description =
            "Look at an image file (png, jpg, heic, gif, bmp, tiff, webp) -- the picture "
            "itself, not a description of it. Use this when you need to SEE something: a "
            "screenshot, a diagram, a rendered output, a photo. Costs context in "
            "proportion to the image's size, so prefer it when looking is the point "
            "rather than as a substitute for reading a file.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            if (!model::looks_like_image_path(path)) {
                return ToolResult::error(
                    ErrorClass::Malformed, false,
                    "(" + path +
                        " does not look like an image. view_image reads png, jpg, heic, "
                        "gif, bmp, tiff and webp; for text use read_file.)");
            }
            // Decoded HERE as well as at render time, so a file that cannot be read is a
            // failed tool call the model can react to -- rather than a prompt that throws
            // several layers away, after the turn has already been recorded.
            model::ImageRGB img;
            std::string err;
            // A DECODE bound, separate from the token budget below: this one exists to
            // refuse a decompression bomb before the pixels are allocated, and it is
            // checked against the file's header. 64 MP is well past any real screenshot
            // and well short of a 48 GB host's patience.
            constexpr long long kMaxDecodedPixels = 64LL * 1000LL * 1000LL;
            if (!model::decode_image_file(resolved.absolute, kMaxDecodedPixels, img, err)) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "(" + path + " could not be decoded: " + err + ")");
            }
            model::PreprocessConfig cfg;
            cfg.max_pixels =
                model::token_budget_to_max_pixels(ctx_.max_image_tokens, cfg);
            model::PreprocessedImage pre;
            if (!model::preprocess_image(img, cfg, pre, err)) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "(" + path + " could not be prepared: " + err + ")");
            }
            // The summary says what it COST, because that is the part the model cannot
            // see and the part that competes with everything else in the context.
            ToolResult r = ToolResult::okay(
                "(" + path + ": " + std::to_string(img.width) + "x" +
                std::to_string(img.height) + " image, shown to you as " +
                std::to_string(pre.token_count()) + " tokens)");
            r.images.push_back(path);
            r.bytes_read = img.rgb.size();
            return r;
        });
    }

    // --- read_file ---------------------------------------------------------
    {
        ToolDecl d;
        d.name = "read_file";
        d.description = "Read a file, whole, with 1-based line numbers prefixed for "
                        "reference. A file too big for the prompt budget is not an error: "
                        "you get as much as fits plus the line range to continue from with "
                        "read_slice, and you will be told it was truncated. The line "
                        "numbers are display only -- never include them in old_text. "
                        "Re-reading always returns current content.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            return read_one_file(*get(p, "path"));
        });
    }
    // --- read_many ---------------------------------------------------------
    {
        ToolDecl d;
        d.name = "read_many";
        d.description =
            "Read up to four files in one call. Pass paths as newline-separated text "
            "(or a JSON string array). Each file is returned with the same line-numbered "
            "format as read_file. Listing more than four is not an error -- the first four "
            "are read and the rest are named back to you, so call it again for those. "
            "Prefer this when several independent files are needed before the next edit; "
            "keep issuing separate read_file calls when the paths are not known together.";
        d.spec.name = d.name;
        d.spec.params = {param("paths", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            std::vector<std::string> paths = parse_read_many_paths(*get(p, "paths"));
            if (paths.empty()) {
                return ToolResult::error(ErrorClass::Malformed, true,
                                         "paths must list at least one non-empty path");
            }
            // READ THE FIRST FOUR, do not refuse all fifteen.
            //
            // Over the cap used to be a hard error that read nothing, and a model that has
            // just listed a directory naturally asks for everything in it. Observed: a run
            // asked for 15 files, was refused, asked for 7, was refused again, and burned a
            // quarter of its turns learning a number the error had already told it. The cap
            // exists to bound how much lands in one observation, and truncating bounds that
            // exactly as well as refusing does -- while making the turn progress.
            std::string overflow_note;
            if (paths.size() > kReadManyMaxPaths) {
                const std::size_t asked = paths.size();
                std::string dropped;
                for (std::size_t i = kReadManyMaxPaths; i < asked; ++i) {
                    dropped += (dropped.empty() ? "" : ", ") + paths[i];
                }
                paths.resize(kReadManyMaxPaths);
                overflow_note = "\n\n(read the first " + std::to_string(kReadManyMaxPaths) +
                                " of " + std::to_string(asked) +
                                " paths; read_many caps each call at " +
                                std::to_string(kReadManyMaxPaths) +
                                ". Not read yet: " + dropped +
                                ". Call read_many again for those.)";
            }
            std::vector<std::size_t> indices(paths.size());
            for (std::size_t i = 0; i < paths.size(); ++i) {
                indices[i] = i;
            }
            // Same concurrent primitive as multi-call batches. One path stays on this
            // thread; two or more fan out.
            const std::vector<ToolResult> parts = run_calls_concurrently(
                indices, [this, &paths](std::size_t i) { return read_one_file(paths[i]); });
            std::string summary;
            std::size_t bytes = 0;
            bool any_error = false;
            bool all_refused = true;
            for (std::size_t i = 0; i < parts.size(); ++i) {
                if (i > 0) {
                    summary += "\n\n";
                }
                summary += "=== ";
                summary += paths[i];
                summary += " ===\n";
                summary += parts[i].summary;
                bytes += parts[i].bytes_read;
                if (!parts[i].ok()) {
                    any_error = true;
                }
                if (parts[i].status != Status::Refused) {
                    all_refused = false;
                }
            }
            summary += overflow_note;

            // BOUND THE AGGREGATE. Each part is already inside max_model_read_bytes, but
            // four of them are not, and the context store's door asserts on the total.
            // Truncating here is what the assert's message asks for -- "the tool layer
            // must bound it" -- and it is the tool layer that knows which files were read
            // and can say so.
            if (ctx_.max_observation_bytes > 0 &&
                summary.size() > ctx_.max_observation_bytes) {
                const std::size_t kept = ctx_.max_observation_bytes;
                // Leave room for the note rather than emitting something the store will
                // then clamp again, which would cut the explanation off mid-sentence.
                const std::string note =
                    "\n\n(read_many truncated: the files above total " +
                    std::to_string(summary.size()) + " bytes, over the " +
                    std::to_string(kept) +
                    "-byte observation budget. Read fewer paths per call, or use "
                    "read_slice for the parts you need.)";
                const std::size_t room = kept > note.size() ? kept - note.size() : 0;
                summary.resize(room);
                summary += note;
            }
            if (all_refused && !parts.empty()) {
                ToolResult r = ToolResult::refused(summary);
                r.bytes_read = bytes;
                return r;
            }
            if (any_error) {
                return measured_read(
                    ToolResult::error(ErrorClass::Transient, true, summary), bytes);
            }
            return measured_read(ToolResult::okay(std::move(summary)), bytes);
        });
    }
    // --- read_slice --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "read_slice";
        d.description = "Read lines [start_line, end_line] of a file (1-based, "
                        "inclusive), each prefixed with its own line number. The slice is "
                        "exact: whole lines, never a partial byte range. The line numbers "
                        "are display only -- never include them in old_text.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("start_line", ParamType::Number, true),
                         param("end_line", ParamType::Number, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            fsx::FileContents f =
                workspace_fs_.read_file_whole(path, ctx_.max_read_bytes);
            if (!f.ok()) {
                if (f.status == fsx::FsStatus::Symlink ||
                    f.status == fsx::FsStatus::InvalidPath) {
                    return refused_path(path, f.error);
                }
                return ToolResult::error(f.status == fsx::FsStatus::NotFound
                                             ? ErrorClass::NotFound
                                             : ErrorClass::Malformed,
                                         false, f.error);
            }
            const long start = std::strtol(get(p, "start_line")->c_str(), nullptr, 10);
            const long end = std::strtol(get(p, "end_line")->c_str(), nullptr, 10);
            if (start < 1 || end < start) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "start_line must be >= 1 and <= end_line");
            }
            std::string outp;
            long line = 1;
            std::size_t at = 0;
            // The line the budget stopped at, so the model is told where to resume rather
            // than left to guess. A 5,000-line slice costs the context as much as a whole
            // file, so the bound applies here too.
            long stopped_at = 0;
            while (at <= f.bytes.size() && line <= end) {
                const std::size_t nl = f.bytes.find('\n', at);
                const std::size_t stop =
                    nl == std::string::npos ? f.bytes.size() : nl + 1;
                if (line >= start) {
                    if (outp.size() >= ctx_.max_model_read_bytes) {
                        stopped_at = line;
                        break;
                    }
                    outp += std::to_string(line);
                    outp += '\t';
                    outp.append(f.bytes, at, stop - at);
                }
                if (nl == std::string::npos) {
                    break;
                }
                at = stop;
                ++line;
            }
            if (line < start) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "file has only " + std::to_string(line) +
                                             " line(s); start_line was " +
                                             std::to_string(start));
            }
            if (stopped_at != 0) {
                if (!outp.empty() && outp.back() != '\n') {
                    outp += '\n';
                }
                outp += "[budget reached at line " + std::to_string(stopped_at) +
                        "; continue with read_slice(path, " +
                        std::to_string(stopped_at) + ", ...)]\n";
            }
            // Whole-file digest: the slice is a view, but the concurrency claim is on the
            // file bytes replace_in_file / write_file will touch.
            const std::string version = fsx::content_sha256_hex(f.bytes);
            note_read_version(resolved.absolute, f.bytes);
            return measured_read(
                ToolResult::okay(with_content_version(std::move(outp), version)),
                f.bytes.size());
        });
    }
    // --- list_dir -----------------------------------------------------------
    {
        ToolDecl d;
        d.name = "list_dir";
        d.description = "List a directory: one entry per line, directories suffixed "
                        "with '/'.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            fsx::DirectoryContents dir = workspace_fs_.list_directory(path);
            if (!dir.ok()) {
                if (dir.status == fsx::FsStatus::Symlink ||
                    dir.status == fsx::FsStatus::InvalidPath) {
                    return refused_path(path, dir.error);
                }
                return ToolResult::error(ErrorClass::NotFound, false, dir.error);
            }
            std::vector<std::string> names;
            for (const fsx::DirectoryEntry& entry : dir.entries) {
                names.push_back(entry.kind == fsx::DirectoryEntryKind::Directory
                                    ? entry.name + "/"
                                    : entry.name);
            }
            std::sort(names.begin(), names.end());
            std::string outp;
            for (const std::string& n : names) {
                outp += n + "\n";
            }
            const std::size_t bytes = outp.size();
            // SAY IT, rather than returning nothing and letting the loop's generic floor
            // say "list_dir succeeded and produced no output" -- which a model reads as
            // "this directory is empty" whether or not that is what happened. It cost a
            // whole run: a shared-offset bug made every listing of the workspace root come
            // back with zero entries, the model was told the project was empty, and it
            // spent fourteen turns re-listing a directory it had been told was not there.
            if (names.empty()) {
                return measured_read(
                    ToolResult::okay("(" + path + " is an empty directory)"), 0);
            }
            return measured_read(ToolResult::okay(std::move(outp)), bytes);
        });
    }
    // --- search -------------------------------------------------------------
    {
        ToolDecl d;
        d.name = "search";
        d.description = "Literal substring search of file CONTENTS across the workspace. "
                        "Returns path:line: text matches, capped; narrow with subdir. To "
                        "find files by NAME or extension, use find_files instead.";
        d.spec.name = d.name;
        d.spec.params = {param("text", ParamType::Text, true),
                         param("subdir", ParamType::Text, false)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string* sub = get(p, "subdir");
            const std::string where =
                (sub != nullptr && !sub->empty()) ? *sub : std::string(".");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(where);
            if (!resolved.ok()) {
                return contained_failure(where, resolved);
            }
            const std::string& needle = *get(p, "text");
            std::string output;
            std::size_t matches = 0;
            (void)walk_regular_files(
                workspace_fs_, resolved.relative, ctx_.max_read_bytes,
                [&](const std::string& path, std::string_view bytes) {
                    if (bytes.find('\0') != std::string_view::npos) {
                        return true;
                    }
                    std::size_t at = 0;
                    std::size_t line_number = 1;
                    while (at <= bytes.size()) {
                        const std::size_t nl = bytes.find('\n', at);
                        const std::size_t stop =
                            nl == std::string_view::npos ? bytes.size() : nl;
                        const std::string_view line = bytes.substr(at, stop - at);
                        if (line.find(needle) != std::string_view::npos) {
                            const std::string hit =
                                path + ":" + std::to_string(line_number) + ":" +
                                std::string(line) + "\n";
                            if (matches >= 200 ||
                                output.size() + hit.size() > ctx_.max_result_bytes) {
                                return false;
                            }
                            output += hit;
                            ++matches;
                        }
                        if (nl == std::string_view::npos) {
                            break;
                        }
                        at = nl + 1;
                        ++line_number;
                    }
                    return true;
                });
            return ToolResult::okay(output.empty() ? "(no matches)" : std::move(output));
        });
    }
    // --- find_files ---------------------------------------------------------
    //
    // The gap `search` kept being asked to fill. `search` reads CONTENTS, so a model
    // looking for the Swift files in a project reaches for search(".swift"), gets no
    // matches -- correctly, since no line of code contains that string -- and concludes
    // the files are not there. Watched twice in one run.
    {
        ToolDecl d;
        d.name = "find_files";
        d.description =
            "Find files by NAME. `pattern` is a glob when it contains * or ? (e.g. "
            "'*.swift'), and a plain substring otherwise (e.g. '.swift' or 'Dashboard'). "
            "A pattern containing '/' is matched against the whole relative path, "
            "otherwise against the file name alone; '*' spans directory separators, so "
            "'Sources/*.swift' also matches files in subdirectories of Sources. Returns "
            "one relative path per line, capped; narrow with subdir.";
        d.spec.name = d.name;
        d.spec.params = {param("pattern", ParamType::Text, true),
                         param("subdir", ParamType::Text, false)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string* sub = get(p, "subdir");
            const std::string where =
                (sub != nullptr && !sub->empty()) ? *sub : std::string(".");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(where);
            if (!resolved.ok()) {
                return contained_failure(where, resolved);
            }
            const std::string& pattern = *get(p, "pattern");
            if (pattern.empty()) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "'pattern' must not be empty");
            }
            const bool is_glob = pattern.find_first_of("*?") != std::string::npos;
            const bool whole_path = pattern.find('/') != std::string::npos;
            std::vector<std::string> hits;
            // Names only -- nothing is opened, so this stays cheap on a tree where
            // `search` would be expensive.
            (void)walk_file_names(
                workspace_fs_, resolved.relative, [&](const std::string& path) {
                    const std::size_t slash = path.rfind('/');
                    const std::string_view subject =
                        whole_path || slash == std::string::npos
                            ? std::string_view(path)
                            : std::string_view(path).substr(slash + 1);
                    const bool hit = is_glob ? glob_match(pattern, subject)
                                             : subject.find(pattern) != std::string_view::npos;
                    if (hit) {
                        hits.push_back(path);
                    }
                    return hits.size() < kFindFilesMax;
                });
            std::sort(hits.begin(), hits.end());
            std::string output;
            bool capped = hits.size() >= kFindFilesMax;
            for (const std::string& h : hits) {
                if (output.size() + h.size() + 1 > ctx_.max_result_bytes) {
                    capped = true;
                    break;
                }
                output += h + "\n";
            }
            // SAY that the list is partial. A silently truncated list is worse than a short
            // one: the model treats "18 files" as the whole set and reasons about what is
            // missing as if it were absent from the workspace.
            if (capped) {
                output += "(truncated -- narrow the pattern or pass subdir)\n";
            }
            const std::size_t bytes = output.size();
            return measured_read(
                ToolResult::okay(output.empty() ? "(no files matched)" : std::move(output)),
                bytes);
        });
    }
    // --- locate_symbol ------------------------------------------------------
    {
        ToolDecl d;
        d.name = "locate_symbol";
        d.description = "Find likely definition sites of a symbol (class, function, "
                        "variable) as path:line: text. Language-agnostic.";
        d.spec.name = d.name;
        d.spec.params = {param("symbol", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            std::string sym;
            for (char c : *get(p, "symbol")) {
                if ((std::isalnum(static_cast<unsigned char>(c)) != 0) || c == '_') {
                    sym += c;
                }
            }
            if (sym.empty()) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "symbol must be an identifier");
            }
            // Editor language features when the client advertised provides_code_intel.
            // Headless keeps the ranked walk below. Not routed through syntax_check and
            // not a sidecar-launched clangd (P2 §10).
            if (code_intel_sink_) {
                CodeIntelQuery q;
                q.op = "workspace_symbols"; // protocol op name, not a tool
                q.query = sym;
                const CodeIntelOutcome o = code_intel_sink_(q);
                if (o.ok) {
                    return ToolResult::okay(o.result_text.empty()
                                                ? ("(no workspace symbols for '" + sym +
                                                   "')")
                                                : o.result_text);
                }
                // Fall through to the walk when the editor has no provider / failed.
            }
            std::string candidates;
            std::size_t candidate_count = 0;
            (void)walk_regular_files(
                workspace_fs_, ".", ctx_.max_read_bytes,
                [&](const std::string& path, std::string_view bytes) {
                    if (bytes.find('\0') != std::string_view::npos) {
                        return true;
                    }
                    std::size_t at = 0;
                    long line_number = 1;
                    while (at <= bytes.size()) {
                        const std::size_t nl = bytes.find('\n', at);
                        const std::size_t stop =
                            nl == std::string_view::npos ? bytes.size() : nl;
                        const std::string_view line = bytes.substr(at, stop - at);
                        if (definition_shaped(line, sym)) {
                            candidates += path + ":" + std::to_string(line_number) +
                                          ":" + std::string(line) + "\n";
                            if (++candidate_count >= 400) {
                                return false;
                            }
                        }
                        if (nl == std::string_view::npos) {
                            break;
                        }
                        at = nl + 1;
                        ++line_number;
                    }
                    return true;
                });
            if (candidates.empty()) {
                return ToolResult::okay("(no definition-shaped lines found for '" + sym +
                                        "'; try search)");
            }
            // Ranked here, not in the pipeline. `head -60` used to hand back whichever
            // lines the filesystem walk reached first, so a symbol with fifty call sites
            // buried its own definition. The grep now casts a wider net and the ordering
            // is decided where it can be asserted.
            std::size_t suppressed = 0;
            const std::vector<SymbolHit> hits =
                rank_symbol_hits(candidates, sym, 40, suppressed);
            std::string out;
            for (const SymbolHit& h : hits) {
                out += h.path + ":" + std::to_string(h.line) + ":" + h.text + "\n";
            }
            if (suppressed > 0) {
                out += "[" + std::to_string(suppressed) +
                       " lower-ranked match(es) not shown]\n";
            }
            return ToolResult::okay(std::move(out));
        });
    }
    // --- write_file ---------------------------------------------------------
    {
        ToolDecl d;
        d.name = "write_file";
        // The last sentence is a FACT ABOUT THE GATE, not encouragement. "Prefer
        // replace_in_file" was already here, and already in the persona, and a real run
        // still made 11 whole-file writes against 0 partial edits -- so the description was
        // saying which tool is nicer without saying what the other one costs. Overwriting
        // content this run did not write raises an approval card whatever the auto-approve
        // setting says (it destroys data), and a model that knows the price can weigh it.
        d.description = "Create or fully replace a file with the given content. "
                        "Atomic: a crash leaves the old file or the new one, never a "
                        "prefix. Overwriting an existing file requires a prior "
                        "read_file/read_slice in this run (content version); a create "
                        "requires the path to still be absent. For a partial change use "
                        "replace_in_file: it is not just tidier, it is cheaper -- "
                        "replacing the whole of a file this run did not write destroys "
                        "the existing content and stops for a human decision, while a "
                        "scoped edit does not.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("content", ParamType::Text, true),
                         param("expected_version", ParamType::Text, false)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            fsx::WritePrecondition pre;
            const std::string_view content = *get(p, "content");
            // Presence, not would_overwrite_existing: an empty file still exists, so a
            // rewrite must carry a version rather than expected_absent (which would
            // conflict against the empty inode).
            const fsx::FileContents cur =
                workspace_fs_.read_file_whole(path, ctx_.max_read_bytes);
            if (cur.status == fsx::FsStatus::Symlink ||
                cur.status == fsx::FsStatus::InvalidPath) {
                return refused_path(path, cur.error);
            }
            if (cur.status == fsx::FsStatus::IsDirectory) {
                return ToolResult::error(ErrorClass::Malformed, false, cur.error);
            }
            const bool exists =
                cur.ok() || cur.status == fsx::FsStatus::TooLarge;
            if (exists) {
                if (cur.ok() && cur.bytes == content) {
                    return ToolResult::no_change(
                        path + " already contained exactly these " +
                        std::to_string(content.size()) +
                        " bytes; nothing was written and the file is unchanged.");
                }
                pre.expected_version = resolve_expected_version(resolved.absolute, p);
                if (pre.expected_version.empty()) {
                    return need_read_before_write(path);
                }
            } else if (cur.status == fsx::FsStatus::NotFound) {
                pre.expected_absent = true;
            } else {
                return ToolResult::error(ErrorClass::Malformed, false, cur.error);
            }
            const CommitOutcome w = commit_write(resolved, content, pre);
            if (!w.ok()) {
                return write_failure(w.write);
            }
            // NAMES THE NON-EVENT. "wrote 5327 bytes" for a write that did not happen is
            // the sentence that let a model believe it had just fixed something; the model
            // reads this result and nothing else about what its edit did. The fact and
            // nothing more -- what to do about an edit that already exists is the model's
            // call, and an instruction bolted onto every no-op is prompt noise.
            if (w.unchanged) {
                return ToolResult::no_change(
                    *get(p, "path") + " already contained exactly these " +
                    std::to_string(get(p, "content")->size()) +
                    " bytes; nothing was written and the file is unchanged.");
            }
            return measured_edit(
                ToolResult::okay("wrote " +
                                 std::to_string(get(p, "content")->size()) +
                                 " bytes to " + *get(p, "path")),
                get(p, "content")->size());
        });
    }
    // --- replace_in_file ----------------------------------------------------
    {
        ToolDecl d;
        d.name = "replace_in_file";
        d.description = "Replace exactly one occurrence of old_text with new_text. "
                        "Whitespace-tolerant matching; refuses when the match is "
                        "ambiguous (listing candidate sites) or absent, and the file "
                        "is left untouched in both cases. On success it returns the "
                        "applied hunk with the file's current line numbers, so you can "
                        "see exactly what changed and where WITHOUT reading the file "
                        "back — if the hunk shows the edit landed somewhere you did not "
                        "intend, send a corrected replace_in_file rather than re-reading.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("old_text", ParamType::Text, true),
                         param("new_text", ParamType::Text, true),
                         param("expected_version", ParamType::Text, false)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            fsx::FileContents f =
                workspace_fs_.read_file_whole(path, ctx_.max_read_bytes);
            if (!f.ok()) {
                if (f.status == fsx::FsStatus::Symlink ||
                    f.status == fsx::FsStatus::InvalidPath) {
                    return refused_path(path, f.error);
                }
                return ToolResult::error(f.status == fsx::FsStatus::NotFound
                                             ? ErrorClass::NotFound
                                             : ErrorClass::Malformed,
                                         false, f.error);
            }
            // This tool's own read is the preimage. Prefer an explicit expected_version
            // when the model supplies one; otherwise bind to the bytes just observed.
            note_read_version(resolved.absolute, f.bytes);
            fsx::WritePrecondition pre;
            pre.expected_version = resolve_expected_version(resolved.absolute, p);
            if (pre.expected_version.empty()) {
                pre.expected_version = fsx::content_sha256_hex(f.bytes);
            }
            // Accept what read_file and read_slice displayed. See strip_line_numbers.
            //
            // ASYMMETRIC ON PURPOSE. Stripping old_text is safe in the worst case: a
            // wrong strip means NoMatch and the file is left untouched, which is graft's
            // contract. Stripping new_text is NOT -- a TSV whose first column is a row
            // number is exactly the shape this matches, and a wrong strip there silently
            // writes corrupted content. So new_text is only stripped when old_text ALSO
            // carried numbers, i.e. when the model is demonstrably echoing back a view it
            // was shown rather than authoring numbered data.
            const std::string& raw_old = *get(p, "old_text");
            const std::string old_text = strip_line_numbers(raw_old);
            const bool echoing_a_view = old_text != raw_old;
            const std::string new_text = echoing_a_view ? strip_line_numbers(*get(p, "new_text"))
                                                        : *get(p, "new_text");
            const graft::Result g = graft::apply(f.bytes, old_text, new_text);
            if (g.status == graft::Status::NoMatch) {
                std::string snippet = "\n\n[Current ground-truth lines from " + *get(p, "path") + "]:\n```\n";
                std::string target_line;
                std::size_t nl = 0;
                while (nl < old_text.size()) {
                    std::size_t next_nl = old_text.find('\n', nl);
                    std::string line = old_text.substr(nl, next_nl == std::string::npos ? std::string::npos : next_nl - nl);
                    while (!line.empty() && (line.front() == ' ' || line.front() == '\t')) line.erase(line.begin());
                    while (!line.empty() && (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) line.pop_back();
                    if (!line.empty()) {
                        target_line = line;
                        break;
                    }
                    if (next_nl == std::string::npos) break;
                    nl = next_nl + 1;
                }

                long approx_line = 1;
                if (!target_line.empty()) {
                    std::size_t found = f.bytes.find(target_line);
                    if (found != std::string::npos) {
                        approx_line = 1;
                        for (std::size_t i = 0; i < found; ++i) {
                            if (f.bytes[i] == '\n') approx_line++;
                        }
                    }
                }

                long start_line = std::max(1L, approx_line - 5);
                long end_line = start_line + 40;
                long curr_line = 1;
                std::size_t at = 0;
                while (at <= f.bytes.size() && curr_line <= end_line) {
                    std::size_t next_nl = f.bytes.find('\n', at);
                    std::size_t stop = (next_nl == std::string::npos) ? f.bytes.size() : next_nl + 1;
                    if (curr_line >= start_line) {
                        snippet += std::to_string(curr_line) + "\t";
                        snippet.append(f.bytes, at, stop - at);
                    }
                    if (next_nl == std::string::npos) break;
                    at = stop;
                    curr_line++;
                }
                snippet += "```";
                // Nearest regions are DIAGNOSTICS ONLY — similarity never authorizes a write.
                snippet += edit_diagnostics::format_nearest(*get(p, "path"), f.bytes, old_text);
                return ToolResult::error(ErrorClass::NotFound, false,
                                         "old_text not found in " + *get(p, "path") +
                                             "; check the current lines below and update old_text:" + snippet);
            }
            if (g.status == graft::Status::Ambiguous) {
                std::string sites;
                for (const graft::Match& m : g.matches) {
                    sites += (sites.empty() ? "" : ", ") + std::to_string(m.line);
                }
                return ToolResult::error(ErrorClass::Conflict, false,
                                         "old_text matches more than one site (lines " +
                                             sites + "); include more context");
            }
            const CommitOutcome w = commit_write(resolved, g.result, pre);
            if (!w.ok()) {
                return write_failure(w.write);
            }
            // graft matched, so old_text was there -- and the file still came out
            // identical, which means new_text equals old_text. The model has sent an edit
            // that says "replace this with itself", and reporting it as a replacement is
            // how a run spends four turns re-applying it.
            if (w.unchanged) {
                return ToolResult::no_change(
                    "old_text matched in " + *get(p, "path") +
                    " but new_text is identical to it, so the file is unchanged.");
            }
            // THE APPLIED HUNK, not a receipt. "replaced one occurrence in <path>" told the
            // model that something happened and nothing about what, which left re-reading
            // the file as its only way to find out -- see edit_diagnostics::applied_hunk
            // for the run that spent six reads and half its budget doing exactly that.
            return measured_edit(
                ToolResult::okay(edit_diagnostics::format_applied(*get(p, "path"), f.bytes,
                                                                  g.result)),
                contiguous_edit_bytes(f.bytes, g.result));
        });
    }
    // --- apply_patch --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "apply_patch";
        d.description =
            "Apply an exact structured patch (*** Begin Patch / *** End Patch) with "
            "Add/Update/Delete File sections. Matching is byte-exact context/preimage "
            "only — never fuzzy. Each existing file binds to the content version from "
            "the last read_file/read_slice (or this tool's own preimage read); all hunks "
            "for a file apply or none do. Prefer this for multi-hunk edits; keep "
            "replace_in_file for a single old_text→new_text swap. Pass the patch as raw "
            "multiline text in `patch`. On success it returns the applied hunk for each "
            "UPDATED file with that file's current line numbers, so you can see what "
            "changed without reading the files back.";
        d.spec.name = d.name;
        d.spec.params = {param("patch", ParamType::Text, true)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& patch = *get(p, "patch");
            const apply_patch::Result parsed = apply_patch::parse(patch);
            if (parsed.status != apply_patch::Status::Applied) {
                return ToolResult::error(ErrorClass::Malformed, false,
                                         "apply_patch parse error: " + parsed.failure.reason);
            }

            // One coherent snapshot per path: hunks and content versions share these bytes.
            std::map<std::string, std::string> snapshot;
            std::map<std::string, fsx::ContainedPath> resolved_by_path;
            for (const apply_patch::FileOp& op : parsed.ops) {
                const fsx::ContainedPath resolved = workspace_fs_.contained_path(op.path);
                if (!resolved.ok()) {
                    return contained_failure(op.path, resolved);
                }
                resolved_by_path.emplace(op.path, resolved);
                const fsx::FileContents f =
                    workspace_fs_.read_file_whole(op.path, ctx_.max_read_bytes);
                if (op.kind == apply_patch::FileOpKind::Add) {
                    if (f.ok() || f.status == fsx::FsStatus::TooLarge) {
                        // Presence makes Add a conflict; seed snapshot so apply_to sees it.
                        snapshot.emplace(op.path, f.ok() ? f.bytes : std::string());
                    } else if (f.status == fsx::FsStatus::Symlink ||
                               f.status == fsx::FsStatus::InvalidPath) {
                        return refused_path(op.path, f.error);
                    } else if (f.status != fsx::FsStatus::NotFound) {
                        return ToolResult::error(ErrorClass::Malformed, false, f.error);
                    }
                    continue;
                }
                if (!f.ok()) {
                    if (f.status == fsx::FsStatus::Symlink ||
                        f.status == fsx::FsStatus::InvalidPath) {
                        return refused_path(op.path, f.error);
                    }
                    if (f.status == fsx::FsStatus::NotFound) {
                        return ToolResult::error(ErrorClass::NotFound, false,
                                                 op.path + ": not found for patch apply");
                    }
                    return ToolResult::error(ErrorClass::Malformed, false, f.error);
                }
                note_read_version(resolved.absolute, f.bytes);
                snapshot.emplace(op.path, f.bytes);
            }

            auto read_opt = [&snapshot](const std::string& rel) -> std::optional<std::string> {
                const auto it = snapshot.find(rel);
                if (it == snapshot.end()) {
                    return std::nullopt;
                }
                return it->second;
            };
            const apply_patch::Result applied = apply_patch::apply_to(parsed.ops, read_opt);
            if (applied.status != apply_patch::Status::Applied) {
                std::string msg = "apply_patch refused on " + applied.failure.path + ": " +
                                  applied.failure.reason;
                if (!applied.failure.nearby.empty()) {
                    msg += "\n\n[Current nearby lines]:\n```\n";
                    msg += applied.failure.nearby;
                    msg += "```";
                }
                const ErrorClass ec =
                    applied.status == apply_patch::Status::Ambiguous ? ErrorClass::Conflict
                    : applied.status == apply_patch::Status::Conflict ? ErrorClass::Conflict
                    : applied.status == apply_patch::Status::ParseError
                          ? ErrorClass::Malformed
                          : ErrorClass::NotFound;
                return ToolResult::error(ec, false, std::move(msg));
            }

            std::size_t bytes_changed = 0;
            std::vector<std::string> ok_paths;
            // Per-file hunks, for the same reason replace_in_file carries one: a list of
            // paths says a patch landed, not what it did to them. Bounded twice over --
            // each hunk by applied_hunk, and the number of hunks by kPatchHunkFiles, since
            // a patch may touch far more files than a model needs echoed back.
            static constexpr std::size_t kPatchHunkFiles = 4;
            std::string hunks;
            std::size_t hunks_shown = 0;
            std::size_t hunks_suppressed = 0;
            for (const apply_patch::FileChange& ch : applied.changes) {
                const fsx::ContainedPath& resolved = resolved_by_path.at(ch.path);
                fsx::WritePrecondition pre;
                if (ch.delete_file || ch.kind == apply_patch::FileOpKind::Update) {
                    pre.expected_version = resolve_expected_version(resolved.absolute, p);
                    if (pre.expected_version.empty()) {
                        pre.expected_version = fsx::content_sha256_hex(snapshot.at(ch.path));
                    }
                } else {
                    pre.expected_absent = true;
                }
                if (ch.delete_file) {
                    const fsx::RemoveResult removed =
                        workspace_fs_.remove_file(ch.path, pre.expected_version);
                    if (removed.status == fsx::FsStatus::Conflict) {
                        return ToolResult::error(ErrorClass::Conflict, false, removed.error);
                    }
                    if (!removed.ok()) {
                        if (removed.status == fsx::FsStatus::Symlink ||
                            removed.status == fsx::FsStatus::InvalidPath) {
                            return refused_path(ch.path, removed.error);
                        }
                        return ToolResult::error(ErrorClass::Transient, true, removed.error);
                    }
                    read_versions_.erase(resolved.absolute);
                    bytes_changed += removed.removed_size;
                    ok_paths.push_back(ch.path);
                    continue;
                }
                const CommitOutcome w = commit_write(resolved, ch.new_content, pre);
                if (!w.ok()) {
                    return write_failure(w.write);
                }
                if (!w.unchanged) {
                    const auto before = snapshot.find(ch.path);
                    bytes_changed += before == snapshot.end()
                                         ? ch.new_content.size()
                                         : contiguous_edit_bytes(before->second, ch.new_content);
                    // A file the patch CREATED needs no hunk: its whole content is the
                    // model's own new_content, already in the prompt as the call's
                    // argument. Echoing it back is the prompt bloat this change exists to
                    // reduce. An UPDATED file is the case the model cannot see.
                    if (before != snapshot.end()) {
                        if (hunks_shown < kPatchHunkFiles) {
                            hunks += "\n";
                            hunks += edit_diagnostics::format_applied(ch.path, before->second,
                                                                      ch.new_content);
                            ++hunks_shown;
                        } else {
                            ++hunks_suppressed;
                        }
                    }
                }
                ok_paths.push_back(ch.path);
            }

            if (bytes_changed == 0 && !ok_paths.empty()) {
                return ToolResult::no_change(
                    "apply_patch matched but produced no byte changes in " +
                    std::to_string(ok_paths.size()) + " file(s)");
            }
            std::string summary = "apply_patch applied to ";
            summary += std::to_string(ok_paths.size());
            summary += " file(s): ";
            for (std::size_t i = 0; i < ok_paths.size(); ++i) {
                if (i) {
                    summary += ", ";
                }
                summary += ok_paths[i];
            }
            summary += hunks;
            if (hunks_suppressed > 0) {
                summary += "\n(" + std::to_string(hunks_suppressed) +
                           " further updated file(s) applied; hunks not shown)";
            }
            ToolResult r = measured_edit(ToolResult::okay(std::move(summary)), bytes_changed);
            r.structured_json = "[";
            for (std::size_t i = 0; i < ok_paths.size(); ++i) {
                if (i) {
                    r.structured_json += ",";
                }
                r.structured_json += "\"";
                for (char c : ok_paths[i]) {
                    if (c == '"' || c == '\\') {
                        r.structured_json += '\\';
                    }
                    r.structured_json += c;
                }
                r.structured_json += "\"";
            }
            r.structured_json += "]";
            return r;
        });
    }
    // --- append_file --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "append_file";
        d.description = "Append content to the end of a file, creating it if absent. "
                        "An existing file's current bytes become the content version "
                        "checked at apply time.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("content", ParamType::Text, true),
                         param("expected_version", ParamType::Text, false)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            fsx::FileContents f =
                workspace_fs_.read_file_whole(path, ctx_.max_read_bytes);
            std::string content =
                f.ok() ? f.bytes + *get(p, "content") : *get(p, "content");
            if (f.status == fsx::FsStatus::Symlink ||
                f.status == fsx::FsStatus::InvalidPath) {
                return refused_path(path, f.error);
            }
            if (!f.ok() && f.status != fsx::FsStatus::NotFound) {
                return ToolResult::error(ErrorClass::Malformed, false, f.error);
            }
            fsx::WritePrecondition pre;
            if (f.ok()) {
                note_read_version(resolved.absolute, f.bytes);
                pre.expected_version = resolve_expected_version(resolved.absolute, p);
                if (pre.expected_version.empty()) {
                    pre.expected_version = fsx::content_sha256_hex(f.bytes);
                }
            } else {
                pre.expected_absent = true;
            }
            const CommitOutcome w = commit_write(resolved, content, pre);
            if (!w.ok()) {
                return write_failure(w.write);
            }
            // Appending nothing. Reachable only with empty content, which is the shape a
            // degenerating model produces (see commit_write's own MEASURED note about a
            // run that fell into sending empty writes), so it is worth naming rather than
            // reporting as "appended 0 bytes".
            if (w.unchanged) {
                return ToolResult::no_change(*get(p, "path") +
                                             " is unchanged: there was nothing to append.");
            }
            return measured_edit(
                ToolResult::okay("appended " +
                                 std::to_string(get(p, "content")->size()) +
                                 " bytes to " + *get(p, "path")),
                get(p, "content")->size());
        });
    }
    // --- delete_file --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "delete_file";
        d.description = "Delete one file (not a directory). Requires a prior "
                        "read_file/read_slice in this run so the delete carries the "
                        "content version observed then; refuses if the file changed.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("expected_version", ParamType::Text, false)};
        d.mutates_workspace = true;
        // Nothing in the workspace can undo this, so it always asks (S7.2).
        d.irreversible = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string& path = *get(p, "path");
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(path);
            if (!resolved.ok()) {
                return contained_failure(path, resolved);
            }
            const std::string expected = resolve_expected_version(resolved.absolute, p);
            if (expected.empty()) {
                // Path-kind refusals (symlink, directory) must still surface as Refused
                // rather than "read first" -- the model cannot fix a symlink by reading it.
                const fsx::FileContents probe =
                    workspace_fs_.read_file_whole(path, 1);
                if (probe.status == fsx::FsStatus::Symlink ||
                    probe.status == fsx::FsStatus::InvalidPath) {
                    return refused_path(path, probe.error);
                }
                if (probe.status == fsx::FsStatus::NotFound) {
                    return ToolResult::error(ErrorClass::NotFound, false,
                                             path + ": not found");
                }
                if (probe.status == fsx::FsStatus::IsDirectory) {
                    return ToolResult::refused("'" + path +
                                               "' is a directory; this tool deletes files "
                                               "only");
                }
                return need_read_before_write(path);
            }
            const fsx::RemoveResult removed =
                workspace_fs_.remove_file(path, expected);
            if (removed.status == fsx::FsStatus::NotFound) {
                return ToolResult::error(ErrorClass::NotFound, false,
                                         path + ": not found");
            }
            if (removed.status == fsx::FsStatus::IsDirectory) {
                return ToolResult::refused("'" + path +
                                           "' is a directory; this tool deletes files "
                                           "only");
            }
            if (removed.status == fsx::FsStatus::Symlink ||
                removed.status == fsx::FsStatus::InvalidPath) {
                return refused_path(path, removed.error);
            }
            if (removed.status == fsx::FsStatus::Conflict) {
                return ToolResult::error(ErrorClass::Conflict, false, removed.error);
            }
            if (!removed.ok()) {
                return ToolResult::error(ErrorClass::Transient, true,
                                         removed.error);
            }
            read_versions_.erase(resolved.absolute);
            return measured_edit(
                ToolResult::okay("deleted " + path), removed.removed_size);
        });
    }
    // --- shell --------------------------------------------------------------
    {
        ToolDecl d;
        d.name = "shell";
        d.description = "Run a shell command inside the workspace sandbox: writes are "
                        "jailed to the workspace, the network is unreachable, and "
                        "runaway commands are killed at the wall clock. Output is "
                        "compacted; the full log is spooled to an artifact.";
        d.spec.name = d.name;
        d.spec.params = {param("command", ParamType::Text, true)};
        d.executes_commands = true;
        d.needs_execution = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            const std::string& cmd = *get(p, "command");
            const ExecutionGrant grant = grant_execution(tier_for(approved_tier));
            const ExecLimits limits{ctx_.shell_wall_clock_seconds,
                                    ctx_.shell_wall_clock_seconds,
                                    8LL << 30,
                                    1024,
                                    256,
                                    4U << 20};
            ExecOutcome o =
                run_sandboxed(grant, cmd, ctx_.root, ctx_.root, limits, cancel_token_);

            ToolResult r;
            r.status = o.status;
            if (o.status == Status::Cancelled) {
                r.summary = o.output.empty() ? "cancelled" : o.output;
                return r;
            }
            if (o.status == Status::Refused) {
                r.error_class = ErrorClass::Policy;
                r.summary = o.output;
                return r;
            }
            // Compact through the cookoff-merged triage engine (S6.2): anchors first,
            // own-code beats system headers, never head-truncate.
            r.summary = log_triage::compact(o.output, ctx_.max_result_bytes);
            r.exit_code = o.exit_code;
            // Structured triage from the FULL log (counts/paths may sit outside the compact
            // budget). Attached as a short annotation — never replaces the compacted body.
            const log_triage::StructuredTriage triage = log_triage::analyze(o.output);
            const std::string triage_note = log_triage::format_annotation(triage);
            if (o.wall_clock_killed) {
                r.error_class = ErrorClass::Transient;
                r.retryable = false;
                r.summary = "[killed at the " +
                            std::to_string(ctx_.shell_wall_clock_seconds) +
                            "s wall clock]\n" + r.summary;
            } else if (o.signalled) {
                r.error_class = ErrorClass::Transient;
                r.summary = "[terminated by signal " + std::to_string(o.signal) + "]\n" +
                            r.summary;
            } else if (o.exit_code != 0) {
                r.error_class = ErrorClass::Transient;
                r.retryable = true;
                r.summary = "[exit " + std::to_string(o.exit_code) + "]\n" + r.summary;
            }
            // Spool the full output when compaction dropped anything (S14).
            if (o.output.size() > r.summary.size() && !ctx_.spool_dir.empty()) {
                const std::string requested_spool =
                    ctx_.spool_dir + "/shell_" +
                    std::to_string(reinterpret_cast<std::uintptr_t>(&o) ^ o.output.size()) +
                    ".log";
                const fsx::ContainedPath spool =
                    workspace_fs_.contained_path(requested_spool);
                if (spool.ok() &&
                    workspace_fs_.write_file_atomic(spool.relative, o.output).ok()) {
                    r.artifacts.push_back(spool.absolute);
                }
            }
            if (!triage_note.empty()) {
                r.summary += triage_note;
            }
            // LAST, and read off the FULL output rather than the compacted summary: the
            // signature is one line inside a manifest dump that log_triage may well have
            // elided. After the spool decision, so appending it cannot make the summary
            // look long enough to skip spooling the output it is explaining.
            r.summary += nested_sandbox_note(o.output, cmd);
            // What actually ran, when it was not what was asked for. Said even on
            // SUCCESS: a command that quietly ran differently is how a model comes to
            // believe something about its environment that is not true, and the next
            // thing it does is act on that belief.
            if (!o.rewritten_command.empty()) {
                r.summary += "\n[sandbox] Ran `" + o.rewritten_command +
                             "`. macOS will not let SwiftPM start its own sandbox inside "
                             "this one, so the harness added `--disable-sandbox`; the "
                             "workspace jail is unchanged and still in force.";
            }
            return r;
        });
    }
    // --- job_status ---------------------------------------------------------
    {
        ToolDecl d;
        d.name = "job_status";
        d.description = "Report the sandbox and limits a shell command would run "
                        "under, without running anything. Use it to understand why a "
                        "command was refused.";
        d.spec.name = d.name;
        d.spec.params = {param("command", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            const RiskHint hint = classify_command(*get(p, "command"), ctx_.root, ctx_.root);
            std::string s = "tier=" + std::to_string(approved_tier) +
                            " wall_clock=" + std::to_string(ctx_.shell_wall_clock_seconds) +
                            "s network=denied writes=workspace-only\n";
            s += "advisory capabilities (do not gate on these -- the sandbox is the "
                 "authority):";
            const auto& c = hint.caps;
            if (c.writes_outside_workspace) { s += " write_out"; }
            if (c.reads_outside_workspace) { s += " read_out"; }
            if (c.destroys_data) { s += " destroy"; }
            if (c.rewrites_vcs_history) { s += " vcs"; }
            if (c.network_access) { s += " net"; }
            if (c.spawns_unbounded_process) { s += " unbounded"; }
            if (c.signals_foreign_process) { s += " signal"; }
            if (c.escalates_privileges) { s += " priv"; }
            s += hint.status == blast_radius::ParseStatus::Parsed
                     ? "\nstatus=parsed"
                     : (hint.status == blast_radius::ParseStatus::PartiallyParsed
                            ? "\nstatus=partial (effects depend on bytes not in the string)"
                            : "\nstatus=unparseable");
            return ToolResult::okay(std::move(s));
        });
    }
    // --- git_status / git_diff / git_log ------------------------------------
    //
    // An agent that cannot see its own diff has no review surface: it edits files, and
    // neither it nor the human can tell what changed without leaving the loop. These are
    // read-only by construction -- the command is BUILT here from a fixed shape, never
    // taken from the model -- so they carry no approval weight and cannot be turned into
    // `git push` or `git reset --hard` by a crafted argument.
    {
        ToolDecl d;
        d.name = "git_status";
        d.description = "Show which files in the workspace are modified, staged or "
                        "untracked. Read-only.";
        d.spec.name = d.name;
        d.spec.params = {};
        d.needs_execution = true; // run_git shells out; at T0 it can only refuse
        declare(d, [this](const std::vector<ToolParamValue>&, int approved_tier) {
            return run_git("status --short --branch", approved_tier);
        });
    }
    {
        ToolDecl d;
        d.name = "git_diff";
        d.description = "Show the unstaged diff for the workspace, or for one path when "
                        "`path` is given. Read-only. This is how you review your own "
                        "edits before claiming they are correct.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, false)};
        d.needs_execution = true; // run_git shells out; at T0 it can only refuse
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            const std::string* path = get(p, "path");
            if (path == nullptr || path->empty()) {
                return run_git("diff --stat -p", approved_tier);
            }
            const fsx::ContainedPath resolved = workspace_fs_.contained_path(*path);
            if (!resolved.ok()) {
                return contained_failure(*path, resolved);
            }
            return run_git("diff --stat -p -- " + shell_quote(*path), approved_tier);
        });
    }
    {
        ToolDecl d;
        d.name = "git_log";
        d.description = "Show recent commit subjects, most recent first. Read-only. Use "
                        "it to match the repository's conventions before writing code.";
        d.spec.name = d.name;
        d.spec.params = {param("count", ParamType::Text, false)};
        d.needs_execution = true; // run_git shells out; at T0 it can only refuse
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            const std::string* raw = get(p, "count");
            int n = 15;
            if (raw != nullptr && !raw->empty()) {
                n = std::atoi(raw->c_str());
            }
            if (n <= 0 || n > 200) {
                n = 15;
            }
            return run_git("log --oneline -n " + std::to_string(n), approved_tier);
        });
    }
    // --- plan ---------------------------------------------------------------
    //
    // Declared here so it reaches the guidance and the grammar, but EXECUTED by the loop
    // (Agent::step) -- the checklist lives in the context store, which the registry has
    // no business reaching into. The handler is never called; it exists so that the
    // declaration and the execution path cannot drift apart silently.
    {
        ToolDecl d;
        d.name = "remember";
        d.description =
            "Save one durable fact about this project so a later session starts already "
            "knowing it: how it builds, where a subsystem lives, a constraint you had to "
            "discover the hard way. For things that stay true after this mission ends -- "
            "not the current task's progress, which the checklist already carries. Give "
            "`key` a short stable name for what the note is ABOUT (\"build\", "
            "\"test-layout\") and a later note under the same key REPLACES this one "
            "instead of piling up beside it -- use it whenever you are correcting or "
            "refining something you or an earlier session already wrote down. Leave it "
            "empty only for a standalone observation that supersedes nothing.";
        d.spec.name = d.name;
        d.spec.params = {param("fact", ParamType::Text, true),
                         param("key", ParamType::Text, false)};
        // It writes a file inside the workspace, so it is a mutation: it raises a card
        // when the operator has asked to approve writes, and it stays off the parallel
        // read-only dispatch path. Not irreversible -- the note is additive and the file
        // is the agent's own.
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string* k = get(p, "key");
            return remember_fact(*get(p, "fact"), k == nullptr ? "" : *k);
        });
    }
    // --- plan ---------------------------------------------------------------
    {
        ToolDecl d;
        d.name = "plan";
        // A progress display, not a gate. Nothing in the harness reads the list to
        // decide anything; it feeds the operator's sidebar panel and the run report.
        //
        // THIS DESCRIPTION TOLD THE MODEL NOT TO USE THE TOOL, and it obeyed: across
        // fifteen logged runs, `plan` was never called once, so the sidebar's checklist
        // panel stayed empty and the operator watched an opaque run. It read "Optional:
        // ... skip it for small tasks" -- the only description in this registry that
        // opened by excusing itself and closed by granting permission to decline -- and it
        // put the restate mechanic in the frame of a cost ("ticking one item means sending
        // them all") rather than of how the tool works.
        //
        // It got that way in the eighth-pass loop rewrite, which deleted the grammar mask
        // that had made `plan` the only callable tool until the run had a checklist, and
        // softened this text in the same commit. The mask went for a good reason and is
        // not coming back here; the description going with it was collateral. Note what
        // the mask's own comment recorded, because it bounds what this text can achieve:
        // with `plan` merely available and a description saying to call it first, the
        // model did not. What is different now is the mode brief, which did not exist
        // then -- see kWorkingDiscipline in src/loop/turn.cpp, where the instruction to
        // open a multi-step task with a checklist actually lives. This says what the tool
        // IS and how it behaves; the brief says when to reach for it.
        d.description =
            "Your task list for this mission, shown to the operator as a live checklist -- "
            "it is the only thing that tells them what you are doing and how far along you "
            "are. Call it EARLY on any task with more than one step, before you start the "
            "work, listing every item you expect to do. Call it again as each one lands: "
            "restating replaces the whole list, so send all the items every time, '[x] ' "
            "on the ones that are done and '[ ] ' on the rest. One item per line, separated "
            "by REAL newlines -- one string with '\\n' sequences typed into it is one item, "
            "not a list, and is rejected.";
        d.spec.name = d.name;
        d.spec.params = {param("items", ParamType::Text, true)};
        // Not offered to a mode that only reads, asks and hands over -- see
        // ToolDecl::working_run_only.
        d.working_run_only = true;
        declare(d, [](const std::vector<ToolParamValue>&, int) {
            return ToolResult::error(ErrorClass::Malformed, false,
                                     "internal: 'plan' must be handled by the loop");
        });
    }
    // --- ask_user -----------------------------------------------------------
    //
    // Declared here for the grammar and the guidance, EXECUTED by the loop, exactly as
    // `plan` is -- what it does is end the turn loop, and the registry has no way to do
    // that. The explicit tool matters in a conversational mode because it tells the
    // surface a QUESTION is waiting rather than a statement -- the two render
    // differently, and only one of them asks the operator to reply.
    {
        ToolDecl d;
        d.name = "ask_user";
        d.description =
            "Put one question to the human and stop, so they can answer. Use it when the "
            "answer would genuinely change the design and you cannot settle it by reading "
            "the code -- not to confirm something you already believe, and not to ask "
            "permission to continue. Ask ONE question, the most load-bearing one, and say "
            "what each answer would change. The run ends here; their reply continues this "
            "same conversation with everything you have read still in context.";
        d.spec.name = d.name;
        d.spec.params = {param("question", ParamType::Text, true),
                         param("options", ParamType::Text, false)};
        // NOT conversational_only. Needing an answer is not a property of plan mode; it is
        // a property of a question the code cannot settle, and an agent run hits those too.
        // While this was Plan-only, a working run that needed a decision had exactly one way
        // to raise it -- answer in text -- and in a working mode a text-only turn IS the
        // ending, so "I need to know X before I continue" terminated the run instead of
        // asking anything.
        declare(d, [](const std::vector<ToolParamValue>&, int) {
            return ToolResult::error(ErrorClass::Malformed, false,
                                     "internal: 'ask_user' must be handled by the loop");
        });
    }
    // --- ask_question --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "ask_question";
        d.description =
            "Ask the human a question with 2 to 4 selectable options, which they answer by "
            "clicking a card. Pass `question` (text) and `options` (newline-separated "
            "choices, one per line). Prefer this over `ask_user` whenever the answers are a "
            "short closed set -- which approach, which library, which of two designs -- "
            "because clicking one is faster for them than typing it. Ask when the answer "
            "would genuinely change what you build and reading more code would not settle "
            "it; do not ask permission to continue. The run stops here and their choice "
            "continues this same conversation with everything you have read still in "
            "context.";
        d.spec.name = d.name;
        d.spec.params = {param("question", ParamType::Text, true),
                         param("options", ParamType::Text, true)};
        // Available in every mode -- see ask_user above for why.
        declare(d, [](const std::vector<ToolParamValue>&, int) {
            return ToolResult::error(ErrorClass::Malformed, false,
                                     "internal: 'ask_question' must be handled by the loop");
        });
    }
    // --- exit_plan_mode -----------------------------------------------------
    {
        ToolDecl d;
        d.name = "exit_plan_mode";
        d.description =
            "Present the finished plan and ask to start implementing it. Call this only "
            "when the approach is settled and you would not change it based on anything "
            "else you could read. Give `plan` the whole plan as markdown: what is being "
            "changed and why, which files, and how the result will be verified -- it "
            "becomes the mission of the run that implements it, so anything you leave out "
            "is something that run will not know. The human approves or keeps talking; "
            "you do not switch modes yourself.";
        d.spec.name = d.name;
        d.spec.params = {param("plan", ParamType::Text, true)};
        d.conversational_only = true;
        declare(d, [](const std::vector<ToolParamValue>&, int) {
            return ToolResult::error(
                ErrorClass::Malformed, false,
                "internal: 'exit_plan_mode' must be handled by the loop");
        });
    }
    // --- finish ---------------------------------------------------------------
    //
    // The ending a working run had no way to ask for. Every other termination is something
    // that HAPPENS to a run -- a budget, a stall, a question -- and a run that simply
    // finished had to be inferred from several turns of it saying so. Measured before this
    // existed: the work landed on turn 2 and the run ended on turn 6, because the inert
    // counter needs three nudges plus a fourth turn. Worse, a nudge that provoked any
    // tool call which learned something RESET that counter, so a finished run could
    // ping-pong between narrating and re-verifying until the turn budget ran out -- one
    // did, at 12 of 12 turns with `max_turns` on a mission it had completed.
    //
    // It reports; it does not adjudicate. Calling this ends the run as `ended`, which is
    // the same reason a run that narrated its way out already gets, and `completed` is
    // still decided afterwards against the model's own checklist and the operator check.
    // The model saying it is done does not make the report say the work is right.
    {
        ToolDecl d;
        d.name = "finish";
        d.description =
            "Report that the mission is done and end the run. Call this the moment you have "
            "nothing left to do -- not a turn later, and instead of writing a message that "
            "says you are finished, which does not end anything. Give `summary` the "
            "handback: what you changed, and how you know it works. Do not call it while "
            "any part of the work is unstarted, unverified, or blocked -- to act, call the "
            "tool that acts; to raise something you cannot decide alone, call "
            "`ask_question`. Your checklist is checked against this: items still open are "
            "reported to the human as unfinished.";
        d.spec.name = d.name;
        d.spec.params = {param("summary", ParamType::Text, true)};
        // Plan mode ends with `exit_plan_mode`; there is no work there to declare finished.
        d.working_run_only = true;
        declare(d, [](const std::vector<ToolParamValue>&, int) {
            return ToolResult::error(ErrorClass::Malformed, false,
                                     "internal: 'finish' must be handled by the loop");
        });
    }
}

} // namespace lmp::tools
