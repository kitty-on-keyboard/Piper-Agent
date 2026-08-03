#include "src/tools/registry.hpp"

#include "src/tools/symbol_index.hpp"
#include "src/tools/text_view.hpp"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cstdlib>

#include <algorithm>
#include <cstdio>

#include "src/platform/fs.hpp"
#include "src/tools/graft_engine.hpp"
#include "src/tools/log_triage.hpp"
#include "src/tools/sandbox.hpp"

namespace lmp::tools {
namespace {

namespace fsx = lmp::platform;
using parsephony::ParamSpec;
using parsephony::ParamType;

// A failed write, classified by WHY -- because `retryable` is what decides whether the
// loop's BreakRepeat corrective may fire (see choose_corrective: an unrecoverable repeat
// is `!ok() && !retryable`).
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
            return ToolResult::error(ErrorClass::Malformed, false, w.error);
        case fsx::FsStatus::Ok:
        case fsx::FsStatus::IoError:
            break;
    }
    return ToolResult::error(ErrorClass::Transient, true, w.error);
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

[[nodiscard]] std::string nested_sandbox_note(std::string_view output) {
    if (output.find(kSandboxExecPrefix) == std::string_view::npos ||
        output.find(kNotPermitted) == std::string_view::npos) {
        return {};
    }
    return "\n[sandbox] That failed because the command tried to start a sandbox of its "
           "own inside this one, and macOS refuses to nest them -- it is not a problem "
           "with your code, your caches or your build directory, and no amount of "
           "cleaning or reconfiguring will change it. Swift Package Manager does this to "
           "compile Package.swift. Re-run the same command with `--disable-sandbox` "
           "(`swift build --disable-sandbox`, `swift test --disable-sandbox`); this "
           "workspace jail stays in force around it either way.";
}

ToolResult refused_path(const std::string& p) {
    return ToolResult::refused("path '" + p + "' resolves outside the workspace root; "
                               "only paths inside the workspace are reachable");
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
    const std::string abs = fsx::resolve_against(root, rel);
    return fsx::is_within(root, abs) ? abs : std::string();
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
    const ExecOutcome o = run_sandboxed(grant, "git " + args, ctx_.root, ctx_.root, limits);

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
                             const std::vector<ToolParamValue>& params, int approved_tier) {
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
    std::string out;
    for (const ToolDecl& d : decls_) {
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
            out += "\"" + p.name + "\": {\"type\": \"" +
                   (p.type == ParamType::Number   ? "number"
                    : p.type == ParamType::Boolean ? "boolean"
                    : p.type == ParamType::Object  ? "object"
                    : p.type == ParamType::Array   ? "array"
                                                   : "string") +
                   "\"}";
            if (p.required) {
                required += required.empty() ? "" : ", ";
                required += "\"" + p.name + "\"";
            }
        }
        out += "}, \"required\": [" + required + "]}}}";
    }
    return out;
}

Registry::Registry(WorkspaceContext ctx) : ctx_(std::move(ctx)) {
    // --- read_file ---------------------------------------------------------
    {
        ToolDecl d;
        d.name = "read_file";
        d.description = "Read a file, whole, with 1-based line numbers prefixed for "
                        "reference. Fails honestly with the real size if it exceeds the "
                        "prompt budget; use read_slice for a line range instead. The line "
                        "numbers are display only -- never include them in old_text.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string* path = get(p, "path");
            const std::string abs = resolve_contained(ctx_.root, *path);
            if (abs.empty()) {
                return refused_path(*path);
            }
            fsx::FileContents f = fsx::read_file_whole(abs, ctx_.max_read_bytes);
            if (!f.ok()) {
                const ErrorClass ec = f.status == fsx::FsStatus::NotFound
                                          ? ErrorClass::NotFound
                                          : ErrorClass::Malformed;
                return ToolResult::error(ec, false, f.error);
            }
            // The PROMPT budget, not the read budget. These two used to be one number at
            // 4 MiB, so a single read could hand the context store more bytes than the
            // whole 96k-token window holds -- while this description already claimed it
            // "fails honestly with the real size". It does now.
            if (f.bytes.size() > ctx_.max_model_read_bytes) {
                return ToolResult::error(
                    ErrorClass::Malformed, false,
                    *path + " is " + std::to_string(f.bytes.size()) + " bytes over " +
                        std::to_string(count_lines(f.bytes)) +
                        " lines, above the " + std::to_string(ctx_.max_model_read_bytes) +
                        "-byte prompt budget. Use read_slice(path, start_line, end_line) "
                        "to read a range.");
            }
            // An empty file is a FACT, and it has to be stated. Returning "" makes a
            // successful read indistinguishable from nothing happening at all -- and an
            // empty observation is dropped from the rendered prompt entirely
            // (context.cpp), so the next turn cannot see that the call was ever made.
            //
            // MEASURED: a run wrote 0 bytes to src/kv_store.py, then read it back 17 turns
            // running, getting "" each time and re-issuing the identical call, until the
            // budget ended the run.
            if (f.bytes.empty()) {
                return ToolResult::okay("(" + *path + " exists and is empty: 0 bytes)");
            }
            return ToolResult::okay(number_lines(f.bytes, 1));
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
            const std::string abs = resolve_contained(ctx_.root, *get(p, "path"));
            if (abs.empty()) {
                return refused_path(*get(p, "path"));
            }
            fsx::FileContents f = fsx::read_file_whole(abs, ctx_.max_read_bytes);
            if (!f.ok()) {
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
            return ToolResult::okay(std::move(outp));
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
            const std::string abs = resolve_contained(ctx_.root, *get(p, "path"));
            if (abs.empty()) {
                return refused_path(*get(p, "path"));
            }
            DIR* dir = ::opendir(abs.c_str());
            if (dir == nullptr) {
                return ToolResult::error(ErrorClass::NotFound, false, abs + ": cannot open");
            }
            std::vector<std::string> names;
            while (dirent* e = ::readdir(dir)) {
                const std::string n = e->d_name;
                if (n == "." || n == "..") {
                    continue;
                }
                names.push_back(e->d_type == DT_DIR ? n + "/" : n);
            }
            ::closedir(dir);
            std::sort(names.begin(), names.end());
            std::string outp;
            for (const std::string& n : names) {
                outp += n + "\n";
            }
            return ToolResult::okay(std::move(outp));
        });
    }
    // --- search -------------------------------------------------------------
    {
        ToolDecl d;
        d.name = "search";
        d.description = "Literal substring search across the workspace. Returns "
                        "path:line: text matches, capped; narrow with subdir.";
        d.spec.name = d.name;
        d.spec.params = {param("text", ParamType::Text, true),
                         param("subdir", ParamType::Text, false)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            // grep does this better than any reimplementation; it runs under the same
            // sandbox as shell so search cannot become the unsandboxed escape hatch.
            const std::string* sub = get(p, "subdir");
            const std::string where =
                (sub != nullptr && !sub->empty()) ? *sub : std::string(".");
            const std::string abs = resolve_contained(ctx_.root, where);
            if (abs.empty()) {
                return refused_path(where);
            }
            std::string cmd = "grep -rn --binary-files=without-match -F -e ";
            cmd += "'" ;
            for (char c : *get(p, "text")) {
                if (c == '\'') { cmd += "'\\''"; } else { cmd += c; }
            }
            cmd += "' -- '" + abs + "' | head -200";
            const ExecutionGrant grant = grant_execution(
                approved_tier == 0 ? SandboxTier::T0_NoExec : SandboxTier::T1_Seatbelt);
            const ExecLimits limits{30, 30, 2LL << 30, 256, 64, ctx_.max_result_bytes};
            ExecOutcome o = run_sandboxed(grant, cmd, ctx_.root, ctx_.root, limits);
            if (o.status == Status::Refused) {
                return ToolResult::refused(o.output);
            }
            // grep exits 1 on "no matches" -- that is a successful empty search.
            if (o.exit_code == 1 && o.output.empty()) {
                return ToolResult::okay("(no matches)");
            }
            if (o.status != Status::Ok && o.exit_code != 1) {
                return ToolResult::error(ErrorClass::Transient, true, o.output);
            }
            return ToolResult::okay(std::move(o.output));
        });
    }
    // --- locate_symbol ------------------------------------------------------
    {
        ToolDecl d;
        d.name = "locate_symbol";
        d.description = "Find likely definition sites of a symbol (class, function, "
                        "variable) as path:line: text. Grep-based, language-agnostic.";
        d.spec.name = d.name;
        d.spec.params = {param("symbol", ParamType::Text, true)};
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
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
            const std::string cmd =
                "grep -rnE --binary-files=without-match "
                "'(class|struct|enum|def|fn|func|function|const|let|var|void|int|bool|auto"
                "|std::string)[^;]*\\b" + sym + "\\b' -- '" + ctx_.root + "' | head -400";
            const ExecutionGrant grant = grant_execution(
                approved_tier == 0 ? SandboxTier::T0_NoExec : SandboxTier::T1_Seatbelt);
            const ExecLimits limits{30, 30, 2LL << 30, 256, 64, ctx_.max_result_bytes};
            ExecOutcome o = run_sandboxed(grant, cmd, ctx_.root, ctx_.root, limits);
            if (o.status == Status::Refused) {
                return ToolResult::refused(o.output);
            }
            if (o.output.empty()) {
                return ToolResult::okay("(no definition-shaped lines found for '" + sym +
                                        "'; try search)");
            }
            // Ranked here, not in the pipeline. `head -60` used to hand back whichever
            // lines the filesystem walk reached first, so a symbol with fifty call sites
            // buried its own definition. The grep now casts a wider net and the ordering
            // is decided where it can be asserted.
            std::size_t suppressed = 0;
            const std::vector<SymbolHit> hits =
                rank_symbol_hits(o.output, sym, 40, suppressed);
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
                        "prefix. For a partial change use replace_in_file: it is not just "
                        "tidier, it is cheaper -- replacing the whole of a file this run "
                        "did not write destroys the existing content and stops for a "
                        "human decision, while a scoped edit does not.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("content", ParamType::Text, true)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string abs = resolve_contained(ctx_.root, *get(p, "path"));
            if (abs.empty()) {
                return refused_path(*get(p, "path"));
            }
            const fsx::WriteResult w = commit_write(abs, *get(p, "content"));
            if (!w.ok()) {
                return write_failure(w);
            }
            return ToolResult::okay("wrote " +
                                    std::to_string(get(p, "content")->size()) +
                                    " bytes to " + *get(p, "path"));
        });
    }
    // --- replace_in_file ----------------------------------------------------
    {
        ToolDecl d;
        d.name = "replace_in_file";
        d.description = "Replace exactly one occurrence of old_text with new_text. "
                        "Whitespace-tolerant matching; refuses when the match is "
                        "ambiguous (listing candidate sites) or absent, and the file "
                        "is left untouched in both cases.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("old_text", ParamType::Text, true),
                         param("new_text", ParamType::Text, true)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string abs = resolve_contained(ctx_.root, *get(p, "path"));
            if (abs.empty()) {
                return refused_path(*get(p, "path"));
            }
            fsx::FileContents f = fsx::read_file_whole(abs, ctx_.max_read_bytes);
            if (!f.ok()) {
                return ToolResult::error(f.status == fsx::FsStatus::NotFound
                                             ? ErrorClass::NotFound
                                             : ErrorClass::Malformed,
                                         false, f.error);
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
                return ToolResult::error(ErrorClass::NotFound, false,
                                         "old_text not found in " + *get(p, "path") +
                                             "; re-read the file and try again");
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
            const fsx::WriteResult w = commit_write(abs, g.result);
            if (!w.ok()) {
                return write_failure(w);
            }
            return ToolResult::okay("replaced one occurrence in " + *get(p, "path"));
        });
    }
    // --- append_file --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "append_file";
        d.description = "Append content to the end of a file, creating it if absent.";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true),
                         param("content", ParamType::Text, true)};
        d.mutates_workspace = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string abs = resolve_contained(ctx_.root, *get(p, "path"));
            if (abs.empty()) {
                return refused_path(*get(p, "path"));
            }
            fsx::FileContents f = fsx::read_file_whole(abs, ctx_.max_read_bytes);
            std::string content =
                f.ok() ? f.bytes + *get(p, "content") : *get(p, "content");
            if (!f.ok() && f.status != fsx::FsStatus::NotFound) {
                return ToolResult::error(ErrorClass::Malformed, false, f.error);
            }
            const fsx::WriteResult w = commit_write(abs, content);
            if (!w.ok()) {
                return write_failure(w);
            }
            return ToolResult::okay("appended " +
                                    std::to_string(get(p, "content")->size()) +
                                    " bytes to " + *get(p, "path"));
        });
    }
    // --- delete_file --------------------------------------------------------
    {
        ToolDecl d;
        d.name = "delete_file";
        d.description = "Delete one file (not a directory).";
        d.spec.name = d.name;
        d.spec.params = {param("path", ParamType::Text, true)};
        d.mutates_workspace = true;
        // Nothing in the workspace can undo this, so it always asks (S7.2).
        d.irreversible = true;
        declare(d, [this](const std::vector<ToolParamValue>& p, int) {
            const std::string abs = resolve_contained(ctx_.root, *get(p, "path"));
            if (abs.empty()) {
                return refused_path(*get(p, "path"));
            }
            struct stat st {};
            if (::stat(abs.c_str(), &st) != 0) {
                return ToolResult::error(ErrorClass::NotFound, false,
                                         *get(p, "path") + ": not found");
            }
            if (S_ISDIR(st.st_mode)) {
                return ToolResult::refused("'" + *get(p, "path") +
                                           "' is a directory; this tool deletes files "
                                           "only");
            }
            if (::unlink(abs.c_str()) != 0) {
                return ToolResult::error(ErrorClass::Transient, true,
                                         *get(p, "path") + ": unlink failed");
            }
            return ToolResult::okay("deleted " + *get(p, "path"));
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
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            const std::string& cmd = *get(p, "command");
            const ExecutionGrant grant = grant_execution(tier_for(approved_tier));
            const ExecLimits limits{ctx_.shell_wall_clock_seconds,
                                    ctx_.shell_wall_clock_seconds,
                                    8LL << 30,
                                    1024,
                                    256,
                                    4U << 20};
            ExecOutcome o = run_sandboxed(grant, cmd, ctx_.root, ctx_.root, limits);

            ToolResult r;
            r.status = o.status;
            if (o.status == Status::Refused) {
                r.error_class = ErrorClass::Policy;
                r.summary = o.output;
                return r;
            }
            // Compact through the cookoff-merged triage engine (S6.2): anchors first,
            // own-code beats system headers, never head-truncate.
            r.summary = log_triage::compact(o.output, ctx_.max_result_bytes);
            r.exit_code = o.exit_code;
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
                const std::string spool =
                    ctx_.spool_dir + "/shell_" +
                    std::to_string(reinterpret_cast<std::uintptr_t>(&o) ^ o.output.size()) +
                    ".log";
                if (fsx::write_file_atomic(spool, o.output).ok()) {
                    r.artifacts.push_back(spool);
                }
            }
            // LAST, and read off the FULL output rather than the compacted summary: the
            // signature is one line inside a manifest dump that log_triage may well have
            // elided. After the spool decision, so appending it cannot make the summary
            // look long enough to skip spooling the output it is explaining.
            r.summary += nested_sandbox_note(o.output);
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
        declare(d, [this](const std::vector<ToolParamValue>& p, int approved_tier) {
            const std::string* path = get(p, "path");
            if (path == nullptr || path->empty()) {
                return run_git("diff --stat -p", approved_tier);
            }
            if (resolve_contained(ctx_.root, *path).empty()) {
                return refused_path(*path);
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
        d.description =
            "State or restate the checklist for this mission: one item per line, each "
            "prefixed '[ ] ' for open or '[x] ' for done. Call it first, before doing "
            "the work, and call it again to tick items off as you finish them. Give "
            "`verify_with` the exact shell command that proves the mission is complete "
            "(a test or build command); a run is only finished when that command has "
            "been seen to pass.";
        d.spec.name = d.name;
        d.spec.params = {param("items", ParamType::Text, true),
                         param("verify_with", ParamType::Text, false)};
        declare(d, [](const std::vector<ToolParamValue>&, int) {
            return ToolResult::error(ErrorClass::Malformed, false,
                                     "internal: 'plan' must be handled by the loop");
        });
    }
}

} // namespace lmp::tools
