#include "src/tools/syntax_check.hpp"

#include <simdjson.h>

#include <algorithm>

#include "src/platform/fs.hpp"
#include "src/tools/log_triage.hpp"
#include "src/tools/sandbox.hpp"

namespace lmp::tools {
namespace {

namespace fsx = lmp::platform;

// A check must not become the thing that ends the turn. Fifteen seconds is generous for
// every contract below and short enough that a hung one costs less than a token cap.
constexpr int kWallClockSeconds = 15;

std::string quote(std::string_view s) {
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

std::string extension_of(std::string_view path) {
    const std::size_t slash = path.find_last_of('/');
    const std::string_view base =
        slash == std::string_view::npos ? path : path.substr(slash + 1);
    const std::size_t dot = base.find_last_of('.');
    if (dot == std::string_view::npos || dot == 0) {
        return {};
    }
    std::string ext(base.substr(dot));
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

const SyntaxContract* contract_for(std::string_view rel_path) {
    const std::string ext = extension_of(rel_path);
    if (ext.empty()) {
        return nullptr;
    }
    for (const SyntaxContract& c : syntax_contracts()) {
        if (std::find(c.extensions.begin(), c.extensions.end(), ext) != c.extensions.end()) {
            return &c;
        }
    }
    return nullptr;
}

} // namespace

const std::vector<SyntaxContract>& syntax_contracts() {
    // Every command here is a SYNTAX check, and each one's description says so. None of
    // them type-checks, and claiming otherwise in a tool description is the lie the
    // tool-honesty ratchet exists to catch.
    static const std::vector<SyntaxContract> kContracts = {
        // ast.parse rather than py_compile: py_compile writes a __pycache__ beside the
        // file, and a check that MUTATES the workspace to report on it is the wrong shape
        // -- it also fails outright wherever the sandbox denies that write, which is where
        // this first showed up.
        {"python",
         {".py"},
         "python3 -c \"import ast,sys;ast.parse(open(sys.argv[1]).read(),sys.argv[1])\" "
         "{file}",
         false},
        {"json",
         {".json"},
         "python3 -c \"import json,sys;json.load(open(sys.argv[1]))\" {file}",
         false},
        {"javascript", {".js", ".mjs", ".cjs"}, "node --check {file}", false},
        // Deliberately last and deliberately gated: see rule 2 in the header.
        {"cxx", {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"}, "", true},
    };
    return kContracts;
}

std::string compile_db_syntax_command(const std::string& root,
                                      const std::string& abs_path) {
    const std::string db = root + "/build/compile_commands.json";
    const fsx::FileContents f = fsx::read_file_whole(db, 64U << 20);
    if (!f.ok()) {
        return {};
    }
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(f.bytes).get(doc) != simdjson::SUCCESS) {
        return {};
    }
    simdjson::dom::array entries;
    if (doc.get_array().get(entries) != simdjson::SUCCESS) {
        return {};
    }
    for (simdjson::dom::element e : entries) {
        std::string_view file;
        if (e["file"].get_string().get(file) != simdjson::SUCCESS || file != abs_path) {
            continue;
        }
        std::string_view command;
        if (e["command"].get_string().get(command) != simdjson::SUCCESS) {
            return {};
        }
        // Drop the output argument -- with -fsyntax-only there is nothing to write, and a
        // stale -o would have the check clobber a real object file.
        std::string out;
        std::size_t at = 0;
        const std::string_view cmd = command;
        while (at < cmd.size()) {
            const std::size_t sp = cmd.find(' ', at);
            const std::size_t stop = sp == std::string_view::npos ? cmd.size() : sp;
            const std::string_view tok = cmd.substr(at, stop - at);
            if (tok == "-o") {
                at = stop + 1;
                const std::size_t sp2 = cmd.find(' ', at);
                at = sp2 == std::string_view::npos ? cmd.size() : sp2 + 1;
                continue;
            }
            if (tok == "-c") {
                at = stop + 1;
                continue;
            }
            if (!tok.empty()) {
                out += out.empty() ? "" : " ";
                out.append(tok);
            }
            at = stop + 1;
        }
        out += " -fsyntax-only";
        return out;
    }
    return {};
}

SyntaxVerdict SyntaxChecker::check(const std::string& rel_path, int approved_tier) const {
    SyntaxVerdict v;
    if (approved_tier <= 0) {
        return v; // no execution granted; a Plan-mode run has nothing to check
    }
    const SyntaxContract* c = contract_for(rel_path);
    if (c == nullptr) {
        return v;
    }
    const std::string abs = fsx::resolve_against(root_, rel_path);
    if (!fsx::is_within(root_, abs)) {
        return v;
    }

    std::string command;
    if (c->needs_compile_db) {
        command = compile_db_syntax_command(root_, abs);
        if (command.empty()) {
            return v; // no entry: stay silent rather than emit a false cascade
        }
    } else {
        command = c->command;
        const std::size_t at = command.find("{file}");
        if (at == std::string::npos) {
            return v;
        }
        command.replace(at, 6, quote(abs));
    }

    const ExecutionGrant grant = grant_execution(
        approved_tier == 1 ? SandboxTier::T1_Seatbelt
        : approved_tier == 2 ? SandboxTier::T2_Container
                             : SandboxTier::T3_HostUnsandboxed);
    const ExecLimits limits{kWallClockSeconds, kWallClockSeconds, 4LL << 30, 256, 64,
                            budget_ * 4};
    const ExecOutcome o = run_sandboxed(grant, command, root_, root_, limits);
    if (o.status == Status::Refused) {
        return v; // the sandbox said no; that is not a fact about the edit
    }

    v.ran = true;
    v.language = c->language;
    v.clean = o.exit_code == 0 && !o.signalled && !o.wall_clock_killed;
    if (!v.clean) {
        v.diagnostics = log_triage::compact(o.output, budget_);
    }
    return v;
}

} // namespace lmp::tools
