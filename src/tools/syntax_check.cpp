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

void replace_all(std::string& text, std::string_view from, std::string_view to) {
    if (from.empty()) {
        return;
    }
    std::size_t at = 0;
    while ((at = text.find(from, at)) != std::string::npos) {
        text.replace(at, from.size(), to);
        at += to.size();
    }
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

bool split_shell_command_safe(std::string_view cmd, std::vector<std::string>& argv) {
    argv.clear();
    std::string cur;
    bool in_tok = false;
    char quote = '\0';
    for (std::size_t i = 0; i < cmd.size(); ++i) {
        const char c = cmd[i];
        if (c == '\0' || c == '\n' || c == '\r') {
            return false;
        }
        if (quote != '\0') {
            if (quote == '\'' ) {
                if (c == '\'') {
                    quote = '\0';
                } else {
                    cur.push_back(c);
                }
                continue;
            }
            // Double-quoted: reject unescaped expansions/subshells inside compile DB.
            if (c == '$' || c == '`') {
                return false;
            }
            if (c == '\\' && i + 1 < cmd.size()) {
                cur.push_back(cmd[++i]);
                continue;
            }
            if (c == '"') {
                quote = '\0';
                continue;
            }
            cur.push_back(c);
            continue;
        }
        if (c == ';' || c == '|' || c == '&' || c == '`' || c == '>' || c == '<' ||
            c == '$' || c == '(' || c == ')') {
            return false;
        }
        if (c == '\'' || c == '"') {
            quote = c;
            in_tok = true;
            continue;
        }
        if (c == '\\' && i + 1 < cmd.size()) {
            cur.push_back(cmd[++i]);
            in_tok = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(c)) != 0) {
            if (in_tok) {
                argv.push_back(std::move(cur));
                cur.clear();
                in_tok = false;
            }
            continue;
        }
        cur.push_back(c);
        in_tok = true;
    }
    if (quote != '\0') {
        return false;
    }
    if (in_tok) {
        argv.push_back(std::move(cur));
    }
    return !argv.empty();
}

bool is_allowed_compiler_argv0(std::string_view tok) {
    const std::size_t slash = tok.find_last_of('/');
    const std::string_view base =
        slash == std::string_view::npos ? tok : tok.substr(slash + 1);
    return base == "clang" || base == "clang++" || base == "cc" || base == "c++" ||
           base == "gcc" || base == "g++" || base.rfind("clang-", 0) == 0 ||
           base.rfind("clang++-", 0) == 0 || base.rfind("gcc-", 0) == 0 ||
           base.rfind("g++-", 0) == 0;
}

bool is_disallowed_compiler_arg(std::string_view tok) {
    if (tok.empty()) return false;
    if (tok.front() == '@') return true; // response file
    return tok == "-Xclang" || tok == "-load" || tok == "-B" || tok == "-wrapper" ||
           tok == "--specs" || tok.rfind("-fplugin", 0) == 0 ||
           tok.rfind("-B", 0) == 0 || tok.rfind("--specs=", 0) == 0;
}

static std::string compile_db_syntax_command(const fsx::WorkspaceFs& workspace,
                                             const std::string& abs_path) {
    const fsx::FileContents f =
        workspace.read_file_whole("build/compile_commands.json", 64U << 20);
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
        std::vector<std::string> raw_argv;
        simdjson::dom::array args_arr;
        std::string_view command;
        if (e["arguments"].get_array().get(args_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element arg_el : args_arr) {
                std::string_view sv;
                if (arg_el.get_string().get(sv) != simdjson::SUCCESS ||
                    sv.find('\0') != std::string_view::npos) {
                    return {};
                }
                raw_argv.emplace_back(sv);
            }
        } else if (e["command"].get_string().get(command) == simdjson::SUCCESS) {
            if (!split_shell_command_safe(command, raw_argv)) {
                return {};
            }
        } else {
            return {};
        }
        if (raw_argv.empty() || !is_allowed_compiler_argv0(raw_argv[0])) {
            return {};
        }

        // Drop output/compile flags and reject compiler plugin/loader flags; quote every
        // surviving token individually so /bin/sh cannot interpret repo data as shell.
        std::string out;
        for (std::size_t i = 0; i < raw_argv.size(); ++i) {
            const std::string& tok = raw_argv[i];
            if (is_disallowed_compiler_arg(tok)) {
                return {};
            }
            if (tok == "-o") {
                if (i + 1 < raw_argv.size()) ++i;
                continue;
            }
            if (tok == "-c") {
                continue;
            }
            if (!out.empty()) out += ' ';
            out += quote(tok);
        }
        if (out.empty()) {
            return {};
        }
        out += " -fsyntax-only";
        return out;
    }
    return {};
}

std::string compile_db_syntax_command(const std::string& root,
                                      const std::string& abs_path) {
    const fsx::WorkspaceFs workspace(root);
    return compile_db_syntax_command(workspace, abs_path);
}

SyntaxChecker::SyntaxChecker(std::string root, std::size_t budget_bytes)
    : root_(std::move(root)), workspace_fs_(root_), budget_(budget_bytes) {
    if (workspace_fs_.valid()) {
        root_ = workspace_fs_.root();
    }
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
    const fsx::ContainedPath path = workspace_fs_.contained_path(rel_path);
    if (!path.ok()) {
        return v;
    }
    fsx::OpenedFile source =
        workspace_fs_.open_file_readonly(path.relative, approved_tier != 2);
    if (!source.ok()) {
        return v;
    }

    std::string command;
    if (c->needs_compile_db) {
        command = compile_db_syntax_command(workspace_fs_, path.absolute);
        if (command.empty()) {
            return v; // no entry: stay silent rather than emit a false cascade
        }
        if (approved_tier != 2) {
            // The compiler gets the already-open source descriptor. Add the original
            // directory as a quote include root because /dev/fd has no source directory.
            const std::string fd_path = "/dev/fd/" + std::to_string(source.fd);
            if (command.find(path.absolute) == std::string::npos) {
                return v; // cannot bind an unknown command shape safely
            }
            replace_all(command, path.absolute, fd_path);
            const std::size_t slash = path.absolute.find_last_of('/');
            const std::string parent =
                slash == std::string::npos ? root_ : path.absolute.substr(0, slash);
            command += " -iquote " + quote(parent);
            command += extension_of(rel_path) == ".c" ? " -x c" : " -x c++";
        }
    } else {
        command = c->command;
        const std::size_t at = command.find("{file}");
        if (at == std::string::npos) {
            return v;
        }
        const std::string input =
            approved_tier == 2
                ? path.absolute
                : "/dev/fd/" + std::to_string(source.fd);
        command.replace(at, 6, quote(input));
    }

    const ExecutionGrant grant = grant_execution(tier_for(approved_tier));
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
