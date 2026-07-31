#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>

namespace blast_radius {

enum class ParseStatus : std::uint8_t {
    Parsed = 0,
    PartiallyParsed = 1,
    Unparseable = 2,
};

struct Capabilities {
    bool writes_outside_workspace = false;
    bool reads_outside_workspace = false;
    bool destroys_data = false;
    bool rewrites_vcs_history = false;
    bool network_access = false;
    bool spawns_unbounded_process = false;
    bool signals_foreign_process = false;
    bool escalates_privileges = false;
};

struct Verdict {
    Capabilities capabilities{};
    ParseStatus status = ParseStatus::Parsed;
};

struct CommandContext {
    std::string_view command;
    std::string_view workspace_root;
    std::string_view cwd;
};

namespace detail {

template <std::size_t MaxDepth = 128>
struct PathSegments {
    std::string_view segments[MaxDepth];
    std::size_t count = 0;
    bool absolute = false;
    bool overflow = false;

    void push(std::string_view seg) noexcept {
        if (overflow) return;
        if (seg.empty() || seg == ".") return;
        if (seg == "..") {
            if (count > 0 && segments[count - 1] != "..") {
                count--;
            } else if (!absolute) {
                if (count < MaxDepth) {
                    segments[count++] = "..";
                } else {
                    overflow = true;
                }
            }
            return;
        }
        if (count < MaxDepth) {
            segments[count++] = seg;
        } else {
            overflow = true;
        }
    }

    void parse(std::string_view path) noexcept {
        if (path.empty()) return;
        if (path[0] == '/') {
            absolute = true;
            count = 0;
        }
        std::size_t start = 0;
        while (start < path.size()) {
            std::size_t end = path.find('/', start);
            if (end == std::string_view::npos) {
                push(path.substr(start));
                break;
            }
            push(path.substr(start, end - start));
            start = end + 1;
        }
    }
};

inline bool is_inside_workspace(std::string_view workspace_root, std::string_view cwd, std::string_view target_path) noexcept {
    PathSegments<128> ws;
    ws.parse(workspace_root);

    PathSegments<128> target;
    target.parse(cwd);
    target.parse(target_path); // absolute target_path will reset target

    if (ws.overflow || target.overflow) {
        return false; // If we overflow, assume outside for safety
    }

    if (ws.absolute != target.absolute) {
        return false;
    }

    if (target.count < ws.count) {
        return false;
    }

    for (std::size_t i = 0; i < ws.count; ++i) {
        if (ws.segments[i] != target.segments[i]) {
            return false;
        }
    }

    return true;
}

enum class TokenType {
    Word,
    Pipe,           // |
    LogicalAnd,     // &&
    LogicalOr,      // ||
    Background,     // &
    Semicolon,      // ;
    RedirectOut,    // >
    RedirectAppend, // >>
    SubshellStart,  // (
    SubshellEnd,    // )
    Eof
};

struct Token {
    TokenType type;
    std::string_view value;
};

class Scanner {
    std::string_view input;
    std::size_t pos = 0;
    bool partially_parsed = false;
    bool unparseable = false;

    void skip_whitespace() noexcept {
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n' || input[pos] == '\r')) {
            pos++;
        }
    }

    bool is_operator_char(char c) const noexcept {
        return c == '|' || c == '&' || c == ';' || c == '>' || c == '(' || c == ')';
    }

public:
    explicit Scanner(std::string_view str) noexcept : input(str) {}

    bool is_partially_parsed() const noexcept { return partially_parsed; }
    bool is_unparseable() const noexcept { return unparseable; }
    void mark_partially_parsed() noexcept { partially_parsed = true; }

    Token next() noexcept {
        skip_whitespace();
        if (pos >= input.size()) return {TokenType::Eof, {}};

        char c = input[pos];
        if (c == '|') {
            if (pos + 1 < input.size() && input[pos + 1] == '|') {
                pos += 2;
                return {TokenType::LogicalOr, "||"};
            }
            pos++;
            return {TokenType::Pipe, "|"};
        }
        if (c == '&') {
            if (pos + 1 < input.size() && input[pos + 1] == '&') {
                pos += 2;
                return {TokenType::LogicalAnd, "&&"};
            }
            pos++;
            return {TokenType::Background, "&"};
        }
        if (c == ';') {
            pos++;
            return {TokenType::Semicolon, ";"};
        }
        if (c == '>') {
            if (pos + 1 < input.size() && input[pos + 1] == '>') {
                pos += 2;
                return {TokenType::RedirectAppend, ">>"};
            }
            pos++;
            return {TokenType::RedirectOut, ">"};
        }
        if (c == '(') {
            pos++;
            return {TokenType::SubshellStart, "("};
        }
        if (c == ')') {
            pos++;
            return {TokenType::SubshellEnd, ")"};
        }

        std::size_t start = pos;
        bool in_single_quotes = false;
        bool in_double_quotes = false;
        bool escaped = false;

        while (pos < input.size()) {
            c = input[pos];
            if (escaped) {
                escaped = false;
                pos++;
                continue;
            }

            if (c == '\\') {
                if (!in_single_quotes) {
                    escaped = true;
                }
                pos++;
                continue;
            }

            if (c == '\'') {
                if (!in_double_quotes) {
                    in_single_quotes = !in_single_quotes;
                }
                pos++;
                continue;
            }

            if (c == '"') {
                if (!in_single_quotes) {
                    in_double_quotes = !in_double_quotes;
                }
                pos++;
                continue;
            }

            if (!in_single_quotes && !in_double_quotes) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || is_operator_char(c)) {
                    break;
                }
                // Handle substitutions causing partial parsing
                if (c == '$' || c == '`') {
                    partially_parsed = true;
                }
            } else if (in_double_quotes && c == '$') {
                partially_parsed = true;
            } else if (in_double_quotes && c == '`') {
                partially_parsed = true;
            }

            pos++;
        }

        if (in_single_quotes || in_double_quotes) {
            unparseable = true;
        }

        return {TokenType::Word, input.substr(start, pos - start)};
    }
};

struct UnquotedWord {
    char data[256];
    std::size_t len = 0;
    bool overflow = false;

    void append(char c) noexcept {
        if (len < 256) {
            data[len++] = c;
        } else {
            overflow = true;
        }
    }

    std::string_view view() const noexcept {
        return {data, len};
    }

    bool operator==(std::string_view other) const noexcept {
        return view() == other;
    }
};

inline UnquotedWord unquote(std::string_view token) noexcept {
    UnquotedWord word;
    bool in_single = false;
    bool in_double = false;
    bool escaped = false;

    for (char c : token) {
        if (escaped) {
            word.append(c);
            escaped = false;
            continue;
        }
        if (c == '\\') {
            if (!in_single) {
                escaped = true;
                continue;
            }
        }
        if (c == '\'') {
            if (!in_double) {
                in_single = !in_single;
                continue;
            }
        }
        if (c == '"') {
            if (!in_single) {
                in_double = !in_double;
                continue;
            }
        }
        word.append(c);
    }
    return word;
}

// Forward declaration of classify for recursive indirection (e.g. sh -c)
} // namespace detail
[[nodiscard]] Verdict classify(const CommandContext& ctx) noexcept;
namespace detail {

inline void evaluate_command(const UnquotedWord* args, std::size_t argc, const CommandContext& ctx, Verdict& verdict) noexcept;

inline void classify_internal(Scanner& scanner, const CommandContext& ctx, Verdict& verdict, int depth) noexcept {
    if (depth > 64) {
        verdict.status = ParseStatus::Unparseable;
        return;
    }

    UnquotedWord args[256];
    std::size_t argc = 0;
    bool overflowed_arguments = false;
    bool expecting_redirect_target = false;
    bool is_append = false; // To be used if we differentiate > and >> for destroys_data

    auto finish_command = [&]() {
        if (argc > 0) {
            if (overflowed_arguments) {
                verdict.status = ParseStatus::Unparseable;
            } else {
                evaluate_command(args, argc, ctx, verdict);
            }
            argc = 0;
            overflowed_arguments = false;
        }
    };

    while (true) {
        Token t = scanner.next();
        if (t.type == TokenType::Eof) {
            finish_command();
            if (depth > 0) {
                verdict.status = ParseStatus::Unparseable;
            }
            break;
        }

        if (scanner.is_unparseable()) {
            verdict.status = ParseStatus::Unparseable;
            return;
        }

        if (t.type == TokenType::SubshellStart) {
            finish_command();
            classify_internal(scanner, ctx, verdict, depth + 1);
            continue;
        }

        if (t.type == TokenType::SubshellEnd) {
            finish_command();
            if (depth == 0) {
                verdict.status = ParseStatus::Unparseable;
            }
            return;
        }

        if (t.type == TokenType::RedirectOut || t.type == TokenType::RedirectAppend) {
            expecting_redirect_target = true;
            is_append = (t.type == TokenType::RedirectAppend);
            continue;
        }

        if (t.type == TokenType::Word) {
            UnquotedWord word = unquote(t.value);
            if (word.overflow) {
                verdict.status = ParseStatus::Unparseable;
                return;
            }
            if (expecting_redirect_target) {
                if (!is_append) {
                    verdict.capabilities.destroys_data = true; // > destroys data
                }
                expecting_redirect_target = false;
                if (!is_inside_workspace(ctx.workspace_root, ctx.cwd, word.view())) {
                    verdict.capabilities.writes_outside_workspace = true;
                }
            } else {
                if (argc < 256) {
                    args[argc++] = word;
                } else {
                    overflowed_arguments = true;
                }
            }
        } else {
            finish_command();
        }
    }
}

inline void evaluate_command(const UnquotedWord* args, std::size_t argc, const CommandContext& ctx, Verdict& verdict) noexcept {
    if (argc == 0) return;

    std::string_view cmd = args[0].view();

    if (cmd == "rm" || cmd == "rmdir" || cmd == "unlink") {
        verdict.capabilities.destroys_data = true;
        for (std::size_t i = 1; i < argc; ++i) {
            std::string_view arg = args[i].view();
            if (arg.empty() || arg[0] == '-') continue;
            if (!is_inside_workspace(ctx.workspace_root, ctx.cwd, arg)) {
                verdict.capabilities.writes_outside_workspace = true;
            }
        }
    } else if (cmd == "cp" || cmd == "mv" || cmd == "ln" || cmd == "mkdir" || cmd == "touch" || cmd == "dd") {
        for (std::size_t i = 1; i < argc; ++i) {
            std::string_view arg = args[i].view();
            if (arg.empty() || arg[0] == '-') continue;
            if (cmd == "mv" || cmd == "dd") verdict.capabilities.destroys_data = true;
            if (!is_inside_workspace(ctx.workspace_root, ctx.cwd, arg)) {
                // Approximate: if any non-flag argument is outside workspace, assume read/write outside
                verdict.capabilities.writes_outside_workspace = true;
                verdict.capabilities.reads_outside_workspace = true;
            }
        }
    } else if (cmd == "git") {
        if (argc > 1) {
            std::string_view subcmd = args[1].view();
            if (subcmd == "reset") {
                for (std::size_t i = 2; i < argc; ++i) {
                    if (args[i].view() == "--hard") {
                        verdict.capabilities.destroys_data = true;
                    }
                }
                verdict.capabilities.rewrites_vcs_history = true;
            } else if (subcmd == "clean") {
                bool is_dry_run = false;
                for (std::size_t i = 2; i < argc; ++i) {
                    std::string_view arg = args[i].view();
                    if (arg == "-n" || arg == "--dry-run") {
                        is_dry_run = true;
                    }
                }
                if (!is_dry_run) {
                    verdict.capabilities.destroys_data = true;
                }
            } else if (subcmd == "push") {
                for (std::size_t i = 2; i < argc; ++i) {
                    if (args[i].view() == "-f" || args[i].view() == "--force" || args[i].view() == "--force-with-lease") {
                        verdict.capabilities.rewrites_vcs_history = true;
                    }
                }
                verdict.capabilities.network_access = true;
            } else if (subcmd == "commit") {
                for (std::size_t i = 2; i < argc; ++i) {
                    if (args[i].view() == "--amend") {
                        verdict.capabilities.rewrites_vcs_history = true;
                    }
                }
            } else if (subcmd == "rebase") {
                verdict.capabilities.rewrites_vcs_history = true;
            } else if (subcmd == "fetch" || subcmd == "pull" || subcmd == "clone") {
                verdict.capabilities.network_access = true;
            }
        }
    } else if (cmd == "curl" || cmd == "wget" || cmd == "nc" || cmd == "ping" || cmd == "ssh" || cmd == "scp" || cmd == "ftp") {
        verdict.capabilities.network_access = true;
    } else if (cmd == "kill" || cmd == "killall" || cmd == "pkill" || cmd == "xargs") {
        // xargs handled in indirection if needed, but kill definitely signals
        if (cmd != "xargs") {
            verdict.capabilities.signals_foreign_process = true;
        }
    } else if (cmd == "yes" || cmd == "cat" || cmd == "tail" || cmd == "find" || cmd == "grep") {
        // Find / cat might read outside if path is outside
        if (cmd == "yes") {
            verdict.capabilities.spawns_unbounded_process = true; // wait yes produces infinite output, maybe bounded by pipe but let's be safe
        }
        for (std::size_t i = 1; i < argc; ++i) {
            std::string_view arg = args[i].view();
            if (arg.empty() || arg[0] == '-') continue;
            if (!is_inside_workspace(ctx.workspace_root, ctx.cwd, arg)) {
                verdict.capabilities.reads_outside_workspace = true;
            }
        }
    } else if (cmd == "cat" && argc == 1) {
        verdict.capabilities.spawns_unbounded_process = true; // hangs on stdin
    }

    // Indirection Handling
    if (cmd == "sudo" || cmd == "su" || cmd == "doas") {
        verdict.capabilities.escalates_privileges = true;
        // Shift arguments and re-evaluate
        std::size_t offset = 1;
        while (offset < argc) {
            std::string_view view = args[offset].view();
            if (view.starts_with("-")) {
                if (view == "-u" || view == "-g" || view == "-s" || view == "-c") {
                    offset += 2;
                } else {
                    offset++;
                }
            } else {
                break;
            }
        }
        if (offset < argc) {
            evaluate_command(args + offset, argc - offset, ctx, verdict);
        }
    } else if (cmd == "xargs") {
        std::size_t offset = 1;
        while (offset < argc && args[offset].view().starts_with("-")) {
            if (args[offset].view() == "-I" || args[offset].view() == "-i" || args[offset].view() == "-n" || args[offset].view() == "-P" || args[offset].view() == "-E" || args[offset].view() == "-d" || args[offset].view() == "-s") {
                offset += 2; // these take an arg, typically
            } else {
                offset++;
            }
        }
        if (offset < argc) {
            evaluate_command(args + offset, argc - offset, ctx, verdict);
        }
    } else if (cmd == "sh" || cmd == "bash" || cmd == "zsh") {
        bool has_c = false;
        std::size_t c_idx = 0;
        for (std::size_t i = 1; i < argc; ++i) {
            if (args[i].view() == "-c") {
                has_c = true;
                c_idx = i;
                break;
            }
        }
        if (has_c && c_idx + 1 < argc) {
            CommandContext subctx = ctx;
            subctx.command = args[c_idx + 1].view();
            Verdict sub_verdict = classify(subctx);

            // Merge sub_verdict into verdict
            verdict.capabilities.writes_outside_workspace |= sub_verdict.capabilities.writes_outside_workspace;
            verdict.capabilities.reads_outside_workspace |= sub_verdict.capabilities.reads_outside_workspace;
            verdict.capabilities.destroys_data |= sub_verdict.capabilities.destroys_data;
            verdict.capabilities.rewrites_vcs_history |= sub_verdict.capabilities.rewrites_vcs_history;
            verdict.capabilities.network_access |= sub_verdict.capabilities.network_access;
            verdict.capabilities.spawns_unbounded_process |= sub_verdict.capabilities.spawns_unbounded_process;
            verdict.capabilities.signals_foreign_process |= sub_verdict.capabilities.signals_foreign_process;
            verdict.capabilities.escalates_privileges |= sub_verdict.capabilities.escalates_privileges;

            if (sub_verdict.status > verdict.status) {
                verdict.status = sub_verdict.status;
            }
        } else {
            // Unbounded or script we can't analyze easily (like just running a script file)
            // It might read outside workspace, network access, etc. We don't mark anything specific
            // except partially parsed since we can't read the file.
            verdict.status = ParseStatus::PartiallyParsed;
        }
    } else if (cmd == "make" || cmd == "npm" || cmd == "cargo") {
        // These build systems can do arbitrary things, so we consider it partially parsed,
        // as we can't statically analyze the build scripts here.
        verdict.status = ParseStatus::PartiallyParsed;
    }
}

} // namespace detail

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict verdict{};
    detail::Scanner scanner(ctx.command);
    detail::classify_internal(scanner, ctx, verdict, 0);

    if (scanner.is_unparseable()) {
        verdict.status = ParseStatus::Unparseable;
    } else if (scanner.is_partially_parsed()) {
        if (verdict.status == ParseStatus::Parsed) {
            verdict.status = ParseStatus::PartiallyParsed;
        }
    }

    return verdict;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
