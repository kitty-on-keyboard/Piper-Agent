#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
#include <string>
#include <vector>

namespace blast_radius {

namespace detail {

inline std::string normalize_path(std::string_view base, std::string_view path) {
    std::string full_path;
    if (!path.empty() && path.front() == '/') {
        full_path = std::string(path);
    } else {
        full_path = std::string(base);
        if (!full_path.empty() && full_path.back() != '/') {
            full_path += '/';
        }
        full_path += std::string(path);
    }

    std::vector<std::string_view> parts;
    size_t start = 0;
    while (start < full_path.size()) {
        size_t end = full_path.find('/', start);
        if (end == std::string::npos) {
            end = full_path.size();
        }
        std::string_view part(full_path.data() + start, end - start);
        if (part.empty() || part == ".") {
            // Ignore
        } else if (part == "..") {
            if (!parts.empty()) {
                parts.pop_back();
            }
        } else {
            parts.push_back(part);
        }
        start = end + 1;
    }

    if (parts.empty()) {
        return "/";
    }
    std::string result;
    for (auto p : parts) {
        result += '/';
        result += p;
    }
    return result;
}

inline bool is_safely_inside(std::string_view root, std::string_view cwd, std::string_view path) noexcept {
    try {
        std::string norm_root = normalize_path("/", root);
        std::string norm_path = normalize_path(cwd, path);
        if (norm_root == "/") return true;
        if (norm_path == norm_root) return true;
        if (norm_root.back() == '/') {
            return norm_path.starts_with(norm_root);
        } else {
            return norm_path.starts_with(norm_root + "/");
        }
    } catch (...) {
        return false;
    }
}

} // namespace detail

enum class ParseStatus : std::uint8_t {
    Parsed = 0,
    PartiallyParsed = 1,
    Unparseable = 2,
};

namespace detail {

struct Token {
    enum class Type {
        Word,
        Operator,
        Redirection,
        EndOfFile
    };
    Type type;
    std::string_view value;
};

class Lexer {
    std::string_view input;
    size_t pos = 0;
    ParseStatus status = ParseStatus::Parsed;
    int depth = 0;
    const int MAX_DEPTH = 50;

public:
    Lexer(std::string_view s) : input(s) {}

    ParseStatus get_status() const { return status; }
    void mark_partially_parsed() { if (status == ParseStatus::Parsed) status = ParseStatus::PartiallyParsed; }
    void mark_unparseable() { status = ParseStatus::Unparseable; }

    Token next() {
        if (depth > MAX_DEPTH) {
            mark_unparseable();
            return {Token::Type::EndOfFile, ""};
        }

        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n' || input[pos] == '\r')) {
            pos++;
        }

        if (pos >= input.size()) {
            return {Token::Type::EndOfFile, ""};
        }

        char c = input[pos];

        // Operators
        if (c == ';' || c == '|' || c == '&' || c == '(' || c == ')') {
            size_t start = pos;
            pos++;
            if (c == '(') depth++;
            if (c == ')') depth--;

            if (pos < input.size() && ((c == '|' && input[pos] == '|') || (c == '&' && input[pos] == '&'))) {
                pos++;
            }
            return {Token::Type::Operator, input.substr(start, pos - start)};
        }

        if (c == '>' || c == '<') {
            size_t start = pos;
            pos++;
            if (pos < input.size() && input[pos] == c) {
                pos++;
            }
            if (pos < input.size() && input[pos] == '&') {
                pos++;
            }
            return {Token::Type::Redirection, input.substr(start, pos - start)};
        }

        // Word (handling quotes and substitutions)
        size_t start = pos;
        bool in_single_quote = false;
        bool in_double_quote = false;

        while (pos < input.size()) {
            c = input[pos];
            if (c == '\\' && !in_single_quote) {
                pos += 2;
                continue;
            }

            if (c == '\'' && !in_double_quote) {
                in_single_quote = !in_single_quote;
                pos++;
                continue;
            }

            if (c == '"' && !in_single_quote) {
                in_double_quote = !in_double_quote;
                pos++;
                continue;
            }

            if (!in_single_quote && c == '$') {
                mark_partially_parsed();
            }
            if (!in_single_quote && c == '`') {
                mark_partially_parsed();
            }

            if (!in_single_quote && !in_double_quote) {
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                    c == ';' || c == '|' || c == '&' || c == '>' || c == '<' ||
                    c == '(' || c == ')') {
                    break;
                }
            }
            pos++;
        }

        if (in_single_quote || in_double_quote) {
            mark_unparseable();
        }

        return {Token::Type::Word, input.substr(start, pos - start)};
    }
};

} // namespace detail

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

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    try {
        detail::Lexer lexer(ctx.command);

        bool is_indirect = false;

        std::vector<detail::Token> current_command;

        // Strip quotes utility
        auto unquote = [](std::string_view s) -> std::string {
            std::string res;
            for (char c : s) {
                if (c != '"' && c != '\'') res += c;
            }
            return res;
        };

        auto evaluate_command = [&]() {
            if (current_command.empty()) return;

            size_t cmd_idx = 0;
            std::string cmd_str = unquote(current_command[0].value);

            // Handle priv escalation / indirection chaining
            while (cmd_idx < current_command.size()) {
                if (cmd_str == "sudo" || cmd_str == "su" || cmd_str == "doas") {
                    v.capabilities.escalates_privileges = true;
                } else if (cmd_str == "sh" || cmd_str == "bash" || cmd_str == "zsh" || cmd_str == "make" || cmd_str == "npm" || cmd_str == "xargs" || cmd_str == "eval" || cmd_str == "env") {
                    // It's indirection, continue looking for the real command
                } else {
                    // Not an indirection wrapper, break and evaluate this as the primary command
                    break;
                }

                cmd_idx++;
                // Skip flags like -c, --, etc.
                while (cmd_idx < current_command.size() && current_command[cmd_idx].value.starts_with("-")) {
                    cmd_idx++;
                }

                if (cmd_idx < current_command.size()) {
                    cmd_str = unquote(current_command[cmd_idx].value);

                    // If the payload was quoted, it could contain sub-commands (e.g. sh -c "rm -rf /")
                    // If the command contains spaces, it means it's likely a complete command inside a single token.
                    // Let's do a rudimentary recursive classification on the unquoted command string if it contains spaces.
                    if (cmd_str.find(' ') != std::string::npos) {
                        CommandContext sub_ctx{cmd_str, ctx.workspace_root, ctx.cwd};
                        Verdict sub_v = classify(sub_ctx);
                        v.capabilities.writes_outside_workspace |= sub_v.capabilities.writes_outside_workspace;
                        v.capabilities.reads_outside_workspace |= sub_v.capabilities.reads_outside_workspace;
                        v.capabilities.destroys_data |= sub_v.capabilities.destroys_data;
                        v.capabilities.rewrites_vcs_history |= sub_v.capabilities.rewrites_vcs_history;
                        v.capabilities.network_access |= sub_v.capabilities.network_access;
                        v.capabilities.spawns_unbounded_process |= sub_v.capabilities.spawns_unbounded_process;
                        v.capabilities.signals_foreign_process |= sub_v.capabilities.signals_foreign_process;
                        v.capabilities.escalates_privileges |= sub_v.capabilities.escalates_privileges;

                        // We parsed this recursively, so we can consider it handled and we break
                        cmd_idx = current_command.size();
                        break;
                    }
                }
            }

            if (cmd_idx >= current_command.size()) return;

            // cmd_str is now the first non-indirection command
            if (cmd_str == "curl" || cmd_str == "wget" || cmd_str == "nc" || cmd_str == "ping" || cmd_str == "ssh" || cmd_str == "scp") {
                v.capabilities.network_access = true;
            }

            if (cmd_str == "kill" || cmd_str == "killall" || cmd_str == "pkill") {
                v.capabilities.signals_foreign_process = true;
            }

            if (cmd_str == "rm" || cmd_str == "shred" || cmd_str == "dd" || cmd_str == "wipe") {
                v.capabilities.destroys_data = true;
            }

            if (cmd_str == "git") {
                bool has_reset = false;
                bool has_hard = false;
                bool has_clean = false;
                bool has_fdx = false;
                bool has_push = false;
                bool has_force = false;

                for (size_t i = cmd_idx + 1; i < current_command.size(); ++i) {
                    std::string arg = unquote(current_command[i].value);
                    if (arg == "reset") has_reset = true;
                    if (arg == "--hard") has_hard = true;
                    if (arg == "clean") has_clean = true;
                    if (arg == "-fdx" || arg == "-xfd" || arg == "-dfx") has_fdx = true;
                    if (arg == "push") has_push = true;
                    if (arg == "--force" || arg == "-f") has_force = true;
                }

                if (has_reset && has_hard) v.capabilities.destroys_data = true;
                if (has_clean && has_fdx) v.capabilities.destroys_data = true;
                if (has_push && has_force) v.capabilities.rewrites_vcs_history = true;
            }

            // Path containment
            for (size_t i = cmd_idx + 1; i < current_command.size(); ++i) {
                if (current_command[i].type == detail::Token::Type::Word) {
                    std::string_view arg = current_command[i].value;
                    if (!arg.starts_with("-")) {
                        // Any path might be outside if cwd is outside, or if it explicitly traverses out
                        if (!detail::is_safely_inside(ctx.workspace_root, ctx.cwd, arg)) {
                            v.capabilities.reads_outside_workspace = true;
                        }
                    }
                }
            }
        };

        while (true) {
            detail::Token t = lexer.next();
            if (t.type == detail::Token::Type::EndOfFile) {
                evaluate_command();
                break;
            }

            if (t.type == detail::Token::Type::Operator) {
                evaluate_command();
                current_command.clear();

                if (t.value == "&") {
                    v.capabilities.spawns_unbounded_process = true;
                }
            } else if (t.type == detail::Token::Type::Redirection) {
                if (t.value == ">" || t.value == ">&" || t.value == "&>") {
                    v.capabilities.destroys_data = true;

                    // The next word is the target
                    detail::Token target = lexer.next();
                    if (target.type == detail::Token::Type::Word) {
                        if (!detail::is_safely_inside(ctx.workspace_root, ctx.cwd, target.value)) {
                            v.capabilities.writes_outside_workspace = true;
                        }
                    }
                } else if (t.value == ">>") {
                    detail::Token target = lexer.next();
                    if (target.type == detail::Token::Type::Word) {
                        if (!detail::is_safely_inside(ctx.workspace_root, ctx.cwd, target.value)) {
                            v.capabilities.writes_outside_workspace = true;
                        }
                    }
                } else if (t.value == "<") {
                    detail::Token target = lexer.next();
                    if (target.type == detail::Token::Type::Word) {
                        if (!detail::is_safely_inside(ctx.workspace_root, ctx.cwd, target.value)) {
                            v.capabilities.reads_outside_workspace = true;
                        }
                    }
                }
            } else {
                current_command.push_back(t);
            }
        }

        v.status = lexer.get_status();

        // If we have unparseable bits or it's completely unparseable, play it safe?
        // Let's stick to the extracted capabilities.

    } catch (...) {
        v.status = ParseStatus::Unparseable;
    }
    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
