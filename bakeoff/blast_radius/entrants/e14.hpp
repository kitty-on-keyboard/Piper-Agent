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

namespace internal {

// Extremely simplified path normalization for static checking
inline bool is_outside_workspace(std::string_view path, std::string_view workspace_root, std::string_view cwd) noexcept {
    // If it's absolute, check if it starts with workspace_root
    if (!path.empty() && path[0] == '/') {
        if (workspace_root.empty() || workspace_root == "/") return false;

        // Strip trailing slash from workspace root for comparison
        std::string_view root = workspace_root;
        if (root.back() == '/') root.remove_suffix(1);

        if (!path.starts_with(root)) return true;
        if (path.size() > root.size() && path[root.size()] != '/') return true; // e.g. /workspace2 when root is /workspace

        return false;
    }

    // For relative paths, we must be careful with ".."
    // This is very primitive static analysis

    int depth = 0;

    // Roughly estimate current depth from cwd if it's within workspace
    if (!cwd.empty() && cwd.starts_with(workspace_root) && cwd.size() > workspace_root.size()) {
        std::string_view rel_cwd = cwd.substr(workspace_root.size());
        if (rel_cwd.starts_with('/')) rel_cwd.remove_prefix(1);

        size_t pos = 0;
        while (pos < rel_cwd.size()) {
            size_t next = rel_cwd.find('/', pos);
            if (next == std::string_view::npos) {
                if (rel_cwd.size() - pos > 0) depth++;
                break;
            }
            if (next - pos > 0) depth++;
            pos = next + 1;
        }
    }

    size_t pos = 0;
    while (pos < path.size()) {
        size_t next = path.find('/', pos);
        std::string_view part = (next == std::string_view::npos) ? path.substr(pos) : path.substr(pos, next - pos);

        if (part == "..") {
            depth--;
            if (depth < 0) return true; // Popped out of workspace
        } else if (part != "." && !part.empty()) {
            depth++;
        }

        if (next == std::string_view::npos) break;
        pos = next + 1;
    }

    return false;
}

struct Token {
    enum class Type {
        Word,
        Operator,
        Redirection,
        EndOfInput,
        Error
    };
    Type type = Type::EndOfInput;
    std::string_view value;
};

class Tokenizer {
public:
    explicit Tokenizer(std::string_view input) noexcept : input_(input), pos_(0) {}

    Token next() noexcept {
        skip_whitespace();
        if (pos_ >= input_.size()) {
            return {Token::Type::EndOfInput, {}};
        }

        char c = input_[pos_];

        // Basic operators and redirections
        if (c == '|' || c == '&' || c == ';' || c == '<' || c == '>') {
            size_t start = pos_;
            pos_++;
            if (pos_ < input_.size()) {
                char next_c = input_[pos_];
                if ((c == '&' && next_c == '&') ||
                    (c == '|' && next_c == '|') ||
                    (c == '>' && next_c == '>')) {
                    pos_++;
                }
            }
            std::string_view op_val = input_.substr(start, pos_ - start);
            Token::Type type = (op_val == "<" || op_val == ">" || op_val == ">>")
                               ? Token::Type::Redirection
                               : Token::Type::Operator;
            return {type, op_val};
        }

        if (c == '(' || c == ')') {
            size_t start = pos_;
            pos_++;
            return {Token::Type::Operator, input_.substr(start, 1)};
        }

        // Word parsing with quotes, escapes, and substitutions
        size_t start = pos_;
        bool in_single_quote = false;
        bool in_double_quote = false;
        bool escaped = false;
        int subshell_depth = 0;
        int substitution_depth = 0;
        bool in_backticks = false;
        bool has_substitution = false;

        while (pos_ < input_.size()) {
            char wc = input_[pos_];

            if (escaped) {
                escaped = false;
                pos_++;
                continue;
            }

            if (!in_single_quote && !in_backticks && wc == '$' && pos_ + 1 < input_.size()) {
                if (input_[pos_ + 1] == '(') {
                    substitution_depth++;
                    has_substitution = true;
                    pos_ += 2;
                    continue;
                } else if (input_[pos_ + 1] == '{') {
                    substitution_depth++;
                    has_substitution = true;
                    pos_ += 2;
                    continue;
                } else {
                    has_substitution = true;
                }
            }

            if (substitution_depth > 0) {
                if (wc == ')' || wc == '}') {
                    substitution_depth--;
                }
                pos_++;
                continue;
            }

            if (!in_single_quote && wc == '`') {
                in_backticks = !in_backticks;
                has_substitution = true;
                pos_++;
                continue;
            }

            if (wc == '\\' && !in_single_quote) {
                escaped = true;
                pos_++;
                continue;
            }

            if (wc == '\'' && !in_double_quote) {
                in_single_quote = !in_single_quote;
                pos_++;
                continue;
            }

            if (wc == '"' && !in_single_quote) {
                in_double_quote = !in_double_quote;
                pos_++;
                continue;
            }

            if (!in_single_quote && !in_double_quote && !in_backticks && substitution_depth == 0) {
                if (is_space(wc) || wc == '|' || wc == '&' || wc == ';' || wc == '<' || wc == '>' || wc == '(' || wc == ')') {
                    break;
                }
            }

            pos_++;
        }

        if (in_single_quote || in_double_quote || escaped || in_backticks || substitution_depth > 0) {
            // Unclosed quote, escape, or substitution
            return {Token::Type::Error, input_.substr(start, pos_ - start)};
        }

        if (has_substitution) {
            has_substitution_ = true;
        }

        return {Token::Type::Word, input_.substr(start, pos_ - start)};
    }

    bool has_substitution() const noexcept {
        return has_substitution_;
    }

private:
    void skip_whitespace() noexcept {
        while (pos_ < input_.size() && is_space(input_[pos_])) {
            pos_++;
        }
    }

    bool is_space(char c) const noexcept {
        return c == ' ' || c == '\t' || c == '\n' || c == '\r';
    }

    std::string_view input_;
    size_t pos_;
    bool has_substitution_ = false;
};

class Analyzer {
public:
    Analyzer(const CommandContext& ctx) : ctx_(ctx) {}

    Verdict analyze() noexcept {
        Verdict v;
        Tokenizer tokenizer(ctx_.command);

        bool is_first_in_chain = true;
        std::string_view current_command = "";
        bool expecting_redirection_target = false;
        std::string_view current_redirection = "";

        while (true) {
            Token t = tokenizer.next();
            if (t.type == Token::Type::EndOfInput) {
                break;
            }
            if (t.type == Token::Type::Error) {
                v.status = ParseStatus::Unparseable;
                break;
            }

            if (t.type == Token::Type::Operator) {
                is_first_in_chain = true;
                current_command = "";
                expecting_redirection_target = false;
                continue;
            }

            if (t.type == Token::Type::Redirection) {
                if (t.value == ">" || t.value == ">&") {
                    v.capabilities.destroys_data = true;
                }

                expecting_redirection_target = true;
                current_redirection = t.value;
                continue;
            }

            if (t.type == Token::Type::Word) {
                if (expecting_redirection_target) {
                    expecting_redirection_target = false;
                    if (is_outside_workspace(t.value, ctx_.workspace_root, ctx_.cwd)) {
                        if (current_redirection == ">" || current_redirection == ">>" || current_redirection == ">&") {
                            v.capabilities.writes_outside_workspace = true;
                        } else if (current_redirection == "<") {
                            v.capabilities.reads_outside_workspace = true;
                        }
                    }
                } else if (is_first_in_chain) {
                    current_command = t.value;
                    evaluate_command(current_command, v);

                    if (current_command == "sudo" || current_command == "time" || current_command == "env" || current_command == "xargs") {
                        // Keep is_first_in_chain true so the next word is treated as a command
                        is_first_in_chain = true;
                    } else {
                        is_first_in_chain = false;
                    }
                } else {
                    evaluate_argument(current_command, t.value, v);
                }
            }
        }

        if (tokenizer.has_substitution() && v.status != ParseStatus::Unparseable) {
            v.status = ParseStatus::PartiallyParsed;
        }

        return v;
    }

private:
    std::string_view basename(std::string_view path) const noexcept {
        size_t pos = path.find_last_of('/');
        if (pos == std::string_view::npos) return path;
        return path.substr(pos + 1);
    }

    void evaluate_command(std::string_view cmd_path, Verdict& v) noexcept {
        std::string_view cmd = basename(cmd_path);
        if (cmd == "sudo" || cmd == "su") {
            v.capabilities.escalates_privileges = true;
        } else if (cmd == "rm" || cmd == "shred") {
            v.capabilities.destroys_data = true;
        } else if (cmd == "curl" || cmd == "wget" || cmd == "ssh" || cmd == "nc" || cmd == "netcat") {
            v.capabilities.network_access = true;
        } else if (cmd == "kill" || cmd == "pkill" || cmd == "killall") {
            v.capabilities.signals_foreign_process = true;
        } else if (cmd == "make" || cmd == "npm" || cmd == "xargs" || cmd == "sh" || cmd == "bash" || cmd == "zsh") {
            v.capabilities.spawns_unbounded_process = true;
        }
    }

    void evaluate_argument(std::string_view cmd, std::string_view arg, Verdict& v) noexcept {
        if (cmd == "git") {
            if (arg == "reset" || arg == "push" || arg == "clean") {
                // Heuristic: Some git commands are more dangerous than others
                // Real implementation would need to parse full flags for reset --hard, clean -fdx
            }
            if (arg == "--hard" || arg == "--force" || arg == "-f" || arg == "-fdx") {
                v.capabilities.rewrites_vcs_history = true;
            }
        }

        // Very basic attempt at catching file paths in commands like rm, cp, mv, etc.
        // If an argument looks like a path (contains /) or starts with .., check it.
        // It's a heuristic for this static analysis.
        if (arg.find('/') != std::string_view::npos || arg.starts_with("..")) {
            if (is_outside_workspace(arg, ctx_.workspace_root, ctx_.cwd)) {
                // Conservatively mark both read and write
                v.capabilities.reads_outside_workspace = true;
                // Only mark write if we're clearly modifying or we're not sure.
                if (cmd == "cp" || cmd == "mv" || cmd == "rm" || cmd == "tar" || cmd == "zip" || cmd == "echo") {
                    v.capabilities.writes_outside_workspace = true;
                }
            }
        }
    }

private:
    CommandContext ctx_;
};

} // namespace internal

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    internal::Analyzer analyzer(ctx);
    return analyzer.analyze();
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
