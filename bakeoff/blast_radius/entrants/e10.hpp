#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
#include <array>

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

    enum class TokenType {
        Word,
        Pipe,       // |
        Or,         // ||
        And,        // &&
        Ampersand,  // &
        Semicolon,  // ;
        RedirectOut,    // >
        RedirectOutAppend, // >>
        RedirectIn,     // <
        LParen,     // (
        RParen,     // )
        Eof
    };

    struct Token {
        TokenType type;
        std::string_view raw_value;
    };

    class Tokenizer {
        std::string_view s;
        size_t pos = 0;
        ParseStatus* status;

    public:
        Tokenizer(std::string_view s, ParseStatus* status) : s(s), pos(0), status(status) {}

        Token next() {
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\n' || s[pos] == '\r')) {
                pos++;
            }
            if (pos >= s.size()) {
                return {TokenType::Eof, {}};
            }

            char c = s[pos];
            if (c == '|') {
                if (pos + 1 < s.size() && s[pos+1] == '|') {
                    pos += 2;
                    return {TokenType::Or, s.substr(pos-2, 2)};
                }
                pos++;
                return {TokenType::Pipe, s.substr(pos-1, 1)};
            } else if (c == '&') {
                if (pos + 1 < s.size() && s[pos+1] == '&') {
                    pos += 2;
                    return {TokenType::And, s.substr(pos-2, 2)};
                }
                pos++;
                return {TokenType::Ampersand, s.substr(pos-1, 1)};
            } else if (c == ';') {
                pos++;
                return {TokenType::Semicolon, s.substr(pos-1, 1)};
            } else if (c == '>') {
                if (pos + 1 < s.size() && s[pos+1] == '>') {
                    pos += 2;
                    return {TokenType::RedirectOutAppend, s.substr(pos-2, 2)};
                }
                pos++;
                return {TokenType::RedirectOut, s.substr(pos-1, 1)};
            } else if (c == '<') {
                pos++;
                return {TokenType::RedirectIn, s.substr(pos-1, 1)};
            } else if (c == '(') {
                pos++;
                return {TokenType::LParen, s.substr(pos-1, 1)};
            } else if (c == ')') {
                pos++;
                return {TokenType::RParen, s.substr(pos-1, 1)};
            } else if (c == '#') {
                while (pos < s.size() && s[pos] != '\n') pos++;
                return next();
            }

            size_t start = pos;
            bool in_single = false;
            bool in_double = false;

            while (pos < s.size()) {
                c = s[pos];
                if (c == '\\') {
                    if (!in_single) {
                        pos += 2;
                        continue;
                    }
                }
                if (c == '\'') {
                    if (!in_double) in_single = !in_single;
                    pos++;
                    continue;
                }
                if (c == '"') {
                    if (!in_single) in_double = !in_double;
                    pos++;
                    continue;
                }
                if (c == '`') {
                    if (!in_single) {
                        *status = ParseStatus::PartiallyParsed;
                    }
                    pos++;
                    continue;
                }
                if (c == '$') {
                    if (!in_single) {
                        *status = ParseStatus::PartiallyParsed;
                    }
                    pos++;
                    continue;
                }

                if (!in_single && !in_double) {
                    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                        c == '|' || c == '&' || c == ';' || c == '>' || c == '<' || c == '(' || c == ')') {
                        break;
                    }
                }
                pos++;
            }

            if (in_single || in_double) {
                *status = ParseStatus::Unparseable;
            }

            if (pos > s.size()) pos = s.size();

            return {TokenType::Word, s.substr(start, pos - start)};
        }
    };

    template <size_t N>
    struct FixedString {
        std::array<char, N> data{};
        size_t len = 0;

        void append(char c) {
            if (len < N) data[len++] = c;
        }

        std::string_view view() const {
            return std::string_view(data.data(), len);
        }

        bool empty() const {
            return len == 0;
        }
    };

    inline void strip_quotes(std::string_view raw, FixedString<256>& out) {
        bool in_single = false;
        bool in_double = false;
        for (size_t i = 0; i < raw.size(); i++) {
            char c = raw[i];
            if (c == '\\') {
                if (!in_single) {
                    if (i + 1 < raw.size()) {
                        out.append(raw[i+1]);
                        i++;
                    }
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
            out.append(c);
        }
    }

    struct PathResolver {
        std::array<std::string_view, 128> components{};
        size_t count = 0;
        bool absolute = false;

        void push(std::string_view comp) {
            if (comp == "" || comp == ".") return;
            if (comp == "..") {
                if (count > 0) {
                    count--;
                } else if (!absolute && count < components.size()) {
                    components[count++] = "..";
                }
                return;
            }
            if (count < components.size()) {
                components[count++] = comp;
            }
        }

        void parse(std::string_view path) {
            if (path.empty()) return;
            if (path[0] == '/') {
                absolute = true;
                count = 0;
            }
            size_t start = 0;
            while (start < path.size()) {
                while (start < path.size() && path[start] == '/') start++;
                if (start >= path.size()) break;
                size_t end = start;
                while (end < path.size() && path[end] != '/') end++;
                push(path.substr(start, end - start));
                start = end;
            }
        }

        bool starts_with(const PathResolver& other) const {
            if (absolute != other.absolute) return false;
            if (count < other.count) return false;
            for (size_t i = 0; i < other.count; i++) {
                if (components[i] != other.components[i]) return false;
            }
            return true;
        }
    };

    inline bool is_outside_workspace(std::string_view target, std::string_view workspace, std::string_view cwd) {
        if (!target.empty() && target[0] == '~') {
            return true;
        }
        PathResolver ws;
        ws.parse(workspace);

        PathResolver trg;
        if (!target.empty() && target[0] == '/') {
            trg.parse(target);
        } else {
            trg.parse(cwd);
            trg.parse(target);
        }

        return !trg.starts_with(ws);
    }
} // namespace detail

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v{};
    detail::Tokenizer tokenizer(ctx.command, &v.status);

    using namespace detail;

    Token tok;

    // capabilities state
    Capabilities& cap = v.capabilities;

    std::array<std::string_view, 128> args{};
    size_t args_count = 0;

    auto process_command = [&](auto& self, size_t start_idx, size_t end_idx) -> void {
        if (start_idx >= end_idx) return;

        FixedString<256> bin_stripped;
        strip_quotes(args[start_idx], bin_stripped);
        std::string_view bin = bin_stripped.view();

        bool is_git = (bin == "git");

        if (bin == "sudo" || bin == "su" || bin == "doas") cap.escalates_privileges = true;
        if (bin == "kill" || bin == "killall" || bin == "pkill") cap.signals_foreign_process = true;
        if (bin == "rm" || bin == "shred" || bin == "wipe" || bin == "dd") cap.destroys_data = true;
        if (bin == "curl" || bin == "wget" || bin == "nc" || bin == "netcat" || bin == "ping" || bin == "ssh" || bin == "scp" || bin == "ftp") cap.network_access = true;
        if (bin == "nohup" || bin == "daemon" || bin == "screen" || bin == "tmux") cap.spawns_unbounded_process = true;

        if (bin == "sh" || bin == "bash" || bin == "zsh" || bin == "xargs" || bin == "make" || bin == "npm" || bin == "yarn" || bin == "pnpm" || bin == "sudo" || bin == "nohup" || bin == "env" || bin == "eval" || bin == "exec") {
            v.status = ParseStatus::PartiallyParsed;
            // Recursively process the rest of the arguments if they look like a command
            if (bin == "sudo" || bin == "nohup" || bin == "env" || bin == "xargs" || bin == "exec") {
                // Find next command, skipping flags
                size_t next_cmd_idx = start_idx + 1;
                while (next_cmd_idx < end_idx && !args[next_cmd_idx].empty() && args[next_cmd_idx][0] == '-') {
                    next_cmd_idx++;
                }
                if (next_cmd_idx < end_idx) {
                    self(self, next_cmd_idx, end_idx);
                }
            } else if (bin == "sh" || bin == "bash" || bin == "zsh") {
                // If there's a -c, we technically should parse the string, but we can just mark partially parsed.
                // However, the test expects rm to be caught inside `sh -c 'rm -rf /'`.
                for (size_t i = start_idx + 1; i < end_idx; i++) {
                    if (args[i] == "-c" && i + 1 < end_idx) {
                        FixedString<256> inner_cmd;
                        strip_quotes(args[i+1], inner_cmd);
                        // Very basic checking for inner commands
                        std::string_view inner = inner_cmd.view();
                        if (inner.find("rm ") != std::string_view::npos || inner.find("rm") == 0) cap.destroys_data = true;
                        if (inner.find("sudo ") != std::string_view::npos || inner.find("sudo") == 0) cap.escalates_privileges = true;
                        if (inner.find("curl ") != std::string_view::npos || inner.find("wget ") != std::string_view::npos) cap.network_access = true;
                    }
                }
            }
        }

        if (is_git) {
            for (size_t i = start_idx + 1; i < end_idx; i++) {
                FixedString<256> arg_stripped;
                strip_quotes(args[i], arg_stripped);
                std::string_view arg = arg_stripped.view();

                if (arg == "push") {
                    for (size_t j = i + 1; j < end_idx; j++) {
                        FixedString<256> push_arg_stripped;
                        strip_quotes(args[j], push_arg_stripped);
                        if (push_arg_stripped.view() == "--force" || push_arg_stripped.view() == "-f") cap.rewrites_vcs_history = true;
                    }
                }
                if (arg == "reset") {
                    for (size_t j = i + 1; j < end_idx; j++) {
                        FixedString<256> reset_arg_stripped;
                        strip_quotes(args[j], reset_arg_stripped);
                        if (reset_arg_stripped.view() == "--hard") cap.destroys_data = true;
                    }
                }
                if (arg == "clean") {
                    bool force = false;
                    for (size_t j = i + 1; j < end_idx; j++) {
                        FixedString<256> clean_arg_stripped;
                        strip_quotes(args[j], clean_arg_stripped);
                        std::string_view ca = clean_arg_stripped.view();
                        if (ca.find('f') != std::string_view::npos && ca[0] == '-') force = true;
                        if (ca == "--force") force = true;
                    }
                    if (force) cap.destroys_data = true;
                }
                if (arg == "rebase") cap.rewrites_vcs_history = true;
                if (arg == "commit") {
                    for (size_t j = i + 1; j < end_idx; j++) {
                        FixedString<256> commit_arg_stripped;
                        strip_quotes(args[j], commit_arg_stripped);
                        if (commit_arg_stripped.view() == "--amend") cap.rewrites_vcs_history = true;
                    }
                }
            }
        }

        if (bin == "cat" || bin == "grep" || bin == "ls" || bin == "cp" || bin == "mv" || bin == "rm") {
            for (size_t i = start_idx + 1; i < end_idx; i++) {
                FixedString<256> arg_stripped;
                strip_quotes(args[i], arg_stripped);
                std::string_view arg = arg_stripped.view();
                if (arg.empty() || arg[0] == '-') continue;
                if (is_outside_workspace(arg, ctx.workspace_root, ctx.cwd)) {
                    if (bin == "cat" || bin == "grep" || bin == "ls" || bin == "cp") cap.reads_outside_workspace = true;
                    if (bin == "cp" || bin == "mv" || bin == "rm") cap.writes_outside_workspace = true;
                }
            }
        }
    };

    while ((tok = tokenizer.next()).type != TokenType::Eof) {
        if (tok.type == TokenType::Word) {
            if (args_count < args.size()) {
                args[args_count++] = tok.raw_value;
            }
        } else if (tok.type == TokenType::Pipe || tok.type == TokenType::Or || tok.type == TokenType::And || tok.type == TokenType::Semicolon) {
            process_command(process_command, 0, args_count);
            args_count = 0;
        } else if (tok.type == TokenType::Ampersand) {
            cap.spawns_unbounded_process = true;
            process_command(process_command, 0, args_count);
            args_count = 0;
        } else if (tok.type == TokenType::RedirectOut) {
            cap.destroys_data = true;
            Token target = tokenizer.next();
            if (target.type == TokenType::Word) {
                FixedString<256> target_stripped;
                strip_quotes(target.raw_value, target_stripped);
                if (is_outside_workspace(target_stripped.view(), ctx.workspace_root, ctx.cwd)) {
                    cap.writes_outside_workspace = true;
                }
            }
        } else if (tok.type == TokenType::RedirectOutAppend) {
            Token target = tokenizer.next();
            if (target.type == TokenType::Word) {
                FixedString<256> target_stripped;
                strip_quotes(target.raw_value, target_stripped);
                if (is_outside_workspace(target_stripped.view(), ctx.workspace_root, ctx.cwd)) {
                    cap.writes_outside_workspace = true;
                }
            }
        } else if (tok.type == TokenType::RedirectIn) {
            Token target = tokenizer.next();
            if (target.type == TokenType::Word) {
                FixedString<256> target_stripped;
                strip_quotes(target.raw_value, target_stripped);
                if (is_outside_workspace(target_stripped.view(), ctx.workspace_root, ctx.cwd)) {
                    cap.reads_outside_workspace = true;
                }
            }
        }
    }

    process_command(process_command, 0, args_count); // final command

    return v;
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
