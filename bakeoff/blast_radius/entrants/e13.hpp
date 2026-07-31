#ifndef BLAST_RADIUS_HPP
#define BLAST_RADIUS_HPP

#include <cstdint>
#include <string_view>
#include <string>
#include <vector>
#include <span>

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

inline void merge_capabilities(Capabilities& a, const Capabilities& b) {
    a.writes_outside_workspace |= b.writes_outside_workspace;
    a.reads_outside_workspace |= b.reads_outside_workspace;
    a.destroys_data |= b.destroys_data;
    a.rewrites_vcs_history |= b.rewrites_vcs_history;
    a.network_access |= b.network_access;
    a.spawns_unbounded_process |= b.spawns_unbounded_process;
    a.signals_foreign_process |= b.signals_foreign_process;
    a.escalates_privileges |= b.escalates_privileges;
}

struct Verdict {
    Capabilities capabilities{};
    ParseStatus status = ParseStatus::Parsed;
};

inline void merge_verdict(Verdict& a, const Verdict& b) {
    merge_capabilities(a.capabilities, b.capabilities);
    if (b.status == ParseStatus::Unparseable || a.status == ParseStatus::Unparseable) {
        a.status = ParseStatus::Unparseable;
    } else if (b.status == ParseStatus::PartiallyParsed || a.status == ParseStatus::PartiallyParsed) {
        a.status = ParseStatus::PartiallyParsed;
    }
}

struct CommandContext {
    std::string_view command;
    std::string_view workspace_root;
    std::string_view cwd;
};

[[nodiscard]] Verdict classify(const CommandContext& ctx) noexcept;
[[nodiscard]] Verdict classify_impl(const CommandContext& ctx, int depth) noexcept;

enum class TokenType {
    Word,
    Pipe,           // |
    LogicalAnd,     // &&
    LogicalOr,      // ||
    Semi,           // ;
    SemiSemi,       // ;;
    Background,     // &
    RedirOut,       // >
    RedirAppend,    // >>
    RedirIn,        // <
    RedirInAnd,     // <&
    RedirOutAnd,    // >&
    RedirOutClobber,// >|
    RedirHere,      // <<
    LParen,         // (
    RParen,         // )
    Eof
};

struct Token {
    TokenType type;
    std::string text;
    bool is_dynamic = false;
};

class Lexer {
    std::string_view input;
    size_t pos = 0;
    std::string word_buf;
    Verdict& verdict;
    CommandContext ctx;
    int depth = 0;

    void append_char(char c) {
        word_buf += c;
    }

    bool is_alphanum_var(char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_';
    }

    void handle_substitution() {
        if (pos >= input.size()) return;
        if (input[pos] == '`') {
            size_t start = pos;
            pos++;
            while (pos < input.size() && input[pos] != '`') {
                if (input[pos] == '\\') pos += 2;
                else pos++;
            }
            if (pos >= input.size()) { verdict.status = ParseStatus::Unparseable; return; }
            std::string_view sub = input.substr(start + 1, pos - (start + 1));
            pos++; // skip `
            CommandContext sub_ctx = ctx;
            sub_ctx.command = sub;
            Verdict sub_v = classify_impl(sub_ctx, depth + 1);
            merge_verdict(verdict, sub_v);
        } else if (input[pos] == '$') {
            if (pos + 1 < input.size() && input[pos+1] == '(') {
                if (pos + 2 < input.size() && input[pos+2] == '(') {
                    pos += 3;
                    while (pos + 1 < input.size() && !(input[pos] == ')' && input[pos+1] == ')')) pos++;
                    if (pos + 1 >= input.size()) { verdict.status = ParseStatus::Unparseable; pos = input.size(); return; }
                    pos += 2;
                } else {
                    size_t start = pos;
                    pos += 2;
                    int depth_parens = 1;
                    while (pos < input.size() && depth_parens > 0) {
                        char sc = input[pos];
                        if (sc == '\\') pos += 2;
                        else if (sc == '\'') {
                            pos++;
                            while (pos < input.size() && input[pos] != '\'') pos++;
                            pos++;
                        } else if (sc == '"') {
                            pos++;
                            while (pos < input.size() && input[pos] != '"') {
                                if (input[pos] == '\\') pos += 2;
                                else pos++;
                            }
                            pos++;
                        } else if (sc == '(') { depth_parens++; pos++; }
                        else if (sc == ')') { depth_parens--; pos++; }
                        else pos++;
                    }
                    if (depth_parens > 0) { verdict.status = ParseStatus::Unparseable; return; }
                    std::string_view sub = input.substr(start + 2, (pos - 1) - (start + 2));
                    CommandContext sub_ctx = ctx;
                    sub_ctx.command = sub;
                    Verdict sub_v = classify_impl(sub_ctx, depth + 1);
                    merge_verdict(verdict, sub_v);
                }
            } else if (pos + 1 < input.size() && input[pos+1] == '{') {
                pos += 2;
                while (pos < input.size() && input[pos] != '}') pos++;
                if (pos < input.size()) pos++;
                else verdict.status = ParseStatus::Unparseable;
            } else {
                pos++;
                while (pos < input.size() && is_alphanum_var(input[pos])) pos++;
            }
        }
    }

    Token parse_operator() {
        char c = input[pos];
        if (c == '|') {
            pos++;
            if (pos < input.size() && input[pos] == '|') { pos++; return {TokenType::LogicalOr, "||"}; }
            return {TokenType::Pipe, "|"};
        } else if (c == '&') {
            pos++;
            if (pos < input.size() && input[pos] == '&') { pos++; return {TokenType::LogicalAnd, "&&"}; }
            return {TokenType::Background, "&"};
        } else if (c == ';') {
            pos++;
            if (pos < input.size() && input[pos] == ';') { pos++; return {TokenType::SemiSemi, ";;"}; }
            return {TokenType::Semi, ";"};
        } else if (c == '<') {
            pos++;
            if (pos < input.size() && input[pos] == '&') { pos++; return {TokenType::RedirInAnd, "<&"}; }
            if (pos < input.size() && input[pos] == '<') { pos++; return {TokenType::RedirHere, "<<"}; }
            return {TokenType::RedirIn, "<"};
        } else if (c == '>') {
            pos++;
            if (pos < input.size() && input[pos] == '>') { pos++; return {TokenType::RedirAppend, ">>"}; }
            if (pos < input.size() && input[pos] == '&') { pos++; return {TokenType::RedirOutAnd, ">&"}; }
            if (pos < input.size() && input[pos] == '|') { pos++; return {TokenType::RedirOutClobber, ">|"}; }
            return {TokenType::RedirOut, ">"};
        } else if (c == '(') {
            pos++;
            return {TokenType::LParen, "("};
        } else if (c == ')') {
            pos++;
            return {TokenType::RParen, ")"};
        }
        return {TokenType::Eof, ""};
    }

public:
    Lexer(CommandContext ctx, Verdict& v, int depth) : input(ctx.command), verdict(v), ctx(ctx), depth(depth) {}

    Token next_token() {
        while (pos < input.size() && (input[pos] == ' ' || input[pos] == '\t' || input[pos] == '\n' || input[pos] == '\r')) {
            pos++;
        }

        if (pos >= input.size()) return {TokenType::Eof, "", false};

        if (input[pos] == '#') {
            while (pos < input.size() && input[pos] != '\n') pos++;
            return next_token();
        }

        char c = input[pos];
        if (c == '|' || c == '&' || c == ';' || c == '<' || c == '>' || c == '(' || c == ')') {
            return parse_operator();
        }

        word_buf.clear();
        bool is_dynamic = false;

        while (pos < input.size()) {
            c = input[pos];
            if (c == ' ' || c == '\t' || c == '\n' || c == '\r') break;
            if (c == '|' || c == '&' || c == ';' || c == '<' || c == '>' || c == '(' || c == ')') break;

            if (c == '\'') {
                pos++;
                while (pos < input.size() && input[pos] != '\'') {
                    append_char(input[pos]);
                    pos++;
                }
                if (pos >= input.size()) {
                    verdict.status = ParseStatus::Unparseable;
                    break;
                }
                pos++;
            } else if (c == '"') {
                pos++;
                while (pos < input.size() && input[pos] != '"') {
                    if (input[pos] == '\\') {
                        pos++;
                        if (pos < input.size()) {
                            append_char(input[pos]);
                            pos++;
                        }
                    } else if (input[pos] == '$' || input[pos] == '`') {
                        is_dynamic = true;
                        if (verdict.status != ParseStatus::Unparseable) verdict.status = ParseStatus::PartiallyParsed;
                        handle_substitution();
                    } else {
                        append_char(input[pos]);
                        pos++;
                    }
                }
                if (pos >= input.size()) {
                    verdict.status = ParseStatus::Unparseable;
                    break;
                }
                pos++;
            } else if (c == '\\') {
                pos++;
                if (pos < input.size()) {
                    if (input[pos] != '\n') {
                        append_char(input[pos]);
                    }
                    pos++;
                }
            } else if (c == '$' || c == '`') {
                is_dynamic = true;
                if (verdict.status != ParseStatus::Unparseable) verdict.status = ParseStatus::PartiallyParsed;
                handle_substitution();
            } else {
                append_char(c);
                pos++;
            }
        }

        return {TokenType::Word, word_buf, is_dynamic};
    }
};

class PathNormalizer {
public:
    static bool is_absolute(std::string_view path) {
        return !path.empty() && path[0] == '/';
    }

    static bool path_escapes(std::string_view base, std::string_view cwd, std::string_view path) {
        char resolved[8192];
        size_t r_len = 0;

        auto append = [&](std::string_view part) {
            if (part == "." || part.empty()) return;
            if (part == "..") {
                while (r_len > 0 && resolved[r_len - 1] != '/') r_len--;
                if (r_len > 1) r_len--;
                else if (r_len == 1) {}
            } else {
                if (r_len > 0 && resolved[r_len - 1] != '/' && r_len < sizeof(resolved)) resolved[r_len++] = '/';
                for (char c : part) {
                    if (r_len < sizeof(resolved)) resolved[r_len++] = c;
                }
            }
        };

        if (is_absolute(path)) {
            resolved[0] = '/';
            r_len = 1;
        } else {
            std::string_view start_dir = is_absolute(cwd) ? cwd : base;
            for (char c : start_dir) {
                if (r_len < sizeof(resolved)) resolved[r_len++] = c;
            }
        }

        size_t p = 0;
        if (is_absolute(path)) p = 1;

        while (p < path.size()) {
            size_t next = path.find('/', p);
            if (next == std::string_view::npos) {
                append(path.substr(p));
                break;
            } else {
                append(path.substr(p, next - p));
                p = next + 1;
            }
        }

        std::string_view result(resolved, r_len);
        if (result.size() < base.size()) return true;
        if (result.substr(0, base.size()) != base) return true;
        if (result.size() > base.size() && result[base.size()] != '/') return true;
        return false;
    }
};

class Classifier {
public:
    static void classify_command(const CommandContext& ctx, Verdict& verdict, std::span<const Token> tokens, int depth) {
        if (tokens.empty()) return;

        std::string_view cmd = tokens[0].text;

        if (cmd == "sudo" || cmd == "su" || cmd == "doas" || cmd == "xargs") {
            if (cmd != "xargs") verdict.capabilities.escalates_privileges = true;
            size_t start_idx = 1;
            while (start_idx < tokens.size() && tokens[start_idx].text.starts_with("-")) {
                if (cmd == "xargs" && (tokens[start_idx].text == "-I" || tokens[start_idx].text == "-i" || tokens[start_idx].text == "-a" || tokens[start_idx].text == "-E" || tokens[start_idx].text == "-L" || tokens[start_idx].text == "-n" || tokens[start_idx].text == "-P" || tokens[start_idx].text == "-s" || tokens[start_idx].text == "-x") && start_idx + 1 < tokens.size()) start_idx++;
                start_idx++;
            }
            if (start_idx < tokens.size()) {
                classify_command(ctx, verdict, tokens.subspan(start_idx), depth);
            }
        } else if (cmd == "env" || cmd == "time" || cmd == "watch") {
            size_t start_idx = 1;
            while (start_idx < tokens.size() && (tokens[start_idx].text.starts_with("-") || tokens[start_idx].text.find('=') != std::string::npos)) {
                start_idx++;
            }
            if (start_idx < tokens.size()) {
                classify_command(ctx, verdict, tokens.subspan(start_idx), depth);
            }
        } else if (cmd == "rm" || cmd == "unlink" || cmd == "shred" || cmd == "wipe") {
            verdict.capabilities.destroys_data = true;
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (!tokens[i].text.empty() && tokens[i].text[0] != '-') {
                    if (PathNormalizer::path_escapes(ctx.workspace_root, ctx.cwd, tokens[i].text)) {
                        verdict.capabilities.writes_outside_workspace = true;
                    }
                }
            }
        } else if (cmd == "cp" || cmd == "mv") {
            if (tokens.size() > 2) {
                std::string_view target = tokens.back().text;
                if (!target.empty() && target[0] != '-') {
                    if (PathNormalizer::path_escapes(ctx.workspace_root, ctx.cwd, target)) {
                        verdict.capabilities.writes_outside_workspace = true;
                    }
                }
            }
            if (cmd == "mv") verdict.capabilities.destroys_data = true;
        } else if (cmd == "git") {
            if (tokens.size() > 1) {
                std::string_view subcmd = tokens[1].text;
                if (subcmd == "push" && tokens.size() > 2) {
                    for (size_t i = 2; i < tokens.size(); ++i) {
                        if (tokens[i].text == "-f" || tokens[i].text == "--force") {
                            verdict.capabilities.rewrites_vcs_history = true;
                        }
                    }
                }
                if (subcmd == "reset") {
                    for (size_t i = 2; i < tokens.size(); ++i) {
                        if (tokens[i].text == "--hard") {
                            verdict.capabilities.destroys_data = true;
                        }
                    }
                }
                if (subcmd == "clean") {
                    for (size_t i = 2; i < tokens.size(); ++i) {
                        if (tokens[i].text.find('f') != std::string::npos) {
                            verdict.capabilities.destroys_data = true;
                        }
                    }
                }
                if (subcmd == "clone" || subcmd == "fetch" || subcmd == "pull" || subcmd == "push") {
                    verdict.capabilities.network_access = true;
                }
            }
        } else if (cmd == "curl" || cmd == "wget" || cmd == "nc" || cmd == "netcat" || cmd == "ping" || cmd == "ssh" || cmd == "scp") {
            verdict.capabilities.network_access = true;
        } else if (cmd == "kill" || cmd == "killall" || cmd == "pkill") {
            verdict.capabilities.signals_foreign_process = true;
        } else if (cmd == "nohup" || cmd == "setsid" || cmd == "screen" || cmd == "tmux") {
            verdict.capabilities.spawns_unbounded_process = true;
            if (tokens.size() > 1 && !tokens[1].text.empty()) {
                classify_command(ctx, verdict, tokens.subspan(1), depth);
            }
        } else if (cmd == "make" || cmd == "npm" || cmd == "yarn" || cmd == "sh" || cmd == "bash" || cmd == "zsh" || cmd == "npx" || cmd == "pnpm") {
            if (cmd == "sh" || cmd == "bash" || cmd == "zsh") {
                for (size_t i = 1; i < tokens.size(); ++i) {
                    if (tokens[i].text == "-c" && i + 1 < tokens.size()) {
                        CommandContext sub_ctx = ctx;
                        sub_ctx.command = tokens[i+1].text;
                        Verdict sub_v = blast_radius::classify_impl(sub_ctx, depth + 1);
                        merge_verdict(verdict, sub_v);
                        break;
                    }
                }
            } else if (cmd == "npm" || cmd == "yarn" || cmd == "pnpm") {
                bool is_run = false;
                for (size_t i = 1; i < tokens.size(); ++i) {
                    if (tokens[i].text == "run" || tokens[i].text == "run-script") is_run = true;
                }
                if (is_run || tokens.size() == 1) {
                    verdict.status = ParseStatus::PartiallyParsed;
                }
            } else if (cmd == "make") {
                verdict.status = ParseStatus::PartiallyParsed;
            }
        }

        for (size_t i = 1; i < tokens.size(); ++i) {
            std::string_view arg = tokens[i].text;
            if (!arg.empty() && arg[0] != '-') {
                if (PathNormalizer::path_escapes(ctx.workspace_root, ctx.cwd, arg)) {
                    verdict.capabilities.reads_outside_workspace = true;
                }
            }
        }
    }
};

[[nodiscard]] inline Verdict classify_impl(const CommandContext& ctx, int depth) noexcept {
    Verdict verdict;
    if (depth > 32) {
        verdict.status = ParseStatus::Unparseable;
        return verdict;
    }
    try {
        Lexer lexer(ctx, verdict, depth);

        std::vector<Token> tokens;

        Token t;
        while (true) {
            t = lexer.next_token();
            if (t.type == TokenType::Eof) {
                Classifier::classify_command(ctx, verdict, tokens, depth);
                break;
            }

            if (t.type == TokenType::Semi || t.type == TokenType::LogicalAnd || t.type == TokenType::LogicalOr || t.type == TokenType::Pipe || t.type == TokenType::Background) {
                Classifier::classify_command(ctx, verdict, tokens, depth);
                tokens.clear();
                if (t.type == TokenType::Background) {
                    verdict.capabilities.spawns_unbounded_process = true;
                }
            } else if (t.type == TokenType::RedirOut || t.type == TokenType::RedirOutClobber) {
                verdict.capabilities.destroys_data = true;
                Token target = lexer.next_token();
                if (target.type == TokenType::Word) {
                    if (PathNormalizer::path_escapes(ctx.workspace_root, ctx.cwd, target.text)) {
                        verdict.capabilities.writes_outside_workspace = true;
                    }
                }
            } else if (t.type == TokenType::RedirAppend) {
                Token target = lexer.next_token();
                if (target.type == TokenType::Word) {
                    if (PathNormalizer::path_escapes(ctx.workspace_root, ctx.cwd, target.text)) {
                        verdict.capabilities.writes_outside_workspace = true;
                    }
                }
            } else if (t.type == TokenType::RedirIn) {
                Token target = lexer.next_token();
                if (target.type == TokenType::Word) {
                    if (PathNormalizer::path_escapes(ctx.workspace_root, ctx.cwd, target.text)) {
                        verdict.capabilities.reads_outside_workspace = true;
                    }
                }
            } else if (t.type == TokenType::Word) {
                tokens.push_back(std::move(t));
            }
        }
    } catch (...) {
        verdict.status = ParseStatus::Unparseable;
    }
    return verdict;
}

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    return classify_impl(ctx, 0);
}

} // namespace blast_radius

#endif // BLAST_RADIUS_HPP
