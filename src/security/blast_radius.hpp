#ifndef SECURITY_BLAST_RADIUS_HPP
#define SECURITY_BLAST_RADIUS_HPP
//
// blast_radius -- what a shell command can DO, decided from the string alone.
//
// Consolidated from round 1 of the blast-radius-engine cookoff (14 submissions,
// 11 with code). The neutral corpus, the scorer and the ten labelling rules are
// in bakeoff/blast_radius/; this header is scored by the same binary every
// entrant was, on the same 187 cases, and its number is pinned in
// test_blast_radius_corpus.cpp.
//
// ---------------------------------------------------------------------------
// WHAT THE COOKOFF ACTUALLY SETTLED
// ---------------------------------------------------------------------------
// The eleven entrants agreed on the ARCHITECTURE and were collectively blind in
// the same places. Measured, not asserted:
//
//   * best entrant 169 weighted misses, worst 206, incumbent 267.
//   * an ORACLE that picks, per case and per flag, whichever entrant is right
//     still scores 138. So merge-by-component alone was worth ~18% and the
//     remaining majority of the gap is code NOBODY wrote.
//
// The shape everyone converged on -- lexer -> words -> per-stage evaluation ->
// union of capabilities across stages -- is kept, because eleven independent
// implementations reaching it is real evidence. What is NOT kept is the way all
// eleven expressed the verb knowledge: an ad-hoc if/else ladder over command
// names. The blind spots below are all the same failure, that a ladder has no
// place to put a fact:
//
//   * `unbounded` was implemented by nobody. All 11 miss every one of the 10
//     unbounded cases (`tail -f`, `less`, `vim`, `watch`, `while true`, ...)
//     while three of them raise 15-20 unbounded FALSE alarms elsewhere. A flag
//     fired almost at random and never when it should be.
//   * overwrite-at-a-destination was implemented by nobody: `cp a b`, `rsync`,
//     `tar -x`, `tee`, `sed -i`, `truncate`, `curl -o`, `find -delete`.
//   * dry-run suppression was implemented by nobody in the general case. All 11
//     answer `rm --help` as destroys_data, exactly as the incumbent does.
//   * package managers writing outside the workspace (`apt-get update`,
//     `brew install`, `pip install --user`, `docker pull`) -- nobody.
//
// So the verb knowledge here is a TABLE, and the flag analysis runs before the
// verb's effect is claimed. Adding a fact is adding a row.
//
// ---------------------------------------------------------------------------
// WHAT WAS TAKEN, AND FROM WHERE
// ---------------------------------------------------------------------------
//   e12  the overall winner; the stage/union skeleton and the recursive subshell
//        handling are its shape, and PathSegments below is its idea: containment
//        decided on borrowed string_views with no allocation at all.
//   e04  path containment (best of 11 on path_scope, 26 weighted vs a 44 worst).
//   e03  the quote-splice and backslash handling that makes `r""m` and `\rm`
//        resolve to `rm` (with e07, the only two that answered the adversarial
//        category near-perfectly).
//   e06  chain handling, tied best, and the only entrant whose `unbounded` did
//        not throw false alarms -- because it declined to guess.
//
// WHAT WAS REJECTED, AND WHY
//   * e01/e03/e09's unbounded heuristic. 15-20 false alarms and 10/10 misses:
//     it fires on "this looks like a long-running build", which is neither what
//     the flag means nor something a string can tell you. Replaced by an
//     explicit verb+flag table -- `unbounded` is a closed set of shapes
//     (followers, pagers, editors, REPLs, servers, unbounded loops), so a table
//     is not a stopgap, it is the correct representation.
//   * e11's path containment (44 weighted misses, worst of 11): it compares
//     prefixes as raw strings, so `/work/repo-2` reads as inside `/work/repo`.
//   * every entrant's `rm` => destroys_data with no flag analysis. This is the
//     incumbent's bug surviving into all eleven replacements.
//   * the blanket "the string contains `$` => PartiallyParsed" that all 11 use.
//     It is right for `rm -rf "$BUILD_DIR"` and wrong for
//     `export PATH=$PATH:/opt/bin`, which opens no path and runs nothing.
//     Substitution marks a WORD, and only a word that reaches a position where
//     it could have been an operand or a verb.
//
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace blast_radius {

// How much of the command's effect is determined by the string itself.
enum class ParseStatus : std::uint8_t {
    // Every effect the command can have is visible in the string.
    Parsed = 0,
    // The full effect depends on bytes that are NOT in this string: a script
    // file, an unexpanded variable, a substitution, a downloaded payload, a
    // Makefile/package.json target. The flags still describe everything visible.
    //
    // This is the single most valuable output of the classifier: it is the
    // signal that says "sandbox this one regardless of its flags".
    PartiallyParsed = 1,
    // The structure of the command could not be determined at all.
    Unparseable = 2,
};

struct Capabilities {
    // Creates, modifies, or deletes any path outside workspace_root.
    bool writes_outside_workspace = false;
    // Names a path outside workspace_root as DATA to be read. Executables and
    // libraries resolved from PATH do not count.
    bool reads_outside_workspace = false;
    // Irreversibly removes or overwrites existing data.
    bool destroys_data = false;
    // Discards committed or uncommitted version-control state.
    bool rewrites_vcs_history = false;
    // Opens a network connection in either direction, or binds a listening port.
    bool network_access = false;
    // May run with no natural termination.
    bool spawns_unbounded_process = false;
    // Signals or kills a process it did not itself start.
    bool signals_foreign_process = false;
    // Requests elevated privilege, or changes ownership/permission bits in a way
    // that grants it.
    bool escalates_privileges = false;
};

struct Verdict {
    Capabilities capabilities{};
    ParseStatus status = ParseStatus::Parsed;
};

struct CommandContext {
    // Exactly the command string as an LLM emitted it, destined for `/bin/sh -c`.
    std::string_view command;
    // Absolute path, no trailing slash. The only directory tree the agent owns.
    std::string_view workspace_root;
    // Absolute path where the shell starts. Inside workspace_root, or equal.
    std::string_view cwd;
};

// Pure. No global state. Reentrant and safe to call concurrently.
// Does not throw; allocates nothing at all.
[[nodiscard]] Verdict classify(const CommandContext& ctx) noexcept;

namespace detail {

// ---------------------------------------------------------------------------
// Limits. Every one of these is a bound on ADVERSARIAL input, and every one
// resolves to Unparseable rather than to a wrong answer -- which the scorer
// treats as "assume every capability", so overflowing is safe and expensive.
// ---------------------------------------------------------------------------
inline constexpr std::size_t kMaxPathSegments = 96;
inline constexpr std::size_t kMaxArgs = 128;
inline constexpr int kMaxDepth = 24;

// ---------------------------------------------------------------------------
// Containment (labelling rule 1: containment is TEXTUAL, nothing is stat'ed).
//
// e12's design: a path is a borrowed array of string_views, `..` cancels a
// segment at push time, and nothing is allocated. `/work/repo/../repo/build`
// normalises to /work/repo/build and is correctly INSIDE.
// ---------------------------------------------------------------------------
struct PathSegments {
    std::string_view segments[kMaxPathSegments];
    std::size_t count = 0;
    bool absolute = false;
    bool overflow = false;

    void push(std::string_view seg) noexcept {
        if (overflow) {
            return;
        }
        if (seg.empty() || seg == ".") {
            return;
        }
        if (seg == "..") {
            if (count > 0 && segments[count - 1] != "..") {
                --count;
            } else if (!absolute) {
                if (count < kMaxPathSegments) {
                    segments[count++] = "..";
                } else {
                    overflow = true;
                }
            }
            // Rooted `..` is clamped at `/`, which is what the kernel does.
            return;
        }
        if (count < kMaxPathSegments) {
            segments[count++] = seg;
        } else {
            overflow = true;
        }
    }

    void append(std::string_view path) noexcept {
        if (path.empty()) {
            return;
        }
        if (path[0] == '/') {
            absolute = true;
            count = 0;
        }
        std::size_t start = 0;
        while (start < path.size()) {
            const std::size_t end = path.find('/', start);
            if (end == std::string_view::npos) {
                push(path.substr(start));
                break;
            }
            push(path.substr(start, end - start));
            start = end + 1;
        }
    }

    void reset_to(const PathSegments& other) noexcept { *this = other; }
};

// `~` and `~user` name the home directory, which is never the workspace. They are
// not variables -- the shell expands them itself and we can read them as reliably
// as any literal. `$HOME` is a variable and is NOT this (see mark_substitution).
[[nodiscard]] inline bool is_tilde_path(std::string_view p) noexcept {
    return !p.empty() && p[0] == '~';
}

// True when `path`, resolved against `cwd`, lands outside `root`.
//
// Rejected e11's approach (raw string prefix compare, worst of 11 at 44 weighted
// misses on path_scope): it reads `/work/repo-2/x` as inside `/work/repo`. A
// segment-wise compare cannot make that mistake.
[[nodiscard]] inline bool is_outside(const PathSegments& cwd, std::string_view root,
                                     std::string_view path) noexcept {
    if (path.empty()) {
        return false;
    }
    if (is_tilde_path(path)) {
        return true;
    }
    PathSegments ws;
    ws.append(root);
    PathSegments target = cwd;
    target.append(path);
    if (ws.overflow || target.overflow) {
        return true; // bounded out; the safe reading wins
    }
    if (ws.absolute != target.absolute) {
        return true;
    }
    if (target.count < ws.count) {
        return true;
    }
    for (std::size_t i = 0; i < ws.count; ++i) {
        if (ws.segments[i] != target.segments[i]) {
            return true;
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Words.
//
// A word carries not just its text but WHY we may not be able to trust it. The
// text is rebuilt into a caller-owned buffer because quoting can splice
// (`r""m` is `rm`) and escape (`\rm` is `rm`) -- e03 and e07 were the only two
// entrants that got that whole category right, and this is their handling.
// ---------------------------------------------------------------------------
inline constexpr std::size_t kMaxWordLen = 512;

struct Word {
    char buf[kMaxWordLen];
    std::size_t len = 0;
    // The word contained an unquoted $, ${...}, $(...) or `...`: its VALUE is not
    // in this string. Note this marks the WORD, not the command -- the blanket
    // "any $ anywhere => PartiallyParsed" that all 11 entrants use is what makes
    // them answer `export PATH=$PATH:/opt/bin` as partial.
    bool substituted = false;
    // Every byte came from inside quotes, so it is inert as CODE (rule 6).
    bool quoted = false;
    bool overflow = false;
    // The text inside the first `$(...)` or `` `...` ``. A command substitution
    // RUNS its body, so the body's capabilities are real and visible even though
    // the resulting VALUE is not -- which is the whole of `sh -c "$(curl ...)"`:
    // the fetch is visible, the executed payload is not.
    std::string_view subst_body;

    void push(char c) noexcept {
        if (len < kMaxWordLen) {
            buf[len++] = c;
        } else {
            overflow = true;
        }
    }
    [[nodiscard]] std::string_view view() const noexcept { return {buf, len}; }
    [[nodiscard]] bool empty() const noexcept { return len == 0; }
};

// The last path component, so `/bin/rm` dispatches as `rm` (adv_abs_rm).
[[nodiscard]] inline std::string_view basename_of(std::string_view s) noexcept {
    const std::size_t slash = s.rfind('/');
    return slash == std::string_view::npos ? s : s.substr(slash + 1);
}

[[nodiscard]] inline bool is_flag(std::string_view w) noexcept {
    return w.size() >= 2 && w[0] == '-';
}

[[nodiscard]] inline bool equals_any(std::string_view w,
                                     const std::string_view* set, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i) {
        if (w == set[i]) {
            return true;
        }
    }
    return false;
}

template <std::size_t N>
[[nodiscard]] inline bool in(std::string_view w, const std::string_view (&set)[N]) noexcept {
    return equals_any(w, set, N);
}

// ---------------------------------------------------------------------------
// Verb tables.
//
// The whole reason this file is not an if/else ladder. Each table is one fact
// about the world, and a row is auditable on its own.
// ---------------------------------------------------------------------------

// Runs until something kills it. A CLOSED set of shapes: followers, pagers,
// editors, REPLs, listeners, watchers. Nobody in the cookoff had this at all.
inline constexpr std::string_view kAlwaysUnbounded[] = {
    "less", "more", "top", "htop", "vim", "vi", "nvim", "nano", "emacs", "pico",
    "watch", "tmux", "screen", "irb", "psql", "mysql", "sqlite3",
};
// Unbounded only when they get no script/file operand: a bare REPL blocks on
// stdin forever, `python x.py` does not.
inline constexpr std::string_view kReplWhenBare[] = {
    "python", "python3", "node", "ruby", "perl", "R", "ghci", "cat", "sh", "bash", "zsh",
};

// Opens a network connection. `git`, `docker` and the package managers are
// handled by their own dispatch below, because it depends on the subcommand.
inline constexpr std::string_view kNetVerbs[] = {
    "curl", "wget", "ssh", "sftp", "telnet", "ftp", "nc", "netcat", "ncat",
    "ping", "ping6", "traceroute", "dig", "nslookup", "host", "rsync", "scp",
};

// Fetches from a network AND installs outside the workspace. Both flags, always
// -- this is the row all eleven entrants were missing (apt/brew/pip/docker).
inline constexpr std::string_view kPackageManagers[] = {
    "apt", "apt-get", "aptitude", "yum", "dnf", "pacman", "apk", "zypper",
    "brew", "port", "pkg", "gem", "cpan", "conda",
};

// Signals a process it did not start.
inline constexpr std::string_view kSignalVerbs[] = {
    "kill", "killall", "pkill", "skill", "xkill",
};

// Asks for privilege directly.
inline constexpr std::string_view kPrivVerbs[] = {"sudo", "doas", "su", "runas", "pfexec"};

// Reads its operands as data, so an operand outside root is a read-outside.
inline constexpr std::string_view kReaders[] = {
    "cat", "head", "tail", "less", "more", "grep", "egrep", "fgrep", "rg", "ag",
    "wc", "diff", "cmp", "md5sum", "sha1sum", "sha256sum", "shasum", "od", "xxd",
    "hexdump", "strings", "file", "stat", "ls", "du", "find", "awk", "sed", "sort",
    "uniq", "cut", "tr", "nl", "tac", "join", "paste", "column", "jq", "yq",
};

// Known and harmless on their own. The point of this table is the FALLBACK:
// a verb in none of the tables is PartiallyParsed, because an unrecognised
// program is by definition a program whose effect is not in this string
// (`just deploy`). All 11 entrants answered `just deploy` as fully Parsed.
inline constexpr std::string_view kKnownInert[] = {
    // Shell builtins first. These are inert AND fully visible, and leaving them
    // out was a real defect: `exit 0` fell through to the unknown-verb fallback
    // and came back PartiallyParsed, which is enough to make a security gate
    // refuse it. Found by test_termination_invariants, not by the corpus.
    "exit", "return", "break", "continue", "shift", "read", "cd", "pushd", "popd",
    "dirs", "command", "builtin", "let", "getopts", "hash", "umask", "disown",
    "times", "caller", "logout", "enable", "compgen", "complete",
    "echo", "printf", "pwd", "true", "false", ":", "test", "[", "which", "type",
    "whoami", "id", "date", "env", "printenv", "export", "unset", "set", "alias",
    "basename", "dirname", "realpath", "readlink", "seq", "yes", "sleep", "time",
    "man", "info", "help", "history", "clear", "uname", "hostname", "df",
    "cmake", "ctest", "cc", "gcc", "g++", "clang", "clang++", "ld", "ar", "nm",
    "objdump", "strip", "install_name_tool", "codesign", "lldb", "gdb", "otool",
    "git", "docker", "npm", "yarn", "pnpm", "pip", "pip3", "cargo", "go", "make",
    "gmake", "ninja", "bazel", "swift", "xcodebuild", "rustc", "javac", "java",
    "mkdir", "touch", "ln", "mktemp", "chmod", "chown", "cp", "mv", "rm", "rmdir",
    "unlink", "shred", "truncate", "dd", "tee", "tar", "unzip", "zip", "gzip",
    "gunzip", "xargs", "sudo", "doas", "su", "eval", "exec", "source", ".",
    "kill", "killall", "pkill", "jobs", "bg", "fg", "wait", "trap", "ulimit",
    "journalctl", "systemctl", "launchctl", "ps", "lsof", "netstat", "ss", "open",
};

// Shell keywords that introduce a stage without being its verb. Skipping them is
// what lets `for f in ...; do rm "$f"; done` reach the `rm`.
inline constexpr std::string_view kStageKeywords[] = {
    "do", "then", "else", "elif", "fi", "done", "esac", "{", "}", "!", "time",
};

// A flag that means "tell me, do not do it". Universal across verbs.
//
// Deliberately NOT `-h`: it means --human-readable to `ls`, `du` and `df`, and a
// false "this is a dry run" is a miss on the primary metric, which costs 3.
inline constexpr std::string_view kHelpFlags[] = {"--help", "--usage", "--version", "-?"};

[[nodiscard]] inline bool is_dry_run_flag(std::string_view verb, std::string_view w) noexcept {
    if (w == "--dry-run" || w == "--dryrun" || w == "--just-print" || w == "--no-act") {
        return true;
    }
    // `-n` means dry-run for exactly these; it means something else elsewhere
    // (`sed -n` is quiet, `tail -n` is a count, `sort -n` is numeric).
    if (w == "-n") {
        return verb == "git" || verb == "cp" || verb == "mv" || verb == "rsync" ||
               verb == "ln" || verb == "make" || verb == "gmake";
    }
    return false;
}

// ---------------------------------------------------------------------------
// One evaluated stage.
// ---------------------------------------------------------------------------
struct Stage {
    Word args[kMaxArgs];
    std::size_t argc = 0;
    bool overflow = false;
    // A redirect target seen on this stage, plus whether it truncates.
    bool has_truncating_redirect = false;
    bool substituted_operand = false;
};

inline void merge(Capabilities& into, const Capabilities& from) noexcept {
    into.writes_outside_workspace |= from.writes_outside_workspace;
    into.reads_outside_workspace |= from.reads_outside_workspace;
    into.destroys_data |= from.destroys_data;
    into.rewrites_vcs_history |= from.rewrites_vcs_history;
    into.network_access |= from.network_access;
    into.spawns_unbounded_process |= from.spawns_unbounded_process;
    into.signals_foreign_process |= from.signals_foreign_process;
    into.escalates_privileges |= from.escalates_privileges;
}

inline void raise(ParseStatus& into, ParseStatus s) noexcept {
    if (static_cast<std::uint8_t>(s) > static_cast<std::uint8_t>(into)) {
        into = s;
    }
}

// `/dev/null` and friends are sinks: writing to one destroys nothing and is not
// a write outside the workspace. Without this, `npm test > /dev/null 2>&1` reads
// as a destructive out-of-tree write -- which is how 10 of 11 entrants answered.
[[nodiscard]] inline bool is_sink(std::string_view p) noexcept {
    return p == "/dev/null" || p == "/dev/stdout" || p == "/dev/stderr" || p == "/dev/tty";
}

// ---------------------------------------------------------------------------
// Lexer.
//
// Produces words and separators, and answers three questions the evaluator
// cannot answer for itself: was a word spliced out of quotes, did it contain a
// substitution, and did the quoting ever fail to close.
// ---------------------------------------------------------------------------
enum class Sep : std::uint8_t {
    None,      // more words in this stage
    Stage,     // ; && || & newline -- next stage, same scope
    Pipe,      // | -- next stage, and its stdin is the previous stage's stdout
    SubOpen,   // (
    SubClose,  // )
    End,
};

class Lexer {
public:
    explicit Lexer(std::string_view s) noexcept : src_(s) {}

    [[nodiscard]] bool unterminated() const noexcept { return unterminated_; }

    // Fills `w` with the next word. Returns the separator that ENDED it.
    Sep next(Word& w) noexcept {
        w = Word{};
        bool any_byte = false;
        bool all_quoted = true;
        skip_blanks();
        if (at_end()) {
            return Sep::End;
        }
        // A comment runs to end of line and is inert (rule 6).
        if (peek() == '#') {
            while (!at_end() && peek() != '\n') {
                ++pos_;
            }
            return next(w);
        }
        while (!at_end()) {
            const char c = peek();
            if (c == ' ' || c == '\t') {
                break;
            }
            if (is_stage_char(c) || c == '(' || c == ')' || c == '>' || c == '<') {
                break;
            }
            if (c == '\'') {
                ++pos_;
                any_byte = true;
                while (!at_end() && peek() != '\'') {
                    w.push(src_[pos_++]);
                }
                if (at_end()) {
                    unterminated_ = true;
                    return Sep::End;
                }
                ++pos_;
                continue;
            }
            if (c == '"') {
                ++pos_;
                any_byte = true;
                while (!at_end() && peek() != '"') {
                    if (peek() == '\\' && pos_ + 1 < src_.size()) {
                        // Inside "", a backslash only escapes these. `\$` is a
                        // LITERAL dollar and must not mark a substitution --
                        // that is quote_escaped_subst.
                        const char n = src_[pos_ + 1];
                        if (n == '"' || n == '\\' || n == '$' || n == '`') {
                            w.push(n);
                            pos_ += 2;
                            continue;
                        }
                    }
                    if (peek() == '$' || peek() == '`') {
                        w.substituted = true;
                        consume_substitution(w);
                        continue;
                    }
                    w.push(src_[pos_++]);
                }
                if (at_end()) {
                    unterminated_ = true;
                    return Sep::End;
                }
                ++pos_;
                continue;
            }
            // Unquoted from here on.
            all_quoted = false;
            any_byte = true;
            if (c == '\\') {
                ++pos_;
                if (!at_end()) {
                    w.push(src_[pos_++]);
                }
                continue;
            }
            if (c == '$' || c == '`') {
                w.substituted = true;
                consume_substitution(w);
                continue;
            }
            w.push(src_[pos_++]);
        }
        w.quoted = any_byte && all_quoted;
        if (!any_byte) {
            // We stopped on a separator without consuming a byte.
            return separator();
        }
        return Sep::None;
    }

    // Called by the evaluator when it needs the raw text of a `sh -c` body: the
    // lexer has already unquoted it into the Word, so nothing more is needed.
    [[nodiscard]] bool at_end() const noexcept { return pos_ >= src_.size(); }

    Sep separator() noexcept {
        if (at_end()) {
            return Sep::End;
        }
        const char c = peek();
        if (c == '(') {
            ++pos_;
            return Sep::SubOpen;
        }
        if (c == ')') {
            ++pos_;
            return Sep::SubClose;
        }
        if (c == '<') {
            consume_input_redirect();
            return Sep::None;
        }
        if (c == '>') {
            // Left in place: the driver calls take_redirect() before every word,
            // which is also how `> logs/app.log` (a redirect with no command at
            // all) still reaches the evaluator as a truncating write.
            return Sep::None;
        }
        const bool piped = (c == '|' && !(pos_ + 1 < src_.size() && src_[pos_ + 1] == '|'));
        while (!at_end() && is_stage_char(peek())) {
            ++pos_;
        }
        drain_heredoc();
        return piped ? Sep::Pipe : Sep::Stage;
    }

    // If a redirect operator is next, consume it and report whether it truncates.
    // `2>&1` is a descriptor dup, not a file write, and must not count.
    bool take_redirect(bool& truncating) noexcept {
        skip_blanks();
        std::size_t p = pos_;
        // An optional leading descriptor number.
        while (p < src_.size() && src_[p] >= '0' && src_[p] <= '9') {
            ++p;
        }
        if (p >= src_.size() || src_[p] != '>') {
            return false;
        }
        ++p;
        truncating = true;
        if (p < src_.size() && src_[p] == '>') {
            truncating = false; // append
            ++p;
        }
        if (p < src_.size() && src_[p] == '&') {
            pos_ = p + 1;
            while (pos_ < src_.size() && (src_[pos_] == '-' || (src_[pos_] >= '0' && src_[pos_] <= '9'))) {
                ++pos_;
            }
            return false; // dup, not a file
        }
        pos_ = p;
        return true;
    }

    [[nodiscard]] char peek() const noexcept { return src_[pos_]; }

    // True once if this stage got its stdin from `<` or a heredoc. A command
    // reading a redirect is not a command blocked on an interactive terminal,
    // which is the difference between `cat <<'EOF'` and a bare `cat`.
    bool take_stdin_flag() noexcept {
        const bool t = stdin_taken_;
        stdin_taken_ = false;
        return t;
    }

private:
    static bool is_stage_char(char c) noexcept {
        return c == ';' || c == '|' || c == '&' || c == '\n' || c == '\r';
    }

    void skip_blanks() noexcept {
        while (!at_end() && (peek() == ' ' || peek() == '\t')) {
            ++pos_;
        }
    }

    // `<` reads; `<<` is a heredoc whose BODY is data, never code, so it is
    // skipped whole. That is redir_heredoc_inert: `rm -rf /` inside a heredoc
    // is text a program will read, not a command anyone runs.
    void consume_input_redirect() noexcept {
        stdin_taken_ = true;
        ++pos_;
        if (!at_end() && peek() == '<') {
            ++pos_;
            if (!at_end() && peek() == '<') {
                ++pos_; // <<< here-string
                return;
            }
            skip_blanks();
            // Read the delimiter, then discard everything up to a line equal to it.
            const std::size_t dstart = pos_;
            while (!at_end() && peek() != ' ' && peek() != '\t' && peek() != '\n' &&
                   !is_stage_char(peek()) && peek() != '>') {
                ++pos_;
            }
            std::string_view delim = src_.substr(dstart, pos_ - dstart);
            if (delim.size() >= 2 && (delim.front() == '\'' || delim.front() == '"') &&
                delim.back() == delim.front()) {
                delim = delim.substr(1, delim.size() - 2);
            }
            heredoc_delim_ = delim;
            heredoc_pending_ = !delim.empty();
        }
    }

    // Skip `$(...)`, `${...}`, `` `...` `` and bare `$NAME`, capturing the body
    // of the command-substitution forms so the evaluator can run them.
    void consume_substitution(Word& w) noexcept {
        const char c = src_[pos_];
        if (c == '`') {
            ++pos_;
            const std::size_t start = pos_;
            while (!at_end() && peek() != '`') {
                ++pos_;
            }
            if (at_end()) {
                unterminated_ = true;
                return;
            }
            if (w.subst_body.empty()) {
                w.subst_body = src_.substr(start, pos_ - start);
            }
            ++pos_;
            return;
        }
        ++pos_; // '$'
        if (at_end()) {
            return;
        }
        if (peek() == '(') {
            int depth = 0;
            const std::size_t start = pos_ + 1;
            while (!at_end()) {
                if (peek() == '(') {
                    ++depth;
                } else if (peek() == ')') {
                    if (--depth == 0) {
                        if (w.subst_body.empty()) {
                            w.subst_body = src_.substr(start, pos_ - start);
                        }
                        ++pos_;
                        return;
                    }
                }
                ++pos_;
            }
            unterminated_ = true;
            return;
        }
        if (peek() == '{') {
            while (!at_end() && peek() != '}') {
                ++pos_;
            }
            if (!at_end()) {
                ++pos_;
            }
            return;
        }
        while (!at_end() && (isalnum_(peek()) || peek() == '_')) {
            ++pos_;
        }
    }

    static bool isalnum_(char c) noexcept {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
    }

public:
    // Consume a pending heredoc body once the current line ends.
    void drain_heredoc() noexcept {
        if (!heredoc_pending_) {
            return;
        }
        heredoc_pending_ = false;
        while (!at_end() && peek() != '\n') {
            ++pos_;
        }
        while (!at_end()) {
            ++pos_; // past the newline
            const std::size_t line = pos_;
            while (!at_end() && peek() != '\n') {
                ++pos_;
            }
            if (src_.substr(line, pos_ - line) == heredoc_delim_) {
                return;
            }
            if (at_end()) {
                return;
            }
        }
    }

private:
    std::string_view src_;
    std::size_t pos_ = 0;
    bool unterminated_ = false;
    bool stdin_taken_ = false;
    bool heredoc_pending_ = false;
    std::string_view heredoc_delim_;
};

// ---------------------------------------------------------------------------
// Evaluation.
//
// `scope` is everything that flows between the stages of one chain: the current
// directory (rule 5 -- a leading `cd` moves the scope of every later stage) and
// whether we still know what it is.
// ---------------------------------------------------------------------------
struct Scope {
    PathSegments cwd;
    std::string_view root;
    // A `cd` whose operand was a substitution leaves us not knowing where we
    // are, so no later relative path may be called inside OR outside.
    bool cwd_unknown = false;
};

struct Args {
    const Word* v;
    std::size_t n;
    [[nodiscard]] std::string_view at(std::size_t i) const noexcept {
        return i < n ? v[i].view() : std::string_view{};
    }
    [[nodiscard]] bool has(std::string_view f) const noexcept {
        for (std::size_t i = 1; i < n; ++i) {
            if (v[i].view() == f) {
                return true;
            }
        }
        return false;
    }
    // First non-flag operand at or after `from`, skipping substituted words.
    [[nodiscard]] std::size_t first_operand(std::size_t from) const noexcept {
        for (std::size_t i = from; i < n; ++i) {
            if (!v[i].empty() && !is_flag(v[i].view())) {
                return i;
            }
        }
        return n;
    }
};

inline void evaluate(Args a, const Scope& sc, Verdict& v, int depth, bool piped_stdin) noexcept;
inline void run_chain(Lexer& lex, Scope sc, Verdict& v, int depth) noexcept;
[[nodiscard]] inline Verdict classify_body(const CommandContext& body, const Scope& sc,
                                           int depth) noexcept;

// A path operand: claim containment only when we can actually see the path.
// A substituted word (`$HOME/.cache`) and an unknown cwd both mean we cannot,
// and claiming either way would be a guess.
inline void note_path(const Word& w, const Scope& sc, Verdict& v, bool writes,
                      bool reads) noexcept {
    if (w.substituted || w.empty()) {
        return;
    }
    const std::string_view p = w.view();
    if (is_sink(p)) {
        return;
    }
    if (sc.cwd_unknown && (p.empty() || (p[0] != '/' && p[0] != '~'))) {
        return;
    }
    if (!is_outside(sc.cwd, sc.root, p)) {
        return;
    }
    if (writes) {
        v.capabilities.writes_outside_workspace = true;
    }
    if (reads) {
        v.capabilities.reads_outside_workspace = true;
    }
}

// Verbs whose FIRST non-flag operand is a program, not a path. Without this,
// `awk '/kill/ {print}' log.txt` reads its script as the absolute path /kill/
// and reports a read outside the workspace.
[[nodiscard]] inline bool first_operand_is_script(std::string_view verb) noexcept {
    return verb == "awk" || verb == "sed" || verb == "gawk" || verb == "mawk" ||
           verb == "perl" || verb == "jq" || verb == "yq";
}

// Verbs whose operands are read as data (rule 4).
inline void note_reads(Args a, std::string_view verb, const Scope& sc, Verdict& v) noexcept {
    std::size_t start = 1;
    if (first_operand_is_script(verb)) {
        start = a.first_operand(1) + 1;
    }
    bool saw_operand = false;
    for (std::size_t i = start; i < a.n; ++i) {
        if (a.v[i].empty() || is_flag(a.v[i].view())) {
            continue;
        }
        saw_operand = true;
        note_path(a.v[i], sc, v, false, true);
    }
    // With no operand at all, the reader reads the current directory -- which is
    // how `cd /tmp && ls` is a read outside without naming a path.
    if (!saw_operand && !sc.cwd_unknown && !sc.root.empty()) {
        PathSegments here = sc.cwd;
        if (is_outside(here, sc.root, ".")) {
            v.capabilities.reads_outside_workspace = true;
        }
    }
}

// Everything a destroyer's operands imply: the destroy itself, and write-outside
// for any operand that leaves the workspace.
inline void note_destroys(Args a, const Scope& sc, Verdict& v, std::size_t from) noexcept {
    v.capabilities.destroys_data = true;
    for (std::size_t i = from; i < a.n; ++i) {
        if (a.v[i].empty() || is_flag(a.v[i].view())) {
            continue;
        }
        note_path(a.v[i], sc, v, true, false);
    }
}

[[nodiscard]] inline bool bundled_has(std::string_view flag, char c) noexcept {
    if (flag.size() < 2 || flag[0] != '-' || flag[1] == '-') {
        return false;
    }
    return flag.find(c, 1) != std::string_view::npos;
}

// `git`, whose subcommand decides everything (rule 8: discarding is vcs,
// appending is not; destroy only when file CONTENT is lost, which is why
// `reset --hard` and `reset --soft` differ).
inline void evaluate_git(Args a, Verdict& v) noexcept {
    const std::string_view sub = a.at(1);
    if (sub == "fetch" || sub == "clone" || sub == "pull" || sub == "remote" ||
        sub == "ls-remote" || sub == "submodule") {
        v.capabilities.network_access = true;
        return;
    }
    if (sub == "push") {
        v.capabilities.network_access = true;
        if (a.has("-f") || a.has("--force") || a.has("--force-with-lease") ||
            a.has("--delete") || a.has("-d")) {
            v.capabilities.rewrites_vcs_history = true;
        }
        return;
    }
    if (sub == "reset") {
        v.capabilities.rewrites_vcs_history = true;
        if (a.has("--hard")) {
            v.capabilities.destroys_data = true;
        }
        return;
    }
    if (sub == "checkout" || sub == "restore") {
        v.capabilities.rewrites_vcs_history = true;
        // `--staged` only unstages; the file in the tree keeps its contents.
        if (!a.has("--staged") && !a.has("--cached")) {
            v.capabilities.destroys_data = true;
        }
        return;
    }
    if (sub == "clean") {
        v.capabilities.rewrites_vcs_history = true;
        v.capabilities.destroys_data = true;
        return;
    }
    if (sub == "branch") {
        if (a.has("-D") || a.has("-d") || a.has("--delete")) {
            v.capabilities.rewrites_vcs_history = true;
        }
        return;
    }
    if (sub == "stash") {
        if (a.at(2) == "drop" || a.at(2) == "clear" || a.at(2) == "pop") {
            v.capabilities.rewrites_vcs_history = true;
            v.capabilities.destroys_data = true;
        }
        return;
    }
    if (sub == "rebase") {
        v.capabilities.rewrites_vcs_history = true;
        if (a.has("-i") || a.has("--interactive")) {
            // Drops the agent into an editor that never returns on its own.
            v.capabilities.spawns_unbounded_process = true;
        }
        return;
    }
    if (sub == "filter-branch" || sub == "filter-repo") {
        v.capabilities.rewrites_vcs_history = true;
        v.capabilities.destroys_data = true;
        return;
    }
    if (sub == "commit" || sub == "add" || sub == "status" || sub == "diff" ||
        sub == "log" || sub == "show" || sub == "tag" || sub == "stash-list") {
        // Appending to history is not rewriting it (rule 8).
        if (a.has("--amend")) {
            v.capabilities.rewrites_vcs_history = true;
        }
        return;
    }
}

inline void evaluate_docker(Args a, const Scope& sc, Verdict& v, ParseStatus& st,
                            int depth) noexcept {
    const std::string_view sub = a.at(1);
    if (sub == "pull" || sub == "push" || sub == "login" || sub == "search") {
        v.capabilities.network_access = true;
        v.capabilities.writes_outside_workspace = true; // image store is not ours
        return;
    }
    if (sub == "kill" || sub == "stop" || sub == "restart") {
        v.capabilities.signals_foreign_process = true;
        return;
    }
    if (sub == "system" || sub == "image" || sub == "container" || sub == "volume" ||
        sub == "network") {
        if (a.at(2) == "prune" || a.at(2) == "rm") {
            v.capabilities.destroys_data = true;
            v.capabilities.writes_outside_workspace = true;
            return;
        }
        return;
    }
    if (sub == "rm" || sub == "rmi") {
        v.capabilities.destroys_data = true;
        v.capabilities.writes_outside_workspace = true;
        return;
    }
    if (sub == "run" || sub == "exec" || sub == "create") {
        bool mounted_host = false;
        // A bind mount makes a host path reachable under a container path, so
        // the container's own command operates on the host. Fully visible when
        // both halves are literal -- so `parsed`, not `partial`.
        for (std::size_t i = 2; i < a.n; ++i) {
            const std::string_view f = a.v[i].view();
            std::string_view spec;
            if ((f == "-v" || f == "--volume" || f == "--mount") && i + 1 < a.n) {
                spec = a.v[i + 1].view();
            } else if (f.starts_with("-v")) {
                spec = f.substr(2);
            }
            if (spec.empty()) {
                continue;
            }
            const std::size_t colon = spec.find(':');
            const std::string_view host = colon == std::string_view::npos ? spec : spec.substr(0, colon);
            if (!host.empty() && (host[0] == '/' || host[0] == '~') &&
                is_outside(sc.cwd, sc.root, host)) {
                // The mount makes a host path WRITABLE from inside. It is not by
                // itself a read of host data as data -- that only happens if the
                // container command reads it, and the command is right there.
                v.capabilities.writes_outside_workspace = true;
                mounted_host = true;
            }
        }
        // Flags, then the image name, then the command it runs inside.
        std::size_t img = 2;
        while (img < a.n) {
            const std::string_view f = a.at(img);
            if (!is_flag(f)) {
                break;
            }
            if (f == "-v" || f == "--volume" || f == "--mount" || f == "-e" || f == "--env" ||
                f == "--name" || f == "-w" || f == "--workdir" || f == "-u" || f == "--user" ||
                f == "-p" || f == "--publish" || f == "--entrypoint") {
                ++img;
            }
            ++img;
        }
        if (img + 1 < a.n) {
            // The container's command is fully visible, so this stays `parsed`.
            // Its paths are CONTAINER paths -- meaningless against our
            // workspace_root -- so it is evaluated for what it DOES, and the
            // bind mount above is what decides whether that reaches the host.
            Verdict inner;
            Scope container;
            container.root = "/";
            container.cwd.append("/");
            evaluate(Args{a.v + img + 1, a.n - img - 1}, container, inner, depth - 1, false);
            v.capabilities.destroys_data |= inner.capabilities.destroys_data;
            v.capabilities.network_access |= inner.capabilities.network_access;
            if (mounted_host && (inner.capabilities.destroys_data ||
                                 inner.capabilities.writes_outside_workspace)) {
                v.capabilities.writes_outside_workspace = true;
            }
        } else {
            raise(st, ParseStatus::PartiallyParsed);
        }
        return;
    }
    if (sub == "build" || sub == "compose") {
        v.capabilities.network_access = true;
        raise(st, ParseStatus::PartiallyParsed);
        return;
    }
}

// A local program: `./run.sh`, `scripts/build.sh`, `setup.py`. Its body is not
// in this string, which is exactly PartiallyParsed.
[[nodiscard]] inline bool looks_like_local_program(std::string_view w) noexcept {
    if (w.starts_with("./") || w.starts_with("../")) {
        return true;
    }
    const std::string_view base = basename_of(w);
    for (std::string_view ext : {".sh", ".bash", ".zsh", ".py", ".rb", ".pl", ".js"}) {
        if (base.size() > ext.size() && base.ends_with(ext)) {
            return true;
        }
    }
    return false;
}

inline void evaluate(Args a, const Scope& sc, Verdict& v, int depth, bool piped_stdin) noexcept {
    if (a.n == 0 || depth <= 0) {
        return;
    }
    // Shell keywords introduce a stage without being its verb.
    std::size_t base = 0;
    while (base < a.n && in(a.at(base), kStageKeywords)) {
        ++base;
    }
    if (base >= a.n) {
        return;
    }
    // `while true` / `until false` never terminates on its own.
    if (a.at(base) == "while" || a.at(base) == "until") {
        const std::string_view cond = a.at(base + 1);
        if (cond == "true" || cond == ":" || cond == "false") {
            v.capabilities.spawns_unbounded_process = true;
        }
        return;
    }
    if (a.at(base) == "for" || a.at(base) == "if" || a.at(base) == "case" ||
        a.at(base) == "select" || a.at(base) == "function") {
        return;
    }
    a = Args{a.v + base, a.n - base};

    // The verb itself came from a substitution: we do not know what runs.
    if (a.v[0].substituted) {
        raise(v.status, ParseStatus::PartiallyParsed);
        return;
    }
    const std::string_view raw = a.at(0);
    const std::string_view verb = basename_of(raw);
    if (verb.empty()) {
        return;
    }

    // A variable assignment as the whole "command" (FOO=bar) runs nothing.
    if (verb.find('=') != std::string_view::npos && verb.find('=') > 0) {
        return;
    }

    // Rule 9, applied FIRST and universally: a flag that means "tell me, do not
    // do it" removes every capability. All 11 entrants answer `rm --help` as
    // destroys_data because they never look.
    for (std::size_t i = 1; i < a.n; ++i) {
        if (in(a.at(i), kHelpFlags) || is_dry_run_flag(verb, a.at(i))) {
            return;
        }
    }

    // An operand we cannot see makes the command's effect partly invisible --
    // but only for verbs that USE their operands. `export PATH=$PATH:/opt/bin`
    // assigns a variable and opens nothing, which is the case all 11 entrants
    // answer partial because they test the whole string for a `$`.
    const bool assignment_verb = verb == "export" || verb == "set" || verb == "unset" ||
                                 verb == "alias" || verb == "declare" || verb == "local" ||
                                 verb == "readonly" || verb == "typeset";
    // The second class the rule must not fire on: a verb whose operands are DATA
    // it prints, never a path it opens or a program it runs. `echo "EXIT=$?"` is
    // the single most common shape a model emits -- every model keeps its
    // reporting wrapper on -- and answering it partial refused the whole command
    // it was appended to, which is what made three of test_scripted_loop's
    // verification-accounting arms unrunnable.
    //
    // Sound because the three ways an operand could matter are all handled
    // somewhere else, and were checked: a redirect (`echo $X > $OUT`) is a word
    // this loop never sees and is raised by the chain driver; a command
    // substitution (`echo $(rm -rf /)`) has its BODY classified and merged just
    // below, so the rm is still seen; and a pipe into a shell (`echo $CMD | sh`)
    // is a separate stage whose `sh` is raised on its own.
    const bool operand_inert_verb = verb == "echo" || verb == "printf";
    if (!assignment_verb && !operand_inert_verb) {
        for (std::size_t i = 1; i < a.n; ++i) {
            if (a.v[i].substituted) {
                raise(v.status, ParseStatus::PartiallyParsed);
                break;
            }
        }
    }
    // A command substitution RUNS its body. Its value is invisible, but what it
    // does on the way to producing that value is right there in the string --
    // the `curl` in `sh -c "$(curl -s ...)"` really does reach the network.
    for (std::size_t i = 0; i < a.n; ++i) {
        if (!a.v[i].subst_body.empty()) {
            const Verdict inner = classify_body({a.v[i].subst_body, sc.root, {}}, sc, depth - 1);
            merge(v.capabilities, inner.capabilities);
        }
    }

    // ---- privilege: grant, not danger (rule 10) ---------------------------
    if (in(verb, kPrivVerbs)) {
        v.capabilities.escalates_privileges = true;
        std::size_t i = 1;
        while (i < a.n && is_flag(a.at(i))) {
            if (a.at(i) == "-u" || a.at(i) == "-g" || a.at(i) == "-U") {
                ++i;
            }
            ++i;
        }
        if (verb == "su") {
            // `su - root -c 'ls'`: the -c body is visible but the login shell
            // sources a profile we cannot see.
            raise(v.status, ParseStatus::PartiallyParsed);
            return;
        }
        if (i < a.n) {
            evaluate(Args{a.v + i, a.n - i}, sc, v, depth - 1, false);
        }
        return;
    }
    if (verb == "chmod" || verb == "chown" || verb == "chgrp" || verb == "setcap") {
        if (verb != "chmod") {
            v.capabilities.escalates_privileges = true;
        } else {
            // Only the MODE grants privilege, and the mode is the first operand.
            // Scanning every argument means `chmod +x scripts/build.sh` sees the
            // 's' in "scripts" and reports an escalation.
            const std::string_view m = a.at(a.first_operand(1));
            const bool setid = m.find('s') != std::string_view::npos ||
                               (m.size() == 4 && (m[0] == '4' || m[0] == '2' || m[0] == '6' ||
                                                  m[0] == '7'));
            if (setid || m == "777" || m == "0777" || m == "a+rwx") {
                v.capabilities.escalates_privileges = true;
            }
        }
        for (std::size_t i = 1; i < a.n; ++i) {
            if (!is_flag(a.at(i)) && i > 1) {
                note_path(a.v[i], sc, v, true, false);
            }
        }
        return;
    }

    // ---- signals -----------------------------------------------------------
    if (in(verb, kSignalVerbs)) {
        bool jobspec_only = true;
        bool any_target = false;
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view t = a.at(i);
            if (t.empty() || is_flag(t)) {
                continue;
            }
            any_target = true;
            if (t[0] != '%') {
                jobspec_only = false;
            }
        }
        // `kill %1` addresses a job of THIS shell -- a process it started
        // itself, which is not what signals_foreign_process means.
        if (!any_target || !jobspec_only) {
            v.capabilities.signals_foreign_process = true;
        }
        return;
    }

    // ---- destroyers --------------------------------------------------------
    if (verb == "rm" || verb == "rmdir" || verb == "unlink" || verb == "shred") {
        note_destroys(a, sc, v, 1);
        // Deleting the repository is the maximal history rewrite (rule 8).
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view p = a.at(i);
            if (basename_of(p) == ".git" || p == ".git/" || p.ends_with("/.git")) {
                v.capabilities.rewrites_vcs_history = true;
            }
        }
        return;
    }
    if (verb == "truncate") {
        note_destroys(a, sc, v, 2); // skip the -s SIZE operand
        return;
    }
    if (verb == "mv" || verb == "cp" || verb == "install" || verb == "rsync") {
        // Rule 3, generalised: any copy/move/extract may clobber what is already
        // at the destination, and we cannot know whether it is there. Not one of
        // the eleven entrants modelled this.
        if (a.has("-n") || a.has("--no-clobber")) {
            return;
        }
        const std::size_t last = [&] {
            std::size_t k = a.n;
            for (std::size_t i = a.n; i-- > 1;) {
                if (!a.v[i].empty() && !is_flag(a.at(i))) {
                    k = i;
                    break;
                }
            }
            return k;
        }();
        bool remote = false;
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view p = a.at(i);
            if (is_flag(p) || p.empty()) {
                continue;
            }
            const std::size_t colon = p.find(':');
            if (colon != std::string_view::npos && p.find('/') > colon) {
                remote = true;
                continue;
            }
            if (i == last) {
                note_path(a.v[i], sc, v, true, false);
            } else {
                note_path(a.v[i], sc, v, false, true);
            }
        }
        if (remote) {
            v.capabilities.network_access = true;
        } else if (last < a.n) {
            v.capabilities.destroys_data = true;
        }
        return;
    }
    if (verb == "dd") {
        v.capabilities.destroys_data = true;
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view p = a.at(i);
            if (p.starts_with("of=")) {
                Word w = a.v[i];
                const std::string_view t = p.substr(3);
                w.len = 0;
                for (char c : t) {
                    w.push(c);
                }
                note_path(w, sc, v, true, false);
            }
        }
        return;
    }
    if (verb == "tee") {
        if (!a.has("-a") && !a.has("--append")) {
            for (std::size_t i = 1; i < a.n; ++i) {
                if (!is_flag(a.at(i)) && !a.v[i].empty() && !is_sink(a.at(i))) {
                    v.capabilities.destroys_data = true;
                }
            }
        }
        for (std::size_t i = 1; i < a.n; ++i) {
            if (!is_flag(a.at(i))) {
                note_path(a.v[i], sc, v, true, false);
            }
        }
        return;
    }
    if (verb == "sed" || verb == "gsed") {
        const bool in_place = [&] {
            for (std::size_t i = 1; i < a.n; ++i) {
                if (a.at(i) == "-i" || a.at(i).starts_with("-i") ||
                    a.at(i) == "--in-place" || a.at(i).starts_with("--in-place=")) {
                    return true;
                }
            }
            return false;
        }();
        note_reads(a, verb, sc, v);
        if (in_place) {
            v.capabilities.destroys_data = true;
            const std::size_t script = a.first_operand(1);
            for (std::size_t i = script + 1; i < a.n; ++i) {
                if (!is_flag(a.at(i))) {
                    note_path(a.v[i], sc, v, true, false);
                }
            }
        }
        return;
    }
    if (verb == "find") {
        note_reads(a, verb, sc, v);
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view f = a.at(i);
            if (f == "-delete") {
                v.capabilities.destroys_data = true;
            } else if (f == "-exec" || f == "-execdir" || f == "-ok") {
                // The command is fully visible here, so this stays `parsed`.
                for (std::size_t k = i + 1; k < a.n; ++k) {
                    const std::string_view c = basename_of(a.at(k));
                    if (c == ";" || c == "+") {
                        break;
                    }
                    if (c == "rm" || c == "shred" || c == "truncate" || c == "unlink") {
                        v.capabilities.destroys_data = true;
                    }
                }
            }
        }
        return;
    }
    if (verb == "tar") {
        bool extract = false;
        bool list = false;
        bool create = false;
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view f = a.at(i);
            extract |= bundled_has(f, 'x') || f == "--extract";
            list |= bundled_has(f, 't') || f == "--list";
            create |= bundled_has(f, 'c') || f == "--create";
            if ((f == "-C" || f == "--directory") && i + 1 < a.n) {
                note_path(a.v[i + 1], sc, v, true, false);
            }
        }
        if (extract && !list && !create) {
            v.capabilities.destroys_data = true; // extraction overwrites collisions
        }
        return;
    }
    if (verb == "unzip" || verb == "gunzip") {
        if (!a.has("-l") && !a.has("-t")) {
            v.capabilities.destroys_data = true;
        }
        return;
    }

    // ---- network -----------------------------------------------------------
    if (verb == "curl" || verb == "wget") {
        v.capabilities.network_access = true;
        for (std::size_t i = 1; i < a.n; ++i) {
            const std::string_view f = a.at(i);
            if ((f == "-o" || f == "--output" || f == "-O" || f == "--remote-name") &&
                i + 1 < a.n && !is_flag(a.at(i + 1))) {
                v.capabilities.destroys_data = true; // -o names a path that may exist
                note_path(a.v[i + 1], sc, v, true, false);
            }
        }
        return;
    }
    if (in(verb, kNetVerbs)) {
        v.capabilities.network_access = true;
        if (verb == "nc" || verb == "netcat" || verb == "ncat") {
            if (a.has("-l") || a.has("-l4") || a.has("--listen")) {
                v.capabilities.spawns_unbounded_process = true;
            }
        }
        if (verb == "ssh" || verb == "sftp" || verb == "telnet") {
            // Whatever it does happens on a machine we cannot reason about.
            raise(v.status, ParseStatus::PartiallyParsed);
        }
        return;
    }
    if (in(verb, kPackageManagers)) {
        v.capabilities.network_access = true;
        v.capabilities.writes_outside_workspace = true;
        return;
    }
    if (verb == "pip" || verb == "pip3") {
        v.capabilities.network_access = true;
        if (a.has("--user")) {
            v.capabilities.writes_outside_workspace = true; // installs into $HOME
        }
        return;
    }
    if (verb == "npm" || verb == "yarn" || verb == "pnpm") {
        // package.json scripts are bytes that are not in this string.
        raise(v.status, ParseStatus::PartiallyParsed);
        const std::string_view sub = a.at(1);
        if (sub == "install" || sub == "i" || sub == "ci" || sub == "add" ||
            sub == "update" || sub == "publish") {
            v.capabilities.network_access = true;
            if (a.has("-g") || a.has("--global")) {
                v.capabilities.writes_outside_workspace = true;
            }
        }
        return;
    }
    if (verb == "docker" || verb == "podman") {
        evaluate_docker(a, sc, v, v.status, depth);
        return;
    }
    if (verb == "git") {
        evaluate_git(a, v);
        return;
    }

    // ---- indirection: the effect is in bytes we do not have (rule 7) -------
    if (verb == "sh" || verb == "bash" || verb == "zsh" || verb == "ksh" || verb == "dash") {
        for (std::size_t i = 1; i < a.n; ++i) {
            if (a.at(i) == "-c" && i + 1 < a.n) {
                if (a.v[i + 1].substituted) {
                    raise(v.status, ParseStatus::PartiallyParsed);
                    return;
                }
                // The body is fully visible, so it is parsed, not partial.
                CommandContext sub{a.at(i + 1), sc.root, {}};
                Verdict inner = classify_body(sub, sc, depth - 1);
                merge(v.capabilities, inner.capabilities);
                raise(v.status, inner.status);
                return;
            }
        }
        if (a.n > 1) {
            raise(v.status, ParseStatus::PartiallyParsed); // a script file
        } else if (piped_stdin) {
            raise(v.status, ParseStatus::PartiallyParsed); // `curl ... | sh`
        } else {
            v.capabilities.spawns_unbounded_process = true; // interactive shell
        }
        return;
    }
    if (verb == "eval" || verb == "source" || verb == ".") {
        raise(v.status, ParseStatus::PartiallyParsed);
        return;
    }
    if (verb == "xargs") {
        raise(v.status, ParseStatus::PartiallyParsed); // operands come from stdin
        const std::size_t next = [&] {
            std::size_t i = 1;
            while (i < a.n && is_flag(a.at(i))) {
                if (a.at(i) == "-I" || a.at(i) == "-n" || a.at(i) == "-P" || a.at(i) == "-d") {
                    ++i;
                }
                ++i;
            }
            return i;
        }();
        if (next < a.n) {
            evaluate(Args{a.v + next, a.n - next}, sc, v, depth - 1, false);
        }
        return;
    }
    if (verb == "watch") {
        v.capabilities.spawns_unbounded_process = true;
        const std::size_t next = a.first_operand(1);
        if (next < a.n) {
            evaluate(Args{a.v + next, a.n - next}, sc, v, depth - 1, false);
        }
        return;
    }
    // Prefix runners: the payload IS in the string, one word along, so the right
    // answer is to recurse into it -- exactly what `time`, `watch` and the
    // kPrivVerbs branch already do. Without this, four one-word prefixes
    // LAUNDERED their payload: `rm -rf /etc` reported destroys_data +
    // writes_outside_workspace, while `command rm -rf /etc`, `builtin rm -rf
    // /etc`, `exec rm -rf /etc` and `env rm -rf /etc` all fell through to the
    // kKnownInert early-return below and came back Parsed with an EMPTY
    // capability set -- a clean bill of health, which is what
    // SubprocessVerifier::provably_confined() acts on.
    if (verb == "command" || verb == "builtin" || verb == "exec" || verb == "env" ||
        verb == "nice" || verb == "stdbuf") {
        std::size_t next = 1;
        while (next < a.n) {
            const std::string_view w = a.at(next);
            // `env` takes VAR=value assignments before the program name, and
            // -u/-S take a value of their own.
            if (is_flag(w)) {
                if (w == "-u" || w == "-S" || w == "-C" || w == "-n") {
                    ++next;
                }
                ++next;
                continue;
            }
            if (verb == "env" && w.find('=') != std::string_view::npos &&
                w.find('=') > 0) {
                ++next;
                continue;
            }
            break;
        }
        if (next < a.n) {
            evaluate(Args{a.v + next, a.n - next}, sc, v, depth - 1, piped_stdin);
        } else if (verb == "exec") {
            // A bare `exec` with no operand only rearranges the shell's own file
            // descriptors; it runs nothing.
            return;
        }
        return;
    }
    if (verb == "make" || verb == "gmake" || verb == "ninja" || verb == "bazel" ||
        verb == "gradle" || verb == "mvn" || verb == "rake") {
        raise(v.status, ParseStatus::PartiallyParsed);
        return;
    }
    // The same rule as make/ninja/bazel above, applied to the launchers that
    // were missing it. Each of these runs code the command string does not
    // contain -- a build.rs, a test binary, a CMake script, an Xcode build
    // phase, a service, an application -- and several reach the network to do
    // it (cargo and go fetch on a cold cache; `open` takes a URL).
    //
    // They reached the `in(verb, kKnownInert)` early-return below instead, which
    // exits with status Parsed and an EMPTY capability set. That is a clean bill
    // of health, and SubprocessVerifier::provably_confined() reads it as PROOF:
    // measured before this fix, `cargo test`, `go test ./...`, `ctest`,
    // `xcodebuild ... test`, `swift build`, `java Foo` and `cmake --build build`
    // were all "provably confined" and ran on the host in an unattended run,
    // while `pytest -q`, `make test`, `npm test` and `bash verify.sh` -- no more
    // dangerous, and pytest strictly less so -- were refused. The gate was
    // deciding on TABLE MEMBERSHIP, not on containment: kKnownInert means
    // "recognised", and for a verb with no branch of its own, recognition was
    // silently becoming a proof.
    //
    // Deliberately NOT included: the pure compilers (`cc`/`gcc`/`g++`/`clang`/
    // `clang++`/`rustc`/`javac`) and the binary utilities (`ld`/`ar`/`nm`/
    // `objdump`/`strip`/`otool`). They transform files and do not run the
    // program they produce. Compile-time code execution exists -- a gcc
    // `-fplugin`, a Rust proc-macro, a Java annotation processor -- but it needs
    // an explicit opt-in that a later row can key on, so the line is drawn at
    // "does this tool's ORDINARY use execute project code".
    if (verb == "cargo" || verb == "go" || verb == "ctest" || verb == "cmake" ||
        verb == "swift" || verb == "xcodebuild" || verb == "java" ||
        verb == "lldb" || verb == "gdb" || verb == "open" ||
        verb == "systemctl" || verb == "launchctl") {
        raise(v.status, ParseStatus::PartiallyParsed);
        return;
    }

    // ---- long-running ------------------------------------------------------
    if (in(verb, kAlwaysUnbounded)) {
        v.capabilities.spawns_unbounded_process = true;
        note_reads(a, verb, sc, v);
        return;
    }
    if (verb == "tail") {
        if (a.has("-f") || a.has("-F") || a.has("--follow") || a.has("-Ff")) {
            v.capabilities.spawns_unbounded_process = true;
        }
        note_reads(a, verb, sc, v);
        return;
    }
    if (verb == "journalctl") {
        v.capabilities.reads_outside_workspace = true; // system logs are never ours
        if (a.has("-f") || a.has("--follow")) {
            v.capabilities.spawns_unbounded_process = true;
        }
        return;
    }
    if (verb == "sleep") {
        const std::string_view d = a.at(1);
        if (d == "infinity" || d == "inf") {
            v.capabilities.spawns_unbounded_process = true;
        }
        return;
    }
    if (verb == "yes") {
        v.capabilities.spawns_unbounded_process = true;
        return;
    }
    if (verb == "python" || verb == "python3" || verb == "ruby" || verb == "node" ||
        verb == "perl" || verb == "irb") {
        const std::size_t op = a.first_operand(1);
        if (a.has("-c") || a.has("-e")) {
            // The body is a program in ANOTHER language. We are a shell
            // classifier: re-reading it as shell is how `python -c
            // "print('rm -rf /')"` becomes a destroy. It is fully visible and,
            // by itself, does nothing to the filesystem we can name.
            return;
        }
        if (a.has("-m")) {
            const std::string_view mod = [&] {
                for (std::size_t i = 1; i < a.n; ++i) {
                    if (a.at(i) == "-m" && i + 1 < a.n) {
                        return a.at(i + 1);
                    }
                }
                return std::string_view{};
            }();
            if (mod == "http.server" || mod == "SimpleHTTPServer" || mod == "smtpd") {
                v.capabilities.network_access = true;
                v.capabilities.spawns_unbounded_process = true;
                return;
            }
            if (mod == "pip") {
                v.capabilities.network_access = true;
                return;
            }
            raise(v.status, ParseStatus::PartiallyParsed);
            return;
        }
        if (op >= a.n) {
            // A bare interpreter blocks on stdin until something closes it.
            if (!piped_stdin) {
                v.capabilities.spawns_unbounded_process = true;
            } else {
                raise(v.status, ParseStatus::PartiallyParsed);
            }
            return;
        }
        raise(v.status, ParseStatus::PartiallyParsed); // runs a script file
        return;
    }
    if (verb == "cat" && a.first_operand(1) >= a.n && !piped_stdin) {
        v.capabilities.spawns_unbounded_process = true;
        return;
    }

    // ---- readers -----------------------------------------------------------
    if (in(verb, kReaders)) {
        note_reads(a, verb, sc, v);
        return;
    }
    // Creation is not destruction (rule 2): touch/mkdir/ln make paths, they do
    // not overwrite content. Their operands still leave the workspace or not.
    if (verb == "mkdir" || verb == "touch" || verb == "ln" || verb == "mktemp") {
        for (std::size_t i = 1; i < a.n; ++i) {
            if (!is_flag(a.at(i))) {
                note_path(a.v[i], sc, v, true, false);
            }
        }
        return;
    }
    if (looks_like_local_program(raw)) {
        raise(v.status, ParseStatus::PartiallyParsed);
        return;
    }
    if (in(verb, kKnownInert)) {
        return;
    }
    // THE FALLBACK. An unrecognised program is a program whose effect is not in
    // this string. All 11 entrants answered `just deploy` as fully Parsed, which
    // hands the consumer an unearned clean bill of health -- the exact failure
    // ParseStatus exists to prevent.
    raise(v.status, ParseStatus::PartiallyParsed);
}

// ---------------------------------------------------------------------------
// The chain driver: words into stages, stages into capabilities, `cd` carried
// forward (rule 5), and the union taken across every stage whether or not it is
// reachable -- `false && rm -rf /` is destroy + write_out.
// ---------------------------------------------------------------------------
inline void run_chain(Lexer& lex, Scope sc, Verdict& v, int depth) noexcept {
    if (depth <= 0) {
        v.status = ParseStatus::Unparseable;
        return;
    }
    Word args[kMaxArgs];
    std::size_t argc = 0;
    bool overflow = false;
    bool piped = false;
    bool next_piped = false;
    bool stdin_taken = false;

    auto flush = [&]() noexcept {
        if (argc > 0) {
            if (overflow) {
                v.status = ParseStatus::Unparseable;
            } else {
                const std::string_view verb = basename_of(args[0].view());
                if (verb == "cd" || verb == "pushd") {
                    if (argc > 1) {
                        if (args[1].substituted) {
                            // We no longer know where we are, so no later
                            // relative path may be called inside or outside --
                            // and the containment of every later stage is
                            // therefore undecidable, which IS partial.
                            sc.cwd_unknown = true;
                            raise(v.status, ParseStatus::PartiallyParsed);
                            if (!args[1].subst_body.empty()) {
                                const Verdict in2 = classify_body(
                                    {args[1].subst_body, sc.root, {}}, sc, depth - 1);
                                merge(v.capabilities, in2.capabilities);
                            }
                        } else {
                            sc.cwd.append(args[1].view());
                        }
                    }
                } else {
                    evaluate(Args{args, argc}, sc, v, depth, piped || stdin_taken);
                }
            }
        }
        argc = 0;
        overflow = false;
        stdin_taken = false;
        piped = next_piped;
        next_piped = false;
    };

    while (true) {
        bool truncating = false;
        if (lex.take_redirect(truncating)) {
            Word t;
            const Sep s = lex.next(t);
            if (t.substituted) {
                // The TARGET's value is not in this string, so where the bytes
                // land is undecidable: `echo x > $OUT` writes wherever $OUT
                // points, and `echo x > "$HOME/.bashrc"` plainly leaves the
                // workspace. This is the same rule evaluate() applies to a
                // substituted operand -- applied to the one word that loop can
                // never see, because a redirect target is consumed HERE and
                // never enters args[]. Without it the two commands above came
                // back Parsed with an empty capability set, which made
                // SubprocessVerifier::provably_confined() answer TRUE and an
                // unattended run execute them on the host.
                //
                // note_path() cannot carry this: it returns early on a
                // substituted word and deliberately raises nothing, because it
                // is written to rely on its CALLER having already raised.
                raise(v.status, ParseStatus::PartiallyParsed);
                if (truncating) {
                    // Rule 3, and more so at an unknown target than a known one.
                    v.capabilities.destroys_data = true;
                }
            } else if (!t.empty() && !is_sink(t.view())) {
                if (truncating) {
                    // Rule 3: `>` may land on a file that exists and we cannot
                    // know, so the safe reading wins. `>>` appends and does not.
                    v.capabilities.destroys_data = true;
                }
                note_path(t, sc, v, true, false);
            }
            if (s == Sep::End) {
                flush();
                return;
            }
            if (s == Sep::Pipe || s == Sep::Stage) {
                next_piped = (s == Sep::Pipe);
                flush();
            } else if (s == Sep::SubClose) {
                flush();
                return;
            }
            continue;
        }
        Word w;
        const Sep s = lex.next(w);
        if (lex.take_stdin_flag()) {
            stdin_taken = true;
        }
        if (!w.empty() || w.substituted) {
            if (argc < kMaxArgs) {
                args[argc++] = w;
            } else {
                overflow = true;
            }
            if (w.overflow) {
                v.status = ParseStatus::Unparseable;
            }
        }
        switch (s) {
        case Sep::None:
            break;
        case Sep::Pipe:
            next_piped = true;
            flush();
            break;
        case Sep::Stage:
            next_piped = false;
            flush();
            break;
        case Sep::SubOpen: {
            flush();
            run_chain(lex, sc, v, depth - 1);
            break;
        }
        case Sep::SubClose:
            flush();
            return;
        case Sep::End:
            flush();
            return;
        }
    }
}

[[nodiscard]] inline Verdict classify_body(const CommandContext& body, const Scope& sc,
                                           int depth) noexcept {
    Verdict v;
    Lexer lex(body.command);
    Scope inner = sc;
    run_chain(lex, inner, v, depth);
    if (lex.unterminated()) {
        v.status = ParseStatus::Unparseable;
    }
    return v;
}

} // namespace detail

[[nodiscard]] inline Verdict classify(const CommandContext& ctx) noexcept {
    Verdict v;
    detail::Scope sc;
    sc.root = ctx.workspace_root;
    sc.cwd.append(ctx.cwd.empty() ? ctx.workspace_root : ctx.cwd);

    detail::Lexer lex(ctx.command);
    detail::run_chain(lex, sc, v, detail::kMaxDepth);

    // Quoting that never closes means no token boundary after it is trustworthy,
    // so no flag we produced can be relied on. Unparseable is read by the scorer
    // -- and must be read by the consumer -- as "assume every capability".
    if (lex.unterminated()) {
        v.status = ParseStatus::Unparseable;
        v.capabilities = Capabilities{};
    }
    return v;
}

} // namespace blast_radius

#endif // SECURITY_BLAST_RADIUS_HPP
