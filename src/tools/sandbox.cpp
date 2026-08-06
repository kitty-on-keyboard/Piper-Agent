#include "src/tools/sandbox.hpp"

#include <fcntl.h>
#include <libproc.h>
#include <signal.h>
#include <spawn.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <climits>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

extern char** environ;

namespace lmp::tools {

RiskHint classify_command(std::string_view command, std::string_view workspace_root,
                          std::string_view cwd) {
    const blast_radius::CommandContext ctx{command, workspace_root, cwd};
    const blast_radius::Verdict v = blast_radius::classify(ctx);
    return RiskHint{v.capabilities, v.status};
}

ExecutionGrant grant_execution(SandboxTier tier) { return ExecutionGrant(tier); }

SandboxTier tier_for(int approved_tier) noexcept {
    switch (approved_tier) {
        case 0:
            return SandboxTier::T0_NoExec;
        case 1:
            return SandboxTier::T1_Seatbelt;
        case 2:
            return SandboxTier::T2_Container;
        case 3:
            return SandboxTier::T3_HostUnsandboxed;
        default:
            break;
    }
    // Below the range is "no execution"; above it is the container. Neither direction
    // lands on T3, which is reachable only by naming it exactly.
    return approved_tier < 0 ? SandboxTier::T0_NoExec : SandboxTier::T2_Container;
}

std::string resolve_real(const std::string& path) {
    // Seatbelt matches subpaths against RESOLVED paths. /tmp is a symlink to
    // /private/tmp on macOS, so a profile written with the unresolved path silently
    // matches nothing -- and a jail that matches nothing denies the workspace's own
    // writes while looking correctly configured. Found by attempting a write inside
    // the root and watching it be refused.
    //
    // A trailing slash is stripped for the same reason: `(subpath "/x/y/")` matches
    // nothing at all. realpath removes it when the path resolves, so this only bites on
    // the paths that do not exist yet -- which is most of the toolchain state below the
    // first time a build runs.
    char buf[PATH_MAX];
    std::string out = ::realpath(path.c_str(), buf) != nullptr ? std::string(buf) : path;
    while (out.size() > 1 && out.back() == '/') {
        out.pop_back();
    }
    return out;
}

namespace {

// A per-user directory the OS names for us, resolved. Empty when the platform declines.
std::string darwin_dir(int name) {
    char buf[PATH_MAX];
    const std::size_t n = ::confstr(name, buf, sizeof(buf));
    if (n == 0 || n > sizeof(buf)) {
        return {};
    }
    return resolve_real(buf);
}

std::string home_dir() {
    const char* home = std::getenv("HOME");
    return home != nullptr && *home != '\0' ? resolve_real(home) : std::string();
}

void allow_subpath(std::string& profile, const std::string& path) {
    // An empty or relative path would produce a rule that matches nothing, or -- worse --
    // one that matches something unintended. Dropped rather than written.
    if (path.empty() || path.front() != '/') {
        return;
    }
    profile += "(allow file-write* (subpath \"" + path + "\"))\n";
}

} // namespace

std::string seatbelt_profile(const std::string& workspace_root) {
    // Deny by default. Writes are the workspace, plus the narrow set of TOOLCHAIN STATE
    // directories below -- no /tmp allowance, and emphatically not the per-user temp root.
    // An earlier version allow-listed /private/var/folders wholesale for build scratch;
    // that is the user's entire temp tree, which is precisely "writes outside the
    // workspace", and the break-out test caught it. It still would: that test writes
    // straight into the temp root, which stays denied here.
    //
    // Network is denied here, in the profile, not by inspecting the command (S7.4).
    // Reads stay open: toolchains legitimately read /usr and the SDKs, and the assets
    // protected at T1 are the user's data and the network. Read confinement beyond
    // that is a T2 property, and S7.2 already requires T2 for unattended runs.
    const std::string root = resolve_real(workspace_root);
    std::string p;
    p += "(version 1)\n";
    p += "(allow default)\n";
    p += "(deny network*)\n";
    p += "(deny file-write*)\n";
    allow_subpath(p, root);

    // --- toolchain state -----------------------------------------------------
    //
    // MEASURED, on the run that prompted this. `swift build` inside the jail died with
    // "You don't have permission to save the file output-file-map.json", and the agent
    // spent thirty turns trying to configure its way out of it.
    //
    // The cause is that macOS's atomic save does not write the destination file: it
    // stages the bytes in the per-user ITEM REPLACEMENT directory and renames them into
    // place. That directory is NOT TMPDIR and cannot be moved by setting TMPDIR -- it is
    // .../T/TemporaryItems, read from confstr(). So a jail that denies it denies every
    // atomic save any tool makes, into the workspace or anywhere else, and the error
    // names the destination file rather than the staging directory that was actually
    // refused. That is what made it unreadable.
    //
    // TemporaryItems ONLY, never .../T itself. mkdtemp, `echo > /var/folders/.../x` and
    // every other ordinary write outside the workspace land in the temp root, which is
    // still denied -- so this buys atomic saves without buying an escape hatch.
    const std::string user_temp = darwin_dir(_CS_DARWIN_USER_TEMP_DIR);
    if (!user_temp.empty()) {
        allow_subpath(p, user_temp + "/TemporaryItems");
    }
    // The per-user CACHE directory (.../C), which is where clang puts its module cache.
    // A cache is derived data by definition: losing it costs a rebuild and nothing else.
    allow_subpath(p, darwin_dir(_CS_DARWIN_USER_CACHE_DIR));

    // Xcode's and SwiftPM's own state. Each of these is build output or a package cache
    // that the tool creates, owns and can rebuild -- none of them is the user's work.
    // Named individually rather than allowing ~/Library, which holds every application's
    // data on the machine.
    const std::string home = home_dir();
    if (!home.empty()) {
        allow_subpath(p, home + "/Library/Developer/Xcode/DerivedData");
        allow_subpath(p, home + "/Library/Caches/org.swift.swiftpm");
        allow_subpath(p, home + "/Library/org.swift.swiftpm");
        allow_subpath(p, home + "/.swiftpm");
        allow_subpath(p, home + "/Library/Caches/clang");
    }

    p += "(allow file-write-data (literal \"/dev/null\"))\n";
    p += "(allow file-write-data (literal \"/dev/stdout\"))\n";
    p += "(allow file-write-data (literal \"/dev/stderr\"))\n";
    return p;
}

namespace {

constexpr std::string_view kDisableSandbox = "--disable-sandbox";

// WHAT USED TO BE HERE: `at_command_position`, which asked whether the character before a
// `swift` token was start-of-string or one of `;&|(`.
//
// It was the whole bug. In `xcrun swift build` the preceding character is the `n` of
// `xcrun`, so `swift` read as an ARGUMENT and the rewrite declined -- and `xcrun` is how
// every model on a Mac spells a toolchain invocation, because it is how Apple's own docs
// spell it.
//
// MEASURED: an 85-turn run whose declared contract was `xcrun swift build`, at T1, on a
// tree whose only real problem was a handful of Mach API type errors. The rewrite fired
// ZERO times. Its first verification was the nesting EPERM, so was every one after it, and
// the model spent the budget rewriting correct code to chase a failure the harness was
// causing. `xcrun` appeared nowhere in this file.
//
// The replacement asks the right question -- "what program does this segment RUN?" -- and
// answers it by skipping the things that stand in front of a program without being one.
// That is strictly more permissive about position and no more permissive about identity:
// `echo swift build` and `grep -r 'swift test' .` still find `echo` and `grep` and are
// still left alone.

// A program whose only job is to run ANOTHER program, so the rewrite must look past it.
struct Launcher {
    std::string_view name;
    // Flags taking a SEPARATE value token, which has to be skipped along with the flag.
    // Space-delimited. Without this, `xcrun -sdk macosx swift build` stops at `macosx`,
    // concludes the program is `macosx`, and declines to rewrite a plain swift build.
    std::string_view value_flags;
};

constexpr Launcher kLaunchers[] = {
    {"xcrun", "-sdk --sdk -toolchain --toolchain -find --find -run"},
    {"env", "-u --unset -S --split-string"},
    {"arch", "-arch"},
    {"nice", "-n"},
    {"stdbuf", "-i -o -e --input --output --error"},
    {"caffeinate", "-t -w"},
    {"time", "-o -f --output --format"},
    {"command", ""},
    {"exec", ""},
    {"setsid", ""},
};

// The basename of a token: `/usr/bin/swift` and `swift` are the same program.
std::string_view basename_of(std::string_view tok) {
    const std::size_t slash = tok.find_last_of('/');
    return slash == std::string_view::npos ? tok : tok.substr(slash + 1);
}

const Launcher* launcher_for(std::string_view name) {
    for (const Launcher& l : kLaunchers) {
        if (l.name == name) {
            return &l;
        }
    }
    return nullptr;
}

bool takes_value(std::string_view value_flags, std::string_view tok) {
    std::size_t at = 0;
    while (at < value_flags.size()) {
        const std::size_t end = value_flags.find(' ', at);
        const std::size_t stop = end == std::string_view::npos ? value_flags.size() : end;
        if (value_flags.substr(at, stop - at) == tok) {
            return true;
        }
        at = stop + 1;
    }
    return false;
}

// The next whitespace-delimited token in [pos, end), and where it ends. Tokens are runs of
// non-whitespace: a program can be `/usr/bin/swift` and a flag can be `--sdk=macosx`, and
// neither is a run of `is_word_char`.
std::string_view next_token(const std::string& s, std::size_t pos, std::size_t end,
                            std::size_t& tok_end) {
    while (pos < end && (std::isspace(static_cast<unsigned char>(s[pos])) != 0)) {
        ++pos;
    }
    const std::size_t begin = pos;
    while (pos < end && (std::isspace(static_cast<unsigned char>(s[pos])) == 0)) {
        ++pos;
    }
    tok_end = pos;
    return std::string_view(s).substr(begin, pos - begin);
}

// The program a segment invokes, and where its token ends. Skips what precedes a program
// without being one: `VAR=value` assignments, `cd DIR`, and launcher prefixes with their
// flags. Stops at the first token that is none of those, so an unrelated program shadows
// anything after it -- which is what keeps `echo swift build` and `grep 'swift test'` out.
std::string_view segment_program(const std::string& s, std::size_t begin, std::size_t end,
                                 std::size_t& program_end) {
    std::size_t at = begin;
    while (at < end) {
        std::size_t tok_end = 0;
        const std::string_view tok = next_token(s, at, end, tok_end);
        if (tok.empty()) {
            return {};
        }
        const std::string_view name = basename_of(tok);
        // An inline assignment: `FOO=1 swift build`. A leading dash means it is a flag
        // like `--sdk=macosx`, not an assignment.
        if (!tok.empty() && tok.front() != '-' &&
            tok.find('=') != std::string_view::npos) {
            at = tok_end;
            continue;
        }
        if (name == "cd") {
            at = tok_end;
            std::size_t dir_end = 0;
            (void)next_token(s, at, end, dir_end);
            at = dir_end;
            continue;
        }
        if (const Launcher* l = launcher_for(name); l != nullptr) {
            at = tok_end;
            // The launcher's own flags, and the value token of any flag that takes one.
            while (at < end) {
                std::size_t flag_end = 0;
                const std::string_view flag = next_token(s, at, end, flag_end);
                if (flag.empty() || flag.front() != '-') {
                    break;
                }
                at = flag_end;
                if (takes_value(l->value_flags, flag)) {
                    std::size_t val_end = 0;
                    (void)next_token(s, at, end, val_end);
                    at = val_end;
                }
            }
            continue;
        }
        program_end = tok_end;
        return name;
    }
    return {};
}

// Top-level segment bounds: the string split on `; | & \n (` outside quotes and outside
// `$(...)`. One rewrite decision is made PER SEGMENT, because `swift build | tee log` and
// `echo hi && swift test` each contain exactly one program that matters, in a different
// place. Empty segments (from the second `&` of an `&&`) yield no program and are skipped.
std::vector<std::pair<std::size_t, std::size_t>> segments(const std::string& s) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    std::size_t begin = 0;
    char quote = '\0';
    int depth = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (quote != '\0') {
            if (c == '\\' && quote == '"' && i + 1 < s.size()) {
                ++i;
            } else if (c == quote) {
                quote = '\0';
            }
            continue;
        }
        if (c == '\'' || c == '"' || c == '`') {
            quote = c;
            continue;
        }
        if (c == '$' && i + 1 < s.size() && s[i + 1] == '(') {
            ++depth;
            ++i;
            continue;
        }
        if (c == ')' && depth > 0) {
            --depth;
            continue;
        }
        if (depth == 0 && (c == ';' || c == '|' || c == '&' || c == '\n' || c == '(')) {
            out.emplace_back(begin, i);
            begin = i + 1;
        }
    }
    out.emplace_back(begin, s.size());
    return out;
}

} // namespace

std::string t1_compat_rewrite(const std::string& command) {
    // Where to splice ` --disable-sandbox`, in ascending order.
    std::vector<std::size_t> insert_at;
    for (const auto& [begin, end] : segments(command)) {
        // Already asked for by hand -- leave this segment exactly as written. A second copy
        // of the flag is harmless to swift and confusing to read, and the model that typed
        // it does not need to be told the harness agrees.
        //
        // PER SEGMENT, not per command: `swift build --disable-sandbox && swift test` has
        // one half already handled and one half that still nests, and a whole-string check
        // silently declined to fix the second.
        const std::string_view seg = std::string_view(command).substr(begin, end - begin);
        if (seg.find(kDisableSandbox) != std::string_view::npos) {
            continue;
        }
        std::size_t program_end = 0;
        // Whole basename, so `swiftc` and `swift-format` are not this program.
        if (segment_program(command, begin, end, program_end) != "swift") {
            continue;
        }
        std::size_t sub_end = 0;
        const std::string_view sub = next_token(command, program_end, end, sub_end);
        // The four SwiftPM verbs that sandbox a manifest compile and accept the flag.
        //
        // `package` was previously excluded on the grounds that it was out of scope. It is
        // not: `swift package resolve` and `swift package update` compile Package.swift the
        // same way `build` does and die at T1 the same way, and dependency resolution is
        // part of building. Both flag positions were checked against SwiftPM 6 on the host
        // -- `swift package --disable-sandbox describe` and `swift package describe
        // --disable-sandbox` are both accepted -- so this keeps the single uniform rule:
        // immediately after the subcommand.
        if (sub != "build" && sub != "test" && sub != "run" && sub != "package") {
            continue;
        }
        // After the subcommand, where swift-argument-parser takes it and where it stays
        // out of the way of anything the caller appended (`2>&1 | tee ...`).
        insert_at.push_back(sub_end);
    }
    if (insert_at.empty()) {
        return {};
    }
    std::string out;
    std::size_t scan = 0;
    for (const std::size_t at : insert_at) {
        out.append(command, scan, at - scan);
        out += ' ';
        out += kDisableSandbox;
        scan = at;
    }
    out.append(command, scan, std::string::npos);
    return out;
}

namespace {

struct Pipe {
    int read_fd = -1;
    int write_fd = -1;
    [[nodiscard]] bool open() {
        int fds[2] = {-1, -1};
        if (::pipe(fds) != 0) {
            return false;
        }
        read_fd = fds[0];
        write_fd = fds[1];
        return true;
    }
};

// RLIMIT_NPROC counts every process owned by the REAL UID -- not the ones in this
// process group, and not the ones this child goes on to spawn. That is BSD semantics and
// macOS inherits them, so `max_processes` cannot be applied as an absolute number: on a
// desktop with an editor or two already open the uid is several hundred processes deep,
// and the very first fork() in the sandbox returns EAGAIN.
//
// It failed exactly that way. Every shell call in the first end-to-end run came back
// `/bin/sh: fork: Resource temporarily unavailable`, so the agent could not run a test,
// a build, or anything else -- it fixed its mission's bug and then reported failure
// because it could not prove it.
//
// So the limit is a HEADROOM over what the uid is already using: a runaway may add
// max_processes and no more, which is the containment the number was always meant to
// express. Computed in the PARENT because it is not async-signal-safe: between fork and
// exec, only async-signal-safe calls are legal.
[[nodiscard]] rlim_t nproc_ceiling(int headroom) {
    int in_use = 0;
    const int bytes = ::proc_listpids(PROC_UID_ONLY, ::getuid(), nullptr, 0);
    if (bytes > 0) {
        in_use = bytes / static_cast<int>(sizeof(pid_t));
    }
    rlim_t want = static_cast<rlim_t>(in_use) + static_cast<rlim_t>(headroom);
    // Raising a hard limit needs privilege we do not have and must not acquire, so the
    // ceiling is clamped to the one we inherited rather than allowed to fail silently.
    rlimit current{};
    if (::getrlimit(RLIMIT_NPROC, &current) == 0 && current.rlim_max != RLIM_INFINITY &&
        want > current.rlim_max) {
        want = current.rlim_max;
    }
    return want;
}

std::string scratch_dir(const std::string& workspace_root) {
    return resolve_real(workspace_root) + "/.lmp_tmp";
}

// SWEEP THE SCRATCH TREE, once per process, before the first command uses it.
//
// TMPDIR points into the workspace so the jail needs no hole in the temp tree, and nothing
// ever emptied it. Every sandboxed toolchain that calls mkdtemp leaves its directory
// behind: an observed workspace had 824 `TemporaryDirectory.*` under `.lmp_tmp`, which is
// also 824 directories every workspace walk then had to descend into.
//
// Age-gated rather than "delete everything" because another sidecar may be mid-command in
// the same workspace, and its scratch directory is not ours to remove. Six hours is far
// longer than any command that survives the wall-clock killer, so anything older is
// abandoned by definition.
void prune_scratch(const std::string& workspace_root) {
    namespace fs = std::filesystem;
    const fs::path tmp(scratch_dir(workspace_root));
    std::error_code ec;
    if (!fs::is_directory(tmp, ec)) {
        return;
    }
    const auto cutoff = fs::file_time_type::clock::now() - std::chrono::hours(6);
    for (fs::directory_iterator it(tmp, fs::directory_options::skip_permission_denied, ec);
         !ec && it != fs::directory_iterator(); it.increment(ec)) {
        std::error_code each;
        const auto when = fs::last_write_time(it->path(), each);
        if (each || when >= cutoff) {
            continue;
        }
        fs::remove_all(it->path(), each);
    }
}

// Once per ROOT, not once per process. A single sidecar only ever has one workspace, but
// std::once_flag would latch on whichever root arrived first and silently never sweep a
// second one -- which is precisely how a test suite, or a future multi-workspace host,
// gets a cleanup that quietly does nothing.
void prune_scratch_once(const std::string& workspace_root) {
    static std::mutex mu;
    static std::vector<std::string> done;
    {
        const std::lock_guard<std::mutex> lock(mu);
        for (const std::string& seen : done) {
            if (seen == workspace_root) {
                return;
            }
        }
        done.push_back(workspace_root);
    }
    prune_scratch(workspace_root);
}

void apply_rlimits_in_child(const ExecLimits& limits, rlim_t nproc) {
    const auto set = [](int what, rlim_t v) {
        rlimit rl{v, v};
        (void)::setrlimit(what, &rl);
    };
    set(RLIMIT_CPU, static_cast<rlim_t>(limits.cpu_seconds));
    set(RLIMIT_AS, static_cast<rlim_t>(limits.memory_bytes));
    set(RLIMIT_NOFILE, static_cast<rlim_t>(limits.max_open_files));
    set(RLIMIT_NPROC, nproc);
}

// Reads until EOF or cap, then drains without storing. The wall-clock killer and the
// CancelToken both run in the same loop -- an unattended run cannot afford a command
// that never returns (S7.3), and a cancelled run cannot afford to wait for that wall
// clock either.
void kill_process_group(pid_t pid) {
    ::kill(-pid, SIGKILL); // the group: a shell's children die too
    ::kill(pid, SIGKILL);
}

void pump_output(int fd, pid_t pid, const ExecLimits& limits, ExecOutcome& out,
                 const model::CancelToken* cancel) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(limits.wall_clock_seconds);
    char buf[8192];
    while (true) {
        if (cancel != nullptr && cancel->cancelled()) {
            kill_process_group(pid);
            out.cancelled = true;
            return;
        }
        if (std::chrono::steady_clock::now() > deadline) {
            kill_process_group(pid);
            out.wall_clock_killed = true;
            return;
        }
        struct timeval tv {0, 200000}; // 200 ms poll so cancel and the deadline are honoured
        fd_set set;
        FD_ZERO(&set);
        FD_SET(fd, &set);
        const int rc = ::select(fd + 1, &set, nullptr, nullptr, &tv);
        if (rc < 0 && errno != EINTR) {
            return;
        }
        if (rc <= 0) {
            continue;
        }
        const ssize_t n = ::read(fd, buf, sizeof(buf));
        if (n == 0) {
            return; // EOF
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return;
        }
        if (out.output.size() < limits.max_output_bytes) {
            const std::size_t room = limits.max_output_bytes - out.output.size();
            out.output.append(buf, std::min(static_cast<std::size_t>(n), room));
            out.output_truncated |= static_cast<std::size_t>(n) > room;
        } else {
            out.output_truncated = true;
        }
    }
}

} // namespace

ExecOutcome run_sandboxed(const ExecutionGrant& grant, const std::string& command,
                          const std::string& workspace_root, const std::string& cwd,
                          const ExecLimits& limits, const model::CancelToken* cancel) {
    ExecOutcome out;
    if (cancel != nullptr && cancel->cancelled()) {
        out.status = Status::Cancelled;
        out.cancelled = true;
        out.output = "cancelled before the command started";
        return out;
    }
    if (grant.tier() == SandboxTier::T0_NoExec) {
        out.status = Status::Refused;
        out.output = "T0: this mode does not execute commands";
        return out;
    }
    // Before the first command claims TMPDIR, not after the last one releases it: a run
    // killed by the wall clock never reaches its own cleanup, and that is exactly the run
    // that leaves scratch behind.
    prune_scratch_once(workspace_root);
    // T2 rewrites the command into a container invocation and then takes the SAME spawn
    // path as everything else: the pipe, the rlimits, the process group and its
    // wall-clock killer, the output cap. A runtime that ignores a limit must never be the
    // only thing enforcing it.
    std::string effective_command = command;
    bool containerised = false;
    if (grant.tier() == SandboxTier::T2_Container) {
        const ContainerRuntime& rt = detect_container_runtime();
        if (!rt.available) {
            // Refusal, not downgrade (S7.2): unattended MUST be containerised, and running
            // it in T1 instead would be the silent unsafe_host default all over again.
            // This stays a refusal when the runtime is MISSING, when it is present but
            // unusable, and when the probe itself fails -- there is no path from here to
            // execution that does not go through a container.
            out.status = Status::Refused;
            out.output = "T2 (container) requested but no container runtime is usable: " +
                         rt.detail +
                         ". Unattended runs require T2; this is a refusal, not a "
                         "downgrade to T1.";
            return out;
        }
        effective_command = container_command(rt, command, workspace_root, cwd, limits);
        containerised = true;
    }

    // T1 ONLY. T3 has no profile to nest inside, and T2's jail is the container, so in
    // both of those the command runs exactly as written -- which is also what makes the
    // tiers comparable when a build fails: if it fails the same way at T1 and T3, the
    // sandbox is not what is failing.
    if (grant.tier() == SandboxTier::T1_Seatbelt) {
        std::string rewritten = t1_compat_rewrite(effective_command);
        if (!rewritten.empty()) {
            effective_command = rewritten;
            out.rewritten_command = effective_command;
        }
    }

    // T3 keeps every OTHER containment the harness has -- rlimits, the process group and
    // its wall-clock killer, the output cap, the workspace cwd. Only the Seatbelt profile
    // is dropped. "Unsandboxed" here means "no filesystem jail and no egress denial", not
    // "no limits at all", and a runaway build is still killed on the same schedule.
    //
    // T2 is unjailed here for a different reason: the container IS the jail, and wrapping
    // the runtime's own client in a Seatbelt profile would deny it the sockets it needs.
    const bool jailed =
        grant.tier() != SandboxTier::T3_HostUnsandboxed && !containerised;
    const std::string& command_to_run = effective_command;
    const std::string profile = jailed ? seatbelt_profile(workspace_root) : std::string();

    Pipe pipe;
    if (!pipe.open()) {
        out.output = std::string("pipe: ") + std::strerror(errno);
        return out;
    }

    // Before the fork: see nproc_ceiling().
    const rlim_t nproc = nproc_ceiling(limits.max_processes);

    const pid_t pid = ::fork();
    if (pid < 0) {
        out.output = std::string("fork: ") + std::strerror(errno);
        ::close(pipe.read_fd);
        ::close(pipe.write_fd);
        return out;
    }
    if (pid == 0) {
        ::close(pipe.read_fd);
        ::dup2(pipe.write_fd, STDOUT_FILENO);
        ::dup2(pipe.write_fd, STDERR_FILENO);
        ::close(pipe.write_fd);
        if (::chdir(cwd.c_str()) != 0) {
            ::_exit(126);
        }
        // Scratch space inside the jail, so no temp-tree hole is needed in the profile.
        const std::string tmp = scratch_dir(workspace_root);
        ::mkdir(tmp.c_str(), 0700);
        ::setenv("TMPDIR", tmp.c_str(), 1);
        // A new process group, so the wall-clock killer can take down the whole tree a
        // shell may have spawned rather than just the shell.
        ::setpgid(0, 0);
        apply_rlimits_in_child(limits, nproc);
        if (jailed) {
            // sandbox-exec applies the Seatbelt profile then execs the shell. Deprecated
            // in the headers, load-bearing across macOS tooling, and the ONLY per-process
            // profile API without an entitlement; the T2 container is the successor path.
            ::execlp("/usr/bin/sandbox-exec", "sandbox-exec", "-p", profile.c_str(),
                     "/bin/sh", "-c", command_to_run.c_str(), static_cast<char*>(nullptr));
        } else {
            ::execlp("/bin/sh", "sh", "-c", command_to_run.c_str(),
                     static_cast<char*>(nullptr));
        }
        ::_exit(127);
    }
    ::setpgid(pid, pid);
    ::close(pipe.write_fd);
    pump_output(pipe.read_fd, pid, limits, out, cancel);
    ::close(pipe.read_fd);

    int wstatus = 0;
    (void)::waitpid(pid, &wstatus, 0);
    if (out.cancelled) {
        out.status = Status::Cancelled;
        return out;
    }
    if (out.wall_clock_killed) {
        out.status = Status::Timeout;
        return out;
    }
    if (WIFSIGNALED(wstatus)) {
        out.signalled = true;
        out.signal = WTERMSIG(wstatus);
        out.status = Status::ToolError;
        return out;
    }
    out.exit_code = WEXITSTATUS(wstatus);
    out.status = out.exit_code == 0 ? Status::Ok : Status::ToolError;
    return out;
}

std::string_view to_string(Status s) noexcept {
    switch (s) {
        case Status::Ok:
            return "Ok";
        case Status::ToolError:
            return "ToolError";
        case Status::Denied:
            return "Denied";
        case Status::Timeout:
            return "Timeout";
        case Status::Refused:
            return "Refused";
        case Status::Cancelled:
            return "Cancelled";
    }
    return "ToolError";
}

std::string_view to_string(ErrorClass e) noexcept {
    switch (e) {
        case ErrorClass::None:
            return "None";
        case ErrorClass::NotFound:
            return "NotFound";
        case ErrorClass::Malformed:
            return "Malformed";
        case ErrorClass::Conflict:
            return "Conflict";
        case ErrorClass::Policy:
            return "Policy";
        case ErrorClass::Transient:
            return "Transient";
    }
    return "None";
}

} // namespace lmp::tools
