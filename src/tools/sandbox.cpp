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
#include <cerrno>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <thread>

extern char** environ;

namespace lmp::tools {

RiskHint classify_command(std::string_view command, std::string_view workspace_root,
                          std::string_view cwd) {
    const blast_radius::CommandContext ctx{command, workspace_root, cwd};
    const blast_radius::Verdict v = blast_radius::classify(ctx);
    return RiskHint{v.capabilities, v.status};
}

ExecutionGrant grant_execution(SandboxTier tier) { return ExecutionGrant(tier); }

std::string resolve_real(const std::string& path) {
    // Seatbelt matches subpaths against RESOLVED paths. /tmp is a symlink to
    // /private/tmp on macOS, so a profile written with the unresolved path silently
    // matches nothing -- and a jail that matches nothing denies the workspace's own
    // writes while looking correctly configured. Found by attempting a write inside
    // the root and watching it be refused.
    char buf[PATH_MAX];
    if (::realpath(path.c_str(), buf) != nullptr) {
        return {buf};
    }
    return path;
}

std::string seatbelt_profile(const std::string& workspace_root) {
    // Deny by default. Writes ONLY under the workspace root -- no /tmp allowance, no
    // /var/folders allowance. An earlier version allow-listed /private/var/folders for
    // build scratch; that is the user's entire temp tree, which is precisely "writes
    // outside the workspace", and the test that attempts the escape caught it. Build
    // tools get scratch space via TMPDIR pointed inside the root instead (see
    // run_sandboxed), so nothing legitimate needs the hole.
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
    p += "(allow file-write* (subpath \"" + root + "\"))\n";
    p += "(allow file-write-data (literal \"/dev/null\"))\n";
    p += "(allow file-write-data (literal \"/dev/stdout\"))\n";
    p += "(allow file-write-data (literal \"/dev/stderr\"))\n";
    return p;
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

// Reads until EOF or cap, then drains without storing. The wall-clock killer runs in
// the same loop -- an unattended run cannot afford a command that never returns (S7.3).
void pump_output(int fd, pid_t pid, const ExecLimits& limits, ExecOutcome& out) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(limits.wall_clock_seconds);
    char buf[8192];
    while (true) {
        if (std::chrono::steady_clock::now() > deadline) {
            ::kill(-pid, SIGKILL); // the group: a shell's children die too
            ::kill(pid, SIGKILL);
            out.wall_clock_killed = true;
            return;
        }
        struct timeval tv {0, 200000}; // 200 ms poll so the deadline is honoured
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
                          const ExecLimits& limits) {
    ExecOutcome out;
    if (grant.tier() == SandboxTier::T0_NoExec) {
        out.status = Status::Refused;
        out.output = "T0: this mode does not execute commands";
        return out;
    }
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
        const std::string tmp = resolve_real(workspace_root) + "/.lmp_tmp";
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
    pump_output(pipe.read_fd, pid, limits, out);
    ::close(pipe.read_fd);

    int wstatus = 0;
    (void)::waitpid(pid, &wstatus, 0);
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
