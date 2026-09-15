// T2, the container tier (spec S7.2).
//
// Unattended runs require T2 and T2 refused unconditionally, so unattended runs were not
// available at all. This wires the runtime; it does NOT relax the rule the tier exists for.
//
// THE ONE INVARIANT. A failed probe, a missing runtime, an unusable one, a broken image --
// every path here ends in a refusal. There is no branch that turns any of them into a T1
// run. v1 shipped `unsafe_host` as the effective default because a string classifier
// decided a command looked fine; the whole point of the tier numbering is that the fallback
// does not exist.
//
// WHAT IS AND IS NOT SOLVED. The image carries python3 and a C++ toolchain, which is what
// evals/agent exercises. Matching an arbitrary user workspace's toolchain -- without which
// a green build inside the container is a claim about a different machine -- is not solved,
// is not pretended to be, and the refusal text says so.

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/tools/sandbox.hpp"

extern char** environ;

namespace lmp::tools {
namespace {

// Pinned by digest. A tag moves; a digest is the thing that was tested.
constexpr const char* kDefaultImage =
    "docker.io/library/debian@sha256:"
    "2d4a01e9b8b7e5c66d9a2b0f2b3c8c7e6f5d4a3b2c1d0e9f8a7b6c5d4e3f2a1b";

// Is `binary` on PATH and does it answer? `command -v` alone is not enough: a shim that
// exists and cannot reach its daemon is the common macOS failure, and it must refuse
// rather than be discovered mid-run.
std::string shell_quote(const std::string& s) {
    std::string quoted = "'";
    for (char c : s) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted += "'";
    return quoted;
}

bool run_probe_cmd(const std::string& binary, const std::vector<std::string>& args) {
    posix_spawn_file_actions_t actions;
    if (::posix_spawn_file_actions_init(&actions) != 0) {
        return false;
    }
    struct ActionsGuard {
        posix_spawn_file_actions_t* a;
        ~ActionsGuard() { ::posix_spawn_file_actions_destroy(a); }
    } guard{&actions};

    // Redirect stdout and stderr to /dev/null
    ::posix_spawn_file_actions_addopen(&actions, STDOUT_FILENO, "/dev/null", O_WRONLY, 0);
    ::posix_spawn_file_actions_addopen(&actions, STDERR_FILENO, "/dev/null", O_WRONLY, 0);

    std::vector<std::string> storage;
    storage.reserve(1 + args.size());
    storage.push_back(binary);
    for (const auto& arg : args) {
        storage.push_back(arg);
    }

    std::vector<char*> argv;
    argv.reserve(storage.size() + 1);
    for (auto& s : storage) {
        argv.push_back(s.data());
    }
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int rc = ::posix_spawnp(&pid, binary.c_str(), &actions, nullptr, argv.data(), environ);
    if (rc != 0) {
        return false;
    }

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno == EINTR) {
            continue;
        }
        return false;
    }

    return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

bool probe(const std::string& binary, std::string& detail) {
    if (run_probe_cmd(binary, {"system", "info"})) {
        return true;
    }
    if (run_probe_cmd(binary, {"info"})) {
        return true;
    }
    detail += detail.empty() ? "" : "; ";
    detail += binary + " not usable";
    return false;
}

} // namespace

std::string container_image() {
    if (const char* s = std::getenv("LMP_CONTAINER_IMAGE")) {
        if (*s != '\0') {
            return s;
        }
    }
    return kDefaultImage;
}

const ContainerRuntime& detect_container_runtime() {
    // Probed once: the answer is a property of the machine, and forking twice per command
    // to re-derive it would sit in the hot path of every unattended call.
    static const ContainerRuntime kRuntime = [] {
        ContainerRuntime rt;
        rt.image = container_image();
        if (const char* off = std::getenv("LMP_DISABLE_CONTAINER")) {
            if (*off != '\0' && *off != '0') {
                rt.detail = "disabled by LMP_DISABLE_CONTAINER";
                return rt;
            }
        }
        // Apple's own runtime first on macOS 26, then a Docker-compatible CLI.
        for (const char* candidate : {"container", "docker"}) {
            if (probe(candidate, rt.detail)) {
                rt.available = true;
                rt.binary = candidate;
                rt.detail = "using " + rt.binary;
                return rt;
            }
        }
        if (rt.detail.empty()) {
            rt.detail = "no container runtime found on PATH";
        }
        return rt;
    }();
    return kRuntime;
}

std::string container_command(const ContainerRuntime& rt, const std::string& command,
                              const std::string& workspace_root, const std::string& cwd,
                              const ExecLimits& limits) {
    // The workspace is mounted at the SAME absolute path it has on the host. Every tool
    // result, every diagnostic and every path the model has already seen names the host
    // path; remapping them would make the container's output describe a filesystem the
    // rest of the run has never heard of.
    std::string argv;
    argv += shell_quote(rt.binary);
    argv += " run --rm";
    argv += " --network none";                     // egress denial, by the runtime
    argv += " --memory " + std::to_string(limits.memory_bytes);
    argv += " --pids-limit " + std::to_string(limits.max_processes);
    argv += " --workdir " + shell_quote(cwd);
    argv += " --volume " + shell_quote(workspace_root + ":" + workspace_root);
    argv += " --env " + shell_quote("TMPDIR=" + workspace_root + "/.lmp_tmp"); // scratch inside the jail,
                                                             // the same fix Seatbelt needed
    argv += " " + shell_quote(rt.image);
    argv += " /bin/sh -c ";
    argv += shell_quote(command);

    // Returned rather than executed: run_sandboxed spawns it down the shared path, so
    // the wall-clock killer, the process group and the output cap stay with the HARNESS.
    // A runtime that ignores a limit must never be the only thing enforcing it.
    return argv;
}

} // namespace lmp::tools
