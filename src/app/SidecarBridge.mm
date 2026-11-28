#import "src/app/SidecarBridge.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

extern char** environ;

namespace {

std::filesystem::path app_stderr_log_path() {
    const char* home = std::getenv("HOME");
    std::filesystem::path dir =
        (home && *home) ? (std::filesystem::path(home) / "Library" / "Logs" / "LM_Pipe")
                        : std::filesystem::temp_directory_path();
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    ::chmod(dir.c_str(), 0700);
    return dir / "app-sidecar-stderr.log";
}

std::string resolve_sidecar_binary() {
    if (const char* override_bin = std::getenv("LMP_SIDECAR"); override_bin && *override_bin) {
        return override_bin;
    }
    NSString* execPath = [[NSBundle mainBundle] executablePath];
    if (execPath) {
        NSString* dir = [execPath stringByDeletingLastPathComponent];
        NSString* bundled = [dir stringByAppendingPathComponent:@"lmp_sidecar"];
        if ([[NSFileManager defaultManager] isExecutableFileAtPath:bundled]) {
            return [bundled UTF8String];
        }
    }
    return "lmp_sidecar";
}

void write_all_fd(int fd, const char* data, std::size_t len, std::mutex& mu) {
    std::lock_guard<std::mutex> lock(mu);
    std::size_t off = 0;
    while (off < len) {
        const ssize_t n = ::write(fd, data + off, len - off);
        if (n > 0) {
            off += static_cast<std::size_t>(n);
        } else if (n < 0 && errno == EINTR) {
            continue;
        } else {
            break;
        }
    }
}

} // namespace

@implementation SidecarBridge {
    std::atomic<bool> _runningFlag;
    std::atomic<pid_t> _pid;
    int _inWriteFd;
    std::mutex _writeMutex;
    std::mutex _stderrMutex;
    std::deque<std::string> _stderrTail;
    std::thread _stdoutThread;
    std::thread _stderrThread;
    std::thread _waitThread;
}

- (instancetype)init {
    self = [super init];
    if (self) {
        _runningFlag.store(false);
        _pid.store(-1);
        _inWriteFd = -1;
    }
    return self;
}

- (void)dealloc {
    [self stopSidecar];
}

- (BOOL)running {
    return _runningFlag.load();
}

- (BOOL)startSidecarWithError:(NSString* _Nullable* _Nullable)errorOut {
    if (_runningFlag.load()) {
        return YES;
    }
    [self joinThreads];

    const std::string binary = resolve_sidecar_binary();
    if (binary.find('/') != std::string::npos &&
        ::access(binary.c_str(), X_OK) != 0) {
        if (errorOut) {
            *errorOut = [NSString stringWithFormat:@"lmp_sidecar not executable at %s", binary.c_str()];
        }
        return NO;
    }

    int inPipe[2] = {-1, -1};
    int outPipe[2] = {-1, -1};
    int errPipe[2] = {-1, -1};
    if (::pipe(inPipe) != 0 || ::pipe(outPipe) != 0 || ::pipe(errPipe) != 0) {
        for (int fd : {inPipe[0], inPipe[1], outPipe[0], outPipe[1], errPipe[0], errPipe[1]}) {
            if (fd >= 0) ::close(fd);
        }
        if (errorOut) {
            *errorOut = @"failed to create stdio pipes for lmp_sidecar";
        }
        return NO;
    }
    for (int fd : {inPipe[0], inPipe[1], outPipe[0], outPipe[1], errPipe[0], errPipe[1]}) {
        ::fcntl(fd, F_SETFD, FD_CLOEXEC);
    }

    posix_spawn_file_actions_t actions;
    ::posix_spawn_file_actions_init(&actions);
    ::posix_spawn_file_actions_adddup2(&actions, inPipe[0], STDIN_FILENO);
    ::posix_spawn_file_actions_adddup2(&actions, outPipe[1], STDOUT_FILENO);
    ::posix_spawn_file_actions_adddup2(&actions, errPipe[1], STDERR_FILENO);
    ::posix_spawn_file_actions_addclose(&actions, inPipe[0]);
    ::posix_spawn_file_actions_addclose(&actions, inPipe[1]);
    ::posix_spawn_file_actions_addclose(&actions, outPipe[0]);
    ::posix_spawn_file_actions_addclose(&actions, outPipe[1]);
    ::posix_spawn_file_actions_addclose(&actions, errPipe[0]);
    ::posix_spawn_file_actions_addclose(&actions, errPipe[1]);

    std::string argv0 = binary;
    char* argv[] = {argv0.data(), nullptr};
    pid_t pid = -1;
    const int rc = ::posix_spawnp(&pid, binary.c_str(), &actions, nullptr, argv, environ);
    ::posix_spawn_file_actions_destroy(&actions);

    ::close(inPipe[0]);
    ::close(outPipe[1]);
    ::close(errPipe[1]);

    if (rc != 0) {
        ::close(inPipe[1]);
        ::close(outPipe[0]);
        ::close(errPipe[0]);
        if (errorOut) {
            *errorOut = [NSString stringWithFormat:@"failed to spawn lmp_sidecar: %s", std::strerror(rc)];
        }
        return NO;
    }

    {
        std::lock_guard<std::mutex> lock(_stderrMutex);
        _stderrTail.clear();
    }
    _inWriteFd = inPipe[1];
    _pid.store(pid);
    _runningFlag.store(true);

    __weak SidecarBridge* weakSelf = self;
    const int stdoutFd = outPipe[0];
    const int stderrFd = errPipe[0];

    _stdoutThread = std::thread([weakSelf, stdoutFd]() {
        std::string acc;
        char buf[16384];
        for (;;) {
            const ssize_t n = ::read(stdoutFd, buf, sizeof(buf));
            if (n > 0) {
                acc.append(buf, static_cast<std::size_t>(n));
                std::size_t nl = 0;
                while ((nl = acc.find('\n')) != std::string::npos) {
                    std::string line = acc.substr(0, nl);
                    acc.erase(0, nl + 1);
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    NSString* nsLine = [[NSString alloc] initWithBytes:line.data()
                                                                length:line.size()
                                                              encoding:NSUTF8StringEncoding];
                    if (nsLine) {
                        SidecarBridge* strong = weakSelf;
                        if (strong && strong.onMessage) {
                            void (^cb)(NSString*) = strong.onMessage;
                            dispatch_async(dispatch_get_main_queue(), ^{
                                cb(nsLine);
                            });
                        }
                    }
                }
            } else if (n == 0) {
                break;
            } else if (errno != EINTR) {
                break;
            }
        }
        ::close(stdoutFd);
    });

    _stderrThread = std::thread([weakSelf, stderrFd]() {
        FILE* logFile = std::fopen(app_stderr_log_path().c_str(), "a");
        std::string acc;
        char buf[8192];
        for (;;) {
            const ssize_t n = ::read(stderrFd, buf, sizeof(buf));
            if (n > 0) {
                if (logFile) {
                    std::fwrite(buf, 1, static_cast<std::size_t>(n), logFile);
                    std::fflush(logFile);
                }
                acc.append(buf, static_cast<std::size_t>(n));
                std::size_t nl = 0;
                while ((nl = acc.find('\n')) != std::string::npos) {
                    std::string line = acc.substr(0, nl);
                    acc.erase(0, nl + 1);
                    SidecarBridge* strong = weakSelf;
                    if (strong) {
                        std::lock_guard<std::mutex> lock(strong->_stderrMutex);
                        strong->_stderrTail.push_back(std::move(line));
                        if (strong->_stderrTail.size() > 80) strong->_stderrTail.pop_front();
                    }
                }
            } else if (n == 0) {
                break;
            } else if (errno != EINTR) {
                break;
            }
        }
        if (!acc.empty()) {
            SidecarBridge* strong = weakSelf;
            if (strong) {
                std::lock_guard<std::mutex> lock(strong->_stderrMutex);
                strong->_stderrTail.push_back(std::move(acc));
                if (strong->_stderrTail.size() > 80) strong->_stderrTail.pop_front();
            }
        }
        if (logFile) std::fclose(logFile);
        ::close(stderrFd);
    });

    _waitThread = std::thread([weakSelf, pid]() {
        int status = 0;
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
        int code = -1;
        if (WIFEXITED(status)) code = WEXITSTATUS(status);
        else if (WIFSIGNALED(status)) code = 128 + WTERMSIG(status);

        SidecarBridge* strong = weakSelf;
        if (!strong) return;
        strong->_runningFlag.store(false);
        strong->_pid.store(-1);

        NSMutableArray<NSString*>* tailArr = [NSMutableArray array];
        {
            std::lock_guard<std::mutex> lock(strong->_stderrMutex);
            for (const auto& s : strong->_stderrTail) {
                NSString* item = [NSString stringWithUTF8String:s.c_str()];
                if (item) [tailArr addObject:item];
            }
        }
        if (strong.onExit) {
            void (^cb)(int, NSArray<NSString*>*) = strong.onExit;
            NSArray<NSString*>* frozen = [tailArr copy];
            dispatch_async(dispatch_get_main_queue(), ^{
                cb(code, frozen);
            });
        }
    });

    return YES;
}

- (BOOL)sendRawJsonLine:(NSString*)jsonLine {
    if (!_runningFlag.load() || _inWriteFd < 0 || !jsonLine) {
        return NO;
    }
    NSData* data = [jsonLine dataUsingEncoding:NSUTF8StringEncoding];
    if (!data) return NO;
    std::string frame(static_cast<const char*>(data.bytes), data.length);
    if (frame.empty() || frame.back() != '\n') frame.push_back('\n');
    write_all_fd(_inWriteFd, frame.data(), frame.size(), _writeMutex);
    return YES;
}

- (void)stopSidecar {
    const pid_t pid = _pid.load();
    if (_inWriteFd >= 0) {
        if (_runningFlag.load()) {
            const char shutdownReq[] =
                "{\"jsonrpc\":\"2.0\",\"id\":\"__app_shutdown\",\"method\":\"lmp/shutdown\",\"params\":{}}\n";
            write_all_fd(_inWriteFd, shutdownReq, sizeof(shutdownReq) - 1, _writeMutex);
        }
        {
            std::lock_guard<std::mutex> lock(_writeMutex);
            if (_inWriteFd >= 0) {
                ::close(_inWriteFd);
                _inWriteFd = -1;
            }
        }
    }

    if (pid > 0 && _runningFlag.load()) {
        for (int i = 0; i < 30 && _runningFlag.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (_runningFlag.load()) {
            ::kill(pid, SIGTERM);
            for (int i = 0; i < 10 && _runningFlag.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        }
        if (_runningFlag.load()) {
            ::kill(pid, SIGKILL);
        }
    }

    [self joinThreads];
    _runningFlag.store(false);
    _pid.store(-1);
}

- (void)joinThreads {
    if (_stdoutThread.joinable()) _stdoutThread.join();
    if (_stderrThread.joinable()) _stderrThread.join();
    if (_waitThread.joinable()) _waitThread.join();
}

@end
