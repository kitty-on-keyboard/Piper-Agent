#include "src/platform/fs.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <dirent.h>
#include <fcntl.h>
#include <limits.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace lmp::platform {
namespace {

[[nodiscard]] FsStatus classify_errno(int e) noexcept {
    switch (e) {
        case ENOENT:
            return FsStatus::NotFound;
        case EACCES:
        case EPERM:
            return FsStatus::PermissionDenied;
        case EISDIR:
            return FsStatus::IsDirectory;
        case ELOOP:
            return FsStatus::Symlink;
        case EINVAL:
        case ENAMETOOLONG:
        case ENOTDIR:
            return FsStatus::InvalidPath;
        default:
            return FsStatus::IoError;
    }
}

[[nodiscard]] std::vector<std::string_view> split_components(std::string_view path);

struct ParsedPath {
    FsStatus status = FsStatus::InvalidPath;
    std::vector<std::string> components;
    std::string relative;
    std::string absolute;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok; }
};

[[nodiscard]] bool strip_root(std::string_view root, std::string_view path,
                              std::string_view& relative) {
    if (path == root) {
        relative = ".";
        return true;
    }
    if (root == "/") {
        if (!path.empty() && path.front() == '/') {
            relative = path.substr(1);
            return true;
        }
        return false;
    }
    if (path.size() > root.size() && path.compare(0, root.size(), root) == 0 &&
        path[root.size()] == '/') {
        relative = path.substr(root.size() + 1);
        return true;
    }
    return false;
}

[[nodiscard]] ParsedPath parse_workspace_path(std::string_view requested_root,
                                              std::string_view canonical_root,
                                              std::string_view input) {
    ParsedPath out;
    if (canonical_root.empty() || input.find('\0') != std::string_view::npos) {
        out.error = "invalid workspace path";
        return out;
    }

    // Reject traversal before normalising. Accepting "a/../b" would make this policy
    // depend on lexical cancellation rather than on the descriptor walk below.
    for (std::string_view c : split_components(input)) {
        if (c == "..") {
            out.status = FsStatus::OutsideRoot;
            out.error = std::string(input) + ": '..' is not allowed in workspace paths";
            return out;
        }
    }

    std::string relative_input(input);
    if (!input.empty() && input.front() == '/') {
        const std::string normalized = lexically_normal(input);
        std::string_view stripped;
        if (!strip_root(canonical_root, normalized, stripped) &&
            !strip_root(requested_root, normalized, stripped)) {
            out.status = FsStatus::OutsideRoot;
            out.error = std::string(input) + ": outside the workspace root";
            return out;
        }
        relative_input.assign(stripped);
    }

    for (std::string_view c : split_components(relative_input)) {
        if (c.empty() || c == ".") {
            continue;
        }
        if (c == "..") {
            out.status = FsStatus::OutsideRoot;
            out.error = std::string(input) + ": '..' is not allowed in workspace paths";
            return out;
        }
        out.components.emplace_back(c);
    }

    out.relative = ".";
    if (!out.components.empty()) {
        out.relative.clear();
        for (const std::string& c : out.components) {
            if (!out.relative.empty()) {
                out.relative.push_back('/');
            }
            out.relative += c;
        }
    }
    out.absolute = std::string(canonical_root);
    if (out.relative != ".") {
        if (out.absolute.size() > 1) {
            out.absolute.push_back('/');
        }
        out.absolute += out.relative;
    }
    out.status = FsStatus::Ok;
    return out;
}

[[nodiscard]] int duplicate_fd(int fd) {
    return ::fcntl(fd, F_DUPFD_CLOEXEC, 0);
}

// A directory fd that OWNS ITS OFFSET, for the one caller that reads the offset: readdir.
//
// duplicate_fd() is a dup, and a dup shares the file description -- so it shares the
// directory offset with whatever it was duplicated from. Everywhere else in this file that
// is fine, because traversal only ever calls openat()/fstatat(), which do not touch the
// offset. Enumeration is the exception, and it was doing exactly the wrong thing: for the
// workspace ROOT, open_directory_components() walks zero components and hands back the dup
// of root_fd_ itself, so the first readdir() ran root_fd_ to EOF and left it there. Every
// later listing of the root returned zero entries with status Ok -- an empty workspace, to
// anything reading the result -- and `search` inherited it through walk_regular_files().
// Concurrently batched calls made it nondeterministic on top: two threads enumerating the
// root split the entries between them.
//
// openat(fd, ".") is a fresh file description at offset 0 every time, which is both the fix
// and the reason this cannot regress into a shared-offset bug again.
[[nodiscard]] int reopen_directory_for_scan(int fd) {
    return ::openat(fd, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
}

[[nodiscard]] int open_directory_components(int root_fd,
                                            const std::vector<std::string>& components,
                                            std::size_t count, FsStatus& status,
                                            std::string& error,
                                            std::string_view display_path) {
    int current = duplicate_fd(root_fd);
    if (current < 0) {
        status = classify_errno(errno);
        error = std::string(display_path) + ": duplicate root: " + std::strerror(errno);
        return -1;
    }
    for (std::size_t i = 0; i < count; ++i) {
        struct stat st {};
        if (::fstatat(current, components[i].c_str(), &st,
                      AT_SYMLINK_NOFOLLOW) != 0) {
            const int e = errno;
            ::close(current);
            status = classify_errno(e);
            error = std::string(display_path) + ": " + std::strerror(e);
            return -1;
        }
        if (S_ISLNK(st.st_mode)) {
            ::close(current);
            status = FsStatus::Symlink;
            error = std::string(display_path) + ": symlink components are not allowed";
            return -1;
        }
        if (!S_ISDIR(st.st_mode)) {
            ::close(current);
            status = FsStatus::InvalidPath;
            error = std::string(display_path) + ": a parent component is not a directory";
            return -1;
        }
        const int next =
            ::openat(current, components[i].c_str(),
                     O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) {
            const int e = errno;
            ::close(current);
            status = classify_errno(e);
            error = std::string(display_path) + ": " + std::strerror(e);
            return -1;
        }
        ::close(current);
        current = next;
    }
    status = FsStatus::Ok;
    return current;
}

[[nodiscard]] bool inspect_existing_components(
    int root_fd, const std::vector<std::string>& components, FsStatus& status,
    std::string& error, std::string_view display_path) {
    int current = duplicate_fd(root_fd);
    if (current < 0) {
        status = classify_errno(errno);
        error = std::string(display_path) + ": duplicate root: " + std::strerror(errno);
        return false;
    }
    for (std::size_t i = 0; i < components.size(); ++i) {
        struct stat st {};
        if (::fstatat(current, components[i].c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
            const int e = errno;
            ::close(current);
            if (e == ENOENT) {
                status = FsStatus::Ok; // a creation may legitimately name a missing suffix
                return true;
            }
            status = classify_errno(e);
            error = std::string(display_path) + ": " + std::strerror(e);
            return false;
        }
        if (S_ISLNK(st.st_mode)) {
            ::close(current);
            status = FsStatus::Symlink;
            error = std::string(display_path) + ": symlink components are not allowed";
            return false;
        }
        if (i + 1 == components.size()) {
            break;
        }
        if (!S_ISDIR(st.st_mode)) {
            ::close(current);
            status = FsStatus::InvalidPath;
            error = std::string(display_path) + ": a parent component is not a directory";
            return false;
        }
        const int next = ::openat(current, components[i].c_str(),
                                  O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0) {
            const int e = errno;
            ::close(current);
            status = classify_errno(e);
            error = std::string(display_path) + ": " + std::strerror(e);
            return false;
        }
        ::close(current);
        current = next;
    }
    ::close(current);
    status = FsStatus::Ok;
    return true;
}

[[nodiscard]] int open_parent_for_write(int root_fd,
                                        const std::vector<std::string>& components,
                                        bool create_parents, FsStatus& status,
                                        std::string& error,
                                        std::string_view display_path) {
    int current = duplicate_fd(root_fd);
    if (current < 0) {
        status = classify_errno(errno);
        error = std::string(display_path) + ": duplicate root: " + std::strerror(errno);
        return -1;
    }
    for (std::size_t i = 0; i + 1 < components.size(); ++i) {
        struct stat st {};
        if (::fstatat(current, components[i].c_str(), &st,
                      AT_SYMLINK_NOFOLLOW) == 0) {
            if (S_ISLNK(st.st_mode)) {
                ::close(current);
                status = FsStatus::Symlink;
                error =
                    std::string(display_path) + ": symlink components are not allowed";
                return -1;
            }
            if (!S_ISDIR(st.st_mode)) {
                ::close(current);
                status = FsStatus::InvalidPath;
                error =
                    std::string(display_path) + ": a parent component is not a directory";
                return -1;
            }
        } else if (errno != ENOENT) {
            const int e = errno;
            ::close(current);
            status = classify_errno(e);
            error = std::string(display_path) + ": " + std::strerror(e);
            return -1;
        }
        int next = ::openat(current, components[i].c_str(),
                            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (next < 0 && errno == ENOENT && create_parents) {
            if (::mkdirat(current, components[i].c_str(), 0755) != 0 && errno != EEXIST) {
                const int e = errno;
                ::close(current);
                status = classify_errno(e);
                error = std::string(display_path) + ": mkdir: " + std::strerror(e);
                return -1;
            }
            next = ::openat(current, components[i].c_str(),
                            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        }
        if (next < 0) {
            const int e = errno;
            ::close(current);
            status = classify_errno(e);
            error = std::string(display_path) + ": " + std::strerror(e);
            return -1;
        }
        ::close(current);
        current = next;
    }
    status = FsStatus::Ok;
    return current;
}

[[nodiscard]] std::vector<std::string_view> split_components(std::string_view path) {
    std::vector<std::string_view> parts;
    std::size_t i = 0;
    while (i < path.size()) {
        const std::size_t next = path.find('/', i);
        const std::size_t end = (next == std::string_view::npos) ? path.size() : next;
        if (end > i) {
            parts.push_back(path.substr(i, end - i));
        }
        i = end + 1;
    }
    return parts;
}

// Applies "." and ".." to a component stack. `absolute` decides whether a ".." that
// would escape is dropped (absolute: there is no above-root) or kept (relative: it is
// meaningful and discarding it would silently relocate the path).
void apply_component(std::vector<std::string_view>& stack, std::string_view c, bool absolute) {
    if (c == ".") {
        return;
    }
    if (c != "..") {
        stack.push_back(c);
        return;
    }
    if (!stack.empty() && stack.back() != "..") {
        stack.pop_back();
        return;
    }
    if (!absolute) {
        stack.push_back("..");
    }
}

[[nodiscard]] std::vector<std::string_view> normalized_components(std::string_view path,
                                                                 bool absolute) {
    std::vector<std::string_view> stack;
    for (std::string_view c : split_components(path)) {
        apply_component(stack, c, absolute);
    }
    return stack;
}

[[nodiscard]] bool read_fd_exactly(int fd, std::size_t size, std::string& out) {
    out.resize(size);
    std::size_t done = 0;
    while (done < size) {
        const ssize_t n = ::read(fd, out.data() + done, size - done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false; // short read is a failure, never a truncated success
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

[[nodiscard]] bool write_fd_all(int fd, std::string_view bytes) {
    std::size_t done = 0;
    while (done < bytes.size()) {
        const ssize_t n = ::write(fd, bytes.data() + done, bytes.size() - done);
        if (n < 0 && errno == EINTR) {
            continue;
        }
        if (n <= 0) {
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
}

[[nodiscard]] FileContents read_open_fd(int fd, std::string_view display_path,
                                        std::size_t max_bytes) {
    FileContents out;
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        const int e = errno;
        out.status = classify_errno(e);
        out.error = std::string(display_path) + ": fstat: " + std::strerror(e);
        ::close(fd);
        return out;
    }
    if (S_ISDIR(st.st_mode)) {
        out.status = FsStatus::IsDirectory;
        out.error = std::string(display_path) + ": is a directory";
        ::close(fd);
        return out;
    }
    if (!S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<std::uintmax_t>(st.st_size) >
            static_cast<std::uintmax_t>(std::numeric_limits<std::size_t>::max())) {
        out.status = FsStatus::IoError;
        out.error = std::string(display_path) + ": is not a regular file";
        ::close(fd);
        return out;
    }
    const auto size = static_cast<std::size_t>(st.st_size);
    if (size > max_bytes) {
        out.status = FsStatus::TooLarge;
        out.actual_size = size;
        out.error = std::string(display_path) + ": " + std::to_string(size) +
                    " bytes exceeds the " + std::to_string(max_bytes) +
                    "-byte limit; not returning a prefix";
        ::close(fd);
        return out;
    }
    if (!read_fd_exactly(fd, size, out.bytes)) {
        out.status = FsStatus::IoError;
        out.error = std::string(display_path) +
                    ": short read; the file changed under us or the device failed";
        out.bytes.clear();
        ::close(fd);
        return out;
    }
    ::close(fd);
    out.status = FsStatus::Ok;
    return out;
}

[[nodiscard]] std::string sha256_hex_of_fd(int fd) {
    if (::lseek(fd, 0, SEEK_SET) != 0) {
        return {};
    }
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    std::array<unsigned char, 1U << 16> buf{};
    while (true) {
        const ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return {};
        }
        if (n == 0) {
            break;
        }
        CC_SHA256_Update(&ctx, buf.data(), static_cast<CC_LONG>(n));
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &ctx);
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex(digest.size() * 2, '\0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        hex[i * 2] = kHex[digest[i] >> 4];
        hex[i * 2 + 1] = kHex[digest[i] & 0x0F];
    }
    return hex;
}

// Re-checks the live destination immediately before rename. Anything else is a
// check-then-write race the optimistic claim is meant to close.
[[nodiscard]] WriteResult check_write_precondition(int parent_fd, std::string_view leaf,
                                                   std::string_view display_path,
                                                   const WritePrecondition& pre) {
    if (!pre.active()) {
        return {FsStatus::Ok, {}};
    }
    const std::string leaf_string(leaf);
    struct stat target {};
    if (::fstatat(parent_fd, leaf_string.c_str(), &target, AT_SYMLINK_NOFOLLOW) != 0) {
        if (errno == ENOENT) {
            if (pre.expected_absent) {
                return {FsStatus::Ok, {}};
            }
            return {FsStatus::Conflict,
                    std::string(display_path) +
                        ": expected content version " + pre.expected_version +
                        " but the file is absent; read it again before editing"};
        }
        const int e = errno;
        return {classify_errno(e),
                std::string(display_path) + ": inspect target: " + std::strerror(e)};
    }
    if (pre.expected_absent) {
        return {FsStatus::Conflict,
                std::string(display_path) +
                    ": expected_absent but the file already exists; read it before "
                    "overwriting"};
    }
    if (S_ISLNK(target.st_mode)) {
        return {FsStatus::Symlink,
                std::string(display_path) + ": refusing to replace a symlink"};
    }
    if (S_ISDIR(target.st_mode)) {
        return {FsStatus::IsDirectory, std::string(display_path) + ": is a directory"};
    }
    if (!S_ISREG(target.st_mode)) {
        return {FsStatus::InvalidPath,
                std::string(display_path) + ": is not a regular file"};
    }
    const int fd =
        ::openat(parent_fd, leaf_string.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        const int e = errno;
        return {classify_errno(e),
                std::string(display_path) + ": open for version check: " +
                    std::strerror(e)};
    }
    const std::string actual = sha256_hex_of_fd(fd);
    ::close(fd);
    if (actual.empty()) {
        return {FsStatus::IoError,
                std::string(display_path) + ": could not hash current contents"};
    }
    if (actual != pre.expected_version) {
        return {FsStatus::Conflict,
                std::string(display_path) +
                    ": content version conflict (expected " + pre.expected_version +
                    ", current " + actual +
                    "); read the file again and retry the edit"};
    }
    return {FsStatus::Ok, {}};
}

[[nodiscard]] WriteResult write_atomic_at(int parent_fd, std::string_view leaf,
                                          std::string_view display_path,
                                          std::string_view bytes,
                                          const WritePrecondition& pre) {
    struct stat target {};
    bool target_exists = false;
    if (::fstatat(parent_fd, std::string(leaf).c_str(), &target,
                  AT_SYMLINK_NOFOLLOW) == 0) {
        target_exists = true;
        if (S_ISLNK(target.st_mode)) {
            return {FsStatus::Symlink,
                    std::string(display_path) + ": refusing to replace a symlink"};
        }
        if (S_ISDIR(target.st_mode)) {
            return {FsStatus::IsDirectory,
                    std::string(display_path) + ": is a directory"};
        }
        if (!S_ISREG(target.st_mode)) {
            return {FsStatus::InvalidPath,
                    std::string(display_path) + ": is not a regular file"};
        }
    } else if (errno != ENOENT) {
        const int e = errno;
        return {classify_errno(e),
                std::string(display_path) + ": inspect target: " + std::strerror(e)};
    }

    const mode_t mode =
        target_exists ? static_cast<mode_t>(target.st_mode & 07777) : mode_t{0644};
    int temp_fd = -1;
    std::string temp_name;
    for (int attempt = 0; attempt < 128; ++attempt) {
        std::uint64_t random[2] = {};
        ::arc4random_buf(random, sizeof(random));
        char suffix[48];
        std::snprintf(suffix, sizeof(suffix), ".lmp-tmp-%016llx%016llx",
                      static_cast<unsigned long long>(random[0]),
                      static_cast<unsigned long long>(random[1]));
        temp_name = suffix;
        temp_fd = ::openat(parent_fd, temp_name.c_str(),
                           O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
        if (temp_fd >= 0) {
            break;
        }
        if (errno != EEXIST) {
            const int e = errno;
            return {classify_errno(e),
                    std::string(display_path) + ": create temporary: " +
                        std::strerror(e)};
        }
    }
    if (temp_fd < 0) {
        return {FsStatus::IoError,
                std::string(display_path) + ": could not allocate a unique temporary"};
    }

    const auto fail_with_cleanup = [&](FsStatus status, const std::string& detail) {
        const int saved = errno;
        if (temp_fd >= 0) {
            ::close(temp_fd);
            temp_fd = -1;
        }
        (void)::unlinkat(parent_fd, temp_name.c_str(), 0);
        errno = saved;
        return WriteResult{status, std::string(display_path) + ": " + detail};
    };

    if (target_exists && ::fchmod(temp_fd, mode) != 0) {
        const int e = errno;
        return fail_with_cleanup(classify_errno(e),
                                 "preserve mode: " + std::string(std::strerror(e)));
    }
    if (!write_fd_all(temp_fd, bytes)) {
        const int e = errno;
        return fail_with_cleanup(FsStatus::IoError,
                                 "write temporary: " + std::string(std::strerror(e)));
    }
    if (::fsync(temp_fd) != 0) {
        const int e = errno;
        return fail_with_cleanup(FsStatus::IoError,
                                 "fsync temporary: " + std::string(std::strerror(e)));
    }
    if (::close(temp_fd) != 0) {
        const int e = errno;
        temp_fd = -1;
        (void)::unlinkat(parent_fd, temp_name.c_str(), 0);
        return {FsStatus::IoError,
                std::string(display_path) + ": close temporary: " + std::strerror(e)};
    }
    temp_fd = -1;

    // IMMEDIATELY before rename: the optimistic claim is about the destination that is
    // about to be replaced, not the one we looked at before writing the temporary.
    const WriteResult pre_check =
        check_write_precondition(parent_fd, leaf, display_path, pre);
    if (!pre_check.ok()) {
        (void)::unlinkat(parent_fd, temp_name.c_str(), 0);
        return pre_check;
    }

    const std::string leaf_string(leaf);
    if (::renameat(parent_fd, temp_name.c_str(), parent_fd, leaf_string.c_str()) != 0) {
        const int e = errno;
        (void)::unlinkat(parent_fd, temp_name.c_str(), 0);
        return {classify_errno(e),
                std::string(display_path) + ": rename: " + std::strerror(e)};
    }
    if (::fsync(parent_fd) != 0) {
        const int e = errno;
        return {FsStatus::IoError,
                std::string(display_path) + ": fsync parent: " + std::strerror(e)};
    }
    return {FsStatus::Ok, {}};
}

} // namespace

std::string_view to_string(FsStatus s) noexcept {
    switch (s) {
        case FsStatus::Ok:
            return "Ok";
        case FsStatus::NotFound:
            return "NotFound";
        case FsStatus::PermissionDenied:
            return "PermissionDenied";
        case FsStatus::IsDirectory:
            return "IsDirectory";
        case FsStatus::TooLarge:
            return "TooLarge";
        case FsStatus::OutsideRoot:
            return "OutsideRoot";
        case FsStatus::InvalidPath:
            return "InvalidPath";
        case FsStatus::Symlink:
            return "Symlink";
        case FsStatus::Conflict:
            return "Conflict";
        case FsStatus::IoError:
            return "IoError";
    }
    return "IoError";
}

std::string content_sha256_hex(std::string_view data) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(data.data(), static_cast<CC_LONG>(data.size()), digest.data());
    constexpr char kHex[] = "0123456789abcdef";
    std::string hex(digest.size() * 2, '\0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        hex[i * 2] = kHex[digest[i] >> 4];
        hex[i * 2 + 1] = kHex[digest[i] & 0x0F];
    }
    return hex;
}

FileContents read_file_whole(const std::string& path, std::size_t max_bytes) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        FileContents out;
        out.status = classify_errno(errno);
        out.error = path + ": " + std::strerror(errno);
        return out;
    }
    return read_open_fd(fd, path, max_bytes);
}

WriteResult write_file_atomic(const std::string& path, std::string_view bytes,
                             const WritePrecondition& pre) {
    const std::size_t slash = path.find_last_of('/');
    const std::string parent =
        slash == std::string::npos ? "." : (slash == 0 ? "/" : path.substr(0, slash));
    const std::string leaf =
        slash == std::string::npos ? path : path.substr(slash + 1);
    if (leaf.empty() || leaf == "." || leaf == "..") {
        return {FsStatus::InvalidPath, path + ": invalid file name"};
    }
    const int parent_fd =
        ::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (parent_fd < 0) {
        const int e = errno;
        return {classify_errno(e), path + ": open parent: " + std::strerror(e)};
    }
    WriteResult out = write_atomic_at(parent_fd, leaf, path, bytes, pre);
    ::close(parent_fd);
    return out;
}

OpenedFile::~OpenedFile() {
    if (fd >= 0) {
        ::close(fd);
    }
}

OpenedFile::OpenedFile(OpenedFile&& other) noexcept
    : fd(std::exchange(other.fd, -1)),
      status(other.status),
      error(std::move(other.error)) {}

OpenedFile& OpenedFile::operator=(OpenedFile&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (fd >= 0) {
        ::close(fd);
    }
    fd = std::exchange(other.fd, -1);
    status = other.status;
    error = std::move(other.error);
    return *this;
}

WorkspaceFs::WorkspaceFs(std::string root) {
    char cwd[PATH_MAX];
    if (!root.empty() && root.front() == '/') {
        requested_root_ = lexically_normal(root);
    } else if (::getcwd(cwd, sizeof(cwd)) != nullptr) {
        requested_root_ = resolve_against(cwd, root);
    }

    char canonical[PATH_MAX];
    if (::realpath(root.c_str(), canonical) == nullptr) {
        setup_error_ = root + ": realpath workspace root: " + std::strerror(errno);
        return;
    }
    canonical_root_ = canonical;
    root_fd_ = ::open(canonical_root_.c_str(),
                      O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (root_fd_ < 0) {
        setup_error_ =
            canonical_root_ + ": open workspace root: " + std::strerror(errno);
        canonical_root_.clear();
    }
}

WorkspaceFs::~WorkspaceFs() {
    if (root_fd_ >= 0) {
        ::close(root_fd_);
    }
}

WorkspaceFs::WorkspaceFs(WorkspaceFs&& other) noexcept
    : root_fd_(std::exchange(other.root_fd_, -1)),
      requested_root_(std::move(other.requested_root_)),
      canonical_root_(std::move(other.canonical_root_)),
      setup_error_(std::move(other.setup_error_)) {}

WorkspaceFs& WorkspaceFs::operator=(WorkspaceFs&& other) noexcept {
    if (this == &other) {
        return *this;
    }
    if (root_fd_ >= 0) {
        ::close(root_fd_);
    }
    root_fd_ = std::exchange(other.root_fd_, -1);
    requested_root_ = std::move(other.requested_root_);
    canonical_root_ = std::move(other.canonical_root_);
    setup_error_ = std::move(other.setup_error_);
    return *this;
}

ContainedPath WorkspaceFs::contained_path(std::string_view path) const {
    const ParsedPath parsed =
        parse_workspace_path(requested_root_, canonical_root_, path);
    if (!parsed.ok()) {
        return {parsed.status, {}, {}, parsed.error};
    }
    if (!valid()) {
        return {FsStatus::InvalidPath, {}, {}, setup_error_};
    }
    FsStatus status = FsStatus::IoError;
    std::string error;
    if (!inspect_existing_components(root_fd_, parsed.components, status, error, path)) {
        return {status, {}, {}, std::move(error)};
    }
    return {FsStatus::Ok, parsed.relative, parsed.absolute, {}};
}

OpenedFile WorkspaceFs::open_file_readonly(std::string_view path,
                                           bool inherit_across_exec) const {
    OpenedFile out;
    const ParsedPath parsed =
        parse_workspace_path(requested_root_, canonical_root_, path);
    if (!parsed.ok() || !valid()) {
        out.status = parsed.ok() ? FsStatus::InvalidPath : parsed.status;
        out.error = parsed.ok() ? setup_error_ : parsed.error;
        return out;
    }
    if (parsed.components.empty()) {
        out.status = FsStatus::IsDirectory;
        out.error = std::string(path) + ": workspace root is a directory";
        return out;
    }
    FsStatus status = FsStatus::IoError;
    std::string error;
    const int parent =
        open_directory_components(root_fd_, parsed.components,
                                  parsed.components.size() - 1, status, error, path);
    if (parent < 0) {
        out.status = status;
        out.error = std::move(error);
        return out;
    }
    const int flags =
        O_RDONLY | O_NOFOLLOW | (inherit_across_exec ? 0 : O_CLOEXEC);
    out.fd = ::openat(parent, parsed.components.back().c_str(), flags);
    if (out.fd < 0) {
        const int e = errno;
        ::close(parent);
        out.status = classify_errno(e);
        out.error = std::string(path) + ": " + std::strerror(e);
        return out;
    }
    ::close(parent);
    struct stat st {};
    if (::fstat(out.fd, &st) != 0) {
        const int e = errno;
        ::close(out.fd);
        out.fd = -1;
        out.status = classify_errno(e);
        out.error = std::string(path) + ": fstat: " + std::strerror(e);
        return out;
    }
    if (S_ISDIR(st.st_mode) || !S_ISREG(st.st_mode)) {
        ::close(out.fd);
        out.fd = -1;
        out.status = S_ISDIR(st.st_mode) ? FsStatus::IsDirectory
                                        : FsStatus::InvalidPath;
        out.error = std::string(path) +
                    (S_ISDIR(st.st_mode) ? ": is a directory"
                                         : ": is not a regular file");
        return out;
    }
    out.status = FsStatus::Ok;
    return out;
}

FileContents WorkspaceFs::read_file_whole(std::string_view path,
                                          std::size_t max_bytes) const {
    const ParsedPath parsed =
        parse_workspace_path(requested_root_, canonical_root_, path);
    if (!parsed.ok() || !valid()) {
        return {parsed.ok() ? FsStatus::InvalidPath : parsed.status, {},
                parsed.ok() ? setup_error_ : parsed.error, 0};
    }
    if (parsed.components.empty()) {
        const int fd = duplicate_fd(root_fd_);
        if (fd < 0) {
            const int e = errno;
            return {classify_errno(e), {},
                    std::string(path) + ": duplicate root: " + std::strerror(e), 0};
        }
        return read_open_fd(fd, path, max_bytes);
    }

    FsStatus status = FsStatus::IoError;
    std::string error;
    const int parent =
        open_directory_components(root_fd_, parsed.components,
                                  parsed.components.size() - 1, status, error, path);
    if (parent < 0) {
        return {status, {}, std::move(error), 0};
    }
    const int fd = ::openat(parent, parsed.components.back().c_str(),
                            O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) {
        const int e = errno;
        ::close(parent);
        return {classify_errno(e), {},
                std::string(path) + ": " + std::strerror(e), 0};
    }
    ::close(parent);
    return read_open_fd(fd, path, max_bytes);
}

WriteResult WorkspaceFs::write_file_atomic(std::string_view path,
                                           std::string_view bytes,
                                           bool create_parents,
                                           const WritePrecondition& pre) const {
    const ParsedPath parsed =
        parse_workspace_path(requested_root_, canonical_root_, path);
    if (!parsed.ok() || !valid()) {
        return {parsed.ok() ? FsStatus::InvalidPath : parsed.status,
                parsed.ok() ? setup_error_ : parsed.error};
    }
    if (parsed.components.empty()) {
        return {FsStatus::IsDirectory,
                std::string(path) + ": workspace root is a directory"};
    }
    FsStatus status = FsStatus::IoError;
    std::string error;
    const int parent = open_parent_for_write(root_fd_, parsed.components,
                                             create_parents, status, error, path);
    if (parent < 0) {
        return {status, std::move(error)};
    }
    WriteResult out =
        write_atomic_at(parent, parsed.components.back(), path, bytes, pre);
    ::close(parent);
    return out;
}

DirectoryContents WorkspaceFs::list_directory(std::string_view path) const {
    const ParsedPath parsed =
        parse_workspace_path(requested_root_, canonical_root_, path);
    if (!parsed.ok() || !valid()) {
        return {parsed.ok() ? FsStatus::InvalidPath : parsed.status, {},
                parsed.ok() ? setup_error_ : parsed.error};
    }
    FsStatus status = FsStatus::IoError;
    std::string error;
    const int walked = open_directory_components(root_fd_, parsed.components,
                                                 parsed.components.size(), status, error,
                                                 path);
    if (walked < 0) {
        return {status, {}, std::move(error)};
    }
    // Never enumerate the fd the walk handed back: at the root it is a dup of root_fd_ and
    // shares its offset. See reopen_directory_for_scan().
    const int fd = reopen_directory_for_scan(walked);
    ::close(walked);
    if (fd < 0) {
        const int e = errno;
        return {classify_errno(e), {},
                std::string(path) + ": reopen for scan: " + std::strerror(e)};
    }
    DIR* dir = ::fdopendir(fd);
    if (dir == nullptr) {
        const int e = errno;
        ::close(fd);
        return {classify_errno(e), {},
                std::string(path) + ": fdopendir: " + std::strerror(e)};
    }

    DirectoryContents out;
    out.status = FsStatus::Ok;
    int read_error = 0;
    while (true) {
        errno = 0;
        dirent* entry = ::readdir(dir);
        if (entry == nullptr) {
            read_error = errno;
            break;
        }
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        DirectoryEntryKind kind = DirectoryEntryKind::Other;
        struct stat st {};
        if (::fstatat(::dirfd(dir), name.c_str(), &st, AT_SYMLINK_NOFOLLOW) == 0) {
            kind = S_ISLNK(st.st_mode)   ? DirectoryEntryKind::Symlink
                   : S_ISDIR(st.st_mode) ? DirectoryEntryKind::Directory
                   : S_ISREG(st.st_mode) ? DirectoryEntryKind::File
                                         : DirectoryEntryKind::Other;
        }
        out.entries.push_back({name, kind});
    }
    ::closedir(dir);
    if (read_error != 0) {
        out.status = classify_errno(read_error);
        out.error = std::string(path) + ": readdir: " + std::strerror(read_error);
        out.entries.clear();
    }
    return out;
}

RemoveResult WorkspaceFs::remove_file(std::string_view path,
                                      std::string_view expected_version) const {
    const ParsedPath parsed =
        parse_workspace_path(requested_root_, canonical_root_, path);
    if (!parsed.ok() || !valid()) {
        return {parsed.ok() ? FsStatus::InvalidPath : parsed.status, 0,
                parsed.ok() ? setup_error_ : parsed.error};
    }
    if (parsed.components.empty()) {
        return {FsStatus::IsDirectory, 0,
                std::string(path) + ": workspace root is a directory"};
    }
    FsStatus status = FsStatus::IoError;
    std::string error;
    const int parent =
        open_directory_components(root_fd_, parsed.components,
                                  parsed.components.size() - 1, status, error, path);
    if (parent < 0) {
        return {status, 0, std::move(error)};
    }
    struct stat st {};
    if (::fstatat(parent, parsed.components.back().c_str(), &st,
                  AT_SYMLINK_NOFOLLOW) != 0) {
        const int e = errno;
        ::close(parent);
        return {classify_errno(e), 0,
                std::string(path) + ": " + std::strerror(e)};
    }
    if (S_ISLNK(st.st_mode)) {
        ::close(parent);
        return {FsStatus::Symlink, 0,
                std::string(path) + ": refusing to remove a symlink"};
    }
    if (S_ISDIR(st.st_mode)) {
        ::close(parent);
        return {FsStatus::IsDirectory, 0,
                std::string(path) + ": is a directory"};
    }
    if (!S_ISREG(st.st_mode)) {
        ::close(parent);
        return {FsStatus::InvalidPath, 0,
                std::string(path) + ": is not a regular file"};
    }
    if (!expected_version.empty()) {
        const int fd = ::openat(parent, parsed.components.back().c_str(),
                                O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        if (fd < 0) {
            const int e = errno;
            ::close(parent);
            return {classify_errno(e), 0,
                    std::string(path) + ": open for version check: " + std::strerror(e)};
        }
        const std::string actual = sha256_hex_of_fd(fd);
        ::close(fd);
        if (actual.empty()) {
            ::close(parent);
            return {FsStatus::IoError, 0,
                    std::string(path) + ": could not hash current contents"};
        }
        if (actual != expected_version) {
            ::close(parent);
            return {FsStatus::Conflict, 0,
                    std::string(path) + ": content version conflict (expected " +
                        std::string(expected_version) + ", current " + actual +
                        "); read the file again before deleting"};
        }
    }
    if (::unlinkat(parent, parsed.components.back().c_str(), 0) != 0) {
        const int e = errno;
        ::close(parent);
        return {classify_errno(e), 0,
                std::string(path) + ": unlink: " + std::strerror(e)};
    }
    const std::size_t size =
        st.st_size > 0 ? static_cast<std::size_t>(st.st_size) : std::size_t{0};
    if (::fsync(parent) != 0) {
        const int e = errno;
        ::close(parent);
        return {FsStatus::IoError, size,
                std::string(path) + ": fsync parent: " + std::strerror(e)};
    }
    ::close(parent);
    return {FsStatus::Ok, size, {}};
}

std::string lexically_normal(std::string_view path) {
    if (path.empty()) {
        return {};
    }
    const bool absolute = path.front() == '/';
    const std::vector<std::string_view> stack = normalized_components(path, absolute);
    std::string out;
    out.reserve(path.size());
    for (std::string_view c : stack) {
        out.push_back('/');
        out.append(c);
    }
    if (!absolute) {
        // Drop the leading separator the loop added; a relative path stays relative.
        return out.empty() ? std::string(".") : out.substr(1);
    }
    return out.empty() ? std::string("/") : out;
}

std::string resolve_against(std::string_view base, std::string_view p) {
    if (!p.empty() && p.front() == '/') {
        return lexically_normal(p);
    }
    std::string joined(base);
    if (!joined.empty() && joined.back() != '/') {
        joined.push_back('/');
    }
    joined.append(p);
    return lexically_normal(joined);
}

bool is_within(std::string_view root, std::string_view path) {
    const std::string nr = lexically_normal(root);
    const std::string np = lexically_normal(path);
    if (nr.empty() || np.empty()) {
        return false;
    }
    const std::vector<std::string_view> rc = normalized_components(nr, nr.front() == '/');
    const std::vector<std::string_view> pc = normalized_components(np, np.front() == '/');
    if ((nr.front() == '/') != (np.front() == '/')) {
        return false; // comparing an absolute path against a relative root is a bug
    }
    if (pc.size() < rc.size()) {
        return false;
    }
    for (std::size_t i = 0; i < rc.size(); ++i) {
        if (rc[i] != pc[i]) {
            return false;
        }
    }
    return true;
}

} // namespace lmp::platform
