#include "src/platform/fs.hpp"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

namespace lmp::platform {
namespace {

[[nodiscard]] FsStatus classify_errno(int e) noexcept {
    switch (e) {
        case ENOENT:
        case ENOTDIR:
            return FsStatus::NotFound;
        case EACCES:
        case EPERM:
            return FsStatus::PermissionDenied;
        case EISDIR:
            return FsStatus::IsDirectory;
        default:
            return FsStatus::IoError;
    }
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
        if (n < 0) {
            return false;
        }
        done += static_cast<std::size_t>(n);
    }
    return true;
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
        case FsStatus::IoError:
            return "IoError";
    }
    return "IoError";
}

FileContents read_file_whole(const std::string& path, std::size_t max_bytes) {
    FileContents out;
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        out.status = classify_errno(errno);
        out.error = path + ": " + std::strerror(errno);
        return out;
    }
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        out.status = FsStatus::IoError;
        out.error = path + ": fstat: " + std::strerror(errno);
        ::close(fd);
        return out;
    }
    if (S_ISDIR(st.st_mode)) {
        out.status = FsStatus::IsDirectory;
        out.error = path + ": is a directory";
        ::close(fd);
        return out;
    }
    const auto size = static_cast<std::size_t>(st.st_size);
    if (size > max_bytes) {
        out.status = FsStatus::TooLarge;
        out.actual_size = size;
        out.error = path + ": " + std::to_string(size) + " bytes exceeds the " +
                    std::to_string(max_bytes) + "-byte limit; not returning a prefix";
        ::close(fd);
        return out;
    }
    if (!read_fd_exactly(fd, size, out.bytes)) {
        out.status = FsStatus::IoError;
        out.error = path + ": short read; the file changed under us or the device failed";
        out.bytes.clear();
        ::close(fd);
        return out;
    }
    ::close(fd);
    out.status = FsStatus::Ok;
    return out;
}

WriteResult write_file_atomic(const std::string& path, std::string_view bytes) {
    const std::string tmp = path + ".tmp." + std::to_string(::getpid());
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return {classify_errno(errno), tmp + ": " + std::strerror(errno)};
    }
    if (!write_fd_all(fd, bytes) || ::fsync(fd) != 0) {
        const std::string err = tmp + ": " + std::strerror(errno);
        ::close(fd);
        ::unlink(tmp.c_str());
        return {FsStatus::IoError, err};
    }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0) {
        const std::string err = path + ": rename: " + std::strerror(errno);
        ::unlink(tmp.c_str());
        return {classify_errno(errno), err};
    }
    return {FsStatus::Ok, {}};
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
