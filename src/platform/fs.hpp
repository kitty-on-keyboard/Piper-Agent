#pragma once
//
// Filesystem primitives (spec S3, L0).
//
// WHOLE FILE OR HONEST ERROR. v1's read primitive returned a dishonest partial on the
// paths it could not fully satisfy, and four separate "rescuer" layers grew on top of it
// to detect and repair the lie. None of them existed because reading files is hard; they
// existed because the primitive under them reported success for a partial result. Here a
// read either yields every byte of the file or a typed failure that names why -- and
// TooLarge carries the real size, so the caller can decide, which is the decision the
// truncating read was silently making for it.
//
// The pure path helpers at the bottom remain lexical predicates. WorkspaceFs is the
// authorization primitive: it opens a canonical workspace root once and resolves every
// component relative to that descriptor with no symlink following.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::platform {

enum class FsStatus : std::uint8_t {
    Ok = 0,
    NotFound,
    PermissionDenied,
    IsDirectory,
    TooLarge,
    OutsideRoot,
    InvalidPath,
    Symlink,
    Conflict, // optimistic-concurrency precondition failed
    IoError,
};

[[nodiscard]] std::string_view to_string(FsStatus s) noexcept;

struct FileContents {
    FsStatus status = FsStatus::IoError;
    // Populated if and only if status == Ok, and then it is EVERY byte of the file.
    // There is no state of this struct that means "some of the file".
    std::string bytes;
    std::string error;
    // Set on TooLarge so the caller can act on the real number rather than guess.
    std::size_t actual_size = 0;

    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok; }
};

// `max_bytes` is required. A file larger than it yields TooLarge with actual_size set,
// never a prefix.
[[nodiscard]] FileContents read_file_whole(const std::string& path, std::size_t max_bytes);

struct WriteResult {
    FsStatus status = FsStatus::IoError;
    std::string error;
    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok; }
};

// SHA-256 of `data`, lowercase hex. Stable across processes; used as the content
// version for optimistic concurrency on workspace edits.
[[nodiscard]] std::string content_sha256_hex(std::string_view data);

// Optimistic-concurrency claim checked immediately before the atomic rename.
//
// Empty / default means unconstrained (internal writers such as spool and memory).
// A workspace tool write always carries either `expected_absent` (create) or a
// non-empty `expected_version` (update) so a newer preimage cannot be replaced silently.
struct WritePrecondition {
    bool expected_absent = false;
    std::string expected_version; // lowercase SHA-256 hex of the bytes that must be there

    [[nodiscard]] bool active() const noexcept {
        return expected_absent || !expected_version.empty();
    }
};

// Writes to a sibling temporary, fsyncs it, then rename()s into place. A reader either
// sees the old file or the new one; a crash mid-write cannot leave a half file that
// parses as a short one. When `pre` is active, the claim is re-checked on the live
// destination immediately before rename.
[[nodiscard]] WriteResult write_file_atomic(const std::string& path, std::string_view bytes,
                                           const WritePrecondition& pre = {});

struct ContainedPath {
    FsStatus status = FsStatus::InvalidPath;
    std::string relative;
    std::string absolute;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok; }
};

enum class DirectoryEntryKind : std::uint8_t {
    File,
    Directory,
    Symlink,
    Other,
};

struct DirectoryEntry {
    std::string name;
    DirectoryEntryKind kind = DirectoryEntryKind::Other;
};

struct DirectoryContents {
    FsStatus status = FsStatus::IoError;
    std::vector<DirectoryEntry> entries;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok; }
};

struct RemoveResult {
    FsStatus status = FsStatus::IoError;
    std::size_t removed_size = 0;
    std::string error;

    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok; }
};

class OpenedFile {
  public:
    OpenedFile() = default;
    ~OpenedFile();
    OpenedFile(const OpenedFile&) = delete;
    OpenedFile& operator=(const OpenedFile&) = delete;
    OpenedFile(OpenedFile&& other) noexcept;
    OpenedFile& operator=(OpenedFile&& other) noexcept;

    [[nodiscard]] bool ok() const noexcept { return status == FsStatus::Ok && fd >= 0; }

    int fd = -1;
    FsStatus status = FsStatus::IoError;
    std::string error;
};

// A descriptor-rooted capability for one workspace. The root may itself be a symlink:
// construction resolves it once, then opens the canonical directory and never traverses
// through the original spelling again. All operation paths reject "..", outside absolute
// paths, and symlink components. The class is move-only so the root descriptor has one
// owner.
class WorkspaceFs {
  public:
    explicit WorkspaceFs(std::string root);
    ~WorkspaceFs();

    WorkspaceFs(const WorkspaceFs&) = delete;
    WorkspaceFs& operator=(const WorkspaceFs&) = delete;
    WorkspaceFs(WorkspaceFs&& other) noexcept;
    WorkspaceFs& operator=(WorkspaceFs&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return root_fd_ >= 0; }
    [[nodiscard]] const std::string& root() const noexcept { return canonical_root_; }
    [[nodiscard]] const std::string& error() const noexcept { return setup_error_; }

    // Validates the spelling and all components that currently exist. Missing suffixes
    // are allowed so callers can authorize creation; the actual operation repeats the
    // walk with openat(), which is what closes the check/open race.
    [[nodiscard]] ContainedPath contained_path(std::string_view path) const;

    [[nodiscard]] FileContents read_file_whole(std::string_view path,
                                               std::size_t max_bytes) const;
    // `inherit_across_exec` exists for syntax checkers that bind a subprocess to this
    // already-open file through /dev/fd rather than handing it a raceable pathname.
    [[nodiscard]] OpenedFile open_file_readonly(
        std::string_view path, bool inherit_across_exec = false) const;
    [[nodiscard]] WriteResult write_file_atomic(
        std::string_view path, std::string_view bytes, bool create_parents = true,
        const WritePrecondition& pre = {}) const;
    [[nodiscard]] DirectoryContents list_directory(std::string_view path) const;
    // When `expected_version` is non-empty, the file's current bytes must hash to it
    // immediately before unlink; otherwise Conflict and the file is left alone.
    [[nodiscard]] RemoveResult remove_file(std::string_view path,
                                           std::string_view expected_version = {}) const;

  private:
    int root_fd_ = -1;
    std::string requested_root_;
    std::string canonical_root_;
    std::string setup_error_;
};

// --- pure path algebra: no syscalls, no globals ----------------------------

// Removes "." and resolves ".." textually. On an absolute path, ".." at the root is
// dropped (matching realpath); on a relative one, a leading ".." is preserved because
// there is nothing to pop.
[[nodiscard]] std::string lexically_normal(std::string_view path);

// If `p` is absolute, normalises `p`; otherwise normalises base + "/" + p.
[[nodiscard]] std::string resolve_against(std::string_view base, std::string_view p);

// Component-wise, so "/work/repo2" is NOT within "/work/repo". A string prefix test
// would say it is, and that is a containment bypass spelled as an optimisation.
// A path equal to root counts as within.
[[nodiscard]] bool is_within(std::string_view root, std::string_view path);

} // namespace lmp::platform
