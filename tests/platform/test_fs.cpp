#include "src/platform/fs.hpp"

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "tests/check.hpp"

using namespace lmp::platform;

namespace {

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_fs_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

} // namespace

TEST(read_returns_every_byte_or_nothing) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/f.bin";

    std::string payload;
    for (int i = 0; i < 5000; ++i) {
        payload.push_back(static_cast<char>(i % 256));
    }
    REQUIRE(write_file_atomic(path, payload).ok());

    const FileContents got = read_file_whole(path, 1 << 20);
    CHECK(got.ok());
    CHECK_EQ(got.bytes.size(), payload.size());
    CHECK_EQ(got.bytes, payload);
}

TEST(too_large_yields_no_prefix_and_names_the_real_size) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/big.txt";
    const std::string payload(4096, 'z');
    REQUIRE(write_file_atomic(path, payload).ok());

    const FileContents got = read_file_whole(path, 1024);
    CHECK(!got.ok());
    CHECK(got.status == FsStatus::TooLarge);
    // The four "rescuer" layers in v1 all existed because this next line was false:
    // the primitive returned 1024 bytes and called it success.
    CHECK(got.bytes.empty());
    CHECK_EQ(got.actual_size, std::size_t{4096});
    CHECK(got.error.find("4096") != std::string::npos);
}

TEST(missing_and_directory_are_distinct_typed_failures) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());

    const FileContents missing = read_file_whole(dir + "/nope", 1024);
    CHECK(missing.status == FsStatus::NotFound);
    CHECK(missing.bytes.empty());

    const FileContents isdir = read_file_whole(dir, 1024);
    CHECK(isdir.status == FsStatus::IsDirectory);
    CHECK(isdir.bytes.empty());
}

TEST(atomic_write_replaces_wholesale) {
    const std::string dir = temp_dir();
    REQUIRE(!dir.empty());
    const std::string path = dir + "/a.txt";
    REQUIRE(write_file_atomic(path, "first").ok());
    REQUIRE(write_file_atomic(path, "second-and-longer").ok());

    const FileContents got = read_file_whole(path, 1 << 20);
    CHECK(got.ok());
    CHECK_EQ(got.bytes, std::string("second-and-longer"));

    // The temporary is not left behind.
    const FileContents leftover =
        read_file_whole(path + ".tmp." + std::to_string(::getpid()), 1 << 20);
    CHECK(leftover.status == FsStatus::NotFound);
}

TEST(workspace_reads_and_writes_refuse_symlinked_files_and_directories) {
    const std::string root = temp_dir();
    const std::string outside = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(!outside.empty());
    REQUIRE(write_file_atomic(outside + "/secret.txt", "outside").ok());
    REQUIRE(::symlink((outside + "/secret.txt").c_str(),
                      (root + "/file-link").c_str()) == 0);
    REQUIRE(::symlink(outside.c_str(), (root + "/dir-link").c_str()) == 0);

    WorkspaceFs workspace(root);
    REQUIRE(workspace.valid());
    CHECK(workspace.read_file_whole("file-link", 1024).status == FsStatus::Symlink);
    CHECK(workspace.read_file_whole("dir-link/secret.txt", 1024).status ==
          FsStatus::Symlink);
    CHECK(workspace.write_file_atomic("file-link", "changed").status ==
          FsStatus::Symlink);
    CHECK(workspace.write_file_atomic("dir-link/new.txt", "escaped").status ==
          FsStatus::Symlink);
    CHECK(workspace.remove_file("file-link").status == FsStatus::Symlink);
    CHECK_EQ(read_file_whole(outside + "/secret.txt", 1024).bytes,
             std::string("outside"));
    CHECK(read_file_whole(outside + "/new.txt", 1024).status == FsStatus::NotFound);
}

TEST(atomic_write_ignores_precreated_predictable_temp_symlinks) {
    const std::string root = temp_dir();
    const std::string outside = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(!outside.empty());
    const std::string target = root + "/target.txt";
    const std::string victim = outside + "/victim.txt";
    REQUIRE(write_file_atomic(victim, "victim").ok());

    // This was the old deterministic temporary name. Precreating it as a symlink used to
    // make O_TRUNC destroy the outside file before rename().
    const std::string planted = target + ".tmp." + std::to_string(::getpid());
    REQUIRE(::symlink(victim.c_str(), planted.c_str()) == 0);
    REQUIRE(write_file_atomic(target, "safe").ok());
    CHECK_EQ(read_file_whole(target, 1024).bytes, std::string("safe"));
    CHECK_EQ(read_file_whole(victim, 1024).bytes, std::string("victim"));
    struct stat st {};
    REQUIRE(::lstat(planted.c_str(), &st) == 0);
    CHECK(S_ISLNK(st.st_mode));
}

TEST(atomic_write_preserves_existing_target_mode) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    const std::string target = root + "/private.txt";
    REQUIRE(write_file_atomic(target, "old").ok());
    REQUIRE(::chmod(target.c_str(), 0600) == 0);
    REQUIRE(write_file_atomic(target, "new").ok());
    struct stat st {};
    REQUIRE(::stat(target.c_str(), &st) == 0);
    CHECK_EQ(static_cast<unsigned>(st.st_mode & 0777), 0600U);
}

TEST(write_precondition_rejects_stale_version_and_create_race) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    WorkspaceFs workspace(root);
    REQUIRE(workspace.valid());

    const std::string original = "preimage\n";
    REQUIRE(workspace.write_file_atomic("f.txt", original).ok());
    const std::string version = content_sha256_hex(original);

    WritePrecondition stale;
    stale.expected_version = content_sha256_hex("other\n");
    CHECK(workspace.write_file_atomic("f.txt", "new\n", true, stale).status ==
          FsStatus::Conflict);
    CHECK_EQ(workspace.read_file_whole("f.txt", 1024).bytes, original);

    WritePrecondition fresh;
    fresh.expected_version = version;
    REQUIRE(workspace.write_file_atomic("f.txt", "new\n", true, fresh).ok());
    CHECK_EQ(workspace.read_file_whole("f.txt", 1024).bytes, std::string("new\n"));

    WritePrecondition create;
    create.expected_absent = true;
    CHECK(workspace.write_file_atomic("f.txt", "again\n", true, create).status ==
          FsStatus::Conflict);
    REQUIRE(workspace.write_file_atomic("g.txt", "born\n", true, create).ok());

    WritePrecondition del;
    del.expected_version = content_sha256_hex("born\n");
    REQUIRE(workspace.remove_file("g.txt", del.expected_version).ok());
    CHECK(workspace.remove_file("f.txt", "deadbeef").status == FsStatus::Conflict);
}

TEST(workspace_rejects_prefix_collisions_and_dotdot_components) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    WorkspaceFs workspace(root);
    REQUIRE(workspace.valid());

    CHECK(!workspace.contained_path(root + "-other/secret").ok());
    CHECK(!workspace.contained_path(root + "2/secret").ok());
    CHECK(!workspace.contained_path("../escape").ok());
    CHECK(!workspace.contained_path("sub/../escape").ok());
    CHECK(workspace.contained_path("sub/new.txt").ok());
}

TEST(workspace_root_symlink_is_canonicalized_once) {
    const std::string actual = temp_dir();
    REQUIRE(!actual.empty());
    const std::string alias = actual + "-alias";
    REQUIRE(::symlink(actual.c_str(), alias.c_str()) == 0);

    WorkspaceFs workspace(alias);
    REQUIRE(workspace.valid());
    REQUIRE(workspace.write_file_atomic("inside.txt", "ok").ok());
    CHECK_EQ(workspace.read_file_whole("inside.txt", 1024).bytes, std::string("ok"));
    CHECK(workspace.contained_path(alias + "/inside.txt").ok());
    const ContainedPath resolved = workspace.contained_path("inside.txt");
    REQUIRE(resolved.ok());
    CHECK(resolved.absolute.find(alias) == std::string::npos);
}

TEST(lexical_normalisation_never_touches_the_disk) {
    CHECK_EQ(lexically_normal("/a/b/../c"), std::string("/a/c"));
    CHECK_EQ(lexically_normal("/a/./b/"), std::string("/a/b"));
    CHECK_EQ(lexically_normal("/a/b/../../.."), std::string("/"));
    CHECK_EQ(lexically_normal("/"), std::string("/"));
    CHECK_EQ(lexically_normal("a/b/../c"), std::string("a/c"));
    CHECK_EQ(lexically_normal("../x"), std::string("../x"));
    CHECK_EQ(lexically_normal("a/../.."), std::string(".."));
    CHECK_EQ(lexically_normal("//a///b//"), std::string("/a/b"));
}

TEST(containment_matches_blast_radius_rule_one) {
    // Rule 1 of bakeoff/blast_radius/README.md, verbatim: containment is textual,
    // nothing is stat()ed, and `/work/repo/../repo/build` is INSIDE.
    CHECK(is_within("/work/repo", "/work/repo/../repo/build"));
    CHECK(is_within("/work/repo", "/work/repo"));
    CHECK(is_within("/work/repo", "/work/repo/src/main.cpp"));

    CHECK(!is_within("/work/repo", "/work/other"));
    CHECK(!is_within("/work/repo", "/etc/passwd"));
    CHECK(!is_within("/work/repo", "/work/repo/../secrets"));
}

TEST(containment_is_component_wise_not_a_string_prefix) {
    // "/work/repo2" starts with "/work/repo". A prefix test says it is contained; that
    // is a containment bypass spelled as an optimisation.
    CHECK(!is_within("/work/repo", "/work/repo2"));
    CHECK(!is_within("/work/repo", "/work/repo2/deep/inside"));
    CHECK(!is_within("/work/repo", "/work/repository"));
}

TEST(containment_refuses_to_compare_across_path_kinds) {
    // An absolute path measured against a relative root is a caller bug, and answering
    // "yes" to it would be the dangerous direction.
    CHECK(!is_within("work/repo", "/work/repo/x"));
    CHECK(!is_within("/work/repo", "work/repo/x"));
    CHECK(!is_within("", "/work/repo"));
}

TEST(resolve_against_handles_both_kinds) {
    CHECK_EQ(resolve_against("/work/repo", "src/a.cpp"), std::string("/work/repo/src/a.cpp"));
    CHECK_EQ(resolve_against("/work/repo", "/etc/passwd"), std::string("/etc/passwd"));
    CHECK_EQ(resolve_against("/work/repo", "../escape"), std::string("/work/escape"));
    CHECK_EQ(resolve_against("/work/repo/", "./x"), std::string("/work/repo/x"));

    // The composition that matters: resolve, then contain.
    CHECK(!is_within("/work/repo", resolve_against("/work/repo", "../escape")));
    CHECK(is_within("/work/repo", resolve_against("/work/repo", "sub/../ok.txt")));
}
