#include "src/platform/fs.hpp"

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
