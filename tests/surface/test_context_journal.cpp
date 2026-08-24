// The adapter between the working context and the durable store -- and specifically the
// path where it cannot open one.
//
// This test exists because of what the failure used to do rather than because of what it
// does. `open()` returned a bare pointer and wrote its reason to `fprintf(stderr)`, which
// in a sidecar is a subprocess pipe no client reads: stdout is the framed protocol
// channel, so the diagnostic could not go there, and stderr was where it went to die. The
// justification recorded at the time was that no caller could use the reason and that
// sidecar.cpp "had no line to spare" -- the second half being the 800-line file ratchet
// talking, which is not a reason for an API to lose information.
//
// So the property under test is not "it fails". It is that a run whose compacted turns
// are about to become unrecoverable can SAY SO, and still runs.

#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/pcc/recall.hpp"
#include "src/platform/fs.hpp"
#include "src/surface/context_journal.hpp"

#include "tests/check.hpp"

namespace {

using lmp::context::ContextStore;
using lmp::context::TurnRecord;
using lmp::surface::ContextJournal;

// A real directory: open() appends kContextDbName to the root it is given, so ":memory:"
// would name a FILE inside a directory called ":memory:" rather than an in-memory
// database. The journal is a workspace-relative dotfile by design and there is no
// in-memory door into it.
std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_cj_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

TurnRecord tool_turn(const std::string& observation, std::uint64_t event) {
    TurnRecord t;
    t.tool_name = "read_file";
    t.observation = observation;
    t.first_event_seq = event;
    t.last_event_seq = event;
    return t;
}

} // namespace

TEST(a_journal_that_cannot_open_comes_back_with_the_reason) {
    ContextStore ctx("mission");
    // A workspace root that does not exist. SQLite will not create the parent directory,
    // so this is the ordinary shape of the failure: a workspace on a volume that went
    // away, or one the sidecar cannot write to.
    const ContextJournal::Result r =
        ContextJournal::open("/nonexistent-volume-lmp-test/workspace", "sess-1", ctx);

    CHECK(r.journal == nullptr);
    // The verbatim SQLite text, not a paraphrase. "unable to open database file" tells an
    // operator which of the two plausible causes they have; "journal unavailable" does not.
    CHECK(!r.error.empty());
}

TEST(a_failed_journal_leaves_compaction_exactly_as_it_was) {
    // The degradation contract: journalling makes a run better, and a run that cannot
    // journal is still a run. If open() had attached a half-built sink before it
    // discovered it could not open the database, this is where it would show -- the sink
    // fires inside compact_oldest(), so a broken one takes the turn down with it.
    ContextStore ctx("mission");
    const ContextJournal::Result r =
        ContextJournal::open("/nonexistent-volume-lmp-test/workspace", "sess-1", ctx);
    REQUIRE(r.journal == nullptr);

    for (std::uint64_t i = 1; i <= 6; ++i) {
        ctx.add_turn(tool_turn("line " + std::to_string(i), i));
    }
    CHECK_EQ(ctx.compact_oldest(2), std::size_t{4});
    CHECK_EQ(ctx.recent().size(), std::size_t{2});
    CHECK_EQ(ctx.compacted_spans().size(), std::size_t{1});
}

TEST(a_journal_database_symlink_is_refused) {
    ContextStore ctx("mission");
    const std::string root = temp_dir();
    const std::string outside = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(!outside.empty());
    const std::string victim = outside + "/victim.db";
    REQUIRE(lmp::platform::write_file_atomic(victim, "keep").ok());
    REQUIRE(::symlink(victim.c_str(),
                      (root + "/" + lmp::surface::kContextDbName).c_str()) == 0);

    const ContextJournal::Result r = ContextJournal::open(root, "sess-1", ctx);
    CHECK(r.journal == nullptr);
    CHECK(!r.error.empty());
    CHECK_EQ(lmp::platform::read_file_whole(victim, 1024).bytes,
             std::string("keep"));
}

TEST(an_opened_journal_keeps_what_the_trim_drops) {
    // The success half, kept beside the failure half on purpose: a test file that only
    // ever exercises the degraded path stops noticing the day open() degrades ALWAYS.
    ContextStore ctx("mission");
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    const ContextJournal::Result r = ContextJournal::open(root, "sess-1", ctx);
    REQUIRE(r.journal != nullptr);
    CHECK(r.error.empty());

    const std::string detail = "the continuation byte is 0x80 exactly";
    ctx.add_turn(tool_turn(detail, 1));
    for (std::uint64_t i = 2; i <= 6; ++i) {
        ctx.add_turn(tool_turn("build ok " + std::to_string(i), i));
    }
    CHECK_EQ(ctx.compact_oldest(1), std::size_t{5});

    // The anchor line the summary keeps is truncated; the store has the whole thing.
    const lmp::pcc::Recall out = lmp::pcc::rehydrate(r.journal->store(), 1, 5, 4000);
    CHECK_EQ(out.included, std::size_t{5});
    CHECK(out.text.find("continuation byte is 0x80") != std::string::npos);
}

TEST(an_opened_journal_keeps_its_database_out_of_git) {
    // The store reached 1.8 MB after one 26-turn task and sat at the workspace root, so
    // it showed up in the USER's `git status` and went into `git add -A`. This is the
    // property that stops that, and the two halves matter separately: the local exclude
    // gains the pattern, and the tracked .gitignore is not touched at all.
    ContextStore ctx("mission");
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(::mkdir((root + "/.git").c_str(), 0755) == 0);
    const std::string tracked = root + "/.gitignore";
    REQUIRE(lmp::platform::write_file_atomic(tracked, "build/\n").ok());

    const ContextJournal::Result r = ContextJournal::open(root, "sess-1", ctx);
    REQUIRE(r.journal != nullptr);
    CHECK(r.git_exclude_written);

    const std::string exclude =
        lmp::platform::read_file_whole(root + "/.git/info/exclude", 1 << 20).bytes;
    for (const char* pattern : lmp::surface::kGitExcludePatterns) {
        CHECK(exclude.find(pattern) != std::string::npos);
    }

    // The user's own file, byte for byte. A tool that "helpfully" edits a tracked file
    // puts a line into a commit the user did not write.
    CHECK_EQ(lmp::platform::read_file_whole(tracked, 1024).bytes, std::string("build/\n"));
}

TEST(the_git_exclude_line_is_written_once_not_once_per_mission) {
    // open() runs per mission. An append that did not check would grow .git/info/exclude
    // by three lines every time the user started a task in the same workspace.
    ContextStore first("mission");
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(::mkdir((root + "/.git").c_str(), 0755) == 0);

    const ContextJournal::Result a = ContextJournal::open(root, "sess-1", first);
    REQUIRE(a.journal != nullptr);
    CHECK(a.git_exclude_written);
    const std::string after_first =
        lmp::platform::read_file_whole(root + "/.git/info/exclude", 1 << 20).bytes;

    ContextStore second("mission");
    const ContextJournal::Result b = ContextJournal::open(root, "sess-2", second);
    REQUIRE(b.journal != nullptr);
    CHECK(!b.git_exclude_written);
    CHECK_EQ(lmp::platform::read_file_whole(root + "/.git/info/exclude", 1 << 20).bytes,
             after_first);
}

TEST(a_workspace_that_is_not_a_repository_is_left_alone) {
    // No .git, nothing to do, and specifically: no .git created. A workspace is the
    // user's directory, not somewhere to leave scaffolding.
    ContextStore ctx("mission");
    const std::string root = temp_dir();
    REQUIRE(!root.empty());

    const ContextJournal::Result r = ContextJournal::open(root, "sess-1", ctx);
    REQUIRE(r.journal != nullptr);
    CHECK(!r.git_exclude_written);
    struct ::stat st {};
    CHECK(::stat((root + "/.git").c_str(), &st) != 0);
}
