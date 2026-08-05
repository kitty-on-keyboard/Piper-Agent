// The tool registry over a real temp workspace: containment, honesty of results, and
// the graft-backed replace tool's refusal semantics.

#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/platform/fs.hpp"
#include "src/tools/registry.hpp"
#include "src/tools/symbol_index.hpp"
#include "src/tools/text_view.hpp"

#include "tests/check.hpp"

using namespace lmp::tools;

namespace {

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_reg_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

Registry make_registry(const std::string& root) {
    WorkspaceContext ctx;
    ctx.root = root;
    ctx.max_read_bytes = 1U << 20;
    ctx.max_model_read_bytes = 16384;
    ctx.max_result_bytes = 8192;
    ctx.spool_dir = root + "/.spool";
    ctx.shell_wall_clock_seconds = 20;
    return Registry(std::move(ctx));
}

std::vector<ToolParamValue> args(std::initializer_list<std::pair<const char*, std::string>> kv) {
    std::vector<ToolParamValue> out;
    for (const auto& [k, v] : kv) {
        out.push_back({k, v});
    }
    return out;
}

} // namespace

TEST(the_registry_declares_the_spec_set_and_no_more) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    // S6.1: resist growth; every tool is permanent surface area. Growing this pin
    // requires editing this test, which is the review moment.
    //
    // 11 -> 15: git_status, git_diff and git_log (an agent that cannot read its own diff
    // has no review surface), plus `plan`, which is declared here but executed by the
    // loop because the checklist lives in the context store.
    //
    // 15 -> 16: `remember`, the only tool whose effect outlives the run. Reviewed as
    // surface area on the same terms as the rest: it takes no path, writing to one fixed
    // file, so it adds a durable prompt input without adding a way to reach the disk.
    //
    // 16 -> 18: `ask_user` and `exit_plan_mode`. Both are declared here for the grammar and
    // the guidance and EXECUTED by the loop, exactly as `plan` is, because what they do is
    // end the run and the registry cannot do that. They are the only way a conversational
    // mode can stop and hand back: before them, a plan-mode run that asked a question was
    // scored as a turn that made no move, and three of those ended it as a stall.
    CHECK_EQ(reg.decls().size(), std::size_t{18});
    CHECK(reg.find("ask_user") != nullptr);
    CHECK(reg.find("exit_plan_mode") != nullptr);
    CHECK(reg.find("git_diff") != nullptr);
    CHECK(reg.find("plan") != nullptr);
    CHECK(reg.find("read_file") != nullptr);
    CHECK(reg.find("shell") != nullptr);
    CHECK(reg.find("remember") != nullptr);
    CHECK(reg.find("no_such_tool") == nullptr);
    // The guard specs mirror the declarations one-to-one -- the grammar constrains
    // exactly the advertised set (S6.4).
    CHECK_EQ(reg.guard_specs().size(), reg.decls().size());
}

// `remember` is the only tool whose effect outlives the run, so its file invariants are
// the thing to test: one fact per line, no duplicates, and a hard byte cap. Break any of
// them and the damage lands in the STABLE part of every future prompt.
TEST(remember_folds_dedupes_and_stays_under_its_cap) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const std::string path = root + "/" + std::string(kMemoryFileName);

    ToolResult a = reg.execute("remember", args({{"fact", "builds with cmake --preset dev"}}), 1);
    REQUIRE(a.ok());
    CHECK(lmp::platform::read_file_whole(path, 1U << 20).bytes ==
          std::string("- builds with cmake --preset dev\n"));

    // A repeat is the common case, not an edge case: a model that re-reads its own notes
    // re-derives the same conclusion. It must not accumulate.
    ToolResult again =
        reg.execute("remember", args({{"fact", "builds with cmake --preset dev"}}), 1);
    CHECK(again.ok());
    CHECK_EQ(lmp::platform::read_file_whole(path, 1U << 20).bytes,
             std::string("- builds with cmake --preset dev\n"));

    // Multi-line input is FOLDED, not refused -- one fact per line is the invariant that
    // dedupe and trimming both rest on, and folding keeps every character.
    ToolResult multi = reg.execute("remember", args({{"fact", "a\nb\tc"}}), 1);
    CHECK(multi.ok());
    const std::string folded = lmp::platform::read_file_whole(path, 1U << 20).bytes;
    CHECK(folded.find("- a b c\n") != std::string::npos);
    CHECK_EQ(folded.find("- a\nb"), std::string::npos);

    // Blank is a malformed note, not a silent no-op that reports success.
    CHECK(!reg.execute("remember", args({{"fact", "   "}}), 1).ok());

    // The cap holds, and it drops the OLDEST notes: a memory frozen at whatever the
    // project learned first would be worse than none.
    for (int i = 0; i < 400; ++i) {
        (void)reg.execute("remember",
                          args({{"fact", "filler note number " + std::to_string(i) +
                                             " padded out to take up real room"}}),
                          1);
    }
    const std::string full = lmp::platform::read_file_whole(path, 1U << 20).bytes;
    CHECK(full.size() <= kMemoryMaxBytes);
    CHECK(full.find("filler note number 399") != std::string::npos);
    CHECK_EQ(full.find("builds with cmake --preset dev"), std::string::npos);
}

TEST(paths_outside_the_root_are_refused_not_errored) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult r = reg.execute("read_file", args({{"path", "../../etc/passwd"}}), 1);
    // Refused (policy), not ToolError -- the file exists, we did not fail to read it,
    // we declined to. The distinction is S6.2's whole point.
    CHECK(r.status == Status::Refused);
    CHECK(r.error_class == ErrorClass::Policy);
}

TEST(write_read_replace_round_trip) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    ToolResult w = reg.execute(
        "write_file", args({{"path", "a.cpp"}, {"content", "int x = 1;\nint y = 2;\n"}}), 1);
    REQUIRE(w.ok());

    ToolResult rd = reg.execute("read_file", args({{"path", "a.cpp"}}), 1);
    REQUIRE(rd.ok());
    // Line-numbered: the numbers are display only, and read_slice takes them as args.
    CHECK_EQ(rd.summary, std::string("1\tint x = 1;\n2\tint y = 2;\n"));

    // Whitespace-tolerant replace: the model wrote `int  x=1;` with different spacing.
    ToolResult rp = reg.execute("replace_in_file",
                                args({{"path", "a.cpp"},
                                      {"old_text", "int  x=1;"},
                                      {"new_text", "int x = 42;"}}),
                                1);
    CHECK(rp.ok());
    rd = reg.execute("read_file", args({{"path", "a.cpp"}}), 1);
    CHECK(rd.summary.find("42") != std::string::npos);
}

// Writing into a directory that does not exist yet is the FIRST thing an agent does in an
// empty workspace, and it used to fail -- on the atomic write's temp file, so the message
// named `src/store.py.tmp.8276: No such file or directory`, a path the model never wrote.
//
// This is the defect that cost a real 40-turn run everything it had: the model produced
// 8 KB of correct code six times against that error before stumbling onto `mkdir -p src`
// in a shell call. The parent is made now, at the one choke point every write tool shares.
TEST(a_write_creates_the_directories_its_path_names) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    const ToolResult w = reg.execute(
        "write_file", args({{"path", "src/store.py"}, {"content", "VALUE = 1\n"}}), 1);
    REQUIRE(w.ok());

    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/src/store.py", 1U << 20);
    CHECK_EQ(f.bytes, std::string("VALUE = 1\n"));

    // Nested, and via append_file -- the other tool that may name a path nothing created.
    const ToolResult a = reg.execute(
        "append_file", args({{"path", "tests/unit/test_store.py"}, {"content", "pass\n"}}), 1);
    REQUIRE(a.ok());
    const lmp::platform::FileContents g =
        lmp::platform::read_file_whole(root + "/tests/unit/test_store.py", 1U << 20);
    CHECK_EQ(g.bytes, std::string("pass\n"));

    // Still contained: a path that escapes is refused before any directory is made.
    const ToolResult esc = reg.execute(
        "write_file", args({{"path", "../escaped/x.py"}, {"content", "x\n"}}), 1);
    CHECK(esc.status == Status::Refused);
}

// A write that failed for a reason the same call will hit again must not be advertised as
// worth retrying: `retryable` is what the loop's BreakRepeat corrective keys off, and a
// retryable error is never counted as an unrecoverable repeat. Every write failure used to
// be Transient+retryable, so a byte-identical failing write could loop until the budget ran
// out -- and did, six times in one run.
// A successful call that says nothing is worse than a failure: an empty observation is
// dropped from the rendered prompt, so the next turn's prompt is byte-identical and a
// fixed seed reproduces the same call forever. Two real runs died this way -- `shell` on
// a silent `mkdir`, and `read_file` on a 0-byte file for 17 consecutive turns.
TEST(no_tool_returns_a_silent_success) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    // A file that exists and is empty says so, rather than returning "".
    REQUIRE(reg.execute("write_file", args({{"path", "empty.py"}, {"content", ""}}), 1).ok());
    const ToolResult rd = reg.execute("read_file", args({{"path", "empty.py"}}), 1);
    REQUIRE(rd.ok());
    CHECK(!rd.summary.empty());
    CHECK(rd.summary.find("empty") != std::string::npos);

    // A command with no output does too.
    const ToolResult sh = reg.execute("shell", args({{"command", "mkdir -p made/here"}}), 1);
    REQUIRE(sh.ok());
    CHECK(!sh.summary.empty());
}

TEST(a_deterministic_write_failure_is_not_reported_as_retryable) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    REQUIRE(reg.execute("write_file", args({{"path", "occupied"}, {"content", "x"}}), 1).ok());

    // `occupied` is a file, so `occupied/inside.py` can never be created, however often
    // it is tried.
    const ToolResult r = reg.execute(
        "write_file", args({{"path", "occupied/inside.py"}, {"content", "y"}}), 1);
    CHECK(r.status == Status::ToolError);
    CHECK(!r.retryable);
    CHECK(r.error_class != ErrorClass::Transient);
}

TEST(ambiguous_replace_refuses_and_leaves_the_file_alone) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const std::string original = "foo();\nbar();\nfoo();\n";
    (void)reg.execute("write_file", args({{"path", "b.cpp"}, {"content", original}}), 1);

    const ToolResult r = reg.execute(
        "replace_in_file",
        args({{"path", "b.cpp"}, {"old_text", "foo();"}, {"new_text", "baz();"}}), 1);
    CHECK(r.status == Status::ToolError);
    CHECK(r.error_class == ErrorClass::Conflict);

    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/b.cpp", 1U << 20);
    CHECK_EQ(f.bytes, original); // untouched -- graft's contract
}

TEST(read_slice_returns_exact_lines) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    (void)reg.execute("write_file",
                      args({{"path", "l.txt"}, {"content", "one\ntwo\nthree\nfour\n"}}), 1);
    const ToolResult r = reg.execute(
        "read_slice",
        args({{"path", "l.txt"}, {"start_line", "2"}, {"end_line", "3"}}), 1);
    REQUIRE(r.ok());
    // ABSOLUTE line numbers, not slice-relative: slice-relative is an off-by-start_line trap.
    CHECK_EQ(r.summary, std::string("2\ttwo\n3\tthree\n"));
}

TEST(shell_reports_exit_codes_and_compacts) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult ok = reg.execute("shell", args({{"command", "echo hello"}}), 1);
    CHECK(ok.ok());
    CHECK(ok.summary.find("hello") != std::string::npos);

    const ToolResult bad = reg.execute("shell", args({{"command", "exit 3"}}), 1);
    CHECK(bad.status == Status::ToolError);
    CHECK(bad.summary.find("[exit 3]") != std::string::npos);
    CHECK(bad.retryable);
}

TEST(a_nested_sandbox_failure_says_what_it_is_and_what_to_do) {
    // The real thing, not a stubbed string: macOS refuses a second Seatbelt profile
    // inside the first, so running sandbox-exec under T1 produces exactly the message
    // Swift Package Manager produces when it compiles Package.swift.
    //
    // Left bare, that message cost a real run thirty turns of inventing cache
    // directories and deleting build output, because nothing in it says the harness is
    // the obstacle and nothing names the flag that gets past it.
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult r = reg.execute(
        "shell",
        args({{"command", "/usr/bin/sandbox-exec -p '(version 1)(allow default)' "
                          "/usr/bin/true"}}),
        1);

    CHECK(r.status == Status::ToolError);
    CHECK(r.summary.find("[sandbox]") != std::string::npos);
    CHECK(r.summary.find("--disable-sandbox") != std::string::npos);
    // It must say the workspace is NOT the problem, which is the wrong conclusion the
    // bare message leads to.
    CHECK(r.summary.find("not a problem with your code") != std::string::npos);
}

TEST(an_ordinary_permission_error_gets_no_sandbox_note) {
    // The note must be specific to the nesting failure. A plain denied write is a fact
    // about the jail the model SHOULD read at face value, and telling it to reach for
    // --disable-sandbox there would send it chasing a flag that changes nothing.
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult r =
        reg.execute("shell", args({{"command", "echo x > /Denied/nope.txt"}}), 1);
    CHECK(!r.ok());
    CHECK(r.summary.find("[sandbox]") == std::string::npos);
}

TEST(shell_at_t0_is_refused_with_the_reason) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult r = reg.execute("shell", args({{"command", "echo hi"}}), 0);
    CHECK(r.status == Status::Refused);
    CHECK(r.summary.find("T0") != std::string::npos);
}

TEST(tools_json_covers_every_declaration) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const std::string json = reg.tools_json();
    for (const ToolDecl& d : reg.decls()) {
        if (json.find("\"name\": \"" + d.name + "\"") == std::string::npos) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      "tools_json is missing " + d.name);
        }
        ++lmp::test::reg().checks;
    }
}

// --- G7b: writes through the extension's edit API ---------------------------
//
// With a sink attached the sidecar must not touch the file itself, so the editor can
// apply a WorkspaceEdit and undo, dirty buffers and diff review all work (S12.4).
TEST(an_attached_edit_sink_receives_the_write_instead_of_the_disk) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    std::string seen_path;
    std::string seen_content;
    reg.set_edit_sink([&](const std::string& p, const std::string& c) {
        seen_path = p;
        seen_content = c;
        return EditOutcome{true, {}};
    });

    const ToolResult w = reg.execute(
        "write_file", args({{"path", "x.py"}, {"content", "print(1)\n"}}), 1);
    CHECK(w.ok());
    CHECK_EQ(seen_content, std::string("print(1)\n"));
    CHECK(seen_path.find("x.py") != std::string::npos);
    // The sidecar did NOT write it -- the editor was supposed to, and in this test the
    // editor is a lambda that only recorded. A file on disk here would mean the handover
    // is half-wired and the editor's buffer and the disk disagree.
    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/x.py", 1U << 20);
    CHECK(!f.ok());
}

TEST(a_refusing_edit_sink_surfaces_as_a_tool_error) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    reg.set_edit_sink([](const std::string&, const std::string&) {
        return EditOutcome{false, "buffer is read-only"};
    });
    const ToolResult w = reg.execute(
        "write_file", args({{"path", "y.py"}, {"content", "print(1)\n"}}), 1);
    CHECK(!w.ok());
    CHECK(w.summary.find("read-only") != std::string::npos);
}

// No sink means write directly, and that is deliberately the OPPOSITE of the approver's
// deny-by-default: an absent approver means nobody is there to ask, an absent edit sink
// means there is no editor to route through. An eval run has no extension and must still
// be able to edit.
TEST(no_edit_sink_writes_directly) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult w = reg.execute(
        "write_file", args({{"path", "z.py"}, {"content", "print(2)\n"}}), 1);
    CHECK(w.ok());
    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/z.py", 1U << 20);
    REQUIRE(f.ok());
    CHECK_EQ(f.bytes, std::string("print(2)\n"));
}

// --- G4: the prompt read budget ---------------------------------------------
TEST(read_file_over_the_prompt_budget_fails_honestly) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root); // max_model_read_bytes = 16384
    std::string big;
    for (int i = 0; i < 4000; ++i) {
        big += "some line of source\n";
    }
    (void)reg.execute("write_file", args({{"path", "big.txt"}, {"content", big}}), 1);

    const ToolResult r = reg.execute("read_file", args({{"path", "big.txt"}}), 1);
    CHECK(!r.ok());
    // The REAL numbers, and the tool to use instead -- which is what read_file's
    // description has always claimed and, until the budget was split, did not do.
    CHECK(r.summary.find(std::to_string(big.size())) != std::string::npos);
    CHECK(r.summary.find("4001 lines") != std::string::npos);
    CHECK(r.summary.find("read_slice") != std::string::npos);
}

TEST(read_slice_over_the_budget_says_where_to_resume) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    std::string big;
    for (int i = 0; i < 4000; ++i) {
        big += "some line of source\n";
    }
    (void)reg.execute("write_file", args({{"path", "big2.txt"}, {"content", big}}), 1);

    const ToolResult r = reg.execute(
        "read_slice",
        args({{"path", "big2.txt"}, {"start_line", "1"}, {"end_line", "4000"}}), 1);
    REQUIRE(r.ok());
    CHECK(r.summary.find("[budget reached at line ") != std::string::npos);
    CHECK(r.summary.find("continue with read_slice") != std::string::npos);
    CHECK(r.summary.size() < 24000);
}

// --- G6: locate_symbol ranking ----------------------------------------------
//
// `head -60` used to hand back whichever definition-shaped line the filesystem walk
// reached first, so a symbol with many call sites buried its own definition.
TEST(symbol_hits_rank_definitions_above_mentions) {
    const std::string_view grep =
        "a.py:10:    total = compute_total(x)\n"
        "b.py:3:def compute_total(x):\n"
        "c.py:99:        compute_total(y)\n";
    std::size_t suppressed = 0;
    const std::vector<SymbolHit> hits =
        rank_symbol_hits(grep, "compute_total", 10, suppressed);
    REQUIRE(hits.size() == 3);
    CHECK_EQ(hits[0].path, std::string("b.py")); // the `def` wins
    CHECK_EQ(hits[0].line, 3L);
    CHECK_EQ(suppressed, std::size_t{0});
}

// Within a score, shallower indentation first: a top-level definition is nearly always
// what "where is this defined" means.
TEST(shallower_definitions_outrank_nested_ones) {
    const std::string_view grep =
        "a.py:20:        def helper(x):\n"
        "a.py:2:def helper(x):\n";
    std::size_t suppressed = 0;
    const std::vector<SymbolHit> hits = rank_symbol_hits(grep, "helper", 10, suppressed);
    REQUIRE(hits.size() == 2);
    CHECK_EQ(hits[0].line, 2L);
}

// A truncated list that does not say it was truncated is how a model concludes a symbol
// has exactly one definition site.
TEST(suppressed_hits_are_counted_not_hidden) {
    std::string grep;
    for (int i = 1; i <= 10; ++i) {
        grep += "f.py:" + std::to_string(i) + ":    thing(i)\n";
    }
    std::size_t suppressed = 0;
    const std::vector<SymbolHit> hits = rank_symbol_hits(grep, "thing", 4, suppressed);
    CHECK_EQ(hits.size(), std::size_t{4});
    CHECK_EQ(suppressed, std::size_t{6});
}

TEST(duplicate_path_line_pairs_collapse) {
    const std::string_view grep = "a.py:3:def f():\na.py:3:def f():\n";
    std::size_t suppressed = 0;
    CHECK_EQ(rank_symbol_hits(grep, "f", 10, suppressed).size(), std::size_t{1});
}

// --- G5 follow-up: numbered text copied back into an edit --------------------
//
// Not anticipated -- MEASURED. The first end-to-end run after line numbering landed spent
// five consecutive turns on replace_in_file ToolError because the model pasted the
// numbers it had just been shown. The failure was safe (NoMatch, file untouched) and
// still cost five prefills and five iterations.
TEST(an_edit_using_the_displayed_line_numbers_still_applies) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    (void)reg.execute("write_file",
                      args({{"path", "n.py"}, {"content", "def f():\n    return 1\n"}}), 1);

    const ToolResult slice = reg.execute(
        "read_slice", args({{"path", "n.py"}, {"start_line", "1"}, {"end_line", "2"}}), 1);
    REQUIRE(slice.ok());

    // Verbatim what the model was just shown, numbers and all.
    const ToolResult rp =
        reg.execute("replace_in_file",
                    args({{"path", "n.py"},
                          {"old_text", "2\t    return 1\n"},
                          {"new_text", "2\t    return 42\n"}}),
                    1);
    CHECK(rp.ok());
    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/n.py", 1U << 20);
    REQUIRE(f.ok());
    CHECK_EQ(f.bytes, std::string("def f():\n    return 42\n"));
}

// One bare line and nothing is stripped -- real code that happens to start a line with a
// number and a tab must survive untouched.
TEST(stripping_needs_every_line_to_carry_a_number) {
    CHECK_EQ(strip_line_numbers("1\tone\nplain\n"), std::string("1\tone\nplain\n"));
    CHECK_EQ(strip_line_numbers("1\tone\n2\ttwo\n"), std::string("one\ntwo\n"));
    CHECK_EQ(strip_line_numbers("no numbers here"), std::string("no numbers here"));
    CHECK_EQ(strip_line_numbers(""), std::string(""));
}

// Stripping is ASYMMETRIC. A wrong strip of old_text costs a NoMatch and leaves the file
// alone; a wrong strip of new_text silently writes corrupted content -- and a TSV whose
// first column is a row number is exactly the shape that matches.
TEST(numbered_new_text_survives_when_old_text_is_not_numbered) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    (void)reg.execute("write_file",
                      args({{"path", "rows.tsv"}, {"content", "header\nPLACEHOLDER\n"}}), 1);

    const ToolResult rp = reg.execute("replace_in_file",
                                      args({{"path", "rows.tsv"},
                                            {"old_text", "PLACEHOLDER"},
                                            {"new_text", "1\talice\n2\tbob"}}),
                                      1);
    REQUIRE(rp.ok());
    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/rows.tsv", 1U << 20);
    REQUIRE(f.ok());
    CHECK_EQ(f.bytes, std::string("header\n1\talice\n2\tbob\n"));
}

// --- the overwrite gate ------------------------------------------------------
//
// Irreversibility is a property of the CALL, not only of the tool. refuse_wipe_workspace
// denied delete_file twice and shell twice -- every declared gate held -- and the run then
// emptied ledger.csv with three write_file calls that nothing asked about.
TEST(overwriting_existing_content_is_recognised) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    (void)reg.execute("write_file",
                      args({{"path", "ledger.csv"}, {"content", "id,amount\n1,10\n"}}), 1);
    (void)reg.execute("write_file", args({{"path", "empty.txt"}, {"content", ""}}), 1);

    CHECK(would_overwrite_existing(root, "ledger.csv"));
    // A file that does not exist yet is a creation, not a destruction.
    CHECK(!would_overwrite_existing(root, "brand_new.txt"));
    // An empty file has no content to destroy.
    CHECK(!would_overwrite_existing(root, "empty.txt"));
    // Outside the root is refused upstream for a different reason; this must not claim it.
    CHECK(!would_overwrite_existing(root, "../../etc/hosts"));
}

// A keyed note REPLACES the earlier note under the same key, in the file the next session
// actually reads. Without this the mirror could only append, so a corrected note left the
// stale one in place and the next prompt carried BOTH -- the exact failure the durable
// store exists to end, reintroduced one layer up.
TEST(a_keyed_note_supersedes_the_earlier_note_under_that_key) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Registry reg = make_registry(root);

    CHECK(reg.execute("remember", {{"fact", "the suite runs with pytest tests/"},
                                   {"key", "test-layout"}}, 1).ok());
    CHECK(reg.execute("remember", {{"fact", "an unrelated standalone note"}}, 1).ok());
    const ToolResult fixed =
        reg.execute("remember", {{"fact", "the suite runs with pytest spec/"},
                                 {"key", "test-layout"}}, 1);
    CHECK(fixed.ok());
    CHECK(fixed.summary.find("replaced") != std::string::npos);

    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/" + kMemoryFileName, 1 << 20);
    REQUIRE(f.ok());
    // The stale line is GONE, not merely followed by a newer one.
    CHECK(f.bytes.find("pytest tests/") == std::string::npos);
    CHECK(f.bytes.find("- [test-layout] the suite runs with pytest spec/") !=
          std::string::npos);
    // An unkeyed note is untouched by any of it.
    CHECK(f.bytes.find("- an unrelated standalone note") != std::string::npos);
}

// A WRITE THAT CHANGES NOTHING IS NOT A WRITE, and until this it reported as one.
//
// MEASURED, and it is the whole reason the write door reads before it writes: a 73-turn
// run cancelled with 6/6 items open made 39 workspace writes, 13 of which re-sent bytes
// already on disk -- 5327 bytes to one file four times, 5437 four times, 5818 three times.
// Every one came back "wrote N bytes to ...", so the model believed it had just fixed
// something and the harness counted a deliverable.
TEST(rewriting_a_file_with_its_own_bytes_is_reported_as_no_change) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    const std::string body = "int x = 1;\nint y = 2;\n";
    const ToolResult first =
        reg.execute("write_file", args({{"path", "a.cpp"}, {"content", body}}), 1);
    REQUIRE(first.ok());
    CHECK(!first.mutation_was_noop);
    CHECK(first.summary.find("wrote") != std::string::npos);

    const ToolResult again =
        reg.execute("write_file", args({{"path", "a.cpp"}, {"content", body}}), 1);
    // Ok, because the file IS what was asked for and nothing failed. The flag is what
    // separates "the work is already done" from "work happened", and the loop reads the
    // flag, never the sentence.
    CHECK(again.ok());
    CHECK(again.mutation_was_noop);
    CHECK(again.summary.find("already contained") != std::string::npos);
    // And it says what to do instead, because "nothing changed" does not imply a next move.
    CHECK(again.summary.find("verification") != std::string::npos);

    // One byte of difference and it is a real write again -- the test is byte identity,
    // not similarity, so nothing here can swallow an edit.
    const ToolResult changed = reg.execute(
        "write_file", args({{"path", "a.cpp"}, {"content", body + "int z = 3;\n"}}), 1);
    CHECK(changed.ok());
    CHECK(!changed.mutation_was_noop);
}

// graft matched, so old_text was there -- and the file came out identical, which means
// new_text equals it. "replaced one occurrence" for an edit that replaced text with itself
// is the same lie one tool over.
TEST(replacing_text_with_itself_is_reported_as_no_change) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    REQUIRE(reg.execute("write_file",
                        args({{"path", "a.cpp"}, {"content", "int x = 1;\n"}}), 1)
                .ok());

    const ToolResult same = reg.execute("replace_in_file",
                                        args({{"path", "a.cpp"},
                                              {"old_text", "int x = 1;"},
                                              {"new_text", "int x = 1;"}}),
                                        1);
    CHECK(same.ok());
    CHECK(same.mutation_was_noop);
    CHECK(same.summary.find("identical") != std::string::npos);

    const ToolResult real = reg.execute("replace_in_file",
                                        args({{"path", "a.cpp"},
                                              {"old_text", "int x = 1;"},
                                              {"new_text", "int x = 42;"}}),
                                        1);
    CHECK(real.ok());
    CHECK(!real.mutation_was_noop);
}

// The no-op check runs BEFORE the sink, so an empty edit never reaches the editor. Routing
// one through would put an undo step in the operator's history for a change that does not
// exist -- and the sidecar cannot tell the difference afterwards.
TEST(a_no_op_write_never_reaches_the_edit_sink) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    REQUIRE(reg.execute("write_file",
                        args({{"path", "x.py"}, {"content", "print(1)\n"}}), 1)
                .ok());

    int sink_calls = 0;
    reg.set_edit_sink([&](const std::string&, const std::string&) {
        ++sink_calls;
        return EditOutcome{true, {}};
    });

    const ToolResult again =
        reg.execute("write_file", args({{"path", "x.py"}, {"content", "print(1)\n"}}), 1);
    CHECK(again.ok());
    CHECK(again.mutation_was_noop);
    CHECK_EQ(sink_calls, 0);

    // A real change still goes through it.
    CHECK(reg.execute("write_file", args({{"path", "x.py"}, {"content", "print(2)\n"}}), 1)
              .ok());
    CHECK_EQ(sink_calls, 1);
}

// --- what a mode is allowed to SEE ------------------------------------------
//
// Mode policy used to be one boolean checked at dispatch time, so a plan-mode model was
// shown write_file, reached for it, and was refused -- one wasted turn per discovery, and
// after two refusals of the same tool BlockRefusedTool dropped it from the grammar and
// recorded it in the trace as though the OPERATOR had denied it. The filtered overload is
// what stops the model being offered a tool the gate would refuse.
//
// Asserted through the same predicate shape the Agent uses -- properties, never a list of
// tool names -- so a tool added later is covered by what it declares.
TEST(tools_json_can_be_filtered_to_what_a_mode_permits) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    const std::string all = reg.tools_json();
    CHECK(all.find("\"write_file\"") != std::string::npos);
    CHECK(all.find("\"shell\"") != std::string::npos);
    CHECK(all.find("\"delete_file\"") != std::string::npos);

    // Plan mode: no writes, no execution, no remote tools.
    const std::string plan = reg.tools_json([](const ToolDecl& d) {
        return !d.mutates_workspace && !d.needs_execution && !d.remote && !d.irreversible;
    });
    CHECK(plan.find("\"write_file\"") == std::string::npos);
    CHECK(plan.find("\"replace_in_file\"") == std::string::npos);
    CHECK(plan.find("\"delete_file\"") == std::string::npos);
    CHECK(plan.find("\"remember\"") == std::string::npos);
    CHECK(plan.find("\"shell\"") == std::string::npos);
    // The git tools shell out through run_git, so they are needs_execution too -- they were
    // the case that nearly slipped through, because they carry no `command` param and so
    // could not be declared executes_commands without handing the command gate an empty
    // string to classify.
    CHECK(plan.find("\"git_diff\"") == std::string::npos);
    CHECK(plan.find("\"git_log\"") == std::string::npos);
    // Reading is the whole of what plan mode does, and it keeps all of it.
    CHECK(plan.find("\"read_file\"") != std::string::npos);
    CHECK(plan.find("\"search\"") != std::string::npos);
    CHECK(plan.find("\"locate_symbol\"") != std::string::npos);

    // Debug mode: writes and execution, but nothing irreversible.
    const std::string debug =
        reg.tools_json([](const ToolDecl& d) { return !d.irreversible; });
    CHECK(debug.find("\"write_file\"") != std::string::npos);
    CHECK(debug.find("\"shell\"") != std::string::npos);
    CHECK(debug.find("\"delete_file\"") == std::string::npos);
}

// `shell` is both; the git tools are only the second. Conflating them would have sent a
// git call into the blast-radius classifier with no command string to read.
TEST(execution_and_command_classification_are_separate_declarations) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    const ToolDecl* shell = reg.find("shell");
    REQUIRE(shell != nullptr);
    CHECK(shell->executes_commands);
    CHECK(shell->needs_execution);

    for (const char* name : {"git_status", "git_diff", "git_log"}) {
        const ToolDecl* d = reg.find(name);
        REQUIRE(d != nullptr);
        CHECK(d->needs_execution);
        CHECK(!d->executes_commands);
    }

    // Nothing declared locally is remote; that flag exists for MCP and must not drift.
    for (const ToolDecl& d : reg.decls()) {
        CHECK(!d.remote);
    }
}
