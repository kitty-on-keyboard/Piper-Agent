// The read side of the durable context store, as the agent reaches it.
//
// WHY THIS FILE EXISTS. src/pcc has been a complete, tested, bi-temporal store with
// budgeted retrieval for some time, and until context_tools.cpp landed there was no way
// for the agent to read one byte of it: the same two tools existed only inside the
// out-of-process pcc_mcp_server, which is not staged into the VSIX. tests/pcc proves the
// store works. This proves the agent can reach it -- which is the half that was missing,
// and the half whose absence unit tests could never have shown.
//
// The property that matters most is the LAST test here: a store written by one session,
// read by another. That is the entire feature, and everything above it is scaffolding.

#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/pcc/recall.hpp"
#include "src/pcc/store.hpp"
#include "src/tools/registry.hpp"

#include "tests/check.hpp"

namespace {

using lmp::tools::Registry;
using lmp::tools::ToolParamValue;
using lmp::tools::ToolResult;
using lmp::tools::WorkspaceContext;
using namespace lmp::pcc;

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_ct_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

WorkspaceContext workspace(const std::string& root) {
    WorkspaceContext w;
    w.root = root;
    w.max_read_bytes = 4U << 20;
    w.max_model_read_bytes = 24U << 10;
    w.max_result_bytes = 8192;
    w.spool_dir = root + "/.lmp_spool";
    w.shell_wall_clock_seconds = 30;
    return w;
}

std::vector<ToolParamValue> args(std::vector<std::pair<std::string, std::string>> kv) {
    std::vector<ToolParamValue> out;
    out.reserve(kv.size());
    for (auto& [name, value] : kv) {
        out.push_back({name, value});
    }
    return out;
}

// A turn, written the way src/surface/context_journal.cpp writes one.
void put_turn(Store& store, const std::string& session, const std::string& title,
              const std::string& body, std::uint64_t event) {
    Record rec;
    rec.session = session;
    rec.kind = kind::kTurn;
    rec.title = title;
    rec.body = body;
    rec.first_event = event;
    rec.last_event = event;
    (void)store.append(std::move(rec));
}

// The sidecar's wiring, minus the sidecar: a source that resolves a store and a session.
Registry::ContextSourceFn source_of(Store* store, std::string session) {
    return [store, session] {
        Registry::ContextSource src;
        src.store = store;
        src.session = session;
        return src;
    };
}

} // namespace

TEST(the_recall_tools_are_absent_until_they_are_declared) {
    // The registry does not gain surface area by existing. A run whose database could
    // not be opened keeps every other tool and simply never declares these two -- which
    // is the degrade-don't-fail contract ContextJournal::open already documents, carried
    // one layer up.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Registry reg(workspace(root));
    CHECK(reg.find("context_recall") == nullptr);
    CHECK(reg.find("context_rehydrate") == nullptr);
    CHECK(reg.find("read_file") != nullptr);
}

TEST(declared_recall_tools_are_read_only_and_execute_nothing) {
    // These three flags are what put a tool on the parallel read-only dispatch path and
    // keep it off the approval, deliverable and verification ledgers
    // (Agent::can_run_in_parallel). They are DECLARED rather than inferred, so they are
    // worth asserting: a tool that reads a database has no business raising a card.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    for (const char* name : {"context_recall", "context_rehydrate"}) {
        const lmp::tools::ToolDecl* d = reg.find(name);
        REQUIRE(d != nullptr);
        CHECK(!d->mutates_workspace);
        CHECK(!d->executes_commands);
        CHECK(!d->irreversible);
    }
}

TEST(declaring_twice_is_refused_rather_than_duplicated) {
    // ensure_registry() REUSES a Registry across missions in the same workspace, and
    // start_mission calls declare_context_tools on every mission. declare() does not
    // check for duplicates: handlers_.emplace would silently keep the first handler while
    // the declaration and grammar-spec lists each grew a second copy, advertising both
    // tools twice in the system prompt.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));
    CHECK(!reg.declare_context_tools(source_of(&store, "s2"), estimate_tokens));

    std::size_t declared = 0;
    for (const lmp::tools::ToolDecl& d : reg.decls()) {
        if (d.name == "context_recall" || d.name == "context_rehydrate") {
            ++declared;
        }
    }
    CHECK_EQ(declared, std::size_t{2});
}

TEST(recall_returns_a_stored_turn_by_content) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    put_turn(store, "s1", "read_file",
             "the retry budget in fetcher.py is seven attempts, not three", 1);
    put_turn(store, "s1", "list_dir", "src tests docs", 2);

    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    const ToolResult r = reg.execute("context_recall", args({{"query", "retry budget"}}), 1);
    CHECK(r.ok());
    CHECK(r.summary.find("seven attempts") != std::string::npos);
}

TEST(a_recall_that_matches_nothing_still_says_something) {
    // NO TOOL MAY RETURN A SILENT RESULT (Registry::execute). An empty summary is dropped
    // from the rendered prompt, so a call that produced no text leaves the next turn's
    // prompt byte-identical -- and at a fixed seed that is a deterministic infinite loop.
    // "nothing matched" is a real answer and has to be said out loud.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    put_turn(store, "s1", "read_file", "unrelated content", 1);

    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    const ToolResult r =
        reg.execute("context_recall", args({{"query", "zzzznonexistent"}}), 1);
    CHECK(r.ok());
    CHECK(!r.summary.empty());
}

TEST(recall_without_a_store_answers_instead_of_crashing) {
    // The tools outlive the journal: ensure_registry keeps a Registry across missions,
    // and a later mission whose database will not open leaves the declarations in place
    // with nothing behind them. The source resolves null and the tool has to cope.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools([] { return Registry::ContextSource{}; },
                                    estimate_tokens));

    const ToolResult r = reg.execute("context_recall", args({{"query", "anything"}}), 1);
    // Not a refusal: a refusal means policy said no and the tool never ran, which would
    // send the model looking for permission it cannot obtain.
    CHECK(!r.ok());
    CHECK(r.status == lmp::tools::Status::ToolError);
    CHECK(r.error_class == lmp::tools::ErrorClass::NotFound);
    CHECK(!r.summary.empty());
}

TEST(recall_honours_the_token_budget_it_is_given) {
    // The budget IS the contract: a search that returns twenty rows has not helped an
    // agent, it has moved the problem. A tiny budget must come back small, with pointers
    // rather than bodies for what did not fit.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    for (std::uint64_t i = 1; i <= 20; ++i) {
        put_turn(store, "s1", "read_file",
                 "the retry budget is discussed at length here, entry number " +
                     std::to_string(i) + ", with a good deal of surrounding prose so "
                                         "that one entry cannot be mistaken for free",
                 i);
    }
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    const ToolResult small = reg.execute(
        "context_recall", args({{"query", "retry budget"}, {"token_budget", "120"}}), 1);
    const ToolResult large = reg.execute(
        "context_recall", args({{"query", "retry budget"}, {"token_budget", "4000"}}), 1);
    CHECK(small.ok());
    CHECK(large.ok());
    CHECK(small.summary.size() < large.summary.size());
    // estimate_tokens is bytes/4, and the tail list is allowed a quarter of the budget on
    // top of the packed entries. Generous, and still far below what an unbudgeted search
    // over twenty matching rows would have returned.
    CHECK(small.summary.size() < std::size_t{120 * 4 * 2});
}

TEST(an_absurd_budget_cannot_blow_the_prompt_open) {
    // ContextStore::add_turn asserts against the observation budget in every
    // configuration this repo builds, and clamps in a real run. A tool that can trip that
    // door is a tool that forgot to bound itself, and tool_result.hpp says bounding is
    // the tool's job.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    const std::string filler(4000, 'x');
    for (std::uint64_t i = 1; i <= 60; ++i) {
        put_turn(store, "s1", "read_file", "retry budget " + filler, i);
    }
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    const ToolResult r = reg.execute(
        "context_recall",
        args({{"query", "retry budget"}, {"token_budget", "100000000"}}), 1);
    CHECK(r.ok());
    CHECK(r.summary.size() <= (24U << 10) + 256);
}

TEST(rehydrate_turns_an_event_range_back_into_turns) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    for (std::uint64_t i = 1; i <= 6; ++i) {
        put_turn(store, "s1", "shell", "step " + std::to_string(i), i);
    }
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    const ToolResult r = reg.execute(
        "context_rehydrate", args({{"first_event", "2"}, {"last_event", "4"}}), 1);
    CHECK(r.ok());
    CHECK(r.summary.find("step 3") != std::string::npos);
    CHECK(r.summary.find("step 6") == std::string::npos);
}

TEST(rehydrate_refuses_a_range_it_cannot_mean) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "s1"), estimate_tokens));

    const ToolResult backwards = reg.execute(
        "context_rehydrate", args({{"first_event", "9"}, {"last_event", "2"}}), 1);
    CHECK(!backwards.ok());
    CHECK(backwards.error_class == lmp::tools::ErrorClass::Malformed);
}

TEST(rehydrate_stays_inside_the_session_that_printed_the_range) {
    // Event sequence numbers restart at 1 in every sidecar PROCESS, so two missions
    // launched separately both contain an event 3. The range only ever appears in the
    // current run's own summary, so an unscoped rehydrate would interleave turns from
    // unrelated runs under a range that means nothing to them.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    put_turn(store, "old-run", "shell", "the older mission ran a migration", 3);
    put_turn(store, "this-run", "shell", "the current mission ran a build", 3);

    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "this-run"), estimate_tokens));

    const ToolResult r = reg.execute(
        "context_rehydrate", args({{"first_event", "1"}, {"last_event", "9"}}), 1);
    CHECK(r.ok());
    CHECK(r.summary.find("current mission ran a build") != std::string::npos);
    CHECK(r.summary.find("older mission ran a migration") == std::string::npos);
}

TEST(recall_reaches_across_sessions_and_can_be_confined_to_one) {
    // THE FEATURE. A mission that ended writes what it learned; a later, separate mission
    // in the same workspace asks a question and gets it back. Without this the store is a
    // write-only log and the agent's effective context is its window -- which is the
    // state this component was in for its whole existence before the tools landed.
    //
    // The default is EVERY session, deliberately: the value of a durable store is what it
    // knows about this repo from last week. Recency is already one of the two fused rank
    // lists, so the current run is favoured without being the only thing visible.
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Store store(root + "/db.sqlite");
    put_turn(store, "monday-run", "read_file",
             "the deploy script requires PGHOST to be set before it will run", 4);
    put_turn(store, "friday-run", "shell", "ran the unit tests, all green", 2);

    Registry reg(workspace(root));
    CHECK(reg.declare_context_tools(source_of(&store, "friday-run"), estimate_tokens));

    const ToolResult across =
        reg.execute("context_recall", args({{"query", "deploy script PGHOST"}}), 1);
    CHECK(across.ok());
    CHECK(across.summary.find("PGHOST to be set") != std::string::npos);

    // And the escape hatch, for the run that wants only its own history.
    const ToolResult confined = reg.execute(
        "context_recall",
        args({{"query", "deploy script PGHOST"}, {"this_session_only", "true"}}), 1);
    CHECK(confined.ok());
    CHECK(confined.summary.find("PGHOST to be set") == std::string::npos);
}

TEST(check_framework_can_still_fail_here) {
    const std::string root = temp_dir();
    REQUIRE(!root.empty());
    Registry reg(workspace(root));
    EXPECT_FAILING_CHECKS(2, {
        CHECK(reg.find("context_recall") != nullptr);
        CHECK_EQ(reg.workspace().root, std::string("somewhere else"));
    });
}
