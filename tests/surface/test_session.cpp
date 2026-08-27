// The model lifecycle: load, refuse, unload (M2).
//
// This is here because of a specific failure, and it is worth stating plainly. Loading
// the weights used to be a side effect of the first mission -- no method of its own, no
// state, no way for a surface to know whether the sidecar held 19 GB or nothing. So "no
// model is loaded" and "the model is thinking" rendered identically: a status line, and
// no output. A user sent a prompt into a freshly opened editor, was told it was thinking,
// and waited on a process that had never opened a checkpoint.
//
// Everything below is model-free on purpose. Both refusals happen at the tokenizer, which
// is a file read, so these assertions hold identically on a machine with MLX and one
// without -- and the gate has no model (S11.1).

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include <cstdio>
#include <cstdlib>

#include <unistd.h>

#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/platform/fs.hpp"
#include "src/surface/session.hpp"

#include "tests/check.hpp"

using lmp::surface::load_model;
using lmp::surface::ModelLoad;
using lmp::surface::Session;
using lmp::surface::unload_model;

namespace {

lmp::platform::ManualClock clock_;

} // namespace

TEST(an_empty_model_dir_is_refused_rather_than_defaulted) {
    // S7.5: no security-relevant input gets a default. An empty model_dir that fell
    // through to "some checkpoint" would be the whole rule going the wrong way.
    Session session;
    const ModelLoad r = load_model(session, "", "", clock_);
    CHECK(!r.ok);
    CHECK(!r.error.empty());
    CHECK(!session.model_ready());
}

TEST(a_missing_checkpoint_refuses_with_the_loaders_own_words) {
    Session session;
    const ModelLoad r = load_model(session, "/nonexistent/checkpoint", "", clock_);
    CHECK(!r.ok);
    // The path is IN the message. A load failure is nearly always a fixable statement
    // about what is on disk, and "load failed" with the path paraphrased away is the one
    // version of this the operator cannot act on.
    CHECK(r.error.find("/nonexistent/checkpoint") != std::string::npos);
    CHECK(!session.model_ready());
}

TEST(a_failed_load_leaves_no_half_loaded_session) {
    // The dangerous shape is a tokenizer for one checkpoint and weights for another:
    // that generates fluent nonsense instead of failing. model_ready() is both halves,
    // and holds() is what the sidecar asks before deciding whether a load will block.
    Session session;
    (void)load_model(session, "/nonexistent/checkpoint", "", clock_);
    CHECK(!session.model_ready());
    CHECK(session.model_dir.empty());
    CHECK(!session.holds("/nonexistent/checkpoint"));
    CHECK(!session.holds(""));
}

TEST(unload_takes_the_conversation_with_the_weights) {
    // A ContextStore whose model has gone is not resumable, so unloading is a full reset
    // of everything downstream of the weights rather than a free() of the weights alone.
    // If ctx survived here, the next prompt would be sent as a follow-up over a session
    // the sidecar can no longer answer for.
    Session session;
    session.model_dir = "/some/checkpoint";
    session.mcp_signature = "sig";
    session.ctx = std::make_unique<lmp::context::ContextStore>("a mission");
    unload_model(session);
    CHECK(!session.model_ready());
    CHECK(session.model_dir.empty());
    CHECK(session.ctx == nullptr);
    CHECK(session.registry == nullptr);
    CHECK(session.mcp_signature.empty());
}

TEST(unload_is_safe_on_a_session_that_never_loaded) {
    // The state a fresh process is in, and the state the surface can ask to unload from
    // -- a button that is only correct when something is loaded is a button that crashes.
    Session session;
    unload_model(session);
    CHECK(!session.model_ready());
}

// --- tools the conventions name and the registry does not have -------------------
//
// A MISSING TOOL IS INVISIBLE FROM THE INSIDE. Tool names are constrained at decode time
// by parsephony::ToolCallGuard, so a model told by AGENTS.md to "call `godot_guide`"
// cannot emit that call when the tool is unregistered -- the mask steers the name into a
// registered one, and there is no error and no counterfactual to log. Measured on
// r-18cecc130e7bc558-31fdd81a: the model's reasoning said "learning the scene spec format
// via godot_guide, let me call that tool now" and the executed call was `git_status`. It
// spent six of fourteen turns rediscovering this through the shell and then stalled.
//
// So the run-start check has two jobs and both are asserted here: notice the name that is
// genuinely absent, and DO NOT notice one that is present under an MCP namespace. The
// second is the one that would quietly destroy the feature -- reporting every connected
// remote tool as missing is the same defect pointed the other way.

namespace {

lmp::tools::Registry conventions_registry() {
    lmp::tools::WorkspaceContext ctx;
    ctx.root = "/tmp";
    ctx.max_read_bytes = 1U << 20;
    ctx.max_model_read_bytes = 16384;
    ctx.max_result_bytes = 8192;
    ctx.max_observation_bytes = lmp::tools::kObservationBudgetBytes;
    ctx.spool_dir = "/tmp/.spool";
    ctx.shell_wall_clock_seconds = 20;
    ctx.model_can_see = false;
    return lmp::tools::Registry(std::move(ctx));
}

} // namespace

TEST(a_tool_the_conventions_tell_the_model_to_call_and_the_registry_lacks_is_reported) {
    const lmp::tools::Registry reg = conventions_registry();
    // The real sentence from the run that exposed this.
    const std::string conventions =
        "Godoer briefs you itself: it is connected over MCP.\n"
        "**Call `godot_guide` before authoring your first scene** -- it returns the scene "
        "spec format, which you cannot guess.\n";
    const std::vector<std::string> got = lmp::surface::unknown_tool_names(conventions, reg);
    REQUIRE(got.size() == std::size_t{1});
    CHECK_EQ(got[0], std::string("godot_guide"));

    // The note has to tell the model what to DO. The failure was never a model that
    // lacked the fact; it was a model that established it by trial and had no move left.
    const std::string note = lmp::surface::unknown_tools_note(got);
    CHECK(note.find("godot_guide") != std::string::npos);
    CHECK(note.find("ask_user") != std::string::npos);
    CHECK(lmp::surface::unknown_tools_note({}).empty());
}

TEST(a_tool_that_is_registered_is_never_reported_missing) {
    const lmp::tools::Registry reg = conventions_registry();
    // Native, and named exactly as the conventions would name it.
    CHECK(lmp::surface::unknown_tool_names("Always call `read_file` before editing.", reg)
              .empty());
    CHECK(lmp::surface::unknown_tool_names("Use `ask_user` for a design choice.", reg)
              .empty());
}

TEST(an_mcp_tool_answers_to_the_name_the_conventions_use) {
    // The host registers `godot_guide` when that name is free. Collision still
    // namespaces (`mcp__godoer__godot_guide`). Matching only the exact registered
    // string would report the namespaced form as absent while AGENTS.md says
    // `godot_guide` -- the suffix alias is the collision path, not the primary one.
    lmp::tools::Registry reg = conventions_registry();
    lmp::tools::ToolDecl d;
    d.name = "mcp__godoer__godot_guide";
    d.description = "returns the scene spec format";
    d.spec.name = d.name;
    // declare_remote is the public door an MCP server comes through, which is exactly
    // the path this test is about.
    REQUIRE(reg.declare_remote(std::move(d),
                               [](const std::vector<lmp::tools::ToolParamValue>&, int) {
                                   return lmp::tools::ToolResult{};
                               }));
    CHECK(lmp::surface::unknown_tool_names("Call `godot_guide` first.", reg).empty());
}

TEST(the_check_is_narrow_enough_not_to_spend_prompt_on_prose) {
    const lmp::tools::Registry reg = conventions_registry();
    // Not backticked, so not a candidate however tool-shaped it looks.
    CHECK(lmp::surface::unknown_tool_names("Call godot_guide first.", reg).empty());
    // Backticked but no underscore: `capture` and `plan` are ordinary words a README
    // quotes, and the underscore is what separates them from a tool name cheaply.
    CHECK(lmp::surface::unknown_tool_names("Then use `capture` on it.", reg).empty());
    // Underscored but nothing nearby says to call it. A quoted identifier in prose is
    // not a claim that a tool exists.
    CHECK(lmp::surface::unknown_tool_names(
              "The field is `max_position_embeddings` in the config.", reg)
              .empty());
    // Paths and globs are excluded by the charset, which is what keeps `game/`,
    // `project.godot` and `godot_*` out of the report.
    CHECK(lmp::surface::unknown_tool_names("Run `godot_*` tools for this.", reg).empty());
    CHECK(lmp::surface::unknown_tool_names("Call `specs/main.json` here.", reg).empty());
}

TEST(the_report_is_deduplicated_and_bounded) {
    const lmp::tools::Registry reg = conventions_registry();
    std::string many = "Call `godot_guide`. Call `godot_guide` again.";
    for (int i = 0; i < 20; ++i) {
        many += " Call `godot_tool_" + std::to_string(i) + "`.";
    }
    const std::vector<std::string> got = lmp::surface::unknown_tool_names(many, reg);
    // Bounded so a file full of snake_case cannot grow the system prompt without limit,
    // and deduplicated so a name repeated for emphasis is reported once.
    CHECK(got.size() <= std::size_t{8});
    CHECK_EQ(got[0], std::string("godot_guide"));
    CHECK(std::count(got.begin(), got.end(), std::string("godot_guide")) == 1);
}

// --- the wiring, end to end ------------------------------------------------------
//
// Both halves above are unit-tested, and both were correct in the version that shipped
// the bug: `parse_mcp_servers` worked, and nothing called it with the project's own file.
// So this asserts the JOIN -- that ensure_registry actually reads `.mcp.json` from the
// workspace it was handed and puts what it found on the record. The server here is a
// command that cannot spawn, deliberately: whether godoer starts is godoer's business,
// and this test is about whether we ever looked.

TEST(ensure_registry_reads_the_projects_own_mcp_json) {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base != nullptr ? base : "/tmp") + "/lmp_wire_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    REQUIRE(made != nullptr);
    const std::string root(made);
    {
        std::FILE* f = std::fopen((root + "/.mcp.json").c_str(), "wb");
        REQUIRE(f != nullptr);
        // `trusted` set, so the same test covers the vouch being refused.
        const std::string body =
            R"({"mcpServers":{"godoer":{"command":"/nonexistent/godoer","args":["mcp"],)"
            R"("trusted":true}}})";
        std::fwrite(body.data(), 1, body.size(), f);
        std::fclose(f);
    }

    const std::string log_path = root + "/events.jsonl";
    lmp::platform::ManualClock c;
    lmp::platform::EventLogWriter log;
    REQUIRE(log.open({log_path, 1 << 20, 3}).ok);

    Session session;
    // A start message carrying no mcp_servers at all -- the shape every run has had.
    const std::string message =
        R"({"method":"lmp/start","params":{"mission":"m","settings":{"model_dir":"/m"}}})";
    lmp::surface::ensure_registry(session, root, message, log, c);
    log.flush();

    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(log_path, 1 << 20);
    REQUIRE(f.ok());
    // The file was read, and the count is of what it actually contained.
    CHECK(f.bytes.find("\"kind\":\"mcp_config_file\"") != std::string::npos);
    CHECK(f.bytes.find("\"servers\":\"1\"") != std::string::npos);
    // The vouch was refused and SAID SO. A silently downgraded trust is the same class of
    // problem as a silently honoured one: the operator cannot tell which they got.
    CHECK(f.bytes.find("\"trusted_ignored\":\"1\"") != std::string::npos);
    // It reached the host as a server to connect, attributed to the file that named it,
    // and carded rather than trusted.
    CHECK(f.bytes.find("\"kind\":\"mcp_server\"") != std::string::npos);
    CHECK(f.bytes.find("\"source\":\".mcp.json\"") != std::string::npos);
    CHECK(f.bytes.find("\"trusted\":\"0\"") != std::string::npos);
}

TEST(a_workspace_with_no_mcp_json_says_nothing_at_all) {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base != nullptr ? base : "/tmp") + "/lmp_wire2_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    REQUIRE(made != nullptr);
    const std::string root(made);

    const std::string log_path = root + "/events.jsonl";
    lmp::platform::ManualClock c;
    lmp::platform::EventLogWriter log;
    REQUIRE(log.open({log_path, 1 << 20, 3}).ok);

    Session session;
    lmp::surface::ensure_registry(
        session, root, R"({"method":"lmp/start","params":{"settings":{}}})", log, c);
    log.flush();

    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(log_path, 1 << 20);
    REQUIRE(f.ok());
    // The common case is a project without one, and it must not cost an event.
    CHECK(f.bytes.find("mcp_config_file") == std::string::npos);
}
