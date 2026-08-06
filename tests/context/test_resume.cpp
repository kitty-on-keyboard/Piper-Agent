// Crash-safe resume: reconstruct ContextStore from the event log without re-running tools.

#include <string>
#include <vector>

#include "src/context/resume.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"

#include "tests/check.hpp"

using namespace lmp;
using namespace lmp::context;
using namespace lmp::platform;

namespace {

Event make(std::uint64_t seq, std::string kind,
           std::vector<EventField> fields) {
    Event ev;
    ev.seq = seq;
    ev.kind = std::move(kind);
    ev.fields = std::move(fields);
    return ev;
}

std::vector<Event> sample_run() {
    return {
        make(1, "run_begin",
             {{"run_id", "r1"},
              {"mission", "fix the bug"},
              {"workspace_root", "/tmp/ws"},
              {"model_identity", "/models/qwen"},
              {"protocol_version", "1.0.0"},
              {"tool_schema_hash", "abc"}}),
        make(2, "tool_result",
             {{"run_id", "r1"},
              {"tool", "read_file"},
              {"status", "ok"},
              {"summary", "file contents here"}}),
        make(3, "tool_result",
             {{"run_id", "r1"},
              {"tool", "search"},
              {"status", "ok"},
              {"summary", "2 matches"}}),
        make(4, "write", {{"run_id", "r1"}, {"path", "src/a.cpp"}}),
        make(5, "steer", {{"run_id", "r1"}, {"text", "also check tests"}}),
        make(6, "run_end",
             {{"run_id", "r1"}, {"termination_reason", "completed"}, {"completed", "1"}}),
    };
}

} // namespace

TEST(parse_event_line_round_trips_serialize) {
    Event ev;
    ev.seq = 42;
    ev.wall_ns = 100;
    ev.mono_us = 200;
    ev.kind = "tool_result";
    ev.fields = {{"tool", "read_file"}, {"summary", "hello\"world"}};
    const std::string line = serialize_event(ev);
    Event back;
    REQUIRE(parse_event_line(line, back));
    CHECK_EQ(back.seq, std::uint64_t{42});
    CHECK_EQ(back.kind, std::string("tool_result"));
    CHECK_EQ(field_or(back, "tool"), std::string("read_file"));
    CHECK_EQ(field_or(back, "summary"), std::string("hello\"world"));
}

TEST(parse_event_line_prefers_b64_sibling) {
    // Lossy display form plus faithful bytes -- resume must take the bytes.
    const std::string raw = "\xF0\x9F"; // invalid standalone UTF-8
    Event ev;
    ev.seq = 1;
    ev.kind = "token";
    ev.fields = {{"bytes", raw}};
    const std::string line = serialize_event(ev);
    Event back;
    REQUIRE(parse_event_line(line, back));
    CHECK_EQ(field_or(back, "bytes"), raw);
}

TEST(reconstruct_restores_observations_without_tools) {
    const ResumeRebuild rebuilt = reconstruct_context(sample_run(), "r1");
    REQUIRE(rebuilt.ok);
    CHECK(!rebuilt.edit_in_flight);
    CHECK_EQ(rebuilt.store.mission(), std::string("fix the bug"));
    CHECK_EQ(rebuilt.identity.workspace_root, std::string("/tmp/ws"));
    REQUIRE(rebuilt.store.recent().size() >= 3);
    CHECK_EQ(rebuilt.store.recent()[0].tool_name, std::string("read_file"));
    CHECK_EQ(rebuilt.store.recent()[0].observation, std::string("file contents here"));
    CHECK_EQ(rebuilt.store.recent()[1].tool_name, std::string("search"));
    CHECK_EQ(rebuilt.store.user_turns().size(), std::size_t{2});
    CHECK_EQ(rebuilt.store.user_turns()[1], std::string("also check tests"));
    CHECK_EQ(rebuilt.store.deliverables().size(), std::size_t{1});
    CHECK_EQ(rebuilt.store.deliverables()[0], std::string("src/a.cpp"));
}

TEST(auto_resume_refuses_edit_in_flight) {
    std::vector<Event> events = sample_run();
    events.insert(events.end() - 1,
                  make(5, "edit_begin", {{"run_id", "r1"}, {"request_id", "r1/edit/1"}}));
    // Remove the earlier steer seq clash by rebuilding a clearer stream.
    events = {
        make(1, "run_begin",
             {{"run_id", "r1"},
              {"mission", "m"},
              {"workspace_root", "/tmp/ws"},
              {"model_identity", "/m"},
              {"protocol_version", "1.0.0"},
              {"tool_schema_hash", "h"}}),
        make(2, "tool_result",
             {{"run_id", "r1"}, {"tool", "read_file"}, {"status", "ok"}, {"summary", "x"}}),
        make(3, "edit_begin", {{"run_id", "r1"}, {"request_id", "r1/edit/9"}}),
    };
    const ResumeRebuild rebuilt = reconstruct_context(events, "r1");
    REQUIRE(rebuilt.ok);
    CHECK(rebuilt.edit_in_flight);

    ResumeIdentity current = rebuilt.identity;
    const ResumeGate gate = can_auto_resume(rebuilt.identity, current, rebuilt.edit_in_flight);
    CHECK(!gate.allowed);
    CHECK(gate.why.find("edit") != std::string::npos);
}

TEST(auto_resume_refuses_workspace_change) {
    const ResumeRebuild rebuilt = reconstruct_context(sample_run(), "r1");
    REQUIRE(rebuilt.ok);
    ResumeIdentity current = rebuilt.identity;
    current.workspace_root = "/tmp/other";
    const ResumeGate gate =
        can_auto_resume(rebuilt.identity, current, rebuilt.edit_in_flight);
    CHECK(!gate.allowed);
    CHECK(gate.why.find("workspace") != std::string::npos);
}

TEST(auto_resume_allows_identical_identity) {
    const ResumeRebuild rebuilt = reconstruct_context(sample_run(), "r1");
    REQUIRE(rebuilt.ok);
    const ResumeGate gate =
        can_auto_resume(rebuilt.identity, rebuilt.identity, false);
    CHECK(gate.allowed);
}

TEST(edit_end_clears_in_flight) {
    const std::vector<Event> events = {
        make(1, "run_begin",
             {{"run_id", "r1"},
              {"mission", "m"},
              {"workspace_root", "/tmp/ws"},
              {"model_identity", "/m"},
              {"protocol_version", "1.0.0"},
              {"tool_schema_hash", "h"}}),
        make(2, "edit_begin", {{"run_id", "r1"}, {"request_id", "e1"}}),
        make(3, "edit_end", {{"run_id", "r1"}, {"request_id", "e1"}, {"applied", "1"}}),
        make(4, "tool_result",
             {{"run_id", "r1"}, {"tool", "write_file"}, {"status", "ok"}, {"summary", "ok"}}),
    };
    const ResumeRebuild rebuilt = reconstruct_context(events, "r1");
    REQUIRE(rebuilt.ok);
    CHECK(!rebuilt.edit_in_flight);
    CHECK_EQ(rebuilt.identity.last_acked_edit_id, std::string("e1"));
}
