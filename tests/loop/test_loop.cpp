// The scripted-loop suite (S11.2): the whole loop, driven with no model, asserting on
// what the harness SENT. Plus the pure cores -- classifier, correctives, completion
// gate, verification falsifiability, context compaction.

#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/agent.hpp"
#include "src/loop/turn.hpp"
#include "src/loop/verification.hpp"

#include "tests/check.hpp"

using namespace lmp;
using namespace lmp::loop;

namespace {

context::TurnRecord turn(const std::string& tool, const std::string& obs, bool err = false) {
    context::TurnRecord t;
    t.tool_name = tool;
    t.observation = obs;
    t.observation_is_error = err;
    t.first_event_seq = 1;
    t.last_event_seq = 2;
    return t;
}

} // namespace

// --- one turn, one outcome (S9.1) -------------------------------------------

TEST(execution_is_asked_first_and_cannot_be_overwritten) {
    model::GenResult gen;
    gen.status = model::GenStatus::LengthCapped; // would otherwise say LengthCapped
    model::QwenTokenizer tok;
    const std::vector<parsephony::ToolSpec> none;
    const model::TurnGrammar g(tok, none);

    // The call EXECUTED. Nothing downstream may erase that -- v1's second observation
    // silently erased the first.
    CHECK(classify_turn(gen, g, /*executed=*/true, /*refused=*/false) ==
          Outcome::ToolCallExecuted);
}

TEST(refused_is_not_executed_and_not_an_error) {
    model::GenResult gen;
    gen.status = model::GenStatus::Complete;
    model::QwenTokenizer tok;
    const std::vector<parsephony::ToolSpec> none;
    const model::TurnGrammar g(tok, none);
    CHECK(classify_turn(gen, g, false, true) == Outcome::ToolCallRefused);
}

TEST(length_capped_is_never_completion) {
    model::GenResult gen;
    gen.status = model::GenStatus::LengthCapped;
    model::QwenTokenizer tok;
    const std::vector<parsephony::ToolSpec> none;
    const model::TurnGrammar g(tok, none);
    const Outcome o = classify_turn(gen, g, false, false);
    CHECK(o == Outcome::LengthCapped);
    CHECK(o != Outcome::TextOnly);
}

// --- mode policy in one place (S9.3) ----------------------------------------

TEST(plan_mode_cannot_execute_or_write) {
    const ModePolicy plan = ModePolicy::for_mode(Mode::Plan);
    CHECK_EQ(plan.sandbox_tier, 0);
    CHECK(!plan.allow_workspace_writes);

    const ModePolicy debug = ModePolicy::for_mode(Mode::Debug);
    CHECK_EQ(debug.sandbox_tier, 1);
    CHECK(!debug.allow_workspace_writes); // can run, cannot mutate

    const ModePolicy agent = ModePolicy::for_mode(Mode::Agent);
    CHECK(agent.allow_workspace_writes);
}

// --- exactly one repeat detector --------------------------------------------

TEST(repeat_detection_keys_on_tool_and_arguments) {
    RepeatDetector d;
    const std::vector<tools::ToolParamValue> a = {{"path", "src/main.cpp"}};
    const std::vector<tools::ToolParamValue> b = {{"path", "src/other.cpp"}};
    CHECK_EQ(d.seen_count("read_file", a), std::size_t{0});
    d.record("read_file", a);
    CHECK_EQ(d.seen_count("read_file", a), std::size_t{1});
    CHECK_EQ(d.seen_count("read_file", b), std::size_t{0});
    CHECK_EQ(d.seen_count("write_file", a), std::size_t{0});
    d.record("read_file", a);
    CHECK_EQ(d.seen_count("read_file", a), std::size_t{2});
}

TEST(at_most_one_corrective_and_budget_outranks_everything) {
    RepeatDetector d;
    TurnResult t;
    t.outcome = Outcome::ToolCallExecuted;
    t.tool_name = "read_file";
    t.tool_params = {{"path", "a"}};
    t.tool_result = tools::ToolResult::okay("ok");
    d.record("read_file", t.tool_params);
    d.record("read_file", t.tool_params);

    Budget budget;
    budget.max_iterations = 40;
    // Repeat alone -> BreakRepeat.
    CHECK(choose_corrective(t, d, 1, budget, false) == Corrective::BreakRepeat);
    // Budget exhausted outranks it; only ONE is returned.
    CHECK(choose_corrective(t, d, 40, budget, false) == Corrective::HaltOnBudget);
    CHECK(choose_corrective(t, d, 1, budget, true) == Corrective::HaltOnBudget);
}

TEST(a_claimed_verification_synthesizes_a_real_one) {
    RepeatDetector d;
    TurnResult t;
    t.outcome = Outcome::TextOnly;
    t.assistant_text = "I fixed the include. The build should pass now.";
    const Budget budget;
    // Mechanism, not prose: the loop MAKES the call the model only described.
    CHECK(choose_corrective(t, d, 1, budget, false) == Corrective::SynthesizeVerification);

    t.assistant_text = "Here is a summary of the file.";
    CHECK(choose_corrective(t, d, 1, budget, false) == Corrective::None);
}

// --- completion gate (S10.4) -------------------------------------------------

TEST(completion_is_driven_by_ledgers_not_by_prose) {
    context::ContextStore ctx("Add a --version flag");
    CHECK(!evaluate_completion(ctx).complete); // no checklist

    ctx.set_checklist({{"add flag", true}, {"test it", false}});
    CHECK(!evaluate_completion(ctx).complete); // an item is open

    ctx.set_checklist({{"add flag", true}, {"test it", true}});
    CHECK(!evaluate_completion(ctx).complete); // no deliverable

    ctx.record_deliverable("src/main.cpp");
    CHECK(!evaluate_completion(ctx).complete); // no verification

    context::VerificationRecord v;
    v.contract = "ctest";
    v.passed = true;
    v.falsifiable = false;
    ctx.record_verification(v);
    const CompletionVerdict unproven = evaluate_completion(ctx);
    // A green that has never been shown capable of red is not evidence (S10.2).
    CHECK(!unproven.complete);
    CHECK(unproven.reason.find("capable of failing") != std::string::npos);
}

TEST(a_proven_green_completes_the_run) {
    context::ContextStore ctx("Add a --version flag");
    ctx.set_checklist({{"add flag", true}});
    ctx.record_deliverable("src/main.cpp");
    context::VerificationRecord v;
    v.contract = "ctest";
    v.passed = true;
    v.falsifiable = true;
    ctx.record_verification(v);
    CHECK(evaluate_completion(ctx).complete);
}

// --- verification identity (S10.2) ------------------------------------------

TEST(a_reporting_wrapper_does_not_mint_a_second_identity) {
    const std::string base = canonicalize_check("cmake --build build");
    CHECK_EQ(canonicalize_check("cmake --build build ; echo $?"), base);
    CHECK_EQ(canonicalize_check("cmake   --build    build"), base);
    CHECK_EQ(canonicalize_check("  cmake --build build 2>&1  "), base);
    CHECK_EQ(canonicalize_check("cmake --build build && echo $? | cat"), base);
    // A genuinely different check keeps its own identity -- the proof is per-check.
    CHECK(canonicalize_check("cmake --build build2") != base);
}

// --- context tiers and compaction (S8.3) ------------------------------------

TEST(a_run_that_trims_twice_still_answers_from_pre_trim_evidence) {
    // Phase 6's exit criterion, verbatim from S17.
    context::ContextStore ctx("Find the port number and report it");

    // The evidence arrives first, then a lot of noise on top of it.
    ctx.add_turn(turn("read_file", "config.yaml: listen_port: 8443"));
    for (int i = 0; i < 30; ++i) {
        ctx.add_turn(turn("list_dir", "file_" + std::to_string(i) + ".txt"));
    }

    CHECK_EQ(ctx.compact_oldest(10), std::size_t{21});
    for (int i = 0; i < 12; ++i) {
        ctx.add_turn(turn("search", "no match " + std::to_string(i)));
    }
    CHECK(ctx.compact_oldest(10) > 0);
    CHECK(ctx.compaction_count() >= 2);

    // The evidence survived two trims -- summarized, not evicted.
    bool found = false;
    for (const std::string& span : ctx.compacted_spans()) {
        found = found || span.find("8443") != std::string::npos;
    }
    CHECK(found);

    // And it is present in what the model would actually be sent.
    const std::vector<context::Message> msgs = ctx.render("");
    bool in_prompt = false;
    for (const context::Message& m : msgs) {
        in_prompt = in_prompt || m.content.find("8443") != std::string::npos;
    }
    CHECK(in_prompt);
}

TEST(compaction_summarizes_rather_than_announces_a_drop) {
    context::ContextStore ctx("m");
    ctx.add_turn(turn("shell", "error: undefined reference to `foo'", true));
    ctx.add_turn(turn("read_file", "int main() {}"));
    (void)ctx.compact_oldest(0);
    REQUIRE(ctx.compacted_spans().size() == 1);
    const std::string& span = ctx.compacted_spans()[0];
    // The anchor survived, and the failure is still marked as one.
    CHECK(span.find("undefined reference") != std::string::npos);
    CHECK(span.find("FAILED") != std::string::npos);
    // It is not a notice that something was dropped.
    CHECK(span.find("truncated") == std::string::npos);
    CHECK(span.find("omitted") == std::string::npos);
}

TEST(the_mission_and_pinned_state_survive_every_trim) {
    context::ContextStore ctx("THE MISSION: ship the parser");
    ctx.set_checklist({{"write it", true}, {"test it", false}});
    ctx.record_deliverable("src/parser.cpp");
    for (int i = 0; i < 50; ++i) {
        ctx.add_turn(turn("list_dir", "noise"));
    }
    (void)ctx.compact_oldest(2);

    const std::vector<context::Message> msgs = ctx.render("");
    REQUIRE(!msgs.empty());
    const std::string& system = msgs[0].content;
    CHECK(system.find("THE MISSION: ship the parser") != std::string::npos);
    CHECK(system.find("- [x] write it") != std::string::npos);
    CHECK(system.find("- [ ] test it") != std::string::npos);
    CHECK(system.find("src/parser.cpp") != std::string::npos);
}

// --- HITL routing (S7.6) -----------------------------------------------------

TEST(risk_routing_is_a_pure_function) {
    HitlThresholds t;

    tools::RiskHint benign;
    benign.status = blast_radius::ParseStatus::Parsed;
    CHECK(route_approval(benign, t) == Approval::AutoApprove);

    tools::RiskHint destructive;
    destructive.status = blast_radius::ParseStatus::Parsed;
    destructive.caps.destroys_data = true;
    destructive.caps.writes_outside_workspace = true;
    destructive.caps.escalates_privileges = true;
    CHECK(route_approval(destructive, t) == Approval::Reject);

    // A command whose effects are NOT in the string escalates on that alone -- that is
    // what PartiallyParsed is for (S7.1).
    tools::RiskHint unseeable;
    unseeable.status = blast_radius::ParseStatus::PartiallyParsed;
    unseeable.caps.network_access = true;
    unseeable.caps.destroys_data = true;
    CHECK(route_approval(unseeable, t) == Approval::Escalate);
}
