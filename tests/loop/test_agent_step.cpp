// Agent::step, driven end to end IN THE GATE (G0's payoff).
//
// docs/PHASES.md names agent.cpp:168 and agent.cpp:287 as surviving mutants and gives one
// reason for both: "Agent::step is not driven end-to-end in the gate -- vocab-fixture
// blocker". The fixture now exists (tests/fixtures/make_mini_vocab.py), and providing it
// without writing the test it was for would have left the mutants exactly where they were
// while claiming the blocker was cleared.
//
// No GPU and no 19 GB load: ScriptedBackend supplies the tokens, the miniature vocabulary
// supplies the ids, and the real Agent does everything in between -- prompt assembly,
// grammar-driven turn classification, tool dispatch, context recording.
//
// tests/loop/test_token_stream.cpp keeps its own end-to-end case and stays `realmodel`,
// because what IT asserts is the byte-fragment property and 944 of the real checkpoint's
// tokens are the subject. That is a genuine need for the real vocab; this is not.

#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/agent.hpp"
#include "src/model/backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/tools/registry.hpp"

#include "tests/check.hpp"

using namespace lmp;

namespace {

const model::QwenTokenizer& mini_vocab() {
    static model::QwenTokenizer tok;
    static model::LoadStatus st = tok.load(LMP_MINI_VOCAB_JSON, model::Family::Qwen3);
    if (!st.ok) {
        static bool said = false;
        if (!said) {
            test::record_failure(__FILE__, __LINE__, "mini vocab load: " + st.error);
            said = true;
        }
    }
    return tok;
}

tools::WorkspaceContext workspace(const std::string& root) {
    tools::WorkspaceContext ws;
    ws.root = root;
    ws.max_read_bytes = 1U << 20;
    ws.max_model_read_bytes = 16384;
    ws.max_result_bytes = 8192;
    ws.spool_dir = root;
    ws.shell_wall_clock_seconds = 5;
    return ws;
}

// A complete text-only turn as ids. The turn OPENS in the Think phase -- `<think>` comes
// from the chat template, not from generation -- so the script starts with reasoning
// content, closes it, answers, and ends on `<|im_end|>`, which is what the grammar
// ACCEPTS on. Stopping is grammar state, never text matching (S5.5).
std::vector<model::TokenId> text_turn(const model::QwenTokenizer& tok,
                                      const std::string& reasoning,
                                      const std::string& answer) {
    std::vector<model::TokenId> script;
    for (model::TokenId id : tok.encode_content(reasoning)) {
        script.push_back(id);
    }
    script.push_back(tok.specials().think_close);
    for (model::TokenId id : tok.encode_content(answer)) {
        script.push_back(id);
    }
    script.push_back(tok.specials().im_end);
    return script;
}

} // namespace

TEST(a_text_only_turn_runs_end_to_end_through_the_real_agent) {
    const model::QwenTokenizer& tok = mini_vocab();
    REQUIRE(tok.loaded());

    model::ScriptedBackend backend;
    backend.enqueue_response(text_turn(tok, "weighing the options", "Here is the answer."));

    tools::Registry registry(workspace("/tmp"));
    context::ContextStore ctx("say something");
    platform::EventLogWriter log;
    platform::SystemClock clock;
    loop::AgentConfig config;
    config.auto_syntax_check = false; // no workspace writes in this turn; keep it hermetic
    loop::Agent agent(tok, backend, registry, ctx, log, clock, config);

    const model::CancelToken cancel;
    const loop::TurnResult turn = agent.step(cancel);

    // ONE turn, ONE outcome (S9.1). No call ran, so this is TextOnly -- and specifically
    // NOT LengthCapped, which the classifier used to conflate with completion.
    CHECK(turn.outcome == loop::Outcome::TextOnly);
    CHECK_EQ(turn.assistant_text, std::string("Here is the answer."));
    // Reasoning is peeled off and surfaced separately; it is never part of the answer and
    // is never carried forward into the next prompt (S5.7).
    CHECK_EQ(turn.reasoning, std::string("weighing the options"));
    CHECK(turn.tool_name.empty());
}

// The prompt actually reaches the backend, and it is the one the context store rendered.
// This is the seam every other loop assertion rests on: if step() built a different prompt
// from the one ContextStore::render produces, every downstream test would still pass.
TEST(the_prompt_the_backend_receives_is_the_rendered_context) {
    const model::QwenTokenizer& tok = mini_vocab();
    REQUIRE(tok.loaded());

    model::ScriptedBackend backend;
    backend.enqueue_response(text_turn(tok, "thinking", "done"));

    tools::Registry registry(workspace("/tmp"));
    context::ContextStore ctx("a distinctive mission string");
    platform::EventLogWriter log;
    platform::SystemClock clock;
    loop::AgentConfig config;
    config.auto_syntax_check = false;
    loop::Agent agent(tok, backend, registry, ctx, log, clock, config);

    const model::CancelToken cancel;
    (void)agent.step(cancel);

    REQUIRE(!backend.received().empty());
    const model::InferenceTask& task = backend.received().front();
    const std::vector<model::TokenId>& sent = task.prompt;
    REQUIRE(!sent.empty());
    // The mission is in there, and it went through the CONTENT encoder -- so a mission
    // containing "<|im_end|>" could not mint a control token (S5.4).
    const std::string decoded = tok.decode(sent);
    CHECK(decoded.find("a distinctive mission string") != std::string::npos);
    // And the stable boundary the KV checkpoint needs is inside the prompt, not past it.
    CHECK(task.checkpoint_at <= sent.size());
}

// A turn that hits the generation cap leaves NOTHING behind -- no answer body, no call,
// and reasoning is not carried forward. The record would be empty, the context unchanged,
// and the next turn would re-render a byte-identical prompt: a deterministic infinite loop
// at a fixed seed. Observed for twelve consecutive turns before the truncation itself was
// made the observation. This asserts the classifier's half of that fix.
TEST(a_length_capped_turn_is_not_completion) {
    const model::QwenTokenizer& tok = mini_vocab();
    REQUIRE(tok.loaded());

    // No think_close, no im_end: the script simply runs out.
    std::vector<model::TokenId> truncated;
    for (model::TokenId id : tok.encode_content("thinking and thinking and")) {
        truncated.push_back(id);
    }
    model::ScriptedBackend backend;
    backend.enqueue_response(truncated);

    tools::Registry registry(workspace("/tmp"));
    context::ContextStore ctx("keep going");
    platform::EventLogWriter log;
    platform::SystemClock clock;
    loop::AgentConfig config;
    config.auto_syntax_check = false;
    config.max_new_tokens = static_cast<std::int32_t>(truncated.size());
    loop::Agent agent(tok, backend, registry, ctx, log, clock, config);

    const model::CancelToken cancel;
    const loop::TurnResult turn = agent.step(cancel);
    CHECK(turn.outcome != loop::Outcome::TextOnly);
    CHECK(turn.outcome != loop::Outcome::ToolCallExecuted);
}
