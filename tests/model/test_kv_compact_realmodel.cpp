// Shadow-compact M3: after compact_to_budget, the next turn's prefill_reused_tokens
// must be ≫ 0 when LMP_SHADOW_COMPACT is on (AgentConfig.shadow_compact), and sampled
// tokens must match the flag-off path (greedy). Flag off is the M0 tax: reuse ~0.
//
// realmodel, jobs=1. Compact percents stay 75/35.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/agent.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/tools/registry.hpp"

#include "tests/check.hpp"

using namespace lmp;
using namespace lmp::model;

namespace {

std::string qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr ? std::string(v) : std::string("");
}

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base != nullptr ? base : "/tmp") + "/lmp_kvsh_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made != nullptr ? std::string(made) : std::string();
}

context::ContextStore fat_context(int turns) {
    context::ContextStore ctx("keep the context honest");
    for (int i = 0; i < turns; ++i) {
        context::TurnRecord rec;
        rec.tool_name = "read_file";
        rec.tool_args_summary = "src/module_" + std::to_string(i) + ".txt";
        rec.assistant_text =
            "Reading module " + std::to_string(i) + " to see what it does.";
        rec.observation =
            "line one of the file body for module " + std::to_string(i) +
            "; line two of the same file; line three, which carries enough "
            "text that a dozen of these are worth compacting away.";
        ctx.add_turn(std::move(rec));
    }
    return ctx;
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

struct CompactRun {
    std::size_t reused_turn2 = 0;
    double ttft_turn2 = 0;
    std::string text;
    bool compacted = false;
};

} // namespace

TEST(shadow_compact_reuses_the_warmed_prefix_and_matches_flag_off) {
    QwenTokenizer tok;
    REQUIRE(tok.load(qwen_dir() + "/tokenizer.json", Family::Qwen3).ok);

    platform::SystemClock clock;
    model::MlxBackend backend(clock);
    model::MlxBackendConfig cfg;
    cfg.model_dir = qwen_dir();
    REQUIRE(backend.load(cfg).ok);

    const auto play = [&](bool shadow, std::int32_t budget) {
        backend.reset_cache();
        const std::string root = temp_dir();
        tools::Registry registry(workspace(root));
        context::ContextStore ctx = fat_context(12);
        platform::EventLogWriter log;
        loop::AgentConfig config;
        config.auto_syntax_check = false;
        config.operator_verify_contract.clear();
        config.context_budget_tokens = budget;
        config.budget.max_iterations = 2;
        config.max_new_tokens = 16;
        config.max_think_tokens = 16;
        config.sampling.temperature = 0.0F;
        config.sampling.seed = 1;
        config.shadow_compact = shadow;
        loop::Agent agent(tok, backend, registry, ctx, log, clock, config);

        CompactRun out;
        std::size_t turn = 0;
        loop::Observer obs;
        obs.on_token = [&](const std::string&, const std::string& text) { out.text += text; };
        obs.on_perf = [&](const model::GenResult& g, std::size_t, std::size_t, std::size_t) {
            ++turn;
            if (turn == 2) {
                out.reused_turn2 = g.prefill_reused_tokens;
                out.ttft_turn2 = g.ttft_ms;
            }
        };
        agent.set_observer(obs);
        const model::CancelToken cancel;
        (void)agent.run(cancel);
        out.compacted = ctx.compaction_count() > 0;
        return out;
    };

    // Calibrate budget: one cold step against a huge cap, then 85% of that prompt.
    std::int32_t huge = 1000000;
    std::size_t prompt_n = 0;
    {
        backend.reset_cache();
        const std::string root = temp_dir();
        tools::Registry registry(workspace(root));
        context::ContextStore ctx = fat_context(12);
        platform::EventLogWriter log;
        loop::AgentConfig config;
        config.auto_syntax_check = false;
        config.operator_verify_contract.clear();
        config.context_budget_tokens = huge;
        config.budget.max_iterations = 1;
        config.max_new_tokens = 8;
        config.max_think_tokens = 8;
        loop::Agent agent(tok, backend, registry, ctx, log, clock, config);
        loop::Observer obs;
        obs.on_perf = [&](const model::GenResult&, std::size_t used, std::size_t, std::size_t) {
            prompt_n = used;
        };
        agent.set_observer(obs);
        const model::CancelToken cancel;
        (void)agent.run(cancel);
    }
    REQUIRE(prompt_n > 0);
    const auto budget = static_cast<std::int32_t>(prompt_n * 100 / 85);

    const CompactRun off = play(false, budget);
    REQUIRE(off.compacted);
    std::printf("  [shadow-off] turn2 reused=%zu ttft=%.0fms\n", off.reused_turn2,
                off.ttft_turn2);

    const CompactRun on = play(true, budget);
    REQUIRE(on.compacted);
    std::printf("  [shadow-on]  turn2 reused=%zu ttft=%.0fms\n", on.reused_turn2,
                on.ttft_turn2);

    CHECK(on.reused_turn2 > 0);
    CHECK(on.reused_turn2 > off.reused_turn2);
    REQUIRE(!on.text.empty());
    CHECK_EQ(on.text, off.text);
}
