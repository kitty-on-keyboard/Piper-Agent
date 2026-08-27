// The cross-turn KV checkpoint, on the real model (G3).
//
// tests/model/test_kv_reuse.cpp asserts the ALGEBRA with no GPU. This asserts the two
// things the algebra cannot: that the mechanism actually fires against real caches, and
// that firing it does not change a single token of output.
//
// The second is the one that matters. A stale-cache bug does not crash and does not
// produce garbage -- it decodes fluent, plausible, WRONG text. Byte-identity between a
// run with reuse and a run without it is the only assertion that catches it, which is why
// docs/PHASES.md names it as the falsifier for the whole item.
//
// Labelled realmodel: excluded from the gate, never run in parallel (S11.6).

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "src/model/backend.hpp"
#include "src/model/chat_template.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

std::string qwen_dir() {
    const char* v = std::getenv("LMP_QWEN_DIR");
    return v != nullptr
               ? std::string(v)
               : std::string("");
}

// Collects every sampled id, so two runs can be compared token for token.
class Collector final : public TokenSink {
  public:
    explicit Collector(int cap) : cap_(cap) {}
    bool on_token(TokenId id) override {
        ids.push_back(id);
        return static_cast<int>(ids.size()) < cap_;
    }
    std::vector<TokenId> ids;

  private:
    int cap_;
};

// A system block the size the real one is.
//
// The first version of this test used a one-line system message and measured 17 reused
// tokens -- true, firing, and a useless demonstration, because in the real agent the
// stable prefix is the persona plus the tools JSON plus the mission plus every recent
// turn, and that is what the checkpoint is for. A test whose fixture is nothing like the
// thing it certifies certifies nothing.
std::string big_system() {
    std::string s =
        "You are Piper, a master software engineer.\n\n"
        "- You do not cut corners. If a job needs six steps you take six steps.\n"
        "- You favour correctness over speed.\n"
        "- You reach for the most specific tool for the job.\n\n# Tools\n\n";
    // Stands in for the <tools> schema block, which is run-constant and sits in the KV
    // prefix (S6.4) -- the single biggest stable region in a real prompt.
    for (int i = 0; i < 40; ++i) {
        s += "{\"type\": \"function\", \"function\": {\"name\": \"tool_" +
             std::to_string(i) +
             "\", \"description\": \"Does the thing numbered " + std::to_string(i) +
             ", carefully, and reports what it observed.\", \"parameters\": "
             "{\"type\": \"object\", \"properties\": {\"path\": {\"type\": "
             "\"string\"}}, \"required\": [\"path\"]}}}\n";
    }
    return s;
}

// A two-message conversation whose SECOND turn extends the first exactly the way a real
// turn does: a new record appended before the live-state block, so the prompts share a
// long prefix and then diverge.
std::vector<Message> turn_one() {
    return {{Role::System, big_system()}, {Role::User, "What is two plus two?"}};
}

std::vector<Message> turn_two() {
    std::vector<Message> m = turn_one();
    m.push_back({Role::Assistant, "Two plus two equals four."});
    m.push_back({Role::User, "What is three plus three?"});
    return m;
}

} // namespace

TEST(the_second_turn_reuses_the_checkpointed_prefix) {
    QwenTokenizer tok;
    const LoadStatus ts = tok.load(qwen_dir() + "/tokenizer.json", Family::Qwen3);
    REQUIRE(ts.ok);

    lmp::platform::SystemClock clock;
    MlxBackend backend(clock);
    MlxBackendConfig cfg;
    cfg.model_dir = qwen_dir();
    const LoadStatus ls = backend.load(cfg);
    REQUIRE(ls.ok);

    const ChatTemplate tmpl(tok);
    const CancelToken cancel;

    const auto run = [&](const std::vector<Message>& msgs, bool checkpoint,
                         std::vector<TokenId>& out) {
        std::vector<std::size_t> offsets;
        InferenceTask task;
        task.prompt = tmpl.render_with_offsets(msgs, "", offsets);
        // The stable boundary: everything except the final (live-state-shaped) message.
        task.checkpoint_at = checkpoint ? offsets[msgs.size() - 1] : 0;
        task.max_new_tokens = 24;
        task.sampling.temperature = 0.0; // greedy, so two runs are comparable at all
        task.sampling.seed = 1;
        Collector sink(24);
        const GenResult r = backend.generate(task, sink, cancel);
        out = sink.ids;
        return r;
    };

    std::vector<TokenId> first;
    const GenResult r1 = run(turn_one(), true, first);
    REQUIRE(r1.error.empty());
    // Nothing was cached before this, so nothing could be reused.
    CHECK_EQ(r1.prefill_reused_tokens, std::size_t{0});

    std::vector<TokenId> second;
    const GenResult r2 = run(turn_two(), true, second);
    REQUIRE(r2.error.empty());

    // THE NUMBER THE ITEM EXISTS FOR. Before the checkpoint, turn two diverged mid-ledger
    // and re-prefilled the whole context; a 0 here means the mechanism never fired and the
    // complexity is not paying for itself.
    CHECK(r2.prefill_reused_tokens > 0);
    // Printed, not asserted: a TTFT threshold would be a machine-specific number pinned
    // in a test. The eval suite is where that gets tracked; here the point is that the
    // number is VISIBLE, so "did this buy anything" has an answer.
    std::printf("  [kv-reuse] turn2 reused %zu prompt tokens; ttft %.0f ms vs %.0f ms\n",
                r2.prefill_reused_tokens, r2.ttft_ms, r1.ttft_ms);
}

// THE FALSIFIER: a restored cache must equal a re-prefilled one.
//
// The first version of this compared reuse against a run with `checkpoint_at = 0` and
// FAILED -- at token 0, deterministically. It was not a stale cache. `checkpoint_at`
// forces a chunk edge, so the two runs were prefilling in different SEGMENTS, and this
// bf16 checkpoint is segmentation-sensitive: measured here, prefilling in 512-token
// chunks and in 2048-token chunks produce different first tokens with no reuse in the
// picture at all. The same property shows up from the speculative-decode side
// ("~4% TV from sequential decode ... a property to state rather than discover").
//
// So byte-identity across DIFFERENT segmentation is not achievable and never was. The
// assertion that isolates the thing this test exists to catch holds segmentation fixed --
// both runs pass the same `checkpoint_at`, so both break their prefill at the same
// boundary -- and varies only whether turn two RESTORED that boundary or re-prefilled it.
// Any difference then is the restore failing to reproduce the cache, which is exactly the
// silent-stale-context failure S5.10 exists to prevent.
TEST(a_restored_cache_equals_a_reprefilled_one) {
    QwenTokenizer tok;
    REQUIRE(tok.load(qwen_dir() + "/tokenizer.json", Family::Qwen3).ok);
    lmp::platform::SystemClock clock;
    MlxBackend backend(clock);
    MlxBackendConfig cfg;
    cfg.model_dir = qwen_dir();
    REQUIRE(backend.load(cfg).ok);
    const ChatTemplate tmpl(tok);
    const CancelToken cancel;

    // `restore` false drops the cache between the turns, so turn two must rebuild the
    // prefix it would otherwise have rolled back to -- same tokens, same chunk edges.
    const auto two_turns = [&](bool restore) {
        std::vector<TokenId> out;
        bool first = true;
        for (const std::vector<Message>& msgs : {turn_one(), turn_two()}) {
            if (!first && !restore) {
                backend.reset_cache();
            }
            first = false;
            std::vector<std::size_t> offsets;
            InferenceTask task;
            task.prompt = tmpl.render_with_offsets(msgs, "", offsets);
            task.checkpoint_at = offsets[1]; // the same boundary in both runs
            task.max_new_tokens = 24;
            task.sampling.temperature = 0.0;
            task.sampling.seed = 1;
            Collector sink(24);
            (void)backend.generate(task, sink, cancel);
            out = sink.ids;
        }
        return out;
    };

    backend.reset_cache();
    const std::vector<TokenId> restored = two_turns(true);
    backend.reset_cache();
    const std::vector<TokenId> rebuilt = two_turns(false);

    REQUIRE(!restored.empty());
    CHECK_EQ(restored.size(), rebuilt.size());
    for (std::size_t i = 0; i < restored.size() && i < rebuilt.size(); ++i) {
        REQUIRE(restored[i] == rebuilt[i]);
    }
}

// The segmentation sensitivity itself, pinned as a PROPERTY of this checkpoint rather
// than left as folklore -- it is the reason the assertion above is shaped the way it is,
// and the next person to write "reuse must be byte-identical to a plain run" should find
// this instead of rediscovering it against the model for an hour.
TEST(prefill_segmentation_changes_the_sampled_tokens) {
    QwenTokenizer tok;
    REQUIRE(tok.load(qwen_dir() + "/tokenizer.json", Family::Qwen3).ok);
    lmp::platform::SystemClock clock;
    MlxBackend backend(clock);
    MlxBackendConfig cfg;
    cfg.model_dir = qwen_dir();
    REQUIRE(backend.load(cfg).ok);
    const ChatTemplate tmpl(tok);
    const CancelToken cancel;

    const auto once = [&](std::size_t boundary) {
        backend.reset_cache();
        std::vector<std::size_t> offsets;
        InferenceTask task;
        task.prompt = tmpl.render_with_offsets(turn_two(), "", offsets);
        task.checkpoint_at = boundary; // forces a chunk edge here, nothing else
        task.max_new_tokens = 8;
        task.sampling.temperature = 0.0;
        task.sampling.seed = 1;
        Collector sink(8);
        (void)backend.generate(task, sink, cancel);
        return sink.ids;
    };

    std::vector<std::size_t> offsets;
    (void)tmpl.render_with_offsets(turn_two(), "", offsets);
    const std::vector<TokenId> a = once(0);          // one un-split prefill
    const std::vector<TokenId> b = once(offsets[1]); // split at the system boundary
    REQUIRE(!a.empty());
    REQUIRE(!b.empty());
    // No assertion that they DIFFER -- that would pin noise. The claim is only that the
    // harness may not assume they agree, and this case exists so the reason is on record.
    std::printf("  [segmentation] unsplit vs split-at-%zu: %s\n", offsets[1],
                a == b ? "same" : "DIFFERENT (expected on this bf16 checkpoint)");
}
