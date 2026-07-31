// The model seam (S2.1.1, S5.12): the loop is driven and asserted with no model.
// ScriptedBackend records what the harness SENT -- the assertion that killed v1's
// "0 of 17 tool calls completed" misattribution.

#include "src/model/backend.hpp"
#include "src/model/kv_cache.hpp"
#include "src/model/sampler.hpp"

#include "tests/check.hpp"

using namespace lmp::model;

namespace {

class Collect final : public TokenSink {
  public:
    std::vector<TokenId> seen;
    TokenId stop_on = kInvalidToken;
    bool on_token(TokenId id) override {
        seen.push_back(id);
        return id != stop_on;
    }
};

} // namespace

TEST(scripted_backend_records_what_was_sent) {
    ScriptedBackend b;
    b.enqueue_response({1, 2, 3});
    Collect sink;
    sink.stop_on = 3;
    CancelToken cancel;

    InferenceTask task;
    task.prompt = {10, 11, 12};
    const GenResult r = b.generate(task, sink, cancel);

    CHECK(r.status == GenStatus::Complete);
    CHECK_EQ(r.tokens_generated, 3);
    REQUIRE(b.received().size() == 1);
    // The harness's exact prompt is inspectable -- "did the model receive this?" is an
    // assertion, not a guess.
    CHECK(b.received()[0].prompt == std::vector<TokenId>({10, 11, 12}));
}

TEST(script_exhaustion_is_length_capped_not_complete) {
    ScriptedBackend b;
    b.enqueue_response({1, 2});
    Collect sink; // never asks to stop
    CancelToken cancel;
    const GenResult r = b.generate({}, sink, cancel);
    // One turn, one outcome (S9.1): running out of tokens is NOT completion.
    CHECK(r.status == GenStatus::LengthCapped);
}

TEST(unscripted_call_is_a_typed_error) {
    ScriptedBackend b;
    Collect sink;
    CancelToken cancel;
    const GenResult r = b.generate({}, sink, cancel);
    CHECK(r.status == GenStatus::BackendError);
    CHECK(!r.error.empty());
}

TEST(cancel_interrupts_mid_stream) {
    class CancelAfter final : public TokenSink {
      public:
        explicit CancelAfter(CancelToken& c) : cancel_(c) {}
        int count = 0;
        bool on_token(TokenId) override {
            if (++count == 2) {
                cancel_.cancel();
            }
            return true;
        }

      private:
        CancelToken& cancel_;
    };
    ScriptedBackend b;
    b.enqueue_response({1, 2, 3, 4, 5});
    CancelToken cancel;
    CancelAfter sink(cancel);
    const GenResult r = b.generate({}, sink, cancel);
    CHECK(r.status == GenStatus::Cancelled);
    CHECK_EQ(sink.count, 2);
}

TEST(replay_backend_plays_turns_in_order) {
    ReplayBackend b({{7, 8}, {9}});
    Collect s1;
    Collect s2;
    CancelToken cancel;
    InferenceTask t;
    t.max_new_tokens = 2; // exactly the turn length => LengthCapped, deterministic
    (void)b.generate(t, s1, cancel);
    CHECK(s1.seen == std::vector<TokenId>({7, 8}));
    t.max_new_tokens = 1;
    (void)b.generate(t, s2, cancel);
    CHECK(s2.seen == std::vector<TokenId>({9}));
}

// --- KV ledger (S5.10) ------------------------------------------------------

TEST(kv_reuse_is_verified_id_by_id) {
    KvCacheLedger led;
    led.append({1, 2, 3, 4});

    // Extension of the cached prefix: reuse all four.
    ReuseDecision d = led.plan_reuse({1, 2, 3, 4, 5, 6});
    CHECK_EQ(d.reusable, std::size_t{4});
    CHECK(!d.divergent);

    // Mid-prompt mismatch: reuse stops at the divergence AND the caller must reset.
    d = led.plan_reuse({1, 2, 99, 4});
    CHECK_EQ(d.reusable, std::size_t{2});
    CHECK(d.divergent);

    // Cache LONGER than the prompt: those extra entries are stale context.
    d = led.plan_reuse({1, 2});
    CHECK_EQ(d.reusable, std::size_t{2});
    CHECK(d.divergent);
}

TEST(kv_hash_tracks_content_not_length) {
    KvCacheLedger a;
    KvCacheLedger b;
    a.append({1, 2, 3});
    b.append({1, 2, 4});
    CHECK(a.content_hash() != b.content_hash());
    CHECK_EQ(a.content_hash(), hash_ids({1, 2, 3}));
    a.clear();
    CHECK_EQ(a.size(), std::size_t{0});
    CHECK_EQ(a.content_hash(), hash_ids({}));
}

// --- sampler (S5.9, mask-first) ---------------------------------------------

TEST(sampler_is_deterministic_under_a_seed) {
    SamplingParams p;
    p.seed = 42;
    std::vector<float> logits = {0.1F, 2.0F, 0.5F, 1.9F};
    Sampler s1(p);
    Sampler s2(p);
    std::vector<float> l1 = logits;
    std::vector<float> l2 = logits;
    CHECK_EQ(s1.sample(l1, nullptr, {}).id, s2.sample(l2, nullptr, {}).id);
}

TEST(mask_is_applied_before_shaping) {
    SamplingParams p;
    p.temperature = 0.0001F; // near-greedy
    p.top_k = 1;
    Sampler s(p);
    std::vector<float> logits = {0.0F, 10.0F, 0.0F, 5.0F};
    // The argmax id 1 is masked out; top-k=1 must then keep id 3, not produce nothing.
    // Masking after top-k would have kept only id 1 and left zero legal candidates.
    const SampleResult r =
        s.sample(logits, [](TokenId id) { return id != 1; }, {});
    CHECK(!r.no_legal_token);
    CHECK_EQ(r.id, TokenId{3});
}

TEST(all_masked_is_a_typed_failure) {
    Sampler s(SamplingParams{});
    std::vector<float> logits = {1.0F, 2.0F};
    const SampleResult r = s.sample(logits, [](TokenId) { return false; }, {});
    CHECK(r.no_legal_token);
    CHECK_EQ(r.id, kInvalidToken);
}

TEST(repetition_penalty_discourages_recent_ids) {
    SamplingParams p;
    p.temperature = 0.0001F;
    p.top_k = 1;
    p.repetition_penalty = 100.0F; // absurd on purpose: the effect must be decisive
    Sampler s(p);
    std::vector<float> logits = {1.0F, 1.01F}; // id 1 barely wins
    const SampleResult r = s.sample(logits, nullptr, {1});
    CHECK_EQ(r.id, TokenId{0});
}
