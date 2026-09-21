// Pulse T1 — pure mask / logits / policy unit tests (gate, no GPU).
//
// Kill bars for shape: mask permits only options; escape fails; flag-off == baseline.

#include <cmath>
#include <string>
#include <vector>

#include "src/loop/pulse.hpp"
#include "tests/check.hpp"

using namespace lmp::loop;
using lmp::model::TokenId;
using lmp::model::TokenMask;

namespace {

std::vector<float> logits_favoring(std::size_t vocab, TokenId winner, float win = 5.0F,
                                   float other = 0.0F) {
    std::vector<float> logits(vocab, other);
    if (winner >= 0 && static_cast<std::size_t>(winner) < vocab) {
        logits[static_cast<std::size_t>(winner)] = win;
    }
    return logits;
}

} // namespace

TEST(pulse_option_mask_permits_only_options) {
    const std::vector<TokenId> opts = {3, 7, 11, 19};
    const TokenMask m = pulse_option_mask(32, opts);
    CHECK_EQ(m.count(), std::size_t{4});
    for (TokenId id : opts) {
        CHECK(m.allows(id));
        CHECK(!pulse_mask_escape(m, id));
    }
    CHECK(pulse_mask_escape(m, 0));
    CHECK(pulse_mask_escape(m, 4));
    CHECK(pulse_mask_escape(m, 31));
    CHECK(!m.allows(1));
    CHECK(!m.allows(8));
}

TEST(pulse_mask_escape_fails_for_free_text_ids) {
    const std::vector<TokenId> opts = {1, 2, 3, 4};
    const TokenMask m = pulse_option_mask(16, opts);
    // Any id outside the closed set is an escape (would be free text under an open mask).
    for (TokenId id = 0; id < 16; ++id) {
        const bool opt = id >= 1 && id <= 4;
        CHECK_EQ(pulse_mask_escape(m, id), !opt);
    }
}

TEST(pulse_decode_from_logits_argmax_and_softmax) {
    const std::vector<TokenId> opts = {2, 5, 8, 9}; // force_tool, nudge, stall, compact
    auto logits = logits_favoring(16, /*winner=*/8, /*win=*/4.0F, /*other=*/0.0F);
    const PulseMicroResult r = pulse_decode_from_logits(logits, opts, /*latency_ms=*/12.5);
    CHECK(r.ok);
    CHECK(r.choice == PulseChoice::Stall);
    CHECK_EQ(r.choice_name, std::string("stall"));
    CHECK(r.p_vec.size() == 4);
    CHECK(r.p > 0.5F);
    CHECK(std::fabs(r.p - r.p_vec[2]) < 1e-5F);
    float sum = 0.0F;
    for (float p : r.p_vec) {
        sum += p;
    }
    CHECK(std::fabs(sum - 1.0F) < 1e-5F);
    CHECK(std::fabs(r.latency_ms - 12.5) < 1e-9);
}

TEST(pulse_decode_rejects_empty_options) {
    std::vector<float> logits(8, 1.0F);
    const PulseMicroResult r = pulse_decode_from_logits(logits, {}, 0.0);
    CHECK(!r.ok);
    CHECK(!r.error.empty());
}

TEST(pulse_policy_falls_back_below_p_min) {
    PulseMicroResult r;
    r.ok = true;
    r.choice = PulseChoice::ForceTool;
    r.choice_name = "force_tool";
    r.p = 0.40F;
    r.p_vec = {0.40F, 0.30F, 0.20F, 0.10F};
    CHECK(apply_pulse_policy(r, 0.55F) == PulsePolicy::Fallback);
    r.p = 0.60F;
    CHECK(apply_pulse_policy(r, 0.55F) == PulsePolicy::ForceTool);
}

TEST(pulse_policy_maps_each_choice) {
    PulseMicroResult r;
    r.ok = true;
    r.p = 0.9F;
    r.choice = PulseChoice::Nudge;
    CHECK(apply_pulse_policy(r) == PulsePolicy::Nudge);
    r.choice = PulseChoice::Stall;
    CHECK(apply_pulse_policy(r) == PulsePolicy::Stall);
    r.choice = PulseChoice::Compact;
    CHECK(apply_pulse_policy(r) == PulsePolicy::Compact);
    r.ok = false;
    CHECK(apply_pulse_policy(r) == PulsePolicy::Fallback);
}

TEST(pulse_enum_mask_source_is_block_stable) {
    const TokenMask bits = pulse_option_mask(64, {10, 20, 30, 40});
    PulseEnumMask src(bits);
    CHECK(src.mask_is_block_stable());
    CHECK(src.mask().allows(10));
    CHECK(pulse_mask_escape(src.mask(), 11));
}

TEST(pulse_t1_forced_prefix_lists_all_options) {
    const std::string prefix = pulse_t1_forced_prefix();
    CHECK(prefix.find("force_tool") != std::string::npos);
    CHECK(prefix.find("nudge") != std::string::npos);
    CHECK(prefix.find("stall") != std::string::npos);
    CHECK(prefix.find("compact") != std::string::npos);
}
