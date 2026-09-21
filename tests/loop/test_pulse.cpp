// Pulse KEEP-seed — Stage-0 + binary force/nudge (gate, no GPU).

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
    const std::vector<TokenId> opts = {3, 7};
    const TokenMask m = pulse_option_mask(32, opts);
    CHECK_EQ(m.count(), std::size_t{2});
    for (TokenId id : opts) {
        CHECK(m.allows(id));
        CHECK(!pulse_mask_escape(m, id));
    }
    CHECK(pulse_mask_escape(m, 0));
    CHECK(pulse_mask_escape(m, 4));
    CHECK(!m.allows(1));
}

TEST(pulse_mask_escape_fails_for_free_text_ids) {
    const std::vector<TokenId> opts = {1, 2}; // A/B letter ids
    const TokenMask m = pulse_option_mask(16, opts);
    for (TokenId id = 0; id < 16; ++id) {
        const bool opt = id == 1 || id == 2;
        CHECK_EQ(pulse_mask_escape(m, id), !opt);
    }
}

TEST(pulse_decode_from_logits_maps_letter_slot_to_enum) {
    // Presentation order: nudge, force_tool → letters A B
    const std::vector<TokenId> opts = {2, 5};
    const std::vector<PulseChoice> choices = {PulseChoice::Nudge, PulseChoice::ForceTool};
    auto logits = logits_favoring(16, /*winner=*/5, /*win=*/4.0F, /*other=*/0.0F);
    const PulseMicroResult r =
        pulse_decode_from_logits(logits, opts, choices, /*latency_ms=*/12.5);
    CHECK(r.ok);
    CHECK(r.choice == PulseChoice::ForceTool);
    CHECK_EQ(r.choice_name, std::string("force_tool"));
    CHECK(r.letter == 'B');
    CHECK_EQ(r.encoding, std::string("letter"));
    CHECK(r.p_vec.size() == 2);
    CHECK(r.p_force > 0.5F);
    CHECK(r.p > 0.5F);
    float sum = 0.0F;
    for (float p : r.p_vec) {
        sum += p;
    }
    CHECK(std::fabs(sum - 1.0F) < 1e-5F);
}

TEST(pulse_stage0_stall_on_consec_or_streak) {
    PulseFeatures f;
    f.consec = 3;
    f.streak = 0;
    auto s = stage0_pulse_policy(f);
    CHECK(s.has_value());
    CHECK(*s == PulsePolicy::Stall);

    f.consec = 2;
    f.streak = 4;
    s = stage0_pulse_policy(f);
    CHECK(s.has_value());
    CHECK(*s == PulsePolicy::Stall);

    f.consec = 2;
    f.streak = 3;
    f.prompt_tok = 100;
    f.reread_max = 0;
    s = stage0_pulse_policy(f);
    CHECK(!s.has_value());
}

TEST(pulse_stage0_compact_on_prompt_or_reread) {
    PulseFeatures f;
    f.consec = 1;
    f.streak = 1;
    f.prompt_tok = 16000;
    auto s = stage0_pulse_policy(f);
    CHECK(s.has_value());
    CHECK(*s == PulsePolicy::Compact);

    f.prompt_tok = 100;
    f.reread_max = 3;
    s = stage0_pulse_policy(f);
    CHECK(s.has_value());
    CHECK(*s == PulsePolicy::Compact);
}

TEST(pulse_binary_policy_force_iff_p_force_ge_p_min) {
    PulseMicroResult r;
    r.ok = true;
    r.choice = PulseChoice::ForceTool;
    r.choice_name = "force_tool";
    r.p = 0.54F;
    r.p_force = 0.54F;
    r.p_vec = {0.54F, 0.46F};
    CHECK(apply_pulse_policy(r, 0.55F) == PulsePolicy::Nudge);
    r.p_force = 0.55F;
    r.p = 0.55F;
    CHECK(apply_pulse_policy(r, 0.55F) == PulsePolicy::ForceTool);
    r.ok = false;
    CHECK(apply_pulse_policy(r, 0.55F) == PulsePolicy::Fallback);
}

TEST(pulse_binary_mask_only_force_and_nudge_letters) {
    const PulseChoiceOrder order = {PulseChoice::ForceTool, PulseChoice::Nudge};
    PulseFeatures f;
    f.consec = 1;
    f.why = "degenerate";
    const std::string prefix = pulse_t1_forced_prefix(order, f);
    CHECK(prefix.find("A = ") != std::string::npos);
    CHECK(prefix.find("B = ") != std::string::npos);
    CHECK(prefix.find("C = ") == std::string::npos);
    CHECK(prefix.find("D = ") == std::string::npos);
    CHECK(prefix.find("stall —") == std::string::npos);
    CHECK(prefix.find("compact —") == std::string::npos);
    CHECK(prefix.find("force_tool") != std::string::npos);
    CHECK(prefix.find("nudge") != std::string::npos);
    CHECK(prefix.find("Features:") != std::string::npos);
}

TEST(pulse_shuffle_is_binary_and_deterministic) {
    const PulseChoiceOrder a = shuffle_t1_choices(42);
    const PulseChoiceOrder b = shuffle_t1_choices(42);
    CHECK(a.size() == 2);
    for (std::size_t i = 0; i < kPulseT1OptionCount; ++i) {
        CHECK(a[i] == b[i]);
    }
    bool saw_force_first = false;
    bool saw_nudge_first = false;
    for (std::uint64_t seed = 1; seed <= 64; ++seed) {
        const PulseChoiceOrder o = shuffle_t1_choices(seed);
        CHECK(o.size() == 2);
        CHECK((o[0] == PulseChoice::ForceTool && o[1] == PulseChoice::Nudge) ||
              (o[0] == PulseChoice::Nudge && o[1] == PulseChoice::ForceTool));
        if (o[0] == PulseChoice::ForceTool) {
            saw_force_first = true;
        } else {
            saw_nudge_first = true;
        }
    }
    CHECK(saw_force_first);
    CHECK(saw_nudge_first);
}

TEST(pulse_strip_hindsight_removes_next_and_outcome) {
    const std::string raw =
        "consec=3 streak=3 prompt_tok=900 why=degenerate next=force_tool outcome=stall "
        "gold=nudge\n"
        "keep_me=1";
    const std::string cleaned = strip_pulse_hindsight(raw);
    CHECK(cleaned.find("next=") == std::string::npos);
    CHECK(cleaned.find("outcome=") == std::string::npos);
    CHECK(cleaned.find("gold=") == std::string::npos);
    CHECK(cleaned.find("consec=3") != std::string::npos);
    CHECK(cleaned.find("why=degenerate") != std::string::npos);
    CHECK(cleaned.find("keep_me=1") != std::string::npos);
}

TEST(pulse_default_p_min_is_per_model) {
    CHECK(std::fabs(pulse_default_p_min("Qwen3.8-27B-MLX-4bit") - kPulsePMin27B) < 1e-6F);
    CHECK(std::fabs(pulse_default_p_min("/models/qwen-27b") - kPulsePMin27B) < 1e-6F);
    CHECK(std::fabs(pulse_default_p_min("Qwen3.6-35B-A3B-MLX-4bit") - kPulsePMinA3B) < 1e-6F);
    CHECK(std::fabs(pulse_default_p_min("/ckpt/foo-a3b-bar") - kPulsePMinA3B) < 1e-6F);
    CHECK(pulse_model_is_a3b_moe("x-A3B-y"));
    CHECK(!pulse_model_is_a3b_moe("Qwen3.8-27B"));
    CHECK(std::fabs(kPulseDefaultPMin - 0.55F) < 1e-6F);
}

TEST(pulse_format_features_is_structured_only) {
    PulseFeatures f;
    f.consec = 1;
    f.streak = 2;
    f.prompt_tok = 50;
    f.reread_max = 3;
    f.think = 10;
    f.text = 20;
    f.tool_tok = 0;
    f.why = "loop_cut";
    const std::string block = format_pulse_features(f);
    CHECK(block.find("consec=1") != std::string::npos);
    CHECK(block.find("streak=2") != std::string::npos);
    CHECK(block.find("why=loop_cut") != std::string::npos);
    CHECK(block.find("essay") == std::string::npos);
}
