// Pulse T1 Iteration A — letter codes, shuffle, hindsight strip (gate, no GPU).

#include <cmath>
#include <set>
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
    const std::vector<TokenId> opts = {1, 2, 3, 4}; // A/B/C/D letter ids
    const TokenMask m = pulse_option_mask(16, opts);
    for (TokenId id = 0; id < 16; ++id) {
        const bool opt = id >= 1 && id <= 4;
        CHECK_EQ(pulse_mask_escape(m, id), !opt);
    }
}

TEST(pulse_decode_from_logits_maps_letter_slot_to_enum) {
    // Presentation order: stall, force_tool, nudge, compact → letters A B C D
    const std::vector<TokenId> opts = {2, 5, 8, 9};
    const std::vector<PulseChoice> choices = {
        PulseChoice::Stall, PulseChoice::ForceTool, PulseChoice::Nudge, PulseChoice::Compact};
    // Winner = letter B (id 5) → force_tool
    auto logits = logits_favoring(16, /*winner=*/5, /*win=*/4.0F, /*other=*/0.0F);
    const PulseMicroResult r =
        pulse_decode_from_logits(logits, opts, choices, /*latency_ms=*/12.5);
    CHECK(r.ok);
    CHECK(r.choice == PulseChoice::ForceTool);
    CHECK_EQ(r.choice_name, std::string("force_tool"));
    CHECK(r.letter == 'B');
    CHECK_EQ(r.encoding, std::string("letter"));
    CHECK(r.p_vec.size() == 4);
    CHECK(r.p > 0.5F);
    CHECK(std::fabs(r.p - r.p_vec[1]) < 1e-5F);
    float sum = 0.0F;
    for (float p : r.p_vec) {
        sum += p;
    }
    CHECK(std::fabs(sum - 1.0F) < 1e-5F);
}

TEST(pulse_decode_rejects_empty_options) {
    std::vector<float> logits(8, 1.0F);
    const PulseMicroResult r = pulse_decode_from_logits(logits, {}, {}, 0.0);
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

TEST(pulse_t1_forced_prefix_is_neutral_with_letter_codes) {
    const PulseChoiceOrder order = {PulseChoice::Stall, PulseChoice::Nudge,
                                    PulseChoice::ForceTool, PulseChoice::Compact};
    PulseFeatures f;
    f.consec = 2;
    f.streak = 2;
    f.prompt_tok = 100;
    f.why = "degenerate";
    const std::string prefix = pulse_t1_forced_prefix(order, f);
    CHECK(prefix.find("text-instead-of-tool") == std::string::npos);
    CHECK(prefix.find("Pick exactly one harness next-step") != std::string::npos);
    CHECK(prefix.find("A = ") != std::string::npos);
    CHECK(prefix.find("B = ") != std::string::npos);
    CHECK(prefix.find("C = ") != std::string::npos);
    CHECK(prefix.find("D = ") != std::string::npos);
    CHECK(prefix.find("stall —") != std::string::npos);       // A
    CHECK(prefix.find("force_tool —") != std::string::npos); // C
    CHECK(prefix.find("Features:") != std::string::npos);
    CHECK(prefix.find("consec=2") != std::string::npos);
    CHECK(prefix.find("why=degenerate") != std::string::npos);
}

TEST(pulse_shuffle_moves_force_off_first_slot_for_some_seeds) {
    bool force_was_first = false;
    bool force_was_not_first = false;
    std::set<std::string> signatures;
    for (std::uint64_t seed = 1; seed <= 64; ++seed) {
        const PulseChoiceOrder o = shuffle_t1_choices(seed);
        std::string sig;
        for (PulseChoice c : o) {
            sig += pulse_choice_name(c);
            sig += ',';
        }
        signatures.insert(sig);
        if (o[0] == PulseChoice::ForceTool) {
            force_was_first = true;
        } else {
            force_was_not_first = true;
        }
    }
    CHECK(force_was_not_first); // required: force not always first
    CHECK(force_was_first);     // and sometimes is (shuffle is a perm, not a ban)
    CHECK(signatures.size() >= 4); // multiple distinct orders across seeds
}

TEST(pulse_shuffle_is_deterministic) {
    const PulseChoiceOrder a = shuffle_t1_choices(42);
    const PulseChoiceOrder b = shuffle_t1_choices(42);
    for (std::size_t i = 0; i < kPulseT1OptionCount; ++i) {
        CHECK(a[i] == b[i]);
    }
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
