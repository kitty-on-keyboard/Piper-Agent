// Tiny gate — Stage-0 + client JSON parse (gate label, no GPU).

#include <cmath>
#include <string>
#include <vector>

#include "src/loop/tiny_gate.hpp"
#include "tests/check.hpp"

using namespace lmp::loop;

TEST(tiny_gate_stage0_stall_on_consec_or_streak) {
    GateFeatures f;
    f.consec = 3;
    f.streak = 0;
    auto s = stage0_gate_policy(f);
    CHECK(s.has_value());
    CHECK(*s == GatePolicy::Stall);

    f.consec = 2;
    f.streak = 4;
    s = stage0_gate_policy(f);
    CHECK(s.has_value());
    CHECK(*s == GatePolicy::Stall);

    f.consec = 2;
    f.streak = 3;
    f.prompt_tok = 100;
    f.reread_max = 0;
    s = stage0_gate_policy(f);
    CHECK(!s.has_value());
}

TEST(tiny_gate_stage0_compact_on_prompt_or_reread) {
    GateFeatures f;
    f.consec = 1;
    f.streak = 1;
    f.prompt_tok = 16000;
    auto s = stage0_gate_policy(f);
    CHECK(s.has_value());
    CHECK(*s == GatePolicy::Compact);

    f.prompt_tok = 100;
    f.reread_max = 3;
    s = stage0_gate_policy(f);
    CHECK(s.has_value());
    CHECK(*s == GatePolicy::Compact);
}

TEST(tiny_gate_binary_policy_force_iff_p_force_ge_p_min) {
    GateMicroResult r;
    r.ok = true;
    r.choice = GateChoice::ForceTool;
    r.choice_name = "force_tool";
    r.p = 0.54F;
    r.p_force = 0.54F;
    r.p_vec = {0.54F, 0.46F};
    CHECK(apply_gate_policy(r, 0.55F) == GatePolicy::Nudge);
    r.p_force = 0.55F;
    r.p = 0.55F;
    CHECK(apply_gate_policy(r, 0.55F) == GatePolicy::ForceTool);
    r.ok = false;
    CHECK(apply_gate_policy(r, 0.55F) == GatePolicy::Fallback);
}

TEST(tiny_gate_binary_prefix_only_force_and_nudge) {
    const GateChoiceOrder order = {GateChoice::ForceTool, GateChoice::Nudge};
    GateFeatures f;
    f.consec = 1;
    f.why = "degenerate";
    const std::string prefix = gate_t1_forced_prefix(order, f);
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

TEST(tiny_gate_shuffle_is_binary_and_deterministic) {
    const GateChoiceOrder a = shuffle_gate_t1_choices(42);
    const GateChoiceOrder b = shuffle_gate_t1_choices(42);
    CHECK(a.size() == 2);
    for (std::size_t i = 0; i < kGateT1OptionCount; ++i) {
        CHECK(a[i] == b[i]);
    }
    bool saw_force_first = false;
    bool saw_nudge_first = false;
    for (std::uint64_t seed = 1; seed <= 64; ++seed) {
        const GateChoiceOrder o = shuffle_gate_t1_choices(seed);
        CHECK(o.size() == 2);
        CHECK((o[0] == GateChoice::ForceTool && o[1] == GateChoice::Nudge) ||
              (o[0] == GateChoice::Nudge && o[1] == GateChoice::ForceTool));
        if (o[0] == GateChoice::ForceTool) {
            saw_force_first = true;
        } else {
            saw_nudge_first = true;
        }
    }
    CHECK(saw_force_first);
    CHECK(saw_nudge_first);
}

TEST(tiny_gate_strip_hindsight_removes_next_and_outcome) {
    const std::string raw =
        "consec=3 streak=3 prompt_tok=900 why=degenerate next=force_tool outcome=stall "
        "gold=nudge\n"
        "keep_me=1";
    const std::string cleaned = strip_gate_hindsight(raw);
    CHECK(cleaned.find("next=") == std::string::npos);
    CHECK(cleaned.find("outcome=") == std::string::npos);
    CHECK(cleaned.find("gold=") == std::string::npos);
    CHECK(cleaned.find("consec=3") != std::string::npos);
    CHECK(cleaned.find("why=degenerate") != std::string::npos);
    CHECK(cleaned.find("keep_me=1") != std::string::npos);
}

TEST(tiny_gate_parse_response_json_ok) {
    const char* body =
        R"({"ok":true,"error":"","letter":"B","choice":"force_tool","p":0.61,)"
        R"("p_force":0.61,"p_vec":[0.39,0.61],"latency_ms":12.5,)"
        R"("model":"/models/Qwen3-0.6B-4bit","encoding":"letter",)"
        R"("order":"nudge,force_tool"})";
    const GateMicroResult r = parse_gate_response_json(body);
    CHECK(r.ok);
    CHECK(r.choice == GateChoice::ForceTool);
    CHECK_EQ(r.choice_name, std::string("force_tool"));
    CHECK(r.letter == 'B');
    CHECK(std::fabs(r.p - 0.61F) < 1e-5F);
    CHECK(std::fabs(r.p_force - 0.61F) < 1e-5F);
    CHECK(r.p_vec.size() == 2);
    CHECK_EQ(r.model, std::string("/models/Qwen3-0.6B-4bit"));
    CHECK_EQ(r.encoding, std::string("letter"));
    CHECK(std::fabs(r.latency_ms - 12.5) < 1e-6);
}

TEST(tiny_gate_parse_response_json_failure) {
    const GateMicroResult empty = parse_gate_response_json("");
    CHECK(!empty.ok);
    CHECK(!empty.error.empty());

    const GateMicroResult bad = parse_gate_response_json(R"({"ok":false,"error":"boom"})");
    CHECK(!bad.ok);
    CHECK_EQ(bad.error, std::string("boom"));

    const GateMicroResult unknown =
        parse_gate_response_json(R"({"ok":true,"choice":"stall","letter":"C"})");
    CHECK(!unknown.ok);
    CHECK(unknown.error.find("unknown choice") != std::string::npos);
}

TEST(tiny_gate_build_request_json_includes_prompt_and_letters) {
    const GateChoiceOrder order = {GateChoice::Nudge, GateChoice::ForceTool};
    const std::string req = build_gate_request_json("Pick A or B\n", order);
    CHECK(req.find("\"prompt\"") != std::string::npos);
    CHECK(req.find("Pick A or B") != std::string::npos);
    CHECK(req.find("force_tool") != std::string::npos);
    CHECK(req.find("nudge") != std::string::npos);
    CHECK(req.find("\"A\"") != std::string::npos);
    CHECK(req.find("\"B\"") != std::string::npos);
}

TEST(tiny_gate_format_features_is_structured_only) {
    GateFeatures f;
    f.consec = 1;
    f.streak = 2;
    f.prompt_tok = 50;
    f.reread_max = 3;
    f.think = 10;
    f.text = 20;
    f.tool_tok = 0;
    f.why = "loop_cut";
    const std::string block = format_gate_features(f);
    CHECK(block.find("consec=1") != std::string::npos);
    CHECK(block.find("streak=2") != std::string::npos);
    CHECK(block.find("why=loop_cut") != std::string::npos);
    CHECK(block.find("essay") == std::string::npos);
}

TEST(tiny_gate_default_p_min_is_055) {
    CHECK(std::fabs(kGateDefaultPMin - 0.55F) < 1e-6F);
}
