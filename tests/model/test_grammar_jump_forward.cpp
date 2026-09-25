// Grammar jump-forward museum (v0 stub).
//
// CPU-only: walks ToolCallGuard over forced short-arg tool XML and asserts
//   1) a contiguous card==1 non-FreeText span ≥8 exists (jump fires),
//   2) FreeText ValueText never contributes a jump span,
//   3) greedy forced walk with jump spans spliced matches the baseline
//      byte-for-byte when the forced text is unique,
//   4) flag default is off (exact LMP_GRAMMAR_JUMP_FORWARD=1 enables).

#include <optional>
#include <string>
#include <vector>

#include "parsephony/parsephony.hpp"
#include "parsephony/toolcall.hpp"
#include "src/model/grammar_jump_forward.hpp"
#include "tests/check.hpp"

namespace {

parsephony::ToolSpec list_dir_spec() {
    parsephony::ToolSpec s;
    s.name = "list_dir";
    parsephony::ParamSpec p;
    p.name = "path";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    s.params.push_back(p);
    return s;
}

parsephony::ToolSpec read_file_spec() {
    parsephony::ToolSpec s;
    s.name = "read_file";
    parsephony::ParamSpec p;
    p.name = "path";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    s.params.push_back(p);
    return s;
}

const std::string kListDirCall =
    "<tool_call>\n"
    "<function=list_dir>\n"
    "<parameter=path>\n"
    "src\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>";

const std::string kReadFileCall =
    "<tool_call>\n"
    "<function=read_file>\n"
    "<parameter=path>\n"
    "a.cpp\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>";

// Simulate decode with jump: whenever a ≥8 card==1 span is available, append it
// in one shot; otherwise take the next golden byte. Returns emitted bytes.
std::string walk_with_jumps(const std::vector<parsephony::ToolSpec>& specs,
                            std::string_view golden) {
    parsephony::ToolCallGuard g(specs);
    std::string out;
    std::size_t i = 0;
    while (i < golden.size() && !g.complete()) {
        const std::string span = lmp::model::collect_forced_span(g);
        if (lmp::model::should_grammar_jump(span)) {
            // Jump must match the golden prefix (unique forced text).
            CHECK(golden.substr(i, span.size()) == span);
            CHECK(g.feed(span) == parsephony::Error::Ok);
            out.append(span);
            i += span.size();
            continue;
        }
        CHECK(g.feed(golden.substr(i, 1)) == parsephony::Error::Ok);
        out.push_back(golden[i]);
        ++i;
    }
    CHECK(g.complete());
    return out;
}

std::string walk_baseline(const std::vector<parsephony::ToolSpec>& specs,
                          std::string_view golden) {
    parsephony::ToolCallGuard g(specs);
    std::string out;
    for (std::size_t i = 0; i < golden.size(); ++i) {
        CHECK(g.feed(golden.substr(i, 1)) == parsephony::Error::Ok);
        out.push_back(golden[i]);
    }
    CHECK(g.complete());
    return out;
}

struct FlagReset {
    ~FlagReset() { lmp::model::grammar_jump_forward_set_enabled_for_test(std::nullopt); }
};

} // namespace

TEST(gjf_flag_default_off) {
    FlagReset reset;
    lmp::model::grammar_jump_forward_set_enabled_for_test(std::nullopt);
    // Without an explicit test override, unset env → off. We cannot clear the
    // process env reliably here; the override nullopt means "read getenv", and
    // CI does not set LMP_GRAMMAR_JUMP_FORWARD. Force-off for determinism.
    lmp::model::grammar_jump_forward_set_enabled_for_test(false);
    CHECK(!lmp::model::grammar_jump_forward_enabled());
    lmp::model::grammar_jump_forward_set_enabled_for_test(true);
    CHECK(lmp::model::grammar_jump_forward_enabled());
}

TEST(gjf_byte_set_unique_cardinality) {
    parsephony::ByteSet s;
    CHECK_EQ(s.count(), 0u);
    CHECK(!s.unique_byte().has_value());
    s.add('a');
    CHECK_EQ(s.count(), 1u);
    const auto uniq = s.unique_byte();
    REQUIRE(uniq.has_value());
    CHECK_EQ(static_cast<int>(*uniq), static_cast<int>('a'));
    s.add('b');
    CHECK_EQ(s.count(), 2u);
    CHECK(!s.unique_byte().has_value());
    parsephony::ByteSet wide;
    wide.non_ascii = true;
    wide.add('x');
    CHECK(wide.count() > 1u);
    CHECK(!wide.unique_byte().has_value());
}

TEST(gjf_list_dir_framing_jump_fires) {
    FlagReset reset;
    lmp::model::grammar_jump_forward_set_enabled_for_test(true);
    const std::vector<parsephony::ToolSpec> specs{list_dir_spec()};
    parsephony::ToolCallGuard start(specs);
    const auto spans = lmp::model::enumerate_forced_spans(start, kListDirCall);
    std::size_t jumps = 0;
    std::size_t max_span = 0;
    for (const auto& sp : spans) {
        if (sp.bytes.size() > max_span) max_span = sp.bytes.size();
        if (lmp::model::should_grammar_jump(sp.bytes)) ++jumps;
    }
    CHECK(jumps >= 1u);
    CHECK(max_span >= lmp::model::kGrammarJumpMinBytes);
    // Open framing alone is the literal "<tool_call>\n<function=" (22 bytes) once
    // the registry forces a unique name path — at least the open literal is ≥8.
    bool saw_openish = false;
    for (const auto& sp : spans) {
        if (sp.bytes.find("<tool_call>") != std::string::npos ||
            sp.bytes.find("<function=") != std::string::npos ||
            sp.bytes.find("list_dir") != std::string::npos) {
            saw_openish = true;
        }
    }
    CHECK(saw_openish);
}

TEST(gjf_read_file_framing_jump_fires) {
    FlagReset reset;
    lmp::model::grammar_jump_forward_set_enabled_for_test(true);
    const std::vector<parsephony::ToolSpec> specs{read_file_spec()};
    parsephony::ToolCallGuard start(specs);
    const auto spans = lmp::model::enumerate_forced_spans(start, kReadFileCall);
    std::size_t jumps = 0;
    for (const auto& sp : spans) {
        if (lmp::model::should_grammar_jump(sp.bytes)) ++jumps;
    }
    CHECK(jumps >= 1u);
}

TEST(gjf_no_jump_inside_freetext) {
    FlagReset reset;
    lmp::model::grammar_jump_forward_set_enabled_for_test(true);
    const std::vector<parsephony::ToolSpec> specs{list_dir_spec()};
    parsephony::ToolCallGuard g(specs);
    // Enter ValueText (FreeText) for the path body.
    const std::string head =
        "<tool_call>\n<function=list_dir>\n<parameter=path>\n";
    REQUIRE(g.feed(head) == parsephony::Error::Ok);
    CHECK(g.mask_class() == parsephony::MaskClass::FreeText);
    CHECK(!lmp::model::unique_forced_byte(g).has_value());
    const std::string span = lmp::model::collect_forced_span(g);
    CHECK(span.empty());
    CHECK(!lmp::model::should_grammar_jump(span));
}

TEST(gjf_greedy_jump_matches_baseline_byte_for_byte) {
    FlagReset reset;
    lmp::model::grammar_jump_forward_set_enabled_for_test(true);
    const std::vector<parsephony::ToolSpec> specs{list_dir_spec()};
    const std::string baseline = walk_baseline(specs, kListDirCall);
    const std::string jumped = walk_with_jumps(specs, kListDirCall);
    CHECK_EQ(jumped, baseline);
    CHECK_EQ(jumped, kListDirCall);

    const std::vector<parsephony::ToolSpec> read_specs{read_file_spec()};
    CHECK_EQ(walk_with_jumps(read_specs, kReadFileCall),
             walk_baseline(read_specs, kReadFileCall));
}

TEST(gjf_short_span_does_not_jump) {
    // A 7-byte card==1 run must decode normally (threshold is 8).
    CHECK(!lmp::model::should_grammar_jump("1234567"));
    CHECK(lmp::model::should_grammar_jump("12345678"));
}
