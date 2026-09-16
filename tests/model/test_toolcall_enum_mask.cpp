// Unit grammar: ToolCallGuard enum_values masking (Phase C).
//
// When enum_values is non-empty, invalid enum tokens must never appear in
// allowed_bytes(); valid literals must complete; golden paths must not hit an
// empty mask (kill criterion).

#include <string>
#include <vector>

#include "parsephony/parsephony.hpp"
#include "parsephony/toolcall.hpp"
#include "tests/check.hpp"

namespace {

parsephony::ToolSpec paint() {
    parsephony::ToolSpec s;
    s.name = "paint";
    parsephony::ParamSpec p;
    p.name = "color";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    p.enum_values = {"red", "green", "blue"};
    s.params.push_back(p);
    return s;
}

parsephony::ToolSpec shared_prefix() {
    // "red" is a shared prefix of "red" and "reddish" — mask must offer both
    // '\n' (exact) and 'd' (longer), and must not empty-mask.
    parsephony::ToolSpec s;
    s.name = "t";
    parsephony::ParamSpec p;
    p.name = "v";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    p.enum_values = {"red", "reddish", "green"};
    s.params.push_back(p);
    return s;
}

// Feed up to (not including) the color value body; leave guard in ValueText.
bool enter_color_value(parsephony::ToolCallGuard& g) {
    const std::string head =
        "<tool_call>\n<function=paint>\n<parameter=color>\n";
    return g.feed(head) == parsephony::Error::Ok && !g.complete();
}

} // namespace

TEST(enum_mask_rejects_first_byte_outside_enums) {
    const std::vector<parsephony::ToolSpec> specs{paint()};
    parsephony::ToolCallGuard g(specs);
    REQUIRE(enter_color_value(g));
    const parsephony::ByteSet allow = g.allowed_bytes();
    CHECK(allow.contains('r'));
    CHECK(allow.contains('g'));
    CHECK(allow.contains('b'));
    CHECK(!allow.contains('y'));
    CHECK(!allow.contains('x'));
    CHECK(!allow.contains('\n'));   // empty value is not an exact enum
    CHECK(g.feed("y") != parsephony::Error::Ok);
}

TEST(enum_mask_offers_newline_only_on_exact_match) {
    const std::vector<parsephony::ToolSpec> specs{paint()};
    parsephony::ToolCallGuard g(specs);
    REQUIRE(enter_color_value(g));
    CHECK(g.feed("gre") == parsephony::Error::Ok);
    {
        const parsephony::ByteSet allow = g.allowed_bytes();
        CHECK(allow.contains('e'));
        CHECK(!allow.contains('\n'));
    }
    CHECK(g.feed("en") == parsephony::Error::Ok);
    {
        const parsephony::ByteSet allow = g.allowed_bytes();
        CHECK(allow.contains('\n'));
        CHECK(!allow.contains('x'));
    }
}

TEST(enum_mask_shared_prefix_keeps_both_continuations) {
    const std::vector<parsephony::ToolSpec> specs{shared_prefix()};
    parsephony::ToolCallGuard g(specs);
    REQUIRE(g.feed("<tool_call>\n<function=t>\n<parameter=v>\n") ==
            parsephony::Error::Ok);
    CHECK(g.feed("re") == parsephony::Error::Ok);
    const parsephony::ByteSet allow = g.allowed_bytes();
    CHECK(allow.contains('d'));    // → red / reddish
    CHECK(!allow.contains('\n'));  // "re" is not exact
    CHECK(g.feed("d") == parsephony::Error::Ok);
    const parsephony::ByteSet after = g.allowed_bytes();
    CHECK(after.contains('\n'));   // "red" exact
    CHECK(after.contains('d'));    // → reddish
}

TEST(enum_mask_valid_path_completes_and_extracts) {
    const std::vector<parsephony::ToolSpec> specs{paint()};
    parsephony::ToolCallGuard g(specs);
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\nblue\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(g.feed(raw) == parsephony::Error::Ok);
    CHECK(g.complete());
    CHECK_EQ(g.tool_name(), std::string("paint"));
    REQUIRE(g.params().size() == 1);
    CHECK_EQ(g.params()[0].value, std::string("blue"));
}

TEST(enum_mask_invalid_token_never_in_mask_during_value) {
    const std::vector<parsephony::ToolSpec> specs{paint()};
    parsephony::ToolCallGuard g(specs);
    REQUIRE(enter_color_value(g));
    // At every step along "green", 'y' must stay out of the mask.
    const std::string good = "green";
    for (size_t i = 0; i < good.size(); ++i) {
        CHECK(!g.allowed_bytes().contains('y'));
        CHECK(g.allowed_bytes().contains(static_cast<unsigned char>(good[i])));
        CHECK(g.feed(good.substr(i, 1)) == parsephony::Error::Ok);
    }
    CHECK(g.allowed_bytes().contains('\n'));
    CHECK(!g.allowed_bytes().contains('y'));
}

TEST(enum_mask_golden_path_no_empty_mask) {
    const std::vector<parsephony::ToolSpec> specs{paint()};
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\nred\n</parameter>\n"
        "</function>\n</tool_call>";
    parsephony::ToolCallGuard g(specs);
    for (size_t i = 0; i < raw.size(); ++i) {
        CHECK(!g.complete());
        CHECK(!g.allowed_bytes().empty());
        CHECK(g.feed(raw.substr(i, 1)) == parsephony::Error::Ok);
    }
    CHECK(g.complete());
    // Done: nothing further is legal (generation stops).
    CHECK(g.allowed_bytes().empty());
}

TEST(enum_mask_probe_mute_advances_prefix) {
    // TokenMask probes mute() and feed multi-byte tokens; enum progress must
    // still advance on the probe copy.
    const std::vector<parsephony::ToolSpec> specs{paint()};
    parsephony::ToolCallGuard base(specs);
    REQUIRE(enter_color_value(base));
    parsephony::ToolCallGuard probe(base);
    probe.mute();
    CHECK(probe.probe_byte('g') == parsephony::Error::Ok);
    CHECK(probe.probe_byte('r') == parsephony::Error::Ok);
    CHECK(probe.probe_byte('e') == parsephony::Error::Ok);
    CHECK(probe.probe_byte('e') == parsephony::Error::Ok);
    CHECK(probe.probe_byte('n') == parsephony::Error::Ok);
    CHECK(probe.allowed_bytes().contains('\n'));
    // Parent unchanged.
    CHECK(base.allowed_bytes().contains('g'));
    CHECK(!base.allowed_bytes().contains('\n'));
}
