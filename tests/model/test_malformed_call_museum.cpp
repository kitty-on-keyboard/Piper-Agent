// Malformed-call museum (Phase C).
//
// Frozen raw <tool_call> strings exercised through ToolCallGuard only (CPU).
// This is the regression lock for grammar / ToolError escapes: bad enums,
// missing required params, unknown tools, truncated framing, and bare text
// must REJECT; golden good calls (including valid enums) must ACCEPT without
// empty-mask stuck states on the allowed_bytes path.

#include <string>
#include <vector>

#include "parsephony/parsephony.hpp"
#include "parsephony/toolcall.hpp"
#include "tests/check.hpp"

namespace {

parsephony::ToolSpec echo_spec() {
    parsephony::ToolSpec s;
    s.name = "echo";
    parsephony::ParamSpec p;
    p.name = "msg";
    p.type = parsephony::ParamType::Text;
    p.required = true;
    s.params.push_back(p);
    return s;
}

parsephony::ToolSpec paint_spec() {
    parsephony::ToolSpec s;
    s.name = "paint";
    parsephony::ParamSpec color;
    color.name = "color";
    color.type = parsephony::ParamType::Text;
    color.required = true;
    color.enum_values = {"red", "green", "blue"};
    s.params.push_back(color);
    parsephony::ParamSpec note;
    note.name = "note";
    note.type = parsephony::ParamType::Text;
    note.required = false;
    s.params.push_back(note);
    return s;
}

parsephony::ToolSpec mode_spec() {
    // Number-typed enum: enum_values hold the JSON spelling.
    parsephony::ToolSpec s;
    s.name = "set_mode";
    parsephony::ParamSpec p;
    p.name = "mode";
    p.type = parsephony::ParamType::Number;
    p.required = true;
    p.enum_values = {"1", "2", "3"};
    s.params.push_back(p);
    return s;
}

bool guard_accepts(const std::string& raw, const std::vector<parsephony::ToolSpec>& specs,
                   parsephony::Options o = {}) {
    parsephony::ToolCallGuard g(specs, o);
    return g.feed(raw) == parsephony::Error::Ok && g.complete();
}

parsephony::Error guard_feed(const std::string& raw,
                             const std::vector<parsephony::ToolSpec>& specs,
                             parsephony::Options o = {}) {
    parsephony::ToolCallGuard g(specs, o);
    const parsephony::Error e = g.feed(raw);
    if (e != parsephony::Error::Ok) return e;
    return g.complete() ? parsephony::Error::Ok : parsephony::Error::UnexpectedEnd;
}

// Walk every prefix of a golden call and assert allowed_bytes() is never empty
// until Done (kill criterion: enum masking must not stick generations).
bool golden_path_never_empty_mask(const std::string& raw,
                                  const std::vector<parsephony::ToolSpec>& specs) {
    parsephony::ToolCallGuard g(specs);
    for (size_t i = 0; i < raw.size(); ++i) {
        if (g.complete()) return false;
        if (g.allowed_bytes().empty()) return false;
        if (g.feed(raw.substr(i, 1)) != parsephony::Error::Ok) return false;
    }
    return g.complete();
}

} // namespace

TEST(museum_well_formed_echo_is_accepted) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    const std::string raw =
        "<tool_call>\n<function=echo>\n<parameter=msg>\nhi\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(guard_accepts(raw, specs));
}

TEST(museum_unknown_tool_is_rejected) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    const std::string raw =
        "<tool_call>\n<function=not_a_tool>\n<parameter=msg>\nhi\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(!guard_accepts(raw, specs));
}

TEST(museum_truncated_call_is_not_accepted) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    const std::string raw =
        "<tool_call>\n<function=echo>\n<parameter=msg>\nhi\n</parameter>";
    CHECK(!guard_accepts(raw, specs));
}

TEST(museum_missing_required_param_is_rejected) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    // echo requires msg; closing without it must fail.
    const std::string raw =
        "<tool_call>\n<function=echo>\n</function>\n</tool_call>";
    CHECK(!guard_accepts(raw, specs));
}

TEST(museum_wrong_param_name_is_rejected) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    const std::string raw =
        "<tool_call>\n<function=echo>\n<parameter=message>\nhi\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(!guard_accepts(raw, specs));
}

TEST(museum_bad_enum_value_is_rejected) {
    const std::vector<parsephony::ToolSpec> specs{paint_spec()};
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\nyellow\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(!guard_accepts(raw, specs));
    CHECK(guard_feed(raw, specs) != parsephony::Error::Ok);
}

TEST(museum_valid_enum_value_is_accepted) {
    const std::vector<parsephony::ToolSpec> specs{paint_spec()};
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\ngreen\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(guard_accepts(raw, specs));
}

TEST(museum_valid_enum_with_optional_note_is_accepted) {
    const std::vector<parsephony::ToolSpec> specs{paint_spec()};
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\nred\n</parameter>\n"
        "<parameter=note>\naccent wall\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(guard_accepts(raw, specs));
}

TEST(museum_number_enum_valid_is_accepted) {
    const std::vector<parsephony::ToolSpec> specs{mode_spec()};
    const std::string raw =
        "<tool_call>\n<function=set_mode>\n<parameter=mode>\n2\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(guard_accepts(raw, specs));
}

TEST(museum_number_enum_invalid_is_rejected) {
    const std::vector<parsephony::ToolSpec> specs{mode_spec()};
    const std::string raw =
        "<tool_call>\n<function=set_mode>\n<parameter=mode>\n9\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(!guard_accepts(raw, specs));
}

TEST(museum_bare_text_is_not_a_tool_call) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    CHECK(!guard_accepts("hello world", specs));
    CHECK(!guard_accepts("<tool_call>", specs));
    CHECK(!guard_accepts("not xml at all", specs));
}

TEST(museum_enum_kill_switch_restores_free_text) {
    const std::vector<parsephony::ToolSpec> specs{paint_spec()};
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\nyellow\n</parameter>\n"
        "</function>\n</tool_call>";
    parsephony::Options off;
    off.enforce_enum_values = false;
    CHECK(guard_accepts(raw, specs, off));
}

TEST(museum_golden_enum_path_never_empty_mask) {
    const std::vector<parsephony::ToolSpec> specs{paint_spec()};
    const std::string raw =
        "<tool_call>\n<function=paint>\n<parameter=color>\nblue\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(golden_path_never_empty_mask(raw, specs));
}

TEST(museum_golden_echo_path_never_empty_mask) {
    const std::vector<parsephony::ToolSpec> specs{echo_spec()};
    const std::string raw =
        "<tool_call>\n<function=echo>\n<parameter=msg>\nhi\n</parameter>\n"
        "</function>\n</tool_call>";
    CHECK(golden_path_never_empty_mask(raw, specs));
}
