// Malformed-call museum stub (PR1 optional tease-out).
//
// Frozen bad <tool_call> strings exercised through ToolCallGuard only (CPU). Feeds
// Phase C corpus work later; this file locks the harness shape without tightening
// grammar behavior.

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

bool guard_accepts(const std::string& raw, const std::vector<parsephony::ToolSpec>& specs) {
    parsephony::ToolCallGuard g(specs);
    return g.feed(raw) == parsephony::Error::Ok && g.complete();
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
