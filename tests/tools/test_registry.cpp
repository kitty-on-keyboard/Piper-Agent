// The tool registry over a real temp workspace: containment, honesty of results, and
// the graft-backed replace tool's refusal semantics.

#include <unistd.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "src/platform/fs.hpp"
#include "src/tools/registry.hpp"

#include "tests/check.hpp"

using namespace lmp::tools;

namespace {

std::string temp_dir() {
    const char* base = std::getenv("TMPDIR");
    std::string tmpl = std::string(base ? base : "/tmp") + "/lmp_reg_XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    const char* made = ::mkdtemp(buf.data());
    return made ? std::string(made) : std::string();
}

Registry make_registry(const std::string& root) {
    WorkspaceContext ctx;
    ctx.root = root;
    ctx.max_read_bytes = 1U << 20;
    ctx.max_result_bytes = 8192;
    ctx.spool_dir = root + "/.spool";
    ctx.shell_wall_clock_seconds = 20;
    return Registry(std::move(ctx));
}

std::vector<ToolParamValue> args(std::initializer_list<std::pair<const char*, std::string>> kv) {
    std::vector<ToolParamValue> out;
    for (const auto& [k, v] : kv) {
        out.push_back({k, v});
    }
    return out;
}

} // namespace

TEST(the_registry_declares_the_spec_set_and_no_more) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    // S6.1: resist growth; every tool is permanent surface area. Growing this pin
    // requires editing this test, which is the review moment.
    CHECK_EQ(reg.decls().size(), std::size_t{11});
    CHECK(reg.find("read_file") != nullptr);
    CHECK(reg.find("shell") != nullptr);
    CHECK(reg.find("no_such_tool") == nullptr);
    // The guard specs mirror the declarations one-to-one -- the grammar constrains
    // exactly the advertised set (S6.4).
    CHECK_EQ(reg.guard_specs().size(), reg.decls().size());
}

TEST(paths_outside_the_root_are_refused_not_errored) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult r = reg.execute("read_file", args({{"path", "../../etc/passwd"}}), 1);
    // Refused (policy), not ToolError -- the file exists, we did not fail to read it,
    // we declined to. The distinction is S6.2's whole point.
    CHECK(r.status == Status::Refused);
    CHECK(r.error_class == ErrorClass::Policy);
}

TEST(write_read_replace_round_trip) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);

    ToolResult w = reg.execute(
        "write_file", args({{"path", "a.cpp"}, {"content", "int x = 1;\nint y = 2;\n"}}), 1);
    REQUIRE(w.ok());

    ToolResult rd = reg.execute("read_file", args({{"path", "a.cpp"}}), 1);
    REQUIRE(rd.ok());
    CHECK_EQ(rd.summary, std::string("int x = 1;\nint y = 2;\n"));

    // Whitespace-tolerant replace: the model wrote `int  x=1;` with different spacing.
    ToolResult rp = reg.execute("replace_in_file",
                                args({{"path", "a.cpp"},
                                      {"old_text", "int  x=1;"},
                                      {"new_text", "int x = 42;"}}),
                                1);
    CHECK(rp.ok());
    rd = reg.execute("read_file", args({{"path", "a.cpp"}}), 1);
    CHECK(rd.summary.find("42") != std::string::npos);
}

TEST(ambiguous_replace_refuses_and_leaves_the_file_alone) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const std::string original = "foo();\nbar();\nfoo();\n";
    (void)reg.execute("write_file", args({{"path", "b.cpp"}, {"content", original}}), 1);

    const ToolResult r = reg.execute(
        "replace_in_file",
        args({{"path", "b.cpp"}, {"old_text", "foo();"}, {"new_text", "baz();"}}), 1);
    CHECK(r.status == Status::ToolError);
    CHECK(r.error_class == ErrorClass::Conflict);

    const lmp::platform::FileContents f =
        lmp::platform::read_file_whole(root + "/b.cpp", 1U << 20);
    CHECK_EQ(f.bytes, original); // untouched -- graft's contract
}

TEST(read_slice_returns_exact_lines) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    (void)reg.execute("write_file",
                      args({{"path", "l.txt"}, {"content", "one\ntwo\nthree\nfour\n"}}), 1);
    const ToolResult r = reg.execute(
        "read_slice",
        args({{"path", "l.txt"}, {"start_line", "2"}, {"end_line", "3"}}), 1);
    REQUIRE(r.ok());
    CHECK_EQ(r.summary, std::string("two\nthree\n"));
}

TEST(shell_reports_exit_codes_and_compacts) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult ok = reg.execute("shell", args({{"command", "echo hello"}}), 1);
    CHECK(ok.ok());
    CHECK(ok.summary.find("hello") != std::string::npos);

    const ToolResult bad = reg.execute("shell", args({{"command", "exit 3"}}), 1);
    CHECK(bad.status == Status::ToolError);
    CHECK(bad.summary.find("[exit 3]") != std::string::npos);
    CHECK(bad.retryable);
}

TEST(shell_at_t0_is_refused_with_the_reason) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const ToolResult r = reg.execute("shell", args({{"command", "echo hi"}}), 0);
    CHECK(r.status == Status::Refused);
    CHECK(r.summary.find("T0") != std::string::npos);
}

TEST(tools_json_covers_every_declaration) {
    const std::string root = temp_dir();
    Registry reg = make_registry(root);
    const std::string json = reg.tools_json();
    for (const ToolDecl& d : reg.decls()) {
        if (json.find("\"name\": \"" + d.name + "\"") == std::string::npos) {
            lmp::test::record_failure(__FILE__, __LINE__,
                                      "tools_json is missing " + d.name);
        }
        ++lmp::test::reg().checks;
    }
}
