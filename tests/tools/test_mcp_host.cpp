// McpHost (M1): MCP servers as ordinary tools in the Registry.
//
// Two halves, and the split is deliberate. tool_spec_from_schema is where UNTRUSTED input
// meets the grammar that constrains generation, so it is tested exhaustively and with no
// server at all. The rest drives the real mcp_demo_server binary in a real subprocess,
// because "the registry is populated from a live server" is not a claim an in-process
// fake can make.

#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "src/tools/mcp_host.hpp"
#include "src/tools/registry.hpp"

#include <filesystem>
#include <unistd.h>

#include "tests/check.hpp"

using namespace lmp::tools;

namespace {

// Set by CMake to $<TARGET_FILE:mcp_demo_server>.
const char* demo_server_path() { return LMP_MCP_DEMO_SERVER; }

McpServerConfig demo(const std::string& name, bool trusted) {
    McpServerConfig c;
    c.name = name;
    c.command = demo_server_path();
    c.trusted = trusted;
    return c;
}

WorkspaceContext workspace() {
    WorkspaceContext ctx;
    ctx.root = "/tmp";
    ctx.max_read_bytes = 4096;
    ctx.max_model_read_bytes = 4096;
    ctx.max_result_bytes = 4096;
    ctx.shell_wall_clock_seconds = 5;
    // A real directory: an image block can only be shown to the model if there is
    // somewhere to put the bytes, and an empty spool_dir correctly degrades to a note
    // instead. Per-process so two suites do not collide.
    ctx.spool_dir = "/tmp/lmp_mcp_spool_" + std::to_string(::getpid());
    return ctx;
}

nlohmann::json obj_schema(nlohmann::json properties, nlohmann::json required) {
    return nlohmann::json{{"type", "object"},
                          {"properties", std::move(properties)},
                          {"required", std::move(required)}};
}

const ToolDecl* find(const Registry& r, const std::string& name) { return r.find(name); }

} // namespace

// --- the pure half: schema -> generation constraint ------------------------

TEST(a_schema_becomes_the_spec_that_constrains_generation) {
    parsephony::ToolSpec spec;
    std::string why;
    const nlohmann::json schema = obj_schema(
        {{"text", {{"type", "string"}}},
         {"count", {{"type", "integer"}}},
         {"ratio", {{"type", "number"}}},
         {"flag", {{"type", "boolean"}}},
         {"body", {{"type", "object"}}},
         {"items", {{"type", "array"}}}},
        nlohmann::json::array({"text", "count"}));

    REQUIRE(tool_spec_from_schema("mcp__s__t", schema, spec, why));
    CHECK_EQ(spec.name, std::string("mcp__s__t"));
    REQUIRE(spec.params.size() == 6);

    // Types map to what the guard will actually enforce, and `required` is carried --
    // that is the half that makes a missing parameter unrepresentable rather than an
    // error discovered after the model has already emitted the call.
    for (const parsephony::ParamSpec& p : spec.params) {
        if (p.name == "text") {
            CHECK(p.type == parsephony::ParamType::Text);
            CHECK(p.required);
        } else if (p.name == "count") {
            CHECK(p.type == parsephony::ParamType::Number);
            CHECK(p.required);
        } else if (p.name == "ratio") {
            CHECK(p.type == parsephony::ParamType::Number);
            CHECK(!p.required);
        } else if (p.name == "flag") {
            CHECK(p.type == parsephony::ParamType::Boolean);
        } else if (p.name == "body") {
            CHECK(p.type == parsephony::ParamType::Object);
        } else if (p.name == "items") {
            CHECK(p.type == parsephony::ParamType::Array);
        }
    }
}

TEST(a_tool_taking_no_arguments_is_registrable) {
    // always_fails has no `properties` at all. An empty parameter list is a legal tool,
    // not a malformed one -- rejecting it would drop a whole class of real MCP tools.
    parsephony::ToolSpec spec;
    std::string why;
    REQUIRE(tool_spec_from_schema("mcp__s__t", nlohmann::json{{"type", "object"}}, spec, why));
    CHECK(spec.params.empty());
}

TEST(names_carrying_the_formats_own_delimiters_are_rejected) {
    // parsephony's tool-call syntax is delimited by '>', '=' and newline. A name carrying
    // one would not merely fail: it would corrupt the grammar for every OTHER tool sharing
    // the guard. A hostile server gets that tool dropped, not the run.
    parsephony::ToolSpec spec;
    std::string why;
    CHECK(!tool_spec_from_schema("mcp__s__a=b", nlohmann::json{{"type", "object"}}, spec, why));
    CHECK(!why.empty());
    CHECK(!tool_spec_from_schema("mcp__s__a>b", nlohmann::json{{"type", "object"}}, spec, why));
    CHECK(!tool_spec_from_schema("", nlohmann::json{{"type", "object"}}, spec, why));

    // ...and the same for a parameter name.
    const nlohmann::json bad = obj_schema({{"a=b", {{"type", "string"}}}},
                                          nlohmann::json::array());
    CHECK(!tool_spec_from_schema("mcp__s__t", bad, spec, why));
}

TEST(nullable_and_enum_constraints_are_preserved_when_reducible) {
    // P2 §12: ["string","null"] keeps Text + nullable rather than collapsing to Json.
    // Multi-type unions and untyped props still travel as Json.
    parsephony::ToolSpec spec;
    std::string why;
    const nlohmann::json schema = obj_schema(
        {{"maybe", {{"type", nlohmann::json::array({"string", "null"})}}},
         {"mode", {{"type", "string"}, {"enum", nlohmann::json::array({"a", "b"})}}},
         {"tags", {{"type", "array"}, {"items", {{"type", "string"}}}}},
         {"cfg",
          {{"type", "object"},
           {"properties", {{"x", {{"type", "number"}}}}},
           {"required", nlohmann::json::array({"x"})}}},
         {"untyped", nlohmann::json::object()},
         {"union", {{"type", nlohmann::json::array({"string", "number"})}}}},
        nlohmann::json::array());
    REQUIRE(tool_spec_from_schema("mcp__s__t", schema, spec, why));
    REQUIRE(spec.params.size() == 6);
    for (const parsephony::ParamSpec& p : spec.params) {
        if (p.name == "maybe") {
            CHECK(p.type == parsephony::ParamType::Text);
            CHECK(p.nullable);
        } else if (p.name == "mode") {
            CHECK(p.type == parsephony::ParamType::Text);
            REQUIRE(p.enum_values.size() == 2);
            CHECK_EQ(p.enum_values[0], std::string("a"));
            CHECK_EQ(p.enum_values[1], std::string("b"));
        } else if (p.name == "tags") {
            CHECK(p.type == parsephony::ParamType::Array);
            CHECK(p.has_items_type);
            CHECK(p.items_type == parsephony::ParamType::Text);
        } else if (p.name == "cfg") {
            CHECK(p.type == parsephony::ParamType::Object);
            CHECK(p.schema_extras_json.find("properties") != std::string::npos);
        } else if (p.name == "untyped" || p.name == "union") {
            CHECK(p.type == parsephony::ParamType::Json);
        }
    }
}

// --- the live half: a real server, a real subprocess -----------------------

TEST(a_live_server_populates_the_registry_under_the_servers_tool_names) {
    Registry registry(workspace());
    const std::size_t native = registry.decls().size();

    McpHost host;
    const auto report = host.connect_and_register({demo("demo", false)}, registry);

    REQUIRE(report.size() == 1);
    CHECK(report[0].connected);
    CHECK(report[0].error.empty());
    CHECK_EQ(report[0].registered, static_cast<std::size_t>(5));
    CHECK(registry.decls().size() == native + 5);

    // The server's name, not a prefix: `godot_guide` is what AGENTS.md and the model
    // write. Collision is the only reason to namespace.
    REQUIRE(find(registry, "echo") != nullptr);
    CHECK(find(registry, "add") != nullptr);
    CHECK(find(registry, "always_fails") != nullptr);
    CHECK(find(registry, "mcp__demo__echo") == nullptr);

    // The schema came across, so generation of a remote call is constrained.
    const ToolDecl* echo = find(registry, "echo");
    REQUIRE(echo->spec.params.size() == 1);
    CHECK_EQ(echo->spec.params[0].name, std::string("text"));
    CHECK(echo->spec.params[0].required);
}

TEST(a_remote_tool_is_namespaced_only_when_its_name_would_shadow) {
    Registry registry(workspace());
    CHECK_EQ(registered_mcp_tool_name(registry, "demo", "echo"), std::string("echo"));
    CHECK_EQ(registered_mcp_tool_name(registry, "demo", "read_file"),
             std::string("mcp__demo__read_file"));
}

TEST(an_untrusted_servers_tools_are_irreversible_and_a_trusted_servers_are_not) {
    // THE containment decision. A remote tool runs in the server's process, which Seatbelt
    // does not cover, so an untrusted server's tools each raise an approval card. The
    // server's own annotations never participate.
    Registry untrusted_reg(workspace());
    McpHost untrusted_host;
    (void)untrusted_host.connect_and_register({demo("demo", false)}, untrusted_reg);
    const ToolDecl* u = find(untrusted_reg, "echo");
    REQUIRE(u != nullptr);
    CHECK(u->irreversible);
    CHECK(u->mutates_workspace);
    CHECK(u->needs_execution);
    CHECK(!u->executes_commands);
    CHECK(u->remote);
    // The model is told where this came from and that we are not containing it.
    CHECK(u->description.find("outside the sandbox") != std::string::npos);

    Registry trusted_reg(workspace());
    McpHost trusted_host;
    (void)trusted_host.connect_and_register({demo("demo", true)}, trusted_reg);
    const ToolDecl* t = find(trusted_reg, "echo");
    REQUIRE(t != nullptr);
    CHECK(!t->irreversible);
    CHECK(!t->mutates_workspace);
    CHECK(!t->needs_execution);
    CHECK(t->remote);
    CHECK(t->description.find("provided by MCP server") != std::string::npos);

    // MCP's default for an omitted readOnlyHint is false: the tool may modify.
    // Trust is not a proxy for "this is a read".
    const ToolDecl* add = find(trusted_reg, "add");
    REQUIRE(add != nullptr);
    CHECK(add->mutates_workspace);
    CHECK(add->needs_execution);
    CHECK(!add->irreversible);

    CHECK(!trusted_host.prompt_instructions().empty());
    CHECK(trusted_host.prompt_instructions().find("Demonstration server") != std::string::npos);
    CHECK(untrusted_host.prompt_instructions().empty());
}

TEST(a_namespaced_call_round_trips_through_the_registry) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);

    const ToolResult echoed =
        registry.execute("echo", {{"text", "hello from lmp"}}, 0);
    CHECK(echoed.ok());
    CHECK_EQ(echoed.summary, std::string("hello from lmp"));

    // Numbers are parsed rather than sent as strings: the guard validated the SHAPE, and
    // turning "2" into 2 is this layer's job.
    const ToolResult sum = registry.execute("add", {{"a", "2"}, {"b", "3"}}, 0);
    CHECK(sum.ok());
    CHECK(sum.summary.find('5') != std::string::npos);
}

TEST(a_remote_tool_that_fails_is_evidence_and_not_a_refusal) {
    // is_error means the tool RAN and failed, which is exactly what the model needs to
    // see to try something else. Mapping it to Refused would tell the agent the call
    // never happened -- the same conflation S6.2 exists to prevent.
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);

    const ToolResult r = registry.execute("always_fails", {}, 0);
    CHECK(!r.ok());
    CHECK(r.status == Status::ToolError);
    CHECK(r.status != Status::Refused);
    CHECK(r.summary.find("on purpose") != std::string::npos);
}

TEST(a_server_that_cannot_start_leaves_its_tools_absent_and_the_run_alive) {
    Registry registry(workspace());
    const std::size_t native = registry.decls().size();

    McpServerConfig missing;
    missing.name = "ghost";
    missing.command = "/nonexistent/mcp-server-that-is-not-there";

    McpHost host;
    const auto report = host.connect_and_register({missing, demo("demo", true)}, registry);

    REQUIRE(report.size() == 2);
    CHECK(!report[0].connected);
    CHECK(!report[0].error.empty());
    CHECK_EQ(report[0].registered, static_cast<std::size_t>(0));

    // ...and the working server beside it still registered. A broken server must not be
    // able to take the run down with it.
    CHECK(report[1].connected);
    CHECK(registry.decls().size() == native + 5);
    CHECK(registry.execute("echo", {{"text", "still here"}}, 0).ok());
}

TEST(a_server_that_dies_mid_run_fails_its_calls_without_stalling_the_turn) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);
    CHECK(registry.execute("echo", {{"text", "before"}}, 0).ok());

    // Drop the connection under the registered handler, the way a crashed server would.
    host.close();

    const ToolResult after = registry.execute("echo", {{"text", "after"}}, 0);
    CHECK(!after.ok());
    // Typed, and it RETURNED -- the point of this test is that it is not a hang.
    CHECK(after.status == Status::ToolError);
    CHECK(after.error_class == ErrorClass::Transient);
}

TEST(a_duplicate_server_name_cannot_displace_the_first_ones_tools) {
    Registry registry(workspace());
    McpHost host;
    const auto report =
        host.connect_and_register({demo("demo", true), demo("demo", false)}, registry);

    REQUIRE(report.size() == 2);
    CHECK(report[0].connected);
    CHECK(!report[1].connected);
    CHECK(!report[1].error.empty());
    // The first server's trust level survived: the second did not silently re-register
    // the same names with different containment.
    const ToolDecl* echo = find(registry, "echo");
    REQUIRE(echo != nullptr);
    CHECK(!echo->irreversible);
}

TEST(a_second_servers_echo_is_namespaced_rather_than_shadowing) {
    Registry registry(workspace());
    McpHost host;
    const auto report =
        host.connect_and_register({demo("alpha", true), demo("beta", true)}, registry);
    REQUIRE(report.size() == 2);
    CHECK(report[0].connected);
    CHECK(report[1].connected);
    REQUIRE(find(registry, "echo") != nullptr);
    REQUIRE(find(registry, "mcp__beta__echo") != nullptr);
    CHECK(find(registry, "mcp__alpha__echo") == nullptr);
    CHECK(registry.execute("echo", {{"text", "a"}}, 0).ok());
    CHECK(registry.execute("mcp__beta__echo", {{"text", "b"}}, 0).ok());
}

// The case that proves these checks can fail (S11.4): every claim above is about a name
// that IS registered, so a lookup that always succeeded would satisfy all of them.
TEST(the_registry_lookup_can_say_no) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);
    CHECK(find(registry, "mcp__demo__no_such_tool") == nullptr);
    CHECK(find(registry, "mcp__other__echo") == nullptr);
    const ToolResult missing = registry.execute("mcp__demo__no_such_tool", {}, 0);
    CHECK(!missing.ok());
    CHECK(missing.error_class == ErrorClass::NotFound);
}

// --- images from a server ------------------------------------------------------
//
// An MCP image arrives as base64 inside a JSON block, and everything else in this
// codebase moves images by PATH. The host spools the bytes into the workspace so the
// picture goes through exactly the same decode -> resize -> patch -> splice path as a file
// the model opened itself.
//
// Before this, every image block became the literal text "[image content omitted]": a
// server whose entire purpose was returning a picture could describe one and never show
// one, and nothing in the run said so.
TEST(an_image_from_a_server_is_spooled_and_offered_as_pixels) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);

    const ToolResult r = registry.execute("screenshot", {}, 0);
    CHECK(r.ok());
    // The text block still reaches the model...
    CHECK(r.summary.find("here is the screenshot") != std::string::npos);
    // ...and the image is a real file on disk, named in `images`.
    REQUIRE(r.images.size() == std::size_t{1});
    CHECK(std::filesystem::exists(r.images[0]));
    CHECK(std::filesystem::file_size(r.images[0]) > 0);
    // The extension follows the declared mime type, so view_image's own check agrees
    // with the bytes.
    CHECK(r.images[0].size() > 4 &&
          r.images[0].compare(r.images[0].size() - 4, 4, ".png") == 0);
    // And the model is TOLD where it went, so it can name the path again later.
    CHECK(r.summary.find("spooled") != std::string::npos);
    CHECK(r.summary.find("omitted") == std::string::npos);
}

// A block this build cannot turn into pixels must still be announced. Silence would read
// as "the tool returned nothing", which is a different and wrong fact.
TEST(a_non_image_block_is_still_named_rather_than_dropped) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);
    const ToolResult r = registry.execute("echo", {{"text", "plain"}}, 0);
    CHECK(r.ok());
    CHECK(r.images.empty());
}

// A STRING PARAMETER IS SENT AS A STRING, whatever its contents happen to parse as.
//
// Every value went through nlohmann::json::parse unconditionally, so a parameter the
// server's own schema declares as `string` arrived as an object, a number or a bool
// whenever its text looked like one -- and the server then rejected the call it had itself
// specified. Measured on r-18ced29746aa7728-2ea858f4: godoer's `godot_build_scene` takes
// `spec` as a STRING holding a scene spec; the model sent one, this layer parsed it into
// an object, and godoer answered "Input should be a valid string [type=string_type,
// input_value={'path': ...}, input_type=dict]".
//
// `echo` returns its `text` argument, so a round trip that comes back byte-identical is
// proof the string was not reinterpreted on the way out.
TEST(a_string_parameter_holding_json_is_not_parsed_into_an_object) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);

    const std::string spec = R"({"path":"res://main.tscn","root":{"type":"Node3D"}})";
    const ToolResult r = registry.execute("echo", {{"text", spec}}, 0);
    CHECK(r.ok());
    CHECK_EQ(r.summary, spec);
}

TEST(a_string_parameter_that_looks_numeric_keeps_its_text) {
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);

    // The same hazard, quieter: a version "1.20" becomes 1.2 and an id "007" becomes 7,
    // and neither failure announces itself the way the object one did.
    CHECK_EQ(registry.execute("echo", {{"text", "007"}}, 0).summary,
             std::string("007"));
    CHECK_EQ(registry.execute("echo", {{"text", "1.20"}}, 0).summary,
             std::string("1.20"));
    CHECK_EQ(registry.execute("echo", {{"text", "true"}}, 0).summary,
             std::string("true"));
}

TEST(a_non_string_parameter_is_still_parsed) {
    // The parse is right for every other declared type: the guard validates SHAPE, not
    // JSON, so an Object, Array, Number or Boolean param arrives as the text the model
    // emitted and has to be parsed. Narrowing this to strings must not take that with it.
    Registry registry(workspace());
    McpHost host;
    (void)host.connect_and_register({demo("demo", true)}, registry);

    const ToolResult sum = registry.execute("add", {{"a", "2"}, {"b", "3"}}, 0);
    CHECK(sum.ok());
    CHECK(sum.summary.find('5') != std::string::npos);
}
