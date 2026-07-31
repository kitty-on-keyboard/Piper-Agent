#pragma once
//
// Tool registry (spec S6). Eleven tools, the spec's set, and it resists growth --
// every tool is permanent surface area.
//
//   read_file  read_slice  list_dir  search  locate_symbol
//   write_file  replace_in_file  append_file  delete_file
//   shell  job_status
//
// Each declaration carries the parsephony::ToolSpec the grammar enforces at decode
// time, so "the model called a tool wrong" is unrepresentable, and the description the
// model reads. Descriptions are honesty-checked in CI (S6.3): any `tool_name` a
// description mentions must resolve to a registered tool -- v1 audited this once and 7
// of 13 tools were lying.
//
// The advertised set is baked into the KV prefix and is therefore run-constant (S6.4):
// the registry is immutable after construction.
//
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <parsephony/toolcall.hpp>

#include "src/tools/tool_result.hpp"

namespace lmp::tools {

struct ToolParamValue {
    std::string name;
    std::string value;
};

struct ToolDecl {
    std::string name;
    std::string description;
    parsephony::ToolSpec spec;
    bool mutates_workspace = false;
    bool executes_commands = false;
};

// Everything a tool invocation may touch. Paths are resolved against `root` and
// containment-checked textually (blast-radius rule 1); a path outside the root is a
// Refused, not an error.
struct WorkspaceContext {
    std::string root;             // absolute, no trailing slash
    std::size_t max_read_bytes;   // whole file or honest TooLarge (S2.2)
    std::size_t max_result_bytes; // model-facing summary budget, pre-compaction
    std::string spool_dir;        // oversized output lands here (S14)
    int shell_wall_clock_seconds;
};

class Registry {
  public:
    explicit Registry(WorkspaceContext ctx);

    [[nodiscard]] const std::vector<ToolDecl>& decls() const noexcept { return decls_; }
    [[nodiscard]] const std::vector<parsephony::ToolSpec>& guard_specs() const noexcept {
        return specs_;
    }
    [[nodiscard]] const ToolDecl* find(const std::string& name) const;

    // The <tools> block for the system prompt: JSON schemas, one per tool.
    [[nodiscard]] std::string tools_json() const;

    // Executes a call that already passed the grammar (so `name` is registered and
    // required params are present -- the guard enforced it). `approved_tier` is the
    // sandbox tier the loop's policy granted for THIS call; only `shell` consumes it.
    [[nodiscard]] ToolResult execute(const std::string& name,
                                     const std::vector<ToolParamValue>& params,
                                     int approved_tier);

  private:
    using Handler =
        std::function<ToolResult(const std::vector<ToolParamValue>&, int approved_tier)>;

    void declare(ToolDecl decl, Handler handler);

    WorkspaceContext ctx_;
    std::vector<ToolDecl> decls_;
    std::vector<parsephony::ToolSpec> specs_;
    std::map<std::string, Handler> handlers_;
};

// Pure helper shared by tools and tests: resolves `rel` against root/cwd and refuses
// anything that escapes. Empty result means refused.
[[nodiscard]] std::string resolve_contained(const std::string& root, const std::string& rel);

} // namespace lmp::tools
