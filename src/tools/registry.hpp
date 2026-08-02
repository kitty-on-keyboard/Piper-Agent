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
    // This tool destroys something that cannot be recovered from the workspace itself.
    //
    // DECLARED, not inferred. The blast-radius classifier reads command STRINGS, so it
    // has nothing to say about a tool call: `delete_file` destroys data without any
    // command ever existing. Told to wipe a workspace, a run did exactly that through
    // this tool -- no card, no risk score, nothing to deny -- while the whole HITL
    // apparatus watched the `shell` tool it never used.
    //
    // A tool knows what it does at declaration time. It does not need to be guessed at.
    bool irreversible = false;
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

// Where cross-session memory lives, and how much of it there is.
//
// A single dotfile at the workspace root, so nothing has to create a directory, and
// hyphenated so the tool_honesty ratchet does not read the name as a tool reference.
// The cap is a PROMPT budget, not a disk one: this text is prepended to the stable part
// of every prompt for the whole of the next session.
inline constexpr const char* kMemoryFileName = ".lmp-memory.md";
inline constexpr std::size_t kMemoryMaxBytes = 16U * 1024;

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

    // Appends one fact to the workspace's memory file, deduplicated and bounded.
    [[nodiscard]] ToolResult remember_fact(const std::string& raw);

    // Runs `git <args>` in the workspace under the same sandbox as any other command.
    // `args` is composed by the registry, never supplied by the model.
    [[nodiscard]] ToolResult run_git(const std::string& args, int approved_tier);

    WorkspaceContext ctx_;
    std::vector<ToolDecl> decls_;
    std::vector<parsephony::ToolSpec> specs_;
    std::map<std::string, Handler> handlers_;
};

// Pure helper shared by tools and tests: resolves `rel` against root/cwd and refuses
// anything that escapes. Empty result means refused.
[[nodiscard]] std::string resolve_contained(const std::string& root, const std::string& rel);

} // namespace lmp::tools
