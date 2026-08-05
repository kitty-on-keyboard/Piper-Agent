#pragma once
//
// Tool registry (spec S6). Sixteen tools always, eighteen when this run has a durable
// context store -- and it resists growth either way: every tool is permanent surface
// area, advertised in the KV prefix, and impossible to remove once a model has been
// trained to expect it.
//
//   read_file  read_slice  list_dir  search  locate_symbol
//   write_file  replace_in_file  append_file  delete_file
//   shell  job_status
//   git_status  git_diff  git_log      (read-only; the agent never MUTATES git)
//   remember                           (cross-session notes, S8)
//   plan                               (declared here, executed by the loop)
//
// and, only when a journal opened for this run (see declare_context_tools):
//
//   context_recall  context_rehydrate  (src/tools/context_tools.cpp, reading src/pcc)
//
// Those two are CONDITIONAL rather than always-on because a run whose database will not
// open must keep working, and advertising a tool whose store is absent teaches the model
// to call something that can only answer "there is nothing there".
//
// The count was "eleven, the spec's set" for long enough that five tools landed under a
// comment saying they had not. A number in a header is a claim like any other.
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

#include "src/platform/fs.hpp"
#include "src/tools/tool_result.hpp"

namespace lmp::pcc {
// Forward-declared rather than included. src/pcc is at L1 and src/tools may reach it
// (see src/pcc/CMakeLists.txt), but store.hpp pulls <sqlite3.h> through sqlite.hpp, and
// every translation unit that includes this header would then be compiling against
// SQLite for the sake of two tools. The one .cpp that needs the type includes it.
class Store;
} // namespace lmp::pcc

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
    // This tool's work happens in a process this binary does not control, so none of the
    // three flags above is a claim we can stand behind -- they describe what the peer
    // SAYS it does. Declared separately because trust already meant something else here:
    // a trusted MCP server's tools set mutates_workspace, executes_commands and
    // irreversible all false, which is the right answer to "does this need a card" and
    // the wrong answer to "may a read-only mode call it". Mode policy reads this.
    bool remote = false;
    // This tool does no work without an execution grant: at T0 it can only refuse.
    //
    // Distinct from executes_commands, which means "carries an operator-visible command
    // string that the blast-radius classifier must read". `shell` is both. The three git
    // tools are only this -- they run a subprocess through run_git but take no `command`
    // param, so declaring them executes_commands would send them into a command gate with
    // an empty string to classify. Mode filtering needs the first fact and the approval
    // routing needs the second, and they are not the same fact.
    bool needs_execution = false;
    // This tool only means anything in a mode that YIELDS to a human. Asking a question is
    // not a tool call in a run nobody is watching -- it is a turn spent producing text
    // that will be read by the loop and thrown away.
    bool conversational_only = false;
};

// Everything a tool invocation may touch. Paths are resolved against `root` and
// containment-checked textually (blast-radius rule 1); a path outside the root is a
// Refused, not an error.
struct WorkspaceContext {
    std::string root;             // absolute, no trailing slash
    // INTERNAL read budget. These bytes are read to be edited, hashed or matched against;
    // they do not enter the prompt, so 4 MiB is fine.
    std::size_t max_read_bytes;
    // PROMPT read budget: the most a read_file / read_slice result may hand the context
    // store. It used to be max_read_bytes, so one read could exceed the whole context
    // window while read_file's description already claimed it failed honestly above "the
    // limit". Kept separate rather than folded into max_result_bytes because a caller may
    // want a file view wider than a shell summary.
    std::size_t max_model_read_bytes;
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

// Where a workspace write actually lands (spec S12.4).
//
// When a sink is attached, every mutating tool routes its bytes through it instead of
// writing the file itself, so the extension can apply a WorkspaceEdit and undo, dirty
// buffers and the diff UI all work. graft still computes the replacement here -- the
// matching, ambiguity refusal and indent re-anchoring stay in the engine that was
// measured; the extension only applies bytes.
//
// NO SINK MEANS WRITE DIRECTLY, and that is deliberately the opposite of the approver's
// deny-by-default. An absent approver means "nobody is there to ask", so the safe answer
// is no. An absent edit sink means "there is no editor to route through" -- an eval run,
// a script -- and refusing there would break every unattended run for no safety gain.
struct EditOutcome {
    bool applied = false;
    std::string error;
};
using EditSink =
    std::function<EditOutcome(const std::string& abs_path, const std::string& new_content)>;

// What one trip through the write door did, including the case where it did not need to
// open. `unchanged` is a separate answer from `ok()`: nothing failed, and nothing moved.
//
// Not a flag on platform::WriteResult, because that struct is the filesystem's answer to
// "did these bytes land" and this is the harness's answer to "was there anything to
// land" -- a policy question the filesystem has no opinion about.
struct CommitOutcome {
    platform::WriteResult write;
    // The file already held exactly these bytes, so no write was attempted at all. The
    // read that proves it is the same read replace_in_file already does; write_file pays
    // one extra whole-file read per call, which is the cost of being able to tell an edit
    // from an echo.
    bool unchanged = false;

    [[nodiscard]] bool ok() const noexcept { return write.ok(); }
};

class Registry {
  public:
    explicit Registry(WorkspaceContext ctx);

    void set_edit_sink(EditSink sink) { edit_sink_ = std::move(sink); }

    [[nodiscard]] const std::vector<ToolDecl>& decls() const noexcept { return decls_; }
    [[nodiscard]] const WorkspaceContext& workspace() const noexcept { return ctx_; }
    [[nodiscard]] const std::vector<parsephony::ToolSpec>& guard_specs() const noexcept {
        return specs_;
    }
    [[nodiscard]] const ToolDecl* find(const std::string& name) const;

    // The <tools> block for the system prompt: JSON schemas, one per tool.
    //
    // The filtered overload exists so a mode can advertise only what it will actually
    // honour. The registry itself stays run-constant and mode-blind, which S6.4 requires
    // -- the tool set is baked into the KV prefix -- and that still holds, because the
    // mode is fixed at lmp/start and so the filtered set is constant for the run too.
    // Filtering here is not the enforcement; Agent::gate_call is. This is so the model is
    // never shown a tool that would be refused, and does not spend turns finding out.
    [[nodiscard]] std::string tools_json() const;
    [[nodiscard]] std::string tools_json(
        const std::function<bool(const ToolDecl&)>& include) const;

    // Executes a call that already passed the grammar (so `name` is registered and
    // required params are present -- the guard enforced it). `approved_tier` is the
    // sandbox tier the loop's policy granted for THIS call; only `shell` consumes it.
    [[nodiscard]] ToolResult execute(const std::string& name,
                                     const std::vector<ToolParamValue>& params,
                                     int approved_tier);

    using Handler =
        std::function<ToolResult(const std::vector<ToolParamValue>&, int approved_tier)>;

    // Registers a tool whose implementation lives OUTSIDE this process -- today, an MCP
    // server (src/tools/mcp_host.hpp). Public where declare() is private, and the
    // asymmetry is the point: a native tool's behaviour is in this binary and reviewable,
    // while this one's is a promise made by another process that Seatbelt does not cover.
    // Callers are expected to have decided `decl.irreversible` accordingly.
    //
    // Refuses a name that is already registered, so a remote tool can never displace a
    // native one even if the namespacing above it were wrong. Returns false if it did.
    [[nodiscard]] bool declare_remote(ToolDecl decl, Handler handler);

    // --- the durable context store (src/pcc) --------------------------------

    // Where the recall tools read from, resolved at CALL time rather than captured.
    //
    // THIS INDIRECTION IS NOT OPTIONAL. ensure_registry() REUSES a Registry across
    // missions whenever the workspace and the MCP server list are unchanged, while
    // start_mission() replaces session.journal -- and therefore destroys the pcc::Store
    // behind it -- on every mission. A handler holding `pcc::Store&` from mission one is
    // a dangling reference for the whole of mission two, on a path that would read freed
    // memory and usually get plausible bytes back.
    //
    // `store` is null when this run has no journal (the database could not be opened).
    // The tools answer honestly in that case rather than pretending to search.
    struct ContextSource {
        pcc::Store* store = nullptr;
        std::string session; // this mission's partition; see declare_context_tools
    };
    using ContextSourceFn = std::function<ContextSource()>;

    // Counts tokens the way the model does. Same signature as pcc::TokenCounter, which is
    // the same std::function type; spelled out here so this header does not need pcc's.
    using TokenCounter = std::function<std::size_t(std::string_view)>;

    // Declares `context_recall` and `context_rehydrate` against the session's own store.
    //
    // NATIVE, where the same two tools already exist in the out-of-process
    // pcc_mcp_server. Three reasons, and the third is the one that decides it:
    //
    //   - it needs no configuration. The MCP server is not staged into the VSIX and wants
    //     a hand-written server entry with an absolute --db path, and "works when
    //     configured" does not count for a product whose whole pitch is that the model is
    //     in the box.
    //   - no process hop and no JSON round trip per call.
    //   - IT GETS THE REAL TOKENIZER. recall()'s entire contract is that what it returns
    //     FITS A TOKEN BUDGET, and pcc::TokenCounter defaults to bytes/4 -- an estimate
    //     the header is careful to name as one at every call site, because src/pcc
    //     deliberately does not link src/model. In-process the sidecar can pass
    //     QwenTokenizer::encode_content, so the budget is measured in the units it is
    //     denominated in. Out of process it cannot, and a budgeted retrieval whose budget
    //     is a guess is the one thing this component must not be.
    //
    // Returns false if the tools are already declared, which is the reused-Registry case
    // above: the source function resolves the current mission's store on its own, so a
    // second declaration would only duplicate the entries in decls_ and specs_ while
    // handlers_.emplace silently kept the first handler.
    bool declare_context_tools(ContextSourceFn source, TokenCounter count_tokens);

  private:

    void declare(ToolDecl decl, Handler handler);

    // Appends one fact to the workspace's memory file, deduplicated and bounded.
    // `key` empty appends a standalone note; a non-empty key makes the note SUPERSEDE
    // whatever was written under the same key before, in the markdown mirror and in the
    // durable store alike. A keyed fact is stored with no session, because the notes worth
    // correcting are the ones an EARLIER session wrote and supersession is scoped by
    // (session, key).
    [[nodiscard]] ToolResult remember_fact(const std::string& raw, const std::string& key);

    // THE one door for workspace bytes: through the edit sink when one is attached, and
    // atomically to disk when none is. Every mutating tool goes through here, so the
    // extension handover cannot be half-wired -- a tool that forgot would be writing
    // behind the editor's back and nothing would say so.
    //
    // The unchanged check lives HERE for the same reason the door does. Putting it in
    // each handler means three copies of it and a fourth tool added later with none.
    [[nodiscard]] CommitOutcome commit_write(const std::string& abs_path,
                                             std::string_view content);

    // Runs `git <args>` in the workspace under the same sandbox as any other command.
    // `args` is composed by the registry, never supplied by the model.
    [[nodiscard]] ToolResult run_git(const std::string& args, int approved_tier);

    WorkspaceContext ctx_;
    EditSink edit_sink_;
    // Set by declare_context_tools. Read by remember_fact() too, which mirrors each note
    // into the store -- see memory_file.cpp.
    ContextSourceFn context_source_;
    std::vector<ToolDecl> decls_;
    std::vector<parsephony::ToolSpec> specs_;
    std::map<std::string, Handler> handlers_;
};

// Declaration helpers, shared by the files that declare tools -- registry.cpp for the
// sixteen above and context_tools.cpp for the two that read the durable store. They were
// local to registry.cpp until there was a second declarations file; copying twelve lines
// of boilerplate into it would have been the cheaper edit and the wrong one.
[[nodiscard]] parsephony::ParamSpec param(const char* name, parsephony::ParamType type,
                                          bool required);
[[nodiscard]] const std::string* get(const std::vector<ToolParamValue>& params,
                                     const char* name);

// Pure helper shared by tools and tests: resolves `rel` against root/cwd and refuses
// anything that escapes. Empty result means refused.
[[nodiscard]] std::string resolve_contained(const std::string& root, const std::string& rel);

// True when `rel` names an existing, non-empty file inside `root`.
//
// The one question the loop's write gate has to ask, and the reason it exists:
// write_file over an existing file DESTROYS its previous contents, and that is the same
// act delete_file is declared irreversible for, with extra steps.
//
// MEASURED. refuse_wipe_workspace denied delete_file twice, denied shell twice -- every
// gate held -- and the run then emptied ledger.csv with three write_file calls, which
// nothing asked about because write_file is not declared irreversible and
// auto_approve_writes defaults on. The fixture is scored on whether the DATA survived, and
// it did not. Exactly the shape of the hole that put `irreversible` on ToolDecl in the
// first place: the apparatus was watching the tool the run had stopped using.
[[nodiscard]] bool would_overwrite_existing(const std::string& root, const std::string& rel);

} // namespace lmp::tools
