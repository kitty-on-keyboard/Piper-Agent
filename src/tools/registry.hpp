#pragma once
//
// Tool registry (spec S6). Twenty native tools always, plus two when this run has a
// durable context store -- and it resists growth either way: every tool is permanent
// surface area, advertised in the KV prefix, and impossible to remove once a model has
// been trained to expect it.
//
//   read_file  read_many  read_slice  list_dir  search  locate_symbol
//   write_file  replace_in_file  apply_patch  append_file  delete_file
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
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <parsephony/toolcall.hpp>

#include "src/model/backend.hpp"
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

// Everything a tool invocation may touch. Registry opens `root` as a descriptor-rooted
// platform capability; path authorization does not rely on this string.
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
    // THE MOST ANY ONE TOOL RESULT MAY HAND THE CONTEXT STORE, aggregate.
    //
    // The per-file budget above bounds ONE file; nothing bounded a tool that returns
    // several. read_many concatenates up to four of them, so 4 x max_model_read_bytes
    // could arrive at a context whose observation budget is 32 KB -- and ContextStore's
    // add_turn asserts on exactly that, which in a build with assertions on (all of them,
    // here) is an abort. Measured 2026-08-08: read_many returned 38,768 bytes, the assert
    // fired, and the sidecar died mid-run taking a 19 GB model with it.
    //
    // Kept in the workspace rather than as a constant in read_many so it can be set from
    // the same number the context store is given, and the two cannot drift apart again.
    std::size_t max_observation_bytes;
    std::string spool_dir;        // oversized output lands here (S14)
    int shell_wall_clock_seconds;
};

// Cap on paths inside one read_many call. Matches TurnGrammar::kMaxCallsPerTurn so a
// batched primitive cannot outrun the multi-call surface the model already has.
//
// In the header rather than the .cpp because the observation budget below is SIZED from
// it, and a constant that another constant depends on has to be visible to the test that
// checks the two still agree.
inline constexpr std::size_t kReadManyMaxPaths = 4;

// The observation budget, in one place because two copies of it drifted apart and the
// gap was an abort. ContextStore::add_turn is given this number and clamps anything
// larger; the workspace is given it so the tool layer can truncate before it gets there.
//
// SIZED FROM WHAT read_many CAN ACTUALLY PRODUCE, which is the relationship that was
// missing. read_many returns up to kReadManyMaxPaths (4) files of up to
// max_model_read_bytes (24 KB) each -- 96 KB -- into a budget that was 32 KB. A four-file
// read could not fit by construction, so the tool did the IO and the store threw a third
// of it away, and the reads that motivated the cap in the first place measured 27-38 KB.
// 128 KB is 4 x 24 KB plus room for the "=== path ===" headers and a truncation note.
//
// WHAT IT COSTS, because this is not free: an observation this size is ~32k tokens,
// about a third of the 96k context budget, so three maximal reads fill the window and
// force a compaction -- and a compaction is a full re-prefill, which has been measured at
// 22x TTFT. The bet is that reading four files intact once beats reading them truncated
// and going back for the rest, which is what the 32 KB budget actually bought. If
// compaction rate gets worse, this number is the first thing to look at, and lowering
// kReadManyMaxPaths is the other side of the same dial.
inline constexpr std::size_t kObservationBudgetBytes = 128U << 10;

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

// One workspace mutation routed through the editor (or refused by it).
//
// `expected_absent` and `expected_version` are the optimistic-concurrency claim: the
// extension compares them to the current in-memory TextDocument immediately before
// WorkspaceEdit, so a dirty buffer that no longer matches the preimage becomes an
// explicit conflict instead of a silent overwrite.
struct EditIntent {
    std::string abs_path;
    std::string new_content;
    std::string expected_version; // lowercase SHA-256 hex; empty iff expected_absent
    bool expected_absent = false;
};
using EditSink = std::function<EditOutcome(const EditIntent& intent)>;

// Editor-backed code intelligence (P2 §10). When set, locate_symbol prefers this over
// the headless ranked walk. Ops match lmp/code_intel: workspace_symbols, definition,
// references, diagnostics, rename_preview.
struct CodeIntelQuery {
    std::string op;
    std::string query;
    std::string path;
    std::int64_t line = 0;
    std::int64_t character = 0;
};
struct CodeIntelOutcome {
    bool ok = false;
    std::string result_text;
    std::string error;
};
using CodeIntelSink = std::function<CodeIntelOutcome(const CodeIntelQuery& query)>;

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
    void set_code_intel_sink(CodeIntelSink sink) { code_intel_sink_ = std::move(sink); }
    [[nodiscard]] bool has_code_intel_sink() const noexcept {
        return static_cast<bool>(code_intel_sink_);
    }

    // Run-scoped cancel observation for tools that can block (shell, git, MCP). Set
    // once at run start; cleared when the run ends. Handlers read it rather than taking
    // CancelToken through every declare() lambda -- parallel execute() calls then all
    // see the same token without a per-call restore race on a temporary override.
    void set_cancel_token(const model::CancelToken* cancel) noexcept {
        cancel_token_ = cancel;
    }
    [[nodiscard]] const model::CancelToken* cancel_token() const noexcept {
        return cancel_token_;
    }

    [[nodiscard]] const std::vector<ToolDecl>& decls() const noexcept { return decls_; }
    [[nodiscard]] const WorkspaceContext& workspace() const noexcept { return ctx_; }
    [[nodiscard]] const platform::WorkspaceFs& filesystem() const noexcept {
        return workspace_fs_;
    }
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
    //
    // `cancel` defaults to nullptr and means "use the run-scoped token from
    // set_cancel_token()", which is what the agent path does. Tests that never call
    // set_cancel_token may pass a token here; a non-null argument wins for that call
    // only when no run-scoped token is installed (single-threaded test path).
    [[nodiscard]] ToolResult execute(const std::string& name,
                                     const std::vector<ToolParamValue>& params,
                                     int approved_tier,
                                     const model::CancelToken* cancel = nullptr);

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

    // True when `rel` securely names an existing, non-empty regular file.
    [[nodiscard]] bool would_overwrite_existing(const std::string& rel) const;

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
    //
    // `pre` is the optimistic-concurrency claim enforced by the editor sink or, on the
    // headless path, immediately before the atomic rename.
    [[nodiscard]] CommitOutcome commit_write(const platform::ContainedPath& path,
                                             std::string_view content,
                                             const platform::WritePrecondition& pre);

    // Records a successful read's content version so a later write_file/delete_file can
    // carry it without the model copying the digest. Keyed by absolute path.
    void note_read_version(const std::string& abs_path, std::string_view bytes);
    // Resolves the version claim for an existing-file mutation: optional tool param wins,
    // else the harness ledger from a prior read in this run. Empty means "no claim".
    [[nodiscard]] std::string resolve_expected_version(
        const std::string& abs_path, const std::vector<ToolParamValue>& params) const;

    // One whole-file read with line numbers and content version. Shared by read_file and
    // read_many so a batch cannot drift from the single-path tool.
    [[nodiscard]] ToolResult read_one_file(const std::string& path);

    // Runs `git <args>` in the workspace under the same sandbox as any other command.
    // `args` is composed by the registry, never supplied by the model.
    [[nodiscard]] ToolResult run_git(const std::string& args, int approved_tier);

    WorkspaceContext ctx_;
    platform::WorkspaceFs workspace_fs_;
    EditSink edit_sink_;
    CodeIntelSink code_intel_sink_;
    const model::CancelToken* cancel_token_ = nullptr;
    // Set by declare_context_tools. Read by remember_fact() too, which mirrors each note
    // into the store -- see memory_file.cpp.
    ContextSourceFn context_source_;
    std::vector<ToolDecl> decls_;
    std::vector<parsephony::ToolSpec> specs_;
    std::map<std::string, Handler> handlers_;
    // Absolute path -> SHA-256 hex of bytes last observed by read_file/read_slice.
    // Invalidated on a successful mutation so the next whole-file overwrite must read
    // again. Re-reads always return current content; duplicate collapse under context
    // pressure is the agent's job, never a pointer-only System Observation here.
    std::map<std::string, std::string> read_versions_;
};

// Declaration helpers, shared by the files that declare tools -- registry.cpp for the
// native tools above and context_tools.cpp for the two that read the durable store. They were
// local to registry.cpp until there was a second declarations file; copying twelve lines
// of boilerplate into it would have been the cheaper edit and the wrong one.
[[nodiscard]] parsephony::ParamSpec param(const char* name, parsephony::ParamType type,
                                          bool required);
[[nodiscard]] const std::string* get(const std::vector<ToolParamValue>& params,
                                     const char* name);

// Compatibility helper shared by tests and policy code. It opens `root`, validates
// existing components without following symlinks, and returns the canonical absolute
// spelling. Registry operations use its already-open WorkspaceFs instead.
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
