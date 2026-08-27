#pragma once
//
// What survives between missions, and the two acts that create and destroy it.
//
// v1 of the sidecar built the tokenizer, the weights, the registry and the context store
// inside one function and dropped all four on the way out, so every mission was a fresh
// one-shot: no history to follow up on, and a ~19 GB reload to ask a second question.
// The agent could be launched and could be aborted, and that was the entire vocabulary.
//
// The context store is the part that makes a conversation; keeping the weights loaded
// alongside it is what makes a follow-up cost a prefill instead of a minute.
//
// WHY THIS IS ITS OWN UNIT. Loading became something the operator asks for by name
// rather than a side effect of starting a mission, so "the model is loaded" turned into
// a state with transitions, a duration and a failure mode -- all of which the surface
// renders. That is a thing worth being able to point at. sidecar.cpp keeps the protocol
// dispatch and the run loop; the lifecycle of the 19 GB lives here.
//
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/loop/agent.hpp"
#include "src/model/mlx_backend.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/surface/context_journal.hpp"
#include "src/tools/mcp_host.hpp"
#include "src/tools/registry.hpp"

namespace lmp::surface {

struct Session {
    std::string model_dir;
    std::string draft_model_dir;
    // The identity of the run currently loaded into this session, minted by the sidecar at
    // lmp/start (see run_id.hpp). Held HERE rather than taken from each message, because
    // the sidecar is the only party that knows which run its context belongs to -- a
    // client that sent the wrong one would append a turn to another run's history.
    std::string run_id;
    std::string workspace;
    // From the checkpoint's text_config.max_position_embeddings. 0 when unknown (no
    // model loaded yet, or the field was absent). Copied into AgentConfig at run start
    // so the loop can refuse a prompt that cannot fit.
    std::int32_t model_max_sequence_tokens = 0;
    std::unique_ptr<model::QwenTokenizer> tok;
    std::unique_ptr<model::MlxBackend> backend;
    std::unique_ptr<tools::Registry> registry;
    // The handlers this installs into the registry share ownership of their clients
    // (McpHost::Connection), so neither destruction order here is a use-after-free --
    // that WAS true of an earlier raw-pointer version and ASan caught it. Kept adjacent
    // to the registry because they are one unit: the registry holds the tools, this holds
    // the processes behind them.
    std::unique_ptr<tools::McpHost> mcp;
    // What the current registry's remote tools were built from. The Registry has no
    // "unregister", so a changed server list means a new Registry, not a patched one.
    std::string mcp_signature;
    std::unique_ptr<ContextJournal> journal; // BEFORE ctx: its sink points here
    std::unique_ptr<context::ContextStore> ctx;
    loop::AgentConfig config;
    // Whether this client can answer lmp/edit. Advertised at lmp/start rather than
    // assumed: the extension can, agent_eval.py cannot, and guessing wrong either wedges
    // an unattended run on a reply that never comes or writes behind the editor's back.
    bool client_applies_edits = false;
    // Whether this client answers lmp/code_intel via editor language features.
    bool client_provides_code_intel = false;

    // Both halves, because either alone is a session that cannot generate. They are only
    // ever assigned together (see load_model), so this can never be half true -- the
    // predicate says so anyway, since the invariant is the point.
    [[nodiscard]] bool model_ready() const noexcept {
        return tok != nullptr && backend != nullptr;
    }

    // Whether a load of `dir` would be a no-op. The caller needs this BEFORE calling
    // load_model, because loading blocks for tens of seconds and the surface has to be
    // told that is about to happen -- afterwards is too late to say "working on it".
    // Both halves, because swapping ONLY the drafter still changes what generates: a
    // no-op here would leave the previous head bound while the client believes it changed.
    [[nodiscard]] bool holds(const std::string& dir,
                             const std::string& draft_dir = {}) const noexcept {
        return model_ready() && model_dir == dir && draft_model_dir == draft_dir;
    }
};

struct ModelLoad {
    bool ok = false;
    // Verbatim from the tokenizer or the backend. A load failure is nearly always a
    // fixable statement about what is on disk -- a missing safetensors, an MLX-less
    // build -- and paraphrasing it into "load failed" throws away the only part the
    // operator can act on.
    std::string error;
    double elapsed_ms = 0.0;
};

// Loads the tokenizer and the weights, replacing whatever was loaded before.
//
// BLOCKS for as long as the checkpoint takes. Emits nothing: the caller owns the wire
// and brackets this with the model_status notifications, because a progress message
// that could only be sent after the work finished would not be progress.
// `draft_model_dir` is optional; when set it must be an MTP draft head matching the
// target, and a bad one FAILS the load rather than quietly generating without it.
[[nodiscard]] ModelLoad load_model(Session& session, const std::string& model_dir,
                                   const std::string& draft_model_dir,
                                   const platform::Clock& clock);

// Gives the memory back. The conversation goes with it -- a ContextStore whose weights
// are gone cannot be resumed -- so this is a full reset of everything downstream of the
// model, not just a free().
void unload_model(Session& session);

// Builds the tool registry for this run, and connects the configured MCP servers into it.
//
// Rebuilt rather than patched when the server list changes: the Registry has no
// unregister, so the alternative is remote tools from a previous run outliving the
// settings that authorised them.
void ensure_registry(Session& session, const std::string& workspace,
                     const std::string& message, platform::EventLogWriter& log,
                     const platform::Clock& clock);

// The repo's own conventions, in the order the surrounding ecosystem settled on. First
// file found wins; they are alternatives, not layers, and concatenating them would let a
// stale `.cursorrules` contradict a current AGENTS.md with no way to tell which won.
[[nodiscard]] std::string load_project_instructions(
    const platform::WorkspaceFs& workspace);

// What the agent told itself, last time it was here. The counterpart to the above: same
// stable-prompt slot, same once-per-session cost, opposite provenance.
[[nodiscard]] std::string load_project_memory(
    const platform::WorkspaceFs& workspace);

// Tool names the conventions tell the model to call that NOTHING IN THIS RUN ANSWERS TO.
//
// A missing tool is invisible from the inside. Tool names are constrained at decode time
// by parsephony::ToolCallGuard, so the model cannot emit a call to an unregistered one --
// the mask steers the name into a registered one instead, and there is no error, no event
// and no counterfactual to log. Measured on r-18cecc130e7bc558-31fdd81a: AGENTS.md said
// "Call `godot_guide` before authoring your first scene", godoer's MCP server was not
// connected, and the model's own reasoning ("let me call that tool now") came out as
// `git_status`. It spent six of fourteen turns rediscovering this through the shell, could
// not do the next checklist item without the tool, restated its plan three times and the
// run ended `stalled`.
//
// So the check runs where the answer IS knowable -- at run start, against the registry --
// and its result goes into the prompt rather than only into the log. Knowing at turn 1 is
// worth more than a postmortem: a run that knows can say so and ask.
//
// DELIBERATELY NARROW, because a false positive spends prompt on a fact about a word that
// was never a tool. A candidate must be inside backticks, be lower_snake_case with at
// least one underscore, and sit within a few words of "call"/"run"/"use"/"tool". A name
// counts as registered if a tool has it exactly OR carries it after an `mcp__server__`
// prefix -- otherwise every connected MCP tool would be reported missing, which is the
// same defect pointed the other way.
[[nodiscard]] std::vector<std::string> unknown_tool_names(const std::string& conventions,
                                                          const tools::Registry& registry);

// The sentence appended to the conventions for those names. Empty when there are none.
[[nodiscard]] std::string unknown_tools_note(const std::vector<std::string>& names);

} // namespace lmp::surface
