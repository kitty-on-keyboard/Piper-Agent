#pragma once
//
// Agent -- the loop (spec S9).
//
// Model -> tools -> observations -> repeat, until the model answers in text or a budget
// ends it. Every phase boundary is an event (S14). The backend is injected, so the
// identical loop runs under ScriptedBackend in the gate and MlxBackend on the real model
// -- which is the whole point of the seam (S2.1.1).
//
// THE HARNESS DOES NOT ADJUDICATE. It surfaces evidence -- tool output, exit statuses,
// repeat counts, the operator's own check -- and enforces only operator-set policy:
// modes, approvals, budgets. The eighth pass deleted the completion gate, the
// falsifiability ledger and the ten Correctives after a measured run deadlocked between
// them (see turn.hpp). What replaced ~2000 lines of steering: a text-only turn is the
// model's final answer, an exact repeat revalidates current state, and the operator's check
// command runs after every writing turn with its output placed in front of the model.
//
#include <functional>
#include <map>
#include <set>
#include <utility>
#include <vector>
#include <memory>
#include <optional>
#include <string>

#include "src/context/context.hpp"
#include "src/loop/turn.hpp"
#include "src/model/backend.hpp"
#include "src/model/chat_template.hpp"
#include "src/model/grammar.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
#include "src/model/image_preprocess.hpp"
#include "src/tools/registry.hpp"
#include "src/tools/sandbox.hpp"
#include "src/tools/syntax_check.hpp"

namespace lmp::loop {

// Routes a call to auto-approve / escalate / reject. A pure function of risk and the
// model's stated confidence, so it is unit-testable on a corpus (S7.6). The test seam
// is the approver callback: a scripted approver makes attended paths testable without
// a human -- v1 had no such seam, which forced every shell-using test into the
// unattended path and straight into the sandbox refusal.
enum class Approval : std::uint8_t { AutoApprove, Escalate, Reject };

struct HitlThresholds {
    double auto_approve_below_risk = 0.35;
    double reject_above_risk = 0.85;
};

[[nodiscard]] double risk_score(const tools::RiskHint& hint);
[[nodiscard]] Approval route_approval(const tools::RiskHint& hint,
                                      const HitlThresholds& thresholds);

// Capabilities you cannot take back: destroying data, writing outside the workspace,
// escalating privileges, rewriting history. These ALWAYS escalate -- above the risk
// score and above the persistent allowlist.
//
// The score cannot express this and should not be asked to. `rm -rf` carries exactly one
// capability, so it scores 0.30 against a 0.35 auto-approve threshold and never raised a
// card at all: the agent was told to delete every file in a workspace and did, on a run
// configured to deny every approval, because no approval was ever requested. Nudging the
// weight to 0.36 fixes that one command until some other combination lands under the bar.
//
// Irreversibility is a PROPERTY, not a quantity. Everything else stays scored.
[[nodiscard]] bool is_irreversible(const tools::RiskHint& hint) noexcept;

// Hint-level properties that must not be waved through by score alone.
// Irreversible capabilities and Unparseable status always force. PartiallyParsed
// alone also reports true here (status-only hints cannot hide), but the command gate
// narrows Partial via `opaque_script_command` so toolchain drivers the classifier
// marks Partial (`swift build`, `cmake --build`) stay low-friction under T1.
[[nodiscard]] bool forces_escalation(const tools::RiskHint& hint) noexcept;

// True for interpreter+script / source / eval / local script-path shapes whose body
// is not in the command string. The Seatbelt wipe hole: `bash wipe.sh` is Partial
// with empty destroy caps and must card after auto_approve_exec.
[[nodiscard]] bool opaque_script_command(const std::string& command) noexcept;

// Persistent prefix allowlists apply only to fully parsed, non-destructive commands.
// Opaque / irreversible calls cannot be waved through by a remembered prefix.
[[nodiscard]] bool allowlist_may_auto_approve(const tools::RiskHint& hint) noexcept;

// Whether "Always allow" on this command could persist a rule that will ever match again.
// Sent to the card so the button appears only where it does something -- see the
// definition for the 19 saved rules that could not match anything.
[[nodiscard]] bool can_persist_allowlist_rule(const std::string& command,
                                              const tools::RiskHint& hint);

// The build command this workspace obviously has, or empty when it is not obvious.
//
// Consulted only when the operator configured no verify_contract and the mode can write,
// because a writing run with no check has no feedback loop at all. Deliberately narrow:
// it recognises Package.swift / Cargo.toml / go.mod and nothing else, and refuses to
// answer when a root carries more than one. See the definition for why CMake, npm and
// make are excluded. Exposed rather than file-static so the mapping is unit-testable
// without spawning a shell.
[[nodiscard]] std::string detected_verify_command(const std::string& workspace_root);

// Whether the operator has already said yes to this exact command.
//
// Matches on equality, or on `entry` followed by a space -- and NEVER on a command
// carrying shell chaining (`;` `&&` `||` `|` backtick `$(` `>` `<` `&`), because
// allowlisting `python3 -m pytest` must not authorise `python3 -m pytest; rm -rf ~`.
// A chained command falls through to normal routing rather than being refused: it may
// be perfectly ordinary, it just cannot be waved through on a prefix.
[[nodiscard]] bool is_allowlisted(const std::string& command,
                                  const std::vector<std::string>& allowed);

// Run-scoped consent key for an opaque (PartiallyParsed / Unparseable) command. Empty
// when the hint is fully parsed: those use the verbatim command string for the run latch.
// Non-empty keys bind workspace, command, classifier capabilities, and digests of
// referenced script files so approving `bash build.sh` does not survive a script rewrite.
[[nodiscard]] std::string opaque_run_consent_key(const std::string& workspace_root,
                                                 const std::string& command,
                                                 const tools::RiskHint& hint);

// One line naming the call and its arguments, each argument truncated for display. Shared
// with the approval gate in approval.cpp, which puts it on the card.
[[nodiscard]] std::string preview_of(const std::string& tool,
                                     const std::vector<tools::ToolParamValue>& params);

// The same call in the GRAMMAR'S surface form, for the assistant message in the next
// prompt -- so the run's transcript shows it emitting calls instead of tool results
// appearing after messages that called nothing. See the note on the definition.
[[nodiscard]] std::string call_surface_form(const std::string& tool,
                                            const std::vector<tools::ToolParamValue>& params);

// Returns true to allow. Injected so tests script it; the UI supplies the real one.
//
// `command` is the shell command verbatim, or empty for a call that is not one (a write,
// say). It is separate from `preview` because preview truncates each argument for
// display, and a truncated string is the wrong thing to build a remembered allowlist
// rule from -- it would either never match again or match something shorter than what
// the operator actually approved.
using Approver = std::function<bool(const std::string& tool, const std::string& command,
                                    const std::string& preview,
                                    const tools::RiskHint& hint)>;

// Anything the user has said since the last time it was asked, oldest first.
//
// Polled at TURN BOUNDARIES, never mid-generation. Cancel is the violent interrupt and
// sets a token in the middle of the token stream; steering is the gentle one, so the
// model finishes the thought it is having and then reads the instruction before choosing
// its next move. Injected for the same reason the approver is: the scripted loop suite
// can hand the run a message at a chosen turn with no transport in the picture.
using SteerSource = std::function<std::vector<std::string>()>;

struct AgentConfig {
    Mode mode = Mode::Agent;
    Budget budget;
    HitlThresholds hitl;
    // Qwen3's own recommended operating point by default (S5.9). Carried here rather
    // than left to InferenceTask's defaults so the editor's settings can actually reach
    // the sampler -- they could not before, which made every sampling knob in the
    // extension inert.
    model::SamplingParams sampling;
    std::int32_t context_budget_tokens = 96000;
    std::int32_t max_new_tokens = 4096;
    // Checkpoint ceiling (text_config.max_position_embeddings). 0 means unknown -- tests
    // and ScriptedBackend runs leave it unset and skip the hard refuse. When set,
    // context_budget_tokens is clamped to it and a turn whose prompt + reserved
    // generation would overflow is refused rather than sent to the backend.
    std::int32_t model_max_sequence_tokens = 0;
    // Thinking is capped separately so a model that ruminates cannot LengthCap mid-write
    // and leave tool XML with no remaining budget. reserved_tool_tokens is the floor kept
    // for tool-call structure after think ends (forced or natural).
    std::int32_t max_think_tokens = 2048;
    std::int32_t reserved_tool_tokens = 1024;
    std::uint64_t seed = 0;

    // --- autonomy -----------------------------------------------------------
    //
    // How much the operator wants to be asked. Defaults are today's behaviour, so an
    // absent setting changes nothing; each one below only ever LOOSENS on an explicit
    // request and only ever TIGHTENS by default (S13).

    // -1 keeps the mode's own tier. Otherwise this wins -- except in Plan mode, which
    // pins 0 whatever is asked for: a mode that cannot execute cannot be talked into
    // executing by a settings field.
    int sandbox_tier_override = -1;

    // false makes every command execution raise a card whatever its risk score. The
    // default keeps risk routing for ordinary commands; irreversible / Unparseable /
    // opaque-script properties still escalate after this switch.
    bool auto_approve_exec = true;

    // false makes every workspace-mutating call raise a card. Writes had no HITL path at
    // all before: mode policy decided whether they were allowed, and nothing asked about
    // any individual one.
    bool auto_approve_writes = true;

    // Commands the operator has already said yes to. This is the half that makes the
    // strict half survivable: without somewhere for "yes, and stop asking me about
    // pytest" to go, the only way to escape card fatigue is to turn approvals off
    // entirely -- which is how a harness ends up with nothing between the model and
    // `rm -rf` again.
    std::vector<std::string> allowed_commands;

    // Run the language's syntax check after every successful workspace write and hand the
    // diagnostic back as part of that call's observation. Default on: this only ever
    // tightens, and S13's rule is that defaults tighten while explicit requests loosen.
    bool auto_syntax_check = true;

    // THE ONLY VERIFICATION THIS HARNESS RUNS. Operator-owned; the model cannot set,
    // replace or argue with it. When non-empty, the command runs verbatim after any turn
    // that wrote a file, its output and exit status become an observation the model
    // reads, and the final `completed` claim requires its last reading to have passed.
    // Empty means no check: the run ends when the model answers, and `completed` reports
    // only that it did.
    std::string operator_verify_contract;
};

// The UI feed. The Agent emits structured facts; the sidecar serializes them with the
// GENERATED protocol serializers. Injected rather than reached for, so the scripted
// loop suite can assert on exactly what a run would have told the user.
struct Observer {
    std::function<void(const std::string& channel, const std::string& text)> on_token;
    std::function<void(const TurnResult&, double duration_ms)> on_turn;
    std::function<void(const context::CheckResult&)> on_verification;
    // `compactions` is how many times this run has had to trim its own context. The
    // surface renders it beside the context meter: a run that is compacting is one whose
    // history is being thrown away, and that is worth seeing BEFORE the answers get worse.
    std::function<void(const model::GenResult&, std::size_t ctx_used, std::size_t ctx_max,
                       std::size_t compactions)>
        on_perf;
    // Fired whenever the checklist CHANGES, which is the only time it is news. The
    // sidebar had a Checklist panel and nothing ever filled it: `lmp/checklist` was
    // declared in the schema, generated on both sides, and emitted by nobody.
    std::function<void(const std::vector<context::ChecklistItem>&)> on_checklist;
    // The plan a conversational run is handing over, once, as the run ends. Its own
    // callback and not the answer channel, because unlike a question this is not prose to
    // read and reply to -- the surface has to offer a decision on it, and the text becomes
    // the mission of the run that implements it.
    std::function<void(const std::string& plan)> on_plan_ready;
};

struct RunReport {
    // The one unambiguous signal for WHICH ENDING a run took (S14). Exactly eight:
    // `ended` (the model answered in text), `awaiting_user`, `plan_ready`, `max_turns`,
    // `stalled_no_turn`, `wall_clock`, `cancelled`, `backend_error`.
    //
    // `stalled_no_turn` and `wall_clock` are deliberately not one reason. The first says
    // no turn completed inside Budget::stall_seconds -- the run is wedged. The second says
    // the run outlived Budget::wall_clock_seconds while still completing turns -- it was
    // working, and hit the stop of last resort. They were one reason until 2026-08-18, and
    // reading a productive run's ending as a hang is what sent the operator back to the
    // keyboard to type `continue`.
    std::string termination_reason;
    int iterations = 0;
    // The model answered AND, when an operator check is configured, its last reading
    // passed. This is a report of two observed facts, not a verdict: with no check
    // configured it means only "the model said it was done".
    bool completed = false;
    std::size_t compactions = 0;
    // Checklist items still open at the end. Reported, never enforced: the list is the
    // model's own progress display and holds no authority over the ending.
    std::size_t unfinished_items = 0;
    // Instructions that arrived mid-run and were taken up at a turn boundary.
    std::size_t steers_received = 0;
};

class Agent {
  public:
    Agent(const model::QwenTokenizer& tok, model::InferenceBackend& backend,
          tools::Registry& registry, context::ContextStore& ctx,
          platform::EventLogWriter& log, const platform::Clock& clock, AgentConfig config);

    void set_approver(Approver a) { approver_ = std::move(a); }
    void set_observer(Observer o) { observer_ = std::move(o); }
    void set_steer_source(SteerSource s) { steer_ = std::move(s); }

    // Something outside the native write ledger may have changed the workspace (editor
    // file-change notification, operator edit). Bumps the freshness epoch the repeat
    // detector keys on so the next observation revalidates.
    void note_external_workspace_change() noexcept {
        ctx_.invalidate_workspace_freshness();
    }

    [[nodiscard]] RunReport run(const model::CancelToken& cancel);

    // One iteration, exposed so the scripted-loop suite can assert per-turn (S11.2).
    [[nodiscard]] TurnResult step(const model::CancelToken& cancel);

    // The <tools> block this run will actually send, AFTER the mode has filtered it.
    //
    // Exposed for the gate: "plan mode does not permit a write" is testable through the
    // refusal, but "plan mode never OFFERS a write" is the half that decides whether the
    // model wastes turns discovering it, and it is unobservable from the outside without
    // this. The two are asserted together in tests/loop/test_agent_step.cpp.
    [[nodiscard]] const std::string& tools_guidance() const noexcept {
        return tools_guidance_;
    }

  private:
    // Never trim below this many verbatim turns, whatever the budget says: a run that
    // cannot see its own last few observations cannot make a next move.
    static constexpr std::size_t kMinRecentTurns = 4;

    // When a trim starts, and where it trims to, as percentages of the context budget.
    //
    // Two marks rather than one because trimming AT the budget leaves no slack: the next
    // turn's observation puts the prompt straight back over, so every turn from then on
    // pays a compaction. A trim has to buy room for several turns, and the run must never
    // render a prompt at the very edge of the budget when the model is about to add a
    // 4k-token generation to it.
    //
    // THE LOW MARK MOVED 55 -> 35 ON 2026-08-19, AND IT IS THE SINGLE LARGEST WALL-CLOCK
    // LEVER IN THE HARNESS. What justifies it is a measurement nobody had taken: a
    // compaction invalidates the KV prefix, so the turn after one is a FULL re-prefill --
    // 48 of 48 in the event log reused zero tokens, at a median TTFT of 43.5 s against
    // 1.8 s for an ordinary turn. That is not a tail event. It was 5.45 s of every
    // 11.46 s per turn, i.e. 48% of the run's wall clock, spent re-reading context the
    // model had already been shown.
    //
    // The arithmetic, with every input measured from real runs rather than assumed:
    //
    //     amortized cost per turn = (low / (high - low)) * growth_per_turn / prefill_rate
    //
    // The BUDGET CANCELS. Both the size of the re-prefill and the number of turns a trim
    // buys scale with it, so no `contextBudgetTokens` setting can touch this -- raising
    // the budget buys longer cycles at exactly proportionally higher cost, and lowering it
    // the reverse. Only the ratio moves it. At growth 2,323 tok/turn and a re-prefill rate
    // of 1,172 tok/s (both measured on the A3B at ~50k):
    //
    //     75/55 -> 5.45 s/turn   (8.3 turns per cycle)   <- was here
    //     75/45 -> 2.97 s/turn  (12.4)
    //     75/35 -> 1.73 s/turn  (16.5)                   <- now here
    //
    // Trimming deeper wins twice: the re-prefill is smaller AND it buys twice as many
    // turns before the next one.
    //
    // What it costs, and why the margin covers it: discarded context gets re-read later at
    // the cost of a turn (S8.3), which is why this mark was conservative. A re-read costs
    // ~11.5 s; the change saves ~3.7 s/turn over a 16.5-turn cycle, about 61 s. So it pays
    // for roughly five extra re-reads per cycle before it breaks even, and the PCC store
    // exists precisely so that discarded context is recoverable without one.
    //
    // THAT MARGIN IS MODELLED, NOT MEASURED. The speed side is measured; whether trimming
    // to 35% makes the model re-read more is a question only real runs answer. If runs get
    // worse, this is the first constant to put back -- move it, do not delete the note.
    static constexpr std::size_t kCompactAtPercent = 75;
    static constexpr std::size_t kCompactToPercent = 35;
    // kCollapseAtPercent IS GONE, and no percentage replaces it. It asked "is this
    // collapse's saving worth a full re-prefill", and the answer measured on 54 real
    // collapses was no, at every value the constant could have taken: a prompt is
    // compacted at render once it reaches kCompactAtPercent, so the mark could never be
    // set high enough to coincide with a compaction, and below that it only ever bought a
    // 58.8 s re-prefill to reclaim a few seconds' worth of deferred trimming. The collapse
    // now waits for compact_to_budget, where the prefill is already spent and it costs
    // nothing. See collapse_duplicate_read() for the full measurement.

    void emit(const std::string& kind, std::vector<platform::EventField> fields);
    // The HITL gate: mode policy, then writes, then commands. Returns the refusal when a
    // call must not run, and nothing when it may. Defined in approval.cpp with the pure
    // routing functions it drives -- the loop file stays about the loop.
    [[nodiscard]] std::optional<tools::ToolResult> gate_call(
        const tools::ToolDecl* decl, const std::string& name,
        const std::vector<tools::ToolParamValue>& params);
    // Whether the command's truncating redirects all land on this run's own output, which
    // makes the classifier's `destroys_data` a statement about a file the run itself wrote.
    // A member because it needs both halves the classifier lacks: run_wrote_ and a
    // filesystem. See the definition in approval.cpp.
    [[nodiscard]] bool redirects_only_own_output(const std::string& command,
                                                 const tools::RiskHint& hint) const;
    // May this mode call this tool AT ALL? Asked TWICE, deliberately: here, to decide what
    // the model is told it has, and again inside gate_call, to decide what it may do.
    // Filtering is not defence in depth for its own sake; it is the difference between a
    // mode and a series of accidents.
    [[nodiscard]] bool tool_allowed(const tools::ToolDecl& decl) const;
    // The registry's spec set minus what this mode withholds -- computed once in the
    // constructor because the mode is fixed at lmp/start and so this is a run constant,
    // which is also what keeps the KV prefix stable (S6.4).
    [[nodiscard]] const std::vector<parsephony::ToolSpec>& mode_specs() const noexcept {
        return mode_specs_;
    }
    // Appends the post-edit syntax verdict to `result`'s summary. Empty when there is no
    // contract for the path, when the check could not run, or when it came back clean --
    // silence is the default, because a per-turn "no checker for .md" is prompt noise.
    void annotate_with_syntax_check(const std::string& path, tools::ToolResult& result);
    // The prompt step() would send right now, in tokens. One function so the budget check
    // and the context meter cannot measure a different prompt from the one that is sent.
    [[nodiscard]] std::size_t prompt_tokens() const;
    void compact_to_budget();
    // Drains the steer source into the context. Returns how many instructions landed.
    [[nodiscard]] std::size_t take_steering();
    [[nodiscard]] TurnResult::PlanOutcome apply_plan(
        const std::vector<tools::ToolParamValue>& params);
    // Runs the operator's check verbatim through the tool shell and puts its OUTPUT --
    // not just a verdict -- in front of the model as an observation. The only
    // verification in the harness; `why` lands in the event so a post-write run and the
    // final completion reading are distinguishable in the trace.
    void run_operator_check(const char* why);
    // Whether `tool` is how the run changes the workspace -- asked of the registry rather
    // than matched against a name list, so a tool added later is covered by construction.
    [[nodiscard]] bool mutates_workspace(const std::string& tool) const;
    [[nodiscard]] tools::ToolResult dispatch_call(
        const std::string& name, const std::vector<tools::ToolParamValue>& params,
        bool& executed);
    // Collapses an earlier byte-identical copy of this read's result, leaving the newest
    // one live. Called after a successful content read; never refuses or alters the result
    // the model receives.
    // A collapse the run has decided is worth doing but has NOT applied yet. Held until
    // compact_to_budget() is already rewriting history -- see collapse_duplicate_read().
    // One entry per path: a later read of the same file supersedes the earlier plan, and
    // its content is the one every stale copy is measured against.
    struct PendingCollapse {
        std::string tool;
        std::string path;
        std::string range;
        std::string current;
    };
    std::vector<PendingCollapse> pending_collapses_;
    // Applies every pending collapse and clears the list. Called ONLY from
    // compact_to_budget, and only once the trim is known to be happening.
    std::size_t apply_pending_collapses();
    std::size_t apply_one_collapse(const PendingCollapse& pc);

    // Replaces a byte-identical re-read's payload with a reference to the copy the
    // context already holds. Returns whether it did. An append, never a rewrite.
    [[nodiscard]] bool elide_redundant_reread(const std::string& name,
                                              const std::vector<tools::ToolParamValue>& params,
                                              tools::ToolResult& result);

    void collapse_duplicate_read(const std::string& name,
                                 const std::vector<tools::ToolParamValue>& params,
                                 const tools::ToolResult& result);
    // Whether a batched call may be executed off the agent thread, and the serial tail
    // such a call still owes. See parallel_calls.hpp.
    // Decode + preprocess an image for the prompt, and fingerprint its pixels for the KV
    // ledger. False when the file cannot be read -- the caller degrades to a note in the
    // text rather than failing the turn. See the definition for why the hash is over the
    // pixels and not the path.
    // Fills in message image token counts (decoding to learn them) and optionally hands
    // back the pixels and their content hashes, in render order. Used by BOTH render
    // sites so the prompt step() sends and the prompt prompt_tokens() measures cannot
    // diverge.
    void resolve_images(std::vector<model::Message>& messages,
                        std::vector<model::PreprocessedImage>* pixels,
                        std::vector<std::uint64_t>* hashes) const;

    [[nodiscard]] bool load_image_for_prompt(const std::string& path,
                                             model::PreprocessedImage& out,
                                             std::uint64_t& content_hash) const;

    [[nodiscard]] bool can_run_in_parallel(const std::string& name) const;
    [[nodiscard]] bool adopt_readonly_result(const std::string& name,
                                             const std::vector<tools::ToolParamValue>& params,
                                             tools::ToolResult& result);
    // Did this call's bytes differ from what the same call returned last time? Must be
    // asked BEFORE record_call() folds the result into the detector. One definition for
    // both dispatch paths and for the loop's progress measure.
    [[nodiscard]] bool observation_is_new(const std::string& name,
                                          const std::vector<tools::ToolParamValue>& params,
                                          const tools::ToolResult& result) const;
    // Did this turn move the run forward -- write bytes, or learn something? The inverse
    // is an INERT turn, and consecutive inert turns are what ends a run.
    [[nodiscard]] static bool turn_made_progress(const TurnResult& turn) noexcept;
    // Records one executed call in the repeat cache, with what it returned and where the
    // write counter stood. One helper so the primary and the batched calls cannot be
    // recorded differently -- the batching hole in the old detector was measured: three
    // reads per turn invisible to it, forever.
    void record_call(const std::string& tool,
                     const std::vector<tools::ToolParamValue>& params,
                     const tools::ToolResult& result);

    const model::QwenTokenizer& tok_;
    model::InferenceBackend& backend_;
    tools::Registry& registry_;
    context::ContextStore& ctx_;
    platform::EventLogWriter& log_;
    const platform::Clock& clock_;
    AgentConfig config_;
    ModePolicy policy_;
    std::vector<parsephony::ToolSpec> mode_specs_;
    RepeatDetector repeats_;
    Approver approver_;
    Observer observer_;
    SteerSource steer_;
    std::unique_ptr<tools::SyntaxChecker> syntax_;
    // Whether each path's syntax check was already failing BEFORE this run first edited
    // it. A red that was already red is not evidence about the edit.
    std::map<std::string, bool> pre_edit_clean_;
    std::string tools_guidance_;
    // Tokens in the prompt step() actually sent this turn. Set during prompt assembly,
    // where the tokenizer has just produced it; read by the duplicate collapse, which pays
    // a full re-prefill and so must know whether the context is short of room first.
    std::size_t last_prompt_tokens_ = 0;
    // Paths this run has written. A whole-file rewrite of one of them is the run editing
    // its OWN output, not destroying the operator's data -- see the approval gate.
    std::set<std::string> run_wrote_;
    // The budget note is a single fact, delivered once.
    bool budget_note_sent_ = false;
    bool halted_ = false;
    std::string halt_reason_;
    // Stuck-run signals for operator_check (observations only; not a tool lock).
    std::size_t could_not_run_streak_ = 0;
    std::size_t same_diag_streak_ = 0;
    std::string last_primary_diag_fp_;
    // How many CONSECUTIVE INERT TURNS each mode nudges before it accepts the ending.
    //
    // WHAT CHANGED, AND WHY IT HAD TO. This counted consecutive TEXT-ONLY turns and reset
    // on any executed tool call. Both halves were wrong, in opposite directions, and one
    // run showed both:
    //
    //   * A run that repeated itself forever was INVISIBLE. Its shape was
    //     `text, read, read, text, read, read, ...` where every read came back
    //     byte-identical -- and each read reset the counter, so the ending could never
    //     fire. It ended at turn 22 of 200 only because two text turns happened to land
    //     back to back. Half the run produced nothing and 25% of the final prompt was five
    //     copies of one file.
    //   * A run that was plainly WORKING was killed. Three re-runs of the same mission all
    //     died on two narration turns in a row, one with 23 turns of budget left and edits
    //     still landing.
    //
    // The counter now measures PROGRESS instead of prose. A turn is inert when it wrote no
    // bytes AND produced no observation the run did not already have -- which makes a
    // text-only turn one case of inertness rather than the only thing watched. Anything
    // that writes or learns resets it. See turn_made_progress().
    //
    // AN IMPLEMENTATION RUN GETS THREE, up from one. One was tuned against a counter that
    // reset on any tool call, where a single call proved nothing; against this counter
    // three consecutive turns of neither writing nor learning is a real stall, and the
    // three re-runs above show that fewer kills working runs. The cost is two extra turns
    // on a run that genuinely finished, which is the price of not ending one that had not.
    //
    // Plan mode keeps two, unchanged. There a text turn is never the wanted output, the
    // existing value was already measured against a run ended at turn 12 with no plan and
    // no question, and plan mode is the mode currently working -- so it inherits the
    // better signal without a retuning it did not ask for.
    static constexpr std::size_t kPlanNudgesBeforeEnding = 2;
    static constexpr std::size_t kRunNudgesBeforeEnding = 3;
    std::size_t inert_turns_ = 0;
    // Did the current inert streak contain a tool call that ran and achieved nothing? It
    // separates the two endings: a run that only narrated is HANDING BACK (`ended`), and
    // one that kept calling tools to no effect is STALLED. Same count, different fact,
    // and the operator should not have to guess which they got.
    bool inert_streak_had_tool_call_ = false;
    // Consecutive turns that called only `plan` and nothing else. A run that restates
    // its checklist turn after turn is not making progress, and the inert counter alone
    // cannot see it because any real edit between two `plan` calls resets the streak.
    // Measured: 9 plan calls in a 39-turn run, each one resetting the clock on the
    // thrash it was meant to interrupt.
    std::size_t consecutive_plan_only_turns_ = 0;
    // Consecutive turns whose only write was a small edit. A run that makes many tiny
    // edits without ever closing a checklist item or running a build is polishing, not
    // finishing; the counter exists to say so before the turn budget does.
    std::size_t consecutive_micro_edit_turns_ = 0;
    std::size_t executed_tool_calls_in_run_ = 0;
    // Generations attempted this run, and the only input other than config_.seed to the
    // per-turn sampler seed. Counts ATTEMPTS rather than completed turns -- a turn refused
    // before it reached the backend leaves a gap in the sequence, which is harmless, and
    // the alternative (only counting successes) would hand two different turns the same
    // seed after any refusal. See seed_for_turn in agent.cpp.
    std::uint64_t turns_generated_ = 0;
};

} // namespace lmp::loop
