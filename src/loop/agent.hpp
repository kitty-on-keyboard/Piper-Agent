#pragma once
//
// Agent -- the ReAct loop (spec S9).
//
// Structured so that every phase boundary is an event (S14) and every corrective is a
// mechanism (S9.2). The backend is injected, so the identical loop runs under
// ScriptedBackend in the gate and MlxBackend on the real model -- which is the whole
// point of the seam (S2.1.1).
//
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>

#include "src/context/context.hpp"
#include "src/loop/turn.hpp"
#include "src/loop/verification.hpp"
#include "src/model/backend.hpp"
#include "src/model/chat_template.hpp"
#include "src/model/grammar.hpp"
#include "src/model/qwen_tokenizer.hpp"
#include "src/platform/clock.hpp"
#include "src/platform/event_log.hpp"
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
// score and above the allowlist.
//
// The score cannot express this and should not be asked to. `rm -rf` carries exactly one
// capability, so it scores 0.30 against a 0.35 auto-approve threshold and never raised a
// card at all: the agent was told to delete every file in a workspace and did, on a run
// configured to deny every approval, because no approval was ever requested. Nudging the
// weight to 0.36 fixes that one command until some other combination lands under the bar.
//
// Irreversibility is a PROPERTY, not a quantity. Everything else stays scored.
[[nodiscard]] bool is_irreversible(const tools::RiskHint& hint) noexcept;

// Whether the operator has already said yes to this exact command.
//
// Matches on equality, or on `entry` followed by a space -- and NEVER on a command
// carrying shell chaining (`;` `&&` `||` `|` backtick `$(` `>` `<` `&`), because
// allowlisting `python3 -m pytest` must not authorise `python3 -m pytest; rm -rf ~`.
// A chained command falls through to normal routing rather than being refused: it may
// be perfectly ordinary, it just cannot be waved through on a prefix.
[[nodiscard]] bool is_allowlisted(const std::string& command,
                                  const std::vector<std::string>& allowed);

// One line naming the call and its arguments, each argument truncated for display. Shared
// with the approval gate in approval.cpp, which puts it on the card.
[[nodiscard]] std::string preview_of(const std::string& tool,
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
    // default keeps risk routing, where only the 0.35-0.85 band escalates.
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

    // THE OPERATOR'S completion criterion. When set it becomes the contract, `plan` may
    // not replace it, and a complete verdict is a claim about the mission rather than
    // about what the model decided to check. Empty leaves the model to declare one, which
    // is what every run did before this existed -- and is why `rename_across_files` could
    // report completed=yes verified=yes while the mission was unmet.
    std::string operator_verify_contract;
};

// The UI feed. The Agent emits structured facts; the sidecar serializes them with the
// GENERATED protocol serializers. Injected rather than reached for, so the scripted
// loop suite can assert on exactly what a run would have told the user.
struct Observer {
    std::function<void(const std::string& channel, const std::string& text)> on_token;
    std::function<void(const TurnResult&, double duration_ms)> on_turn;
    std::function<void(const context::VerificationRecord&)> on_verification;
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
};

struct RunReport {
    // The one unambiguous signal for WHICH ENDING a run took (S14).
    std::string termination_reason;
    int iterations = 0;
    bool completed = false;
    std::size_t compactions = 0;
    // Checklist items still open at the end. Reported, never enforced -- see
    // CompletionVerdict.
    std::size_t unfinished_items = 0;
    // Instructions that arrived mid-run and were taken up at a turn boundary.
    std::size_t steers_received = 0;
    // True when `completed` rests on a contract the MODEL chose rather than one the
    // operator set. Not a failure and not enforced -- most runs have no operator contract
    // -- but it is the difference between "the mission is met" and "the criterion the
    // model picked for itself is met", and those were previously the same word.
    bool self_declared = false;
};

class Agent {
  public:
    Agent(const model::QwenTokenizer& tok, model::InferenceBackend& backend,
          tools::Registry& registry, context::ContextStore& ctx,
          platform::EventLogWriter& log, const platform::Clock& clock, AgentConfig config);

    void set_approver(Approver a) { approver_ = std::move(a); }
    void set_observer(Observer o) { observer_ = std::move(o); }
    void set_steer_source(SteerSource s) { steer_ = std::move(s); }

    [[nodiscard]] RunReport run(const model::CancelToken& cancel);

    // One iteration, exposed so the scripted-loop suite can assert per-turn (S11.2).
    [[nodiscard]] TurnResult step(const model::CancelToken& cancel);

  private:
    // Never trim below this many verbatim turns, whatever the budget says: a run that
    // cannot see its own last few observations cannot make a next move.
    static constexpr std::size_t kMinRecentTurns = 4;

    // Consecutive turns that executed nothing before a run is declared stalled. Three
    // is enough to let the model think out loud between calls, and few enough that
    // neither narration nor repeated token-cap truncation can burn the whole wall clock.
    static constexpr int kMaxConsecutiveNoProgress = 3;

    void emit(const std::string& kind, std::vector<platform::EventField> fields);
    // Logs and surfaces every ledger entry added since `before`. The one place a
    // verification reaches either the log or the UI, so the two cannot disagree.
    void emit_verifications(std::size_t before);
    // The HITL gate: mode policy, then writes, then commands. Returns the refusal when a
    // call must not run, and nothing when it may. Defined in approval.cpp with the pure
    // routing functions it drives -- the loop file stays about the loop.
    [[nodiscard]] std::optional<tools::ToolResult> gate_call(
        const tools::ToolDecl* decl, const std::string& name,
        const std::vector<tools::ToolParamValue>& params);
    // Appends the post-edit syntax verdict to `result`'s summary. Empty when there is no
    // contract for the path, when the check could not run, or when it came back clean --
    // silence is the default, because a per-turn "no checker for .md" is prompt noise.
    void annotate_with_syntax_check(const std::string& path, tools::ToolResult& result);
    void compact_to_budget();
    // Drains the steer source into the context. Returns how many instructions landed.
    [[nodiscard]] std::size_t take_steering();
    [[nodiscard]] TurnResult::PlanOutcome apply_plan(
        const std::vector<tools::ToolParamValue>& params);
    [[nodiscard]] std::string baseline_check();
    [[nodiscard]] tools::ToolResult dispatch_call(
        const std::string& name, const std::vector<tools::ToolParamValue>& params,
        bool& executed);
    // Whether a batched call may be executed off the agent thread, and the serial tail
    // such a call still owes. See parallel_calls.hpp.
    [[nodiscard]] bool can_run_in_parallel(const std::string& name) const;
    [[nodiscard]] bool adopt_readonly_result(const std::string& name,
                                             const tools::ToolResult& result);
    void apply_corrective(Corrective c, const TurnResult& turn);

    const model::QwenTokenizer& tok_;
    model::InferenceBackend& backend_;
    tools::Registry& registry_;
    context::ContextStore& ctx_;
    platform::EventLogWriter& log_;
    const platform::Clock& clock_;
    AgentConfig config_;
    ModePolicy policy_;
    RepeatDetector repeats_;
    RefusalLedger refusals_;
    Approver approver_;
    Observer observer_;
    SteerSource steer_;
    Verifier verifier_;
    std::unique_ptr<tools::SyntaxChecker> syntax_;
    // Whether each path's syntax check was already failing BEFORE this run first edited
    // it. A red that was already red is not evidence about the edit -- the same
    // FAIL_TO_PASS reasoning verification.cpp applies to the declared contract.
    std::map<std::string, bool> pre_edit_clean_;
    std::string tools_guidance_;
    // The last completion verdict logged, so the reason is emitted when it changes rather
    // than once per turn. See the `not_complete` emit in run().
    std::string last_incomplete_reason_;
    // Set by BreakRepeat, consumed by the next step(): the tool that just repeated is
    // dropped from that turn's grammar, which is the "narrow the registry" half of the
    // corrective that was declared but never implemented.
    std::string suppress_tool_next_turn_;
    int consecutive_no_progress_ = 0;
    bool halted_ = false;
    std::string halt_reason_;
};

} // namespace lmp::loop
