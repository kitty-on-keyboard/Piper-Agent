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
#include <set>
#include <utility>
#include <vector>
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
    // The plan a conversational run is handing over, once, as the run ends. Its own
    // callback and not the answer channel, because unlike a question this is not prose to
    // read and reply to -- the surface has to offer a decision on it, and the text becomes
    // the mission of the run that implements it.
    std::function<void(const std::string& plan)> on_plan_ready;
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

    // Consecutive turns that executed nothing before a run is declared stalled. Three
    // is enough to let the model think out loud between calls, and few enough that
    // neither narration nor repeated token-cap truncation can burn the whole wall clock.
    static constexpr int kMaxConsecutiveNoProgress = 3;

    // When a trim starts, and where it trims to, as percentages of the context budget.
    //
    // Two marks rather than one because trimming AT the budget leaves no slack: the next
    // turn's observation puts the prompt straight back over, so every turn from then on
    // pays a compaction. Starting at 75% and going to 55% means a trim buys room for
    // several turns, and the run is never rendering a prompt at the very edge of the
    // budget when the model is about to add a 4k-token generation to it.
    static constexpr std::size_t kCompactAtPercent = 75;
    static constexpr std::size_t kCompactToPercent = 55;
    // Where the duplicate collapse starts paying for itself, as a percentage of the same
    // budget. Deliberately equal to the low-water mark and deliberately NOT the same
    // constant: this one answers "is the saving worth a full re-prefill", which is a
    // different question from "how far should a trim go", and the two must be free to move
    // apart. See collapse_duplicate_read() for the measurement -- a 22x TTFT penalty paid
    // 33 times in a run that never used more than a third of its context.
    static constexpr std::size_t kCollapseAtPercent = 55;

    // How long a repeated tool stays out of the grammar, at most. The window is the
    // repeat count minus one -- second sighting costs one turn, third costs two -- so a
    // tool the run keeps coming back to is held longer each time, and a ping-pong runs out
    // of partners. Capped because a repeated tool is usually the RIGHT tool with wrong
    // arguments, and taking `read_file` away for a dozen turns would end the run.
    static constexpr int kMaxSuppressTurns = 4;

    // The ceiling on the ESCALATED window, once three ineffective firings have falsified
    // the assumption above for this run. The escalation used to be uncapped
    // (`hits * kMaxSuppressTurns`, with `hits` unbounded), which is how a real run came to
    // have `read_file` held for 20 consecutive turns and `list_dir` for 12.
    //
    // A hold long enough to outlast the work is not a corrective, it is an ending. Eight
    // turns is long enough that waiting it out is not a strategy and short enough that a run
    // which genuinely needs the tool is not finished off by it.
    static constexpr int kMaxEscalatedSuppressTurns = 8;

    // How many workspace-changing writes may pile up before the run is made to check its
    // own work. See Corrective::ForceVerification.
    //
    // Three, because two is a model fixing two errors it read off one compiler run -- the
    // ordinary shape of debugging, and taking a build from it would be pure tax -- while
    // the third edit is the first one that cannot have been informed by anything the run
    // has actually observed. The forced check clears the count, so the steady state of a
    // healthy run is one verification per three edits, which is roughly what a careful
    // human does anyway.
    //
    // MEASURED: the run this comes from put FIFTEEN writes between two builds.
    static constexpr int kMaxUnverifiedWrites = 3;

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
    // May this mode call this tool AT ALL? Asked TWICE, deliberately: here, to decide what
    // the model is told it has, and again inside gate_call, to decide what it may do.
    //
    // The gate alone was the whole of mode policy and it is not enough. A model shown
    // write_file in plan mode reaches for it, is refused, and has spent a turn learning
    // something the prompt could have told it for nothing -- and after two such refusals
    // BlockRefusedTool drops the tool from the grammar and attributes it in the trace as
    // if the OPERATOR had denied it. Filtering is not defence in depth for its own sake;
    // it is the difference between a mode and a series of accidents.
    [[nodiscard]] bool tool_allowed(const tools::ToolDecl& decl) const;
    // The registry's spec set minus what this mode withholds -- the BASE the per-turn
    // narrowings start from. Computed once in the constructor because the mode is fixed at
    // lmp/start and so this is a run constant, which is also what keeps the KV prefix
    // stable (S6.4).
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
    [[nodiscard]] std::string baseline_check();
    // Workspace-changing writes made since the last verification of any kind ran.
    //
    // DERIVED, never counted. Every VerificationRecord already stores the write total at
    // the moment it ran, so this is one subtraction against state that cannot drift out of
    // step with itself -- and a second counter that had to be reset from four call sites
    // is exactly the kind of bookkeeping that ends up reset from three.
    [[nodiscard]] std::size_t writes_since_verification() const;
    // Whether any call in this turn was a workspace mutation the run had already made.
    //
    // Routed into ForceVerification rather than BreakRepeat, and the distinction is the
    // point: taking the editor away from a model that has just re-sent an edit leaves it
    // with the same stale evidence and one fewer way to act on it. What it is missing is
    // not a different tool, it is a fresh reading of the code.
    [[nodiscard]] bool turn_repeated_a_mutation(const TurnResult& turn) const;
    // Runs the declared contract now and puts its OUTPUT, not just its verdict, in front
    // of the model. The mechanism behind Corrective::ForceVerification and behind the
    // first rung of the ineffectiveness ladder -- one implementation, so the two cannot
    // drift into verifying different things or reporting them differently.
    void run_contract_now(const char* why);
    // What to do when one corrective has been applied kIneffectiveAfter times against one
    // target and the run has not moved. Returns whether it took the turn's action; false
    // means no better lever exists here and the chosen mechanism should apply as normal.
    [[nodiscard]] bool escalate_ineffective(Corrective c, const std::string& target,
                                            std::size_t hits);
    // Whether `tool` is how the run changes the workspace -- asked of the registry rather
    // than matched against a name list, so a tool added later is covered by construction.
    //
    // Load-bearing for the suppression floor: a mutating tool is never withheld from the
    // grammar, because withholding all of them asks the run to fix a build with no way to
    // change a file. Measured cost of not having this: 34 of 85 turns with the write family
    // unsamplable, and a model driven into writing its source files through shell heredocs.
    [[nodiscard]] bool mutates_workspace(const std::string& tool) const;
    [[nodiscard]] tools::ToolResult dispatch_call(
        const std::string& name, const std::vector<tools::ToolParamValue>& params,
        bool& executed);
    // Collapses an earlier byte-identical copy of this read's result, leaving the newest
    // one live. Called after a successful content read; never refuses or alters the result
    // the model receives.
    void collapse_duplicate_read(const std::string& name,
                                 const std::vector<tools::ToolParamValue>& params,
                                 const tools::ToolResult& result);
    // Whether the declared contract has ever been seen to pass in this run.
    [[nodiscard]] bool contract_has_passed() const;
    // Whether a batched call may be executed off the agent thread, and the serial tail
    // such a call still owes. See parallel_calls.hpp.
    [[nodiscard]] bool can_run_in_parallel(const std::string& name) const;
    [[nodiscard]] bool adopt_readonly_result(const std::string& name,
                                             const std::vector<tools::ToolParamValue>& params,
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
    std::vector<parsephony::ToolSpec> mode_specs_;
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
    // Set by BreakRepeat, consumed by step(): tools that have repeated, each with the
    // number of turns it stays out of the grammar. That is the "narrow the registry" half
    // of the corrective, and it accumulates -- see without_suppressed() for why one tool
    // for one turn was not enough to break a ping-pong between two tools.
    std::vector<std::pair<std::string, int>> suppressed_tools_;
    // Content reads this turn, and how many of them returned bytes the prompt already
    // held. Reset in step(), written at dispatch, read once by run()'s no-progress test --
    // which could not otherwise tell a turn of work from a turn of re-reading, because
    // both call a tool and both come back ToolCallExecuted.
    std::size_t turn_reads_ = 0;
    std::size_t turn_reads_redundant_ = 0;
    // The same question asked about EVERY call in the turn rather than only the reads:
    // how many executed, and how many of those provably added nothing -- a read whose
    // bytes were already in the prompt, or a write the file did not need.
    //
    // This is what the no-progress test reads now, and the widening is the whole point.
    // The old test could only see a turn made entirely of redundant reads, so ONE write
    // anywhere in the turn took it out of the branch entirely -- and a stuck model writes
    // on almost every turn. A 73-turn run cancelled with nothing built held
    // `no_progress_streak` at 0 for 68 of those turns for exactly that reason.
    //
    // Counted rather than inferred from tool names: a shell call, a search, a list_dir
    // increment `turn_calls_` and never `turn_inert_calls_`, so their presence takes the
    // turn out of the test without anything here having to know what they are.
    std::size_t turn_calls_ = 0;
    std::size_t turn_inert_calls_ = 0;
    // Tokens in the prompt step() actually sent this turn. Set during prompt assembly,
    // where the tokenizer has just produced it; read by the duplicate collapse, which pays
    // a full re-prefill and so must know whether the context is short of room first.
    std::size_t last_prompt_tokens_ = 0;
    // Paths this run has written. A whole-file rewrite of one of them is the run editing
    // its OWN output, not destroying the operator's data -- see the approval gate.
    std::set<std::string> run_wrote_;

    // WHAT USED TO BE HERE: `run_read_`, a per-path ledger of which reads this run had
    // already made, which REFUSED a re-read it believed was still in the prompt.
    //
    // The measurement that built it is still true and still worth acting on -- 45 turns
    // produced 65 turn records, 34 of them content reads against 11 writes, one 12.5 KB file
    // read ELEVEN times -- but refusing the read was the wrong lever, and the ledger could
    // not be made correct: it had to predict staleness from a path string, and it answered
    // slice requests from whole-file notes that never recorded how much of the answer
    // survived the byte cap. It cost a 66-turn run everything (see collapse_duplicate_read).
    //
    // The cost it was aimed at is now paid by collapsing the DUPLICATE copy instead of
    // withholding the answer, keyed on byte identity, which needs no ledger and no
    // invalidation rule: a file that changed produces different bytes and nothing collapses.
    // How many times each corrective has fired against the same target, so a mechanism
    // that is firing correctly and achieving nothing can be seen doing it. See the
    // `corrective_ineffective` emit: BreakRepeat suppressed one tool thirteen times in a
    // single run, each firing textbook-correct and none of them changing what happened
    // next. Keyed "corrective:target" because the same tool repeating under two different
    // correctives is two different findings.
    std::map<std::string, std::size_t> corrective_hits_;
    // Canonical contracts RederiveContract has already reported. Once each: the corrective
    // pins the grammar to `plan`, and re-arming it on a contract the run declined to change
    // would leave `plan` as the only legal move for the rest of the run.
    std::set<std::string> disputed_contracts_;
    // Turns of `plan`-only grammar owed to RederiveContract and ReconcileChecklist. Spent
    // in step(), never latched -- see those correctives.
    int replan_turns_ = 0;
    // Whether this run has already been asked to reconcile a green ledger against an
    // unticked list, and whether the checklist has stopped gating completion as a result.
    //
    // ONCE PER RUN, and asking is what spends it. The deleted `must_reconcile` restriction
    // re-derived its own precondition every turn and pinned `plan` for fourteen turns
    // straight; the difference is not the question, it is that this one is asked a single
    // time and then answered by whatever the run does next.
    //
    // The waiver is set in exactly two places: when the run answers the ask without
    // clearing its list AND then stops moving (run(), at the stall break), and never by
    // anything the model can say. A run that answers by DOING the work does not need it --
    // the list clears itself and completion goes through unanimous.
    bool reconcile_asked_ = false;
    bool checklist_waived_ = false;
    // A command that runs the declared contract's program a different way, PASSED, and did
    // so while the contract itself has never passed. The run's own evidence that its
    // criterion is measuring the wrong thing. Cleared when the corrective reports it.
    std::string passing_near_miss_;
    int consecutive_no_progress_ = 0;
    bool halted_ = false;
    std::string halt_reason_;
    // Which of the two budgets was spent when the run was cut off. Both used to end a run
    // as "budget_exhausted", so a run killed by the clock at 45 of 80 turns was
    // indistinguishable from one that ran out of turns -- and the obvious reading of that
    // word is the turn cap, so the wrong dial gets raised. Set in run(), read once by the
    // HaltOnBudget corrective.
    bool out_of_time_ = false;
};

} // namespace lmp::loop
