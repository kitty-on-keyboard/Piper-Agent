#include "src/loop/agent.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib> // getenv/atoi, for the LMP_TRACE_TEXT gate
#include <memory>

#include "src/loop/parallel_calls.hpp"
#include "src/loop/token_stream.hpp"

namespace lmp::loop {
namespace {

// The log records what the harness DID -- every prompt, every result -- and nothing the
// model SAID. That asymmetry is why a run that burned 40 turns without writing a file
// could not be diagnosed from its own trace: `generation tokens=224` followed by a turn
// with no tool_result says a turn produced nothing, and cannot say why.
//
// Off by default because a turn's reasoning is the largest thing in the run and the log
// is also the UI feed. `LMP_TRACE_TEXT=1` turns it on for a diagnostic run; the events
// go to the same writer as everything else, so the ordering against `prompt` and
// `tool_result` is the real one rather than two files to correlate by timestamp.
bool trace_text_enabled() {
    static const bool on = [] {
        const char* s = std::getenv("LMP_TRACE_TEXT");
        return s != nullptr && std::atoi(s) != 0;
    }();
    return on;
}

// Long enough to see a whole argument -- a truncated write_file is exactly the case
// where the interesting part is the end -- and bounded so one traced turn cannot be the
// whole log.
constexpr std::size_t kTraceFieldCap = 8192;

std::string capped(std::string s) {
    if (s.size() <= kTraceFieldCap) {
        return s;
    }
    s.resize(kTraceFieldCap);
    s += "\n[...truncated]";
    return s;
}

} // namespace

Agent::Agent(const model::QwenTokenizer& tok, model::InferenceBackend& backend,
             tools::Registry& registry, context::ContextStore& ctx,
             platform::EventLogWriter& log, const platform::Clock& clock,
             AgentConfig config)
    : tok_(tok), backend_(backend), registry_(registry), ctx_(ctx), log_(log),
      clock_(clock), config_(config), policy_(ModePolicy::for_mode(config.mode)),
      verifier_(registry, ctx) {
    tools_guidance_ = registry_.tools_json();

    // The operator's tier, when they named one. Plan mode is exempt in the one direction
    // that matters: it pins T0, so "no execution" cannot be undone by a settings field.
    if (config_.sandbox_tier_override >= 0 && config_.mode != Mode::Plan) {
        policy_.sandbox_tier = config_.sandbox_tier_override;
    }
    if (!config_.operator_verify_contract.empty()) {
        ctx_.set_verify_contract(config_.operator_verify_contract,
                                 context::ContextStore::ContractSource::Operator);
        emit("operator_contract", {{"contract", config_.operator_verify_contract}});
    }
    if (config_.auto_syntax_check) {
        syntax_ = std::make_unique<tools::SyntaxChecker>(registry_.workspace().root,
                                                         2048);
    }
    emit("policy", {{"mode", std::to_string(static_cast<int>(config_.mode))},
                    {"sandbox_tier", std::to_string(policy_.sandbox_tier)},
                    {"auto_approve_exec", config_.auto_approve_exec ? "1" : "0"},
                    {"auto_approve_writes", config_.auto_approve_writes ? "1" : "0"}});
}

// `plan` is declared by the registry but executed HERE: the checklist lives in the
// context store, which the registry has no business reaching into.
//
// Restating replaces the whole list, so ticking an item off is the same call as writing
// it -- one idempotent operation instead of a second tool and a synchronisation problem.
TurnResult::PlanOutcome Agent::apply_plan(const std::vector<tools::ToolParamValue>& params) {
    std::vector<context::ChecklistItem> items;
    const std::string* raw = nullptr;
    for (const auto& p : params) {
        if (p.name == "items") {
            raw = &p.value;
        } else if (p.name == "verify_with" && !p.value.empty()) {
            // Pinned in the store, not held here: a follow-up run builds a fresh Agent
            // over the SAME context, and a contract that lived in the Agent would be
            // silently lost between the two.
            //
            // Declared MODEL-sourced. When the operator supplied one, the store ignores
            // this -- otherwise a run could talk its way out of the criterion it was given
            // by restating its plan, which is the one move this whole gate exists to stop.
            ctx_.set_verify_contract(p.value, context::ContextStore::ContractSource::Model);
        }
    }
    if (raw == nullptr) {
        return {false, "plan requires 'items'"};
    }
    std::size_t at = 0;
    while (at < raw->size()) {
        std::size_t nl = raw->find('\n', at);
        if (nl == std::string::npos) {
            nl = raw->size();
        }
        std::string line = raw->substr(at, nl - at);
        at = nl + 1;
        // Tolerate a leading "- " and either bracket style; the model writes prose-ish
        // markdown and refusing it over a dash would be theatre.
        std::size_t i = line.find_first_not_of(" \t-*");
        if (i == std::string::npos) {
            continue;
        }
        bool done = false;
        if (line.compare(i, 3, "[x]") == 0 || line.compare(i, 3, "[X]") == 0) {
            done = true;
            i += 3;
        } else if (line.compare(i, 3, "[ ]") == 0) {
            i += 3;
        }
        const std::size_t text_at = line.find_first_not_of(" \t", i);
        if (text_at == std::string::npos) {
            continue;
        }
        items.push_back({line.substr(text_at), done});
    }
    if (items.empty()) {
        return {false, "plan produced no items; give one item per line"};
    }
    const std::size_t open = static_cast<std::size_t>(std::count_if(
        items.begin(), items.end(), [](const context::ChecklistItem& c) { return !c.done; }));
    const std::size_t total = items.size();
    ctx_.set_checklist(std::move(items));
    emit("plan", {{"items", std::to_string(total)},
                  {"open", std::to_string(open)},
                  {"verify_with", ctx_.verify_contract()}});
    if (observer_.on_checklist) {
        observer_.on_checklist(ctx_.checklist());
    }
    std::string s = "checklist set: " + std::to_string(total - open) + "/" +
                    std::to_string(total) + " done";
    if (!ctx_.verify_contract().empty()) {
        s += "; completion requires '" + ctx_.verify_contract() + "' to pass";
        s += baseline_check();
    }
    return {true, std::move(s)};
}

// Runs the declared contract ONCE, at the moment it is declared -- before any edit.
//
// This is pre-patch validation, the FAIL_TO_PASS baseline: if the check is red now and
// green later, that pair is the proof it can fail, captured for the price of one run and
// without reverting anything. Without it the common order of work (fix first, test after)
// never produces a red, so every green stays UNPROVEN and no run can ever complete.
//
// A baseline that comes back GREEN is a finding, not a failure: either the mission is
// already done, or the check does not exercise what the mission is about. Both are worth
// telling the model, and neither is worth pretending otherwise.
std::string Agent::baseline_check() {
    const std::string canon = canonicalize_check(ctx_.verify_contract());
    for (const context::VerificationRecord& v : ctx_.verifications()) {
        if (v.contract == canon) {
            return {}; // already have a reading for this contract
        }
    }
    const bool passed =
        verifier_.run_and_record_as(ctx_.verify_contract(), policy_.sandbox_tier, canon);
    emit("baseline_check", {{"contract", canon}, {"passed", passed ? "1" : "0"}});
    // A contract that could not be executed is neither the red this wants nor the green
    // it warns about, and calling it "FAILS, as expected" -- which is what the red branch
    // below would say -- actively misleads: it reads as confirmation that the criterion
    // is sound, when the criterion is the one thing that is broken.
    if (!ctx_.verifications().empty() && !ctx_.verifications().back().ran) {
        emit_verifications(ctx_.verifications().size() - 1);
        return "\nBaseline: that command could not be executed at all, so it is not a "
               "criterion yet -- it cannot fail and it cannot pass. Re-declare "
               "verify_with with a command that runs in this workspace, and check it "
               "runs before you declare it.";
    }
    emit_verifications(ctx_.verifications().empty() ? 0 : ctx_.verifications().size() - 1);
    // The green branch says what it COSTS, because the cost is the whole point and the
    // model cannot see the ledger. A green baseline leaves the contract unproven, and an
    // unproven green never completes a run (S10.2) -- so a run that declares its contract
    // after the tests already pass has, at that moment, made completion unreachable
    // without a deliberate proof.
    //
    // MEASURED: a run wrote the suite first and declared `python3 -m pytest ...` second.
    // Its baseline was green, and the harness said only "say which before doing anything
    // else". The run worked, its tests passed, and it spent the rest of its budget being
    // told it was not finished; it worked out what to do on turn 38 and the budget ended
    // at 40, mid-proof, with the code deliberately broken and never restored.
    return passed ? "\nBaseline: that command already PASSES, which means it is NOT yet "
                    "evidence -- a check that has never been seen to fail cannot finish a "
                    "run. Do this next, before any other work: break the behaviour it "
                    "covers, run the command and watch it go red, then restore what you "
                    "broke and run it again. That red-then-green is the proof. If it stays "
                    "green while broken, the check does not test the mission and you need "
                    "a different one."
                  : "\nBaseline: that command currently FAILS, as expected. Making it "
                    "pass is now provable evidence rather than an unproven green.";
}

// Non-model feedback on an edit, on the same observation the edit produced.
//
// Deliberately NOT routed through the Verifier: a syntax check is not the contract the run
// declared, and a green from it must never help a run complete (S10.1). The test for this
// asserts the verification ledger is unchanged across a checked edit, because the tidy
// implementation -- reuse run_and_record, it already exists -- would quietly make S10.4
// completion cheaper and nothing else would notice.
void Agent::annotate_with_syntax_check(const std::string& path, tools::ToolResult& result) {
    if (!config_.auto_syntax_check || !syntax_) {
        return;
    }
    const tools::SyntaxVerdict v = syntax_->check(path, policy_.sandbox_tier);
    if (!v.ran) {
        return; // no contract, or it could not be run: say nothing at all
    }
    const auto before = pre_edit_clean_.find(path);
    const bool was_clean = before == pre_edit_clean_.end() || before->second;
    emit("syntax_check",
         {{"path", path}, {"language", v.language}, {"clean", v.clean ? "1" : "0"}});
    if (v.clean) {
        return;
    }
    result.summary += "\n[syntax] " + v.language;
    // A red that was already red is a different fact and a different next move. Without
    // this the model gets told its edit broke a file that arrived broken.
    result.summary += was_clean ? ": FAILED\n" : ": still failing (it was already failing "
                                                 "before this edit)\n";
    result.summary += v.diagnostics;
    result.error_class = tools::ErrorClass::Malformed;
}

void Agent::emit(const std::string& kind, std::vector<platform::EventField> fields) {
    platform::Event ev;
    ev.kind = kind;
    ev.fields = std::move(fields);
    log_.append(ev, clock_);
}

// Every reading that joined the ledger since `before`, as events.
//
// The ledger is what completion turns on -- passed, falsifiable and seq are the three
// gates in evaluate_completion -- and until now none of it reached the log. A run that
// did the work, proved its check red and then green, and still ended
// `text_only_no_progress` could not be diagnosed from its own trace: the events showed
// the shell calls but not what the Verifier made of them. Emitted from the one place the
// records are already being walked, so the log and the surface cannot disagree.
void Agent::emit_verifications(std::size_t before) {
    const auto& vs = ctx_.verifications();
    for (std::size_t i = before; i < vs.size(); ++i) {
        emit("verification", {{"contract", vs[i].contract},
                              {"ran", vs[i].ran ? "1" : "0"},
                              {"passed", vs[i].passed ? "1" : "0"},
                              {"falsifiable", vs[i].falsifiable ? "1" : "0"},
                              {"seq", std::to_string(vs[i].seq)}});
        if (observer_.on_verification) {
            observer_.on_verification(vs[i]);
        }
    }
}

TurnResult Agent::step(const model::CancelToken& cancel) {
    TurnResult turn;

    // --- prompt assembly ---------------------------------------------------
    const model::ChatTemplate tmpl(tok_);
    const std::vector<model::Message> messages = ctx_.render("");
    model::InferenceTask task;
    // render_with_offsets, NOT a second render of a message sub-list: render() appends the
    // generation prompt, so the first k messages rendered alone are not a token prefix of
    // the whole. Asking for offsets is the only correct way to locate the boundary, and
    // getting it wrong reuses a cache against the wrong prefix without crashing (S5.10).
    std::vector<std::size_t> offsets;
    task.prompt = tmpl.render_with_offsets(messages, tools_guidance_, offsets);
    // Everything except the live-state block, which changes every turn. The backend
    // snapshots here so the next turn rolls back instead of re-prefilling the context.
    const std::size_t stable = ctx_.stable_message_count("");
    task.checkpoint_at = stable < offsets.size() ? offsets[stable] : 0;
    task.max_new_tokens = config_.max_new_tokens;
    task.sampling = config_.sampling;
    // config_.seed stays authoritative over the sampling block's own field: it is the
    // one the run is reproducible from.
    task.sampling.seed = config_.seed;

    // Every harness->model append is an event. This invariant is what makes "did the
    // model receive this?" answerable (S8.1, S14).
    emit("prompt", {{"tokens", std::to_string(task.prompt.size())},
                    {"messages", std::to_string(messages.size())},
                    {"compactions", std::to_string(ctx_.compaction_count())}});

    // --- constrained generation --------------------------------------------
    //
    // Until the run has a checklist, `plan` is the ONLY callable tool. That is a
    // mechanism, not a sentence asking the model to plan first (S9.2): the mask makes
    // every other call unsamplable, so a run cannot begin work it has not stated.
    //
    // Needed because the tool alone was not enough. With `plan` merely available and its
    // description saying to call it first, a real run ignored it for all 14 turns, so the
    // checklist stayed empty, no verification contract was ever declared, and completion
    // remained unreachable -- the same symptom as having no mechanism at all.
    //
    // `specs` must outlive `grammar`: TurnGrammar keeps a reference.
    //
    // There used to be a second restriction here, `must_reconcile`: once the declared
    // contract had passed provably and items were still open, `plan` became the only
    // callable tool so the run would tick its list. It existed only because completion
    // required every item ticked, and it DEADLOCKED the moment a run could be continued.
    //
    // Observed on the first real follow-up: the previous run's green satisfies "proven"
    // forever, the new instruction means items are open again, so the grammar allowed
    // nothing but `plan` -- and each `plan` call re-entered the same state. Fourteen
    // consecutive plan turns, no work, budget_exhausted.
    //
    // Deleted rather than repaired, because the case it was built for no longer exists:
    // a run that has fixed the bug and proved the fix now COMPLETES on that evidence
    // without needing the model to agree in checkbox form (S10.4).
    //
    // A steering message is different and still restricts: an instruction the run
    // acknowledges and then does not act on is indistinguishable from one it never
    // received. Making `plan` the only samplable call forces the next turn to restate the
    // checklist in the light of what it was just told (S9.2). It cannot loop, because
    // restating the checklist is exactly what clears the flag.
    const bool must_replan = ctx_.plan_is_stale();

    std::vector<parsephony::ToolSpec> specs;
    if (ctx_.checklist().empty() || must_replan) {
        for (const parsephony::ToolSpec& s : registry_.guard_specs()) {
            if (s.name == "plan") {
                specs.push_back(s);
            }
        }
    } else {
        specs = registry_.guard_specs();
    }
    specs = without_blocked(specs, refusals_, registry_.guard_specs());
    // The one-turn narrowing BreakRepeat asked for. Taken (and cleared) here rather than
    // held across turns, so it costs exactly the next turn and cannot accumulate into a
    // run that has quietly lost half its tools.
    //
    // Applied AFTER the plan gate, so a run that owes a checklist still gets `plan` even
    // if `plan` is what repeated -- otherwise a repeated plan would leave nothing callable
    // and the fallback below would hand `plan` straight back.
    specs = without_suppressed(specs, suppress_tool_next_turn_);
    suppress_tool_next_turn_.clear();
    model::TurnGrammar grammar(tok_, specs);
    task.mask = &grammar;

    // Reasoning is surfaced on its own channel, never inlined into the answer (S5.7).
    // The split happens by TOKEN ID upstream; the streamer only routes it, one token at a
    // time, on its own thread so a slow reader cannot throttle the decode loop.
    std::unique_ptr<TokenStreamer> streamer;
    if (observer_.on_token) {
        streamer = std::make_unique<TokenStreamer>(tok_, observer_.on_token);
    }
    GrammarSink sink(grammar, streamer.get());
    turn.generation = backend_.generate(task, sink, cancel);

    // Drained and joined BEFORE the text below is read, so what the surface showed and
    // what the transcript records cannot disagree about a turn that is already over.
    if (streamer) {
        streamer->finish();
    }

    // Still decoded in one piece for the transcript and the context store. The streamed
    // concatenation is byte-identical to these (test_token_stream asserts it), so this is
    // the same text, not a second opinion about it.
    turn.reasoning = tok_.decode(grammar.think_ids());
    turn.assistant_text = tok_.decode(grammar.text_ids());
    if (observer_.on_perf) {
        observer_.on_perf(turn.generation, task.prompt.size(),
                          static_cast<std::size_t>(config_.max_new_tokens) +
                              task.prompt.size(),
                          ctx_.compaction_count());
    }

    emit("generation", {{"status", std::to_string(static_cast<int>(turn.generation.status))},
                        {"tokens", std::to_string(turn.generation.tokens_generated)},
                        {"ttft_ms", std::to_string(turn.generation.ttft_ms)},
                        {"decode_tok_per_s", std::to_string(turn.generation.decode_tok_per_s)}});

    if (trace_text_enabled()) {
        emit("turn_text", {{"reasoning", capped(turn.reasoning)},
                           {"text", capped(turn.assistant_text)},
                           {"calls", std::to_string(grammar.tool_calls().size())}});
    }

    if (!grammar.has_tool_call()) {
        turn.outcome = classify_turn(turn.generation, grammar, false, false);
        return turn;
    }

    // --- the call(s) --------------------------------------------------------
    //
    // A turn may carry several calls (S9.1 amended: one turn, one OUTCOME, but the model
    // may batch independent work into it). The first call is the turn's outcome; the rest
    // execute in order and each gets its own history record. Reading four files used to
    // cost four full prefill+decode round-trips.
    const auto& calls = grammar.tool_calls();

    // Params up front for every call, because the concurrent pass below needs them all
    // before it starts and the serial pass wants the same values.
    std::vector<std::vector<tools::ToolParamValue>> params(calls.size());
    for (std::size_t i = 0; i < calls.size(); ++i) {
        for (const auto& p : calls[i].params) {
            params[i].push_back({p.name, p.value});
        }
    }

    // The ARGUMENTS, which `tool_result` never carried -- it records what came back, and
    // the summary of a shell call does not contain the command that produced it. Reading
    // "Ok, empty output" three turns running tells you nothing; reading the three
    // commands tells you immediately whether the model is repeating itself.
    if (trace_text_enabled()) {
        for (std::size_t i = 0; i < calls.size(); ++i) {
            std::vector<platform::EventField> fields{{"tool", calls[i].name},
                                                     {"index", std::to_string(i)}};
            for (const tools::ToolParamValue& p : params[i]) {
                fields.push_back({"arg." + p.name, capped(p.value)});
            }
            emit("tool_call", std::move(fields));
        }
    }

    // The read-only calls of this batch run at once; everything else stays exactly where
    // it was. See parallel_calls.hpp for which calls qualify and why the others cannot.
    std::vector<std::size_t> parallel;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        if (can_run_in_parallel(calls[i].name)) {
            parallel.push_back(i);
        }
    }
    std::vector<tools::ToolResult> precomputed;
    if (parallel.size() > 1) {
        precomputed = run_calls_concurrently(parallel, [this, &calls, &params](std::size_t i) {
            // ONLY the registry. Every gate and every ledger write stays on this thread,
            // below, in call order.
            return registry_.execute(calls[i].name, params[i], policy_.sandbox_tier);
        });
    }
    const auto precomputed_for = [&](std::size_t i) -> const tools::ToolResult* {
        for (std::size_t k = 0; k < precomputed.size(); ++k) {
            if (parallel[k] == i) {
                return &precomputed[k];
            }
        }
        return nullptr;
    };

    // Serial from here, in index order, so the emits, the history records and the UI rows
    // are what the fully serial path produced. Parallelism must not be observable.
    for (std::size_t i = 0; i < calls.size(); ++i) {
        bool ran = false;
        tools::ToolResult result;
        if (const tools::ToolResult* done = precomputed_for(i); done != nullptr) {
            result = *done;
            ran = adopt_readonly_result(calls[i].name, result);
        } else {
            result = dispatch_call(calls[i].name, params[i], ran);
        }

        if (i == 0) {
            turn.tool_name = calls[0].name;
            turn.tool_params = params[0];
            turn.tool_result = std::move(result);
            turn.outcome = classify_turn(turn.generation, grammar, ran, !ran);
        } else {
            TurnResult::ExtraCall extra;
            extra.tool_name = calls[i].name;
            extra.params = params[i];
            extra.result = std::move(result);
            turn.extra_calls.push_back(std::move(extra));
        }
    }
    return turn;
}

// One call: mode policy, HITL, the checklist, the verification contract and the
// deliverable ledger -- all applied in ONE place (S9.3), so a batched call is governed
// exactly as a lone one is. `executed` answers "did this actually run?", which is what
// classification turns on (S9.1).
// May this call be run off the agent thread? Only if dispatch_call would have reached
// `Registry::execute` and touched nothing else on the way.
//
// Stated as the properties that make the other branches unreachable, not as a list of tool
// names, so a tool added later is excluded until it is declared harmless: `plan` mutates
// the checklist, `mutates_workspace` opens the write gate and the deliverable ledger,
// `executes_commands` opens the risk classifier, the approver and the verification ledger.
// An unregistered name is not eligible either -- dispatch_call has to be the one to
// produce the typed NotFound.
bool Agent::can_run_in_parallel(const std::string& name) const {
    if (name == "plan") {
        return false;
    }
    const tools::ToolDecl* decl = registry_.find(name);
    if (decl == nullptr) {
        return false;
    }
    return !decl->mutates_workspace && !decl->executes_commands && !decl->irreversible;
}

// The tail dispatch_call would have run for such a call, minus everything the eligibility
// test already proved unreachable: no deliverable to record (nothing was written), no
// ledger, no approval. What remains is the executed flag and the event, and BOTH must
// happen here on the agent thread, in call order.
bool Agent::adopt_readonly_result(const std::string& name, const tools::ToolResult& result) {
    emit("tool_result", {{"tool", name},
                         {"status", std::string(tools::to_string(result.status))},
                         {"summary", result.summary}});
    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    return result.status != tools::Status::Refused;
}

tools::ToolResult Agent::dispatch_call(const std::string& name,
                                       const std::vector<tools::ToolParamValue>& params,
                                       bool& executed) {
    executed = false;

    // `plan` never reaches the registry: the loop owns the checklist.
    if (name == "plan") {
        const TurnResult::PlanOutcome r = apply_plan(params);
        executed = r.ok;
        return r.ok ? tools::ToolResult::okay(r.detail)
                    : tools::ToolResult::error(tools::ErrorClass::Malformed, true, r.detail);
    }

    const tools::ToolDecl* decl = registry_.find(name);
    // Mode policy, then the write gate, then the command gate -- in approval.cpp, with
    // the pure routing functions they drive (S9.3: policy is applied in ONE place).
    if (std::optional<tools::ToolResult> refusal = gate_call(decl, name, params)) {
        return std::move(*refusal);
    }

    // First touch of a path: record whether its syntax check was ALREADY failing, using
    // what is on disk right now -- which is the pre-image, so nothing has to be
    // snapshotted. Costs one extra sandboxed run per file per run, and it is the
    // difference between "your edit broke this" and "this arrived broken".
    if (syntax_ && config_.auto_syntax_check && decl != nullptr &&
        decl->mutates_workspace) {
        const std::string path = param_value(params, "path");
        if (!path.empty() && pre_edit_clean_.find(path) == pre_edit_clean_.end()) {
            const tools::SyntaxVerdict pre = syntax_->check(path, policy_.sandbox_tier);
            if (pre.ran) {
                pre_edit_clean_.emplace(path, pre.clean);
            }
        }
    }


    // A shell call whose command IS the declared verification contract goes through the
    // Verifier, the only thing that may write the ledger (S10.1). Without this the agent
    // ran its own tests through the raw shell tool, saw them pass, and the ledger stayed
    // empty -- so a finished run could never be recognised as finished.
    tools::ToolResult result;
    const std::string cmd = param_value(params, "command");
    // Containment, not equality. The declared contract is `pytest test_stats.py`, and
    // what the model actually runs is `cd /abs/path && python3 -m pytest test_stats.py`.
    // Requiring an exact match meant the check never matched, the Verifier never saw it,
    // and the ledger stayed empty while the agent watched its own tests pass.
    const std::string canon_cmd = canonicalize_check(cmd);
    const std::string canon_contract = canonicalize_check(ctx_.verify_contract());
    const bool is_the_check = name == "shell" && !canon_contract.empty() &&
                              canon_cmd.find(canon_contract) != std::string::npos;
    if (is_the_check) {
        const std::size_t before = ctx_.verifications().size();
        // Filed under the DECLARED contract, so every spelling of the check accumulates
        // history on one identity instead of minting a fresh, historyless one.
        (void)verifier_.run_and_record_as(cmd, policy_.sandbox_tier, canon_contract);
        const context::VerificationRecord& rec = ctx_.verifications().back();
        result = rec.passed ? tools::ToolResult::okay(rec.detail)
                            : tools::ToolResult::error(tools::ErrorClass::Transient, true,
                                                       rec.detail);
        // When the CONTRACT ITSELF cannot be executed, say so and say what to do -- the
        // broken thing is the declared criterion, not the workspace, and no amount of
        // work on the code will change the answer.
        //
        // MEASURED: a run declared `... && python -m pytest ...` on a host with only
        // `python3`. It then ran its real tests with `python3` (green, 26 passing) and
        // its contract with `python` (exit 127) alternately, for the whole budget,
        // re-declaring the same broken contract at turns 1, 2, 6 and 39. It was told
        // "never ran" every time and never inferred that `verify_with` was the field to
        // change. Nothing here fixes it FOR the model -- the criterion is the model's to
        // set -- but "this is unrunnable, restate it" is an observation it can act on.
        if (!rec.ran) {
            result.summary +=
                "\n[contract] This is the run's declared verification contract, and it "
                "could not be executed at all -- so it can never pass, and the run cannot "
                "finish while it stands. Fix the command itself and re-declare it with "
                "plan(verify_with=...): use a command you have already watched run in "
                "this workspace.";
            result.retryable = false;
        }
        emit_verifications(before);
    } else {
        result = registry_.execute(name, params, policy_.sandbox_tier);
    }

    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    executed = result.status != tools::Status::Refused;

    // A successful write IS the deliverable. Nothing recorded these before, so the
    // completion check's "no deliverable was recorded" gate could never be satisfied.
    if (decl != nullptr && decl->mutates_workspace && result.ok()) {
        const std::string path = param_value(params, "path");
        if (!path.empty()) {
            ctx_.record_deliverable(path);
            // The post-edit check goes on the SAME observation rather than becoming a
            // turn of its own: it is a consequence of this edit, not a separate action,
            // and a turn would violate one-turn-one-outcome (S9.1) and burn an iteration.
            annotate_with_syntax_check(path, result);
        }
    }

    emit("tool_result", {{"tool", name},
                         {"status", std::string(tools::to_string(result.status))},
                         {"summary", result.summary}});
    return result;
}

void Agent::compact_to_budget() {
    // Measured in REAL tokens, not an estimate: the prompt is rendered and tokenized
    // anyway, so asking the tokenizer costs nothing extra and a character heuristic
    // would be wrong exactly where it matters (code and diffs tokenize badly).
    model::ChatTemplate tmpl(tok_);
    while (ctx_.recent().size() > kMinRecentTurns) {
        const std::size_t tokens =
            tmpl.render(ctx_.render(tools_guidance_), tools_guidance_).size();
        if (tokens <= static_cast<std::size_t>(config_.context_budget_tokens)) {
            return;
        }
        if (ctx_.compact_oldest(ctx_.recent().size() - 1) == 0) {
            return;
        }
        emit("compaction", {{"tokens_before", std::to_string(tokens)},
                            {"recent_turns", std::to_string(ctx_.recent().size())}});
    }
}

// Takes whatever the user has said since the last turn boundary into the context.
//
// Everything downstream falls out of ContextStore::add_user_message: the text enters the
// ordered stream at the point it actually arrived, the latest one is pinned in live state
// where compaction cannot reach it, the plan goes stale (so the next turn must re-plan),
// and the directive's position is recorded so a green from before it cannot be offered as
// evidence for it.
std::size_t Agent::take_steering() {
    if (!steer_) {
        return 0;
    }
    const std::vector<std::string> messages = steer_();
    for (const std::string& text : messages) {
        if (text.empty()) {
            continue;
        }
        ctx_.add_user_message(text);
        emit("steer", {{"chars", std::to_string(text.size())},
                       {"at_turn", std::to_string(ctx_.recent().size())}});
        // A run that was drifting into narration has just been given something new to
        // act on. Holding the old count against it would end the run on the strength of
        // turns that happened before anyone spoke to it.
        consecutive_no_progress_ = 0;
    }
    return messages.size();
}

void Agent::apply_corrective(Corrective c, const TurnResult& turn) {
    // Every branch here CHANGES STATE or CONTROL FLOW. None composes a sentence asking
    // the model to behave -- that is the S9.2 rule, and run_ratchets.py counts the
    // sites that break it.
    switch (c) {
        case Corrective::None:
            return;
        case Corrective::BreakRepeat: {
            // Mechanism: make the repeated tool UNSAMPLABLE for the next turn, and say so.
            //
            // This used to append the note below and nothing else -- which left the
            // identical call fully available on the very next turn. The declared mechanism
            // ("force a different tool by narrowing the grammar's registry") was never
            // implemented, so the corrective was a sentence asking the model to stop, and
            // that is exactly what S9.2 forbids.
            //
            // MEASURED: a real run in the editor burned all 80 turns alternating
            // `list_dir ResMon` and `list_dir ResMon/`. Even on the turns this fired, the
            // next turn could call list_dir again, and did.
            //
            // ONE turn, not the rest of the run: unlike a twice-refused tool (which the
            // operator has said no to), a repeated tool is usually the RIGHT tool being
            // used with wrong arguments -- taking `read_file` away permanently because it
            // was read twice would end the run. One turn is enough to force a different
            // move and cheap enough to be wrong about.
            emit("corrective", {{"kind", "break_repeat"}, {"tool", turn.tool_name}});
            suppress_tool_next_turn_ = turn.tool_name;
            context::TurnRecord marker;
            marker.tool_name = turn.tool_name;
            // A repeated FAILURE needs a different sentence from a repeated success: the
            // model is not seeing duplicate progress, it is re-sending bytes that cannot
            // work. Naming the arguments as the thing to change is the mechanism's whole
            // point -- suppressing the observation alone would leave it re-deriving the
            // same call from the same context.
            // Says what the mechanism DID, so the next turn is not left guessing why its
            // tool vanished. Describing a real state change is not the prose-corrective
            // the ratchet forbids; describing one that did not happen is.
            marker.observation =
                (turn.tool_result.ok()
                     ? "(repeat suppressed: this exact call already returned this result. "
                     : "(this exact call has already failed the same way; the arguments "
                       "are what must change, not the tool. ") +
                std::string("`") + turn.tool_name +
                "` cannot be called on the next turn -- take a different action.)";
            marker.observation_is_error = !turn.tool_result.ok();
            ctx_.add_turn(std::move(marker));
            return;
        }
        case Corrective::SynthesizeVerification: {
            // Mechanism: make the call the model described but did not make.
            //
            // The command is the contract the run DECLARED through `plan`, not a hardcoded
            // one. It used to be `cmake --build build` unconditionally, which on a Python
            // workspace runs a command that cannot work -- so the mechanism that exists to
            // break a stall filed a guaranteed failure instead.
            //
            // Through verifier_, not a fresh Verifier: proven_ is the falsifiability cache
            // and a new instance starts with an empty one. And filed under the canonical
            // contract, so every spelling of the check accumulates history on ONE identity
            // (S10.1) rather than minting a historyless second.
            const std::string& contract = ctx_.verify_contract();
            emit("corrective",
                 {{"kind", "synthesize_verification"}, {"contract", contract}});
            (void)verifier_.run_and_record_as(contract, policy_.sandbox_tier,
                                              canonicalize_check(contract));
            return;
        }
        case Corrective::BlockRefusedTool: {
            // Mechanism: take the tool off the grammar for the rest of the run, so the
            // next turn cannot sample it at all. Recording the reason matters as much as
            // the block -- a call that silently stops being available is indistinguishable
            // from a model that forgot the tool exists.
            emit("corrective", {{"kind", "block_refused_tool"}, {"tool", turn.tool_name}});
            refusals_.block(turn.tool_name);
            context::TurnRecord marker;
            marker.tool_name = turn.tool_name;
            marker.observation = "(the operator refused `" + turn.tool_name +
                                 "` twice; it is no longer available this run -- take "
                                 "another route or stop and say why you cannot)";
            marker.observation_is_error = true;
            ctx_.add_turn(std::move(marker));
            return;
        }
        case Corrective::BudgetNearlyGone: {
            // Mechanism: an observation stating the remaining turn count, injected into
            // the history like any other. Not a request to hurry -- a fact the run cannot
            // otherwise obtain, because nothing else in the prompt says how many turns are
            // left, and a model that cannot see the edge cannot avoid stopping on the
            // wrong side of it.
            emit("corrective", {{"kind", "budget_nearly_gone"},
                                {"turns_left", std::to_string(kBudgetWarningTurns)}});
            context::TurnRecord marker;
            marker.observation =
                "(" + std::to_string(kBudgetWarningTurns) +
                " turns left before this run is cut off. If anything in the workspace is "
                "deliberately broken right now -- a bug injected to prove a check can fail "
                "-- restore it and re-run the check NOW, before doing anything else: a run "
                "that ends mid-proof leaves the damage behind. Otherwise finish what is in "
                "flight and stop.)";
            marker.observation_is_error = true;
            marker.first_event_seq = log_.events_written();
            marker.last_event_seq = marker.first_event_seq;
            ctx_.add_turn(std::move(marker));
            return;
        }
        case Corrective::HaltOnBudget:
            // Mechanism: end the run.
            emit("corrective", {{"kind", "halt_on_budget"}});
            halted_ = true;
            halt_reason_ = "budget_exhausted";
            return;
    }
}

RunReport Agent::run(const model::CancelToken& cancel) {
    RunReport report;
    const auto started = clock_.mono();

    while (!halted_) {
        if (cancel.cancelled()) {
            report.termination_reason = "cancelled";
            break;
        }
        // The turn boundary, and the only place the user's words enter a live run.
        report.steers_received += take_steering();

        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                                 clock_.mono() - started)
                                 .count();
        const bool out_of_time = elapsed >= config_.budget.wall_clock_seconds;

        const TurnResult turn = step(cancel);
        ++report.iterations;

        if (turn.outcome == Outcome::BackendError) {
            report.termination_reason = "backend_error";
            break;
        }
        if (turn.outcome == Outcome::Cancelled) {
            report.termination_reason = "cancelled";
            break;
        }

        if (observer_.on_turn) {
            observer_.on_turn(turn, turn.generation.ttft_ms);
        }
        const std::size_t before = ctx_.verifications().size();

        // Record the turn -- observations only, nothing inferred (S8.4).
        context::TurnRecord rec;
        rec.assistant_text = turn.assistant_text;
        rec.tool_name = turn.tool_name;
        rec.tool_args_summary = preview_of(turn.tool_name, turn.tool_params);
        rec.observation = turn.tool_result.summary;
        rec.observation_is_error = !turn.tool_result.ok();
        rec.last_event_seq = log_.events_written();

        // The same floor Registry::execute puts under its tools, applied to the paths that
        // do not go through it -- `plan`, and the Verifier, whose detail is the command's
        // output and is empty whenever a passing check prints nothing. render() drops an
        // empty observation, so without this the turn leaves no trace in the next prompt
        // and the model repeats it.
        if (rec.observation.empty() && turn.outcome == Outcome::ToolCallExecuted) {
            rec.observation = "(" + turn.tool_name +
                              (turn.tool_result.ok()
                                   ? " succeeded and produced no output)"
                                   : " failed, with no detail)");
        }

        // A turn that hit the token cap mid-thought leaves NOTHING behind: reasoning is
        // not carried forward (S5.7), there is no answer body and no call ran. The
        // record would be empty, the context would be unchanged, and the next turn would
        // re-render a byte-identical prompt -- which at a fixed seed produces a
        // byte-identical continuation. A deterministic infinite loop at ~50 s a turn.
        //
        // Observed: twelve consecutive turns, prompt `tokens=2044 messages=11` every
        // time, generation `tokens=4096` every time, until the wall clock killed it.
        //
        // So the truncation itself becomes the observation. It is an observed fact about
        // this run, which is exactly what T2 is for, and it perturbs the prompt enough
        // that the next attempt is a different draw rather than the same one.
        if (turn.outcome == Outcome::LengthCapped) {
            rec.observation =
                "(cut off at the generation cap before any tool call was made -- nothing "
                "ran. Reason in fewer tokens, or take a smaller first step.)";
            rec.observation_is_error = true;
        }
        ctx_.add_turn(std::move(rec));

        if (turn.outcome == Outcome::ToolCallExecuted) {
            repeats_.record(turn.tool_name, turn.tool_params);
        }
        // A refusal is not an execution and not an error, so neither ledger above sees
        // it. Counted here so re-asking has somewhere to register (S9.2).
        if (turn.outcome == Outcome::ToolCallRefused) {
            refusals_.record(turn.tool_name);
        }

        // Calls batched behind the first each get their own record and their own UI row.
        for (const TurnResult::ExtraCall& extra : turn.extra_calls) {
            context::TurnRecord er;
            er.tool_name = extra.tool_name;
            er.tool_args_summary = param_value(extra.params, "path");
            if (er.tool_args_summary.empty()) {
                er.tool_args_summary = param_value(extra.params, "command");
            }
            er.observation = extra.result.summary;
            er.observation_is_error = !extra.result.ok();
            er.first_event_seq = log_.events_written();
            er.last_event_seq = er.first_event_seq;
            ctx_.add_turn(std::move(er));

            if (observer_.on_turn) {
                TurnResult as_turn;
                as_turn.outcome = extra.result.status == tools::Status::Refused
                                      ? Outcome::ToolCallRefused
                                      : Outcome::ToolCallExecuted;
                as_turn.tool_name = extra.tool_name;
                as_turn.tool_params = extra.params;
                as_turn.tool_result = extra.result;
                observer_.on_turn(as_turn, 0.0);
            }
        }

        // Compaction, not eviction (S8.3) -- and only when the BUDGET says so.
        //
        // This used to run unconditionally every turn against a turn-count limit, so a
        // run compacted 18 times in 29 turns while holding ~3k tokens against a 96k
        // budget: it discarded its own history at 3% of capacity, then re-read the same
        // files because it no longer remembered reading them. The budget was never
        // consulted at all. Now a trim happens when the prompt actually needs one.
        compact_to_budget();

        // At most ONE corrective per turn, chosen by rank (S9.2).
        apply_corrective(choose_corrective(turn, repeats_, refusals_, report.iterations,
                                           config_.budget, out_of_time,
                                           !ctx_.verify_contract().empty()),
                         turn);

        // Any verification a corrective produced flows to the UI from the ledger --
        // the one choke point, so nothing can report a result that was not recorded.
        emit_verifications(before);

        const CompletionVerdict verdict = evaluate_completion(ctx_);
        if (verdict.complete) {
            report.completed = true;
            report.self_declared = verdict.self_declared();
            // "completed" against a model-chosen contract is a weaker claim than
            // "completed" against the operator's, and until this field existed both were
            // reported with the same word.
            report.termination_reason = "completed";
            emit("completion", {{"reason", verdict.reason},
                                {"open_items", std::to_string(verdict.open_items)},
                                {"self_declared", verdict.self_declared() ? "1" : "0"}});
            break;
        }
        // WHY NOT, on the turn the answer changes. The verdict is computed every turn and
        // used to be logged only when it said yes, so the single most useful sentence
        // about a run that worked and did not finish -- which gate is still shut -- was
        // computed 40 times and written down never. A run then ends `budget_exhausted`
        // with a green, proven ledger and nothing in the trace connecting the two.
        //
        // On change rather than every turn: the reason is stable for long stretches, and
        // 40 copies of the same line is not a trace, it is noise.
        if (verdict.reason != last_incomplete_reason_) {
            last_incomplete_reason_ = verdict.reason;
            emit("not_complete", {{"reason", verdict.reason},
                                  {"open_items", std::to_string(verdict.open_items)}});
        }
        // A run that has stopped calling tools has stopped working. Two ways to see it:
        // no checklist at all, or a checklist it is no longer acting on.
        //
        // The second case is new and was found the hard way: once `plan` was enforced the
        // checklist was never empty, so the old guard stopped firing and a run that fell
        // into narration spun out 20 text-only turns until the wall clock cancelled it.
        // Ending on a plan is not more honest than ending without one.
        // A turn that executed nothing made no move, and WHY it executed nothing does not
        // change that. This used to count only TextOnly, so a run capped at the token
        // limit every turn was never seen as stalled: LengthCapped is not TextOnly, the
        // counter stayed at zero, and the only thing that could end the run was the
        // budget. Twelve turns and 450 seconds of a fixed 600-second wall clock went
        // that way before anything noticed.
        const bool made_no_move = turn.outcome == Outcome::TextOnly ||
                                  turn.outcome == Outcome::LengthCapped;
        consecutive_no_progress_ = made_no_move ? consecutive_no_progress_ + 1 : 0;

        const bool stalled_without_plan = turn.outcome == Outcome::TextOnly &&
                                          report.iterations > 1 &&
                                          ctx_.checklist().empty();
        const bool stalled_narrating = consecutive_no_progress_ >= kMaxConsecutiveNoProgress;
        if (stalled_without_plan || stalled_narrating) {
            // Last look at the inbox before giving up. A human watching a run drift into
            // narration is exactly the human who types "keep going" or "no, try the other
            // file" -- and ending the run a moment after they said it, having already read
            // it off the pipe, would be the worst possible time to stop listening.
            // take_steering() resets the text-only count, so a message genuinely revives
            // the run rather than deferring the same ending by one turn.
            const std::size_t rescued = take_steering();
            report.steers_received += rescued;
            if (rescued > 0) {
                continue;
            }
            // Named for what actually happened: "it narrated" and "it thought until
            // the token cap" are different failures and want different responses.
            report.termination_reason =
                stalled_without_plan
                    ? "text_only_no_plan"
                    : (turn.outcome == Outcome::LengthCapped ? "length_capped_no_progress"
                                                             : "text_only_no_progress");
            break;
        }
    }
    if (report.termination_reason.empty()) {
        report.termination_reason = halted_ ? halt_reason_ : "loop_exit";
    }
    report.compactions = ctx_.compaction_count();
    report.unfinished_items = ctx_.open_checklist_items();
    emit("run_end", {{"termination_reason", report.termination_reason},
                     {"iterations", std::to_string(report.iterations)},
                     {"completed", report.completed ? "true" : "false"},
                     {"unfinished_items", std::to_string(report.unfinished_items)},
                     {"steers_received", std::to_string(report.steers_received)}});
    return report;
}

} // namespace lmp::loop
