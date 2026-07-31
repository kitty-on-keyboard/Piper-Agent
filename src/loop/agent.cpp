#include "src/loop/agent.hpp"

#include <algorithm>
#include <chrono>

namespace lmp::loop {
namespace {

// Collects tokens, walks the grammar, and stops the instant the grammar accepts --
// not a token later (S5.5).
class GrammarSink final : public model::TokenSink {
  public:
    explicit GrammarSink(model::TurnGrammar& g) : g_(g) {}
    bool on_token(model::TokenId id) override {
        last = g_.advance(id);
        return last == model::Advance::Ok;
    }
    model::Advance last = model::Advance::Ok;

  private:
    model::TurnGrammar& g_;
};

std::string preview_of(const std::string& tool,
                       const std::vector<tools::ToolParamValue>& params) {
    std::string s = tool + "(";
    bool first = true;
    for (const tools::ToolParamValue& p : params) {
        if (!first) {
            s += ", ";
        }
        first = false;
        s += p.name + "=" + (p.value.size() > 120 ? p.value.substr(0, 120) + "..." : p.value);
    }
    return s + ")";
}

} // namespace

double risk_score(const tools::RiskHint& hint) {
    // The published weights from bakeoff/blast_radius: write_out, destroy and priv are
    // worth three ordinary capabilities. Partial parse is itself risk -- it is the
    // signal that says "sandbox this regardless of the flags".
    const auto& c = hint.caps;
    double score = 0.0;
    score += c.writes_outside_workspace ? 0.30 : 0.0;
    score += c.destroys_data ? 0.30 : 0.0;
    score += c.escalates_privileges ? 0.30 : 0.0;
    score += c.rewrites_vcs_history ? 0.10 : 0.0;
    score += c.reads_outside_workspace ? 0.10 : 0.0;
    score += c.network_access ? 0.10 : 0.0;
    score += c.spawns_unbounded_process ? 0.10 : 0.0;
    score += c.signals_foreign_process ? 0.15 : 0.0;
    if (hint.status == blast_radius::ParseStatus::PartiallyParsed) {
        score += 0.20;
    } else if (hint.status == blast_radius::ParseStatus::Unparseable) {
        score += 0.40;
    }
    return score > 1.0 ? 1.0 : score;
}

bool is_irreversible(const tools::RiskHint& hint) noexcept {
    const auto& c = hint.caps;
    return c.destroys_data || c.writes_outside_workspace || c.escalates_privileges ||
           c.rewrites_vcs_history;
}

bool is_allowlisted(const std::string& command, const std::vector<std::string>& allowed) {
    const auto trim = [](std::string s) {
        const auto space = [](unsigned char ch) { return std::isspace(ch) == 0; };
        s.erase(s.begin(), std::find_if(s.begin(), s.end(), space));
        s.erase(std::find_if(s.rbegin(), s.rend(), space).base(), s.end());
        return s;
    };
    const std::string cmd = trim(command);
    if (cmd.empty()) {
        return false;
    }
    // Anything that can chain, substitute or redirect is out of scope for prefix
    // matching. `pytest` on the list must never authorise `pytest; rm -rf ~`.
    if (cmd.find_first_of(";|&`<>") != std::string::npos ||
        cmd.find("$(") != std::string::npos) {
        return false;
    }
    for (const std::string& raw : allowed) {
        const std::string entry = trim(raw);
        if (entry.empty()) {
            continue;
        }
        if (cmd == entry) {
            return true;
        }
        // Followed by a space, so `git st` never matches an entry of `git s`.
        if (cmd.size() > entry.size() && cmd.compare(0, entry.size(), entry) == 0 &&
            cmd[entry.size()] == ' ') {
            return true;
        }
    }
    return false;
}

Approval route_approval(const tools::RiskHint& hint, const HitlThresholds& t) {
    const double score = risk_score(hint);
    if (score >= t.reject_above_risk) {
        return Approval::Reject;
    }
    if (score <= t.auto_approve_below_risk) {
        return Approval::AutoApprove;
    }
    return Approval::Escalate;
}

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
            ctx_.set_verify_contract(p.value);
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
    if (observer_.on_verification && !ctx_.verifications().empty()) {
        observer_.on_verification(ctx_.verifications().back());
    }
    return passed ? "\nBaseline: that command already PASSES. Either the mission is "
                    "already satisfied, or it does not test what the mission describes -- "
                    "say which before doing anything else."
                  : "\nBaseline: that command currently FAILS, as expected. Making it "
                    "pass is now provable evidence rather than an unproven green.";
}

void Agent::emit(const std::string& kind, std::vector<platform::EventField> fields) {
    platform::Event ev;
    ev.kind = kind;
    ev.fields = std::move(fields);
    log_.append(ev, clock_);
}

TurnResult Agent::step(const model::CancelToken& cancel) {
    TurnResult turn;

    // --- prompt assembly ---------------------------------------------------
    const model::ChatTemplate tmpl(tok_);
    const std::vector<model::Message> messages = ctx_.render("");
    model::InferenceTask task;
    task.prompt = tmpl.render(messages, tools_guidance_);
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
    model::TurnGrammar grammar(tok_, specs);
    task.mask = &grammar;
    GrammarSink sink(grammar);
    turn.generation = backend_.generate(task, sink, cancel);

    turn.reasoning = tok_.decode(grammar.think_ids());
    turn.assistant_text = tok_.decode(grammar.text_ids());

    // Reasoning is surfaced on its own channel, never inlined into the answer (S5.7).
    // The split happened by TOKEN ID upstream; this only routes it.
    if (observer_.on_token) {
        if (!turn.reasoning.empty()) {
            observer_.on_token("thinking", turn.reasoning);
        }
        if (!turn.assistant_text.empty()) {
            observer_.on_token("answer", turn.assistant_text);
        }
    }
    if (observer_.on_perf) {
        observer_.on_perf(turn.generation, task.prompt.size(),
                          static_cast<std::size_t>(config_.max_new_tokens) +
                              task.prompt.size());
    }

    emit("generation", {{"status", std::to_string(static_cast<int>(turn.generation.status))},
                        {"tokens", std::to_string(turn.generation.tokens_generated)},
                        {"ttft_ms", std::to_string(turn.generation.ttft_ms)},
                        {"decode_tok_per_s", std::to_string(turn.generation.decode_tok_per_s)}});

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
    turn.tool_name = calls.front().name;
    for (const auto& p : calls.front().params) {
        turn.tool_params.push_back({p.name, p.value});
    }
    bool executed = false;
    turn.tool_result = dispatch_call(turn.tool_name, turn.tool_params, executed);
    turn.outcome = classify_turn(turn.generation, grammar, executed, !executed);

    for (std::size_t i = 1; i < calls.size(); ++i) {
        TurnResult::ExtraCall extra;
        extra.tool_name = calls[i].name;
        for (const auto& p : calls[i].params) {
            extra.params.push_back({p.name, p.value});
        }
        bool ran = false;
        extra.result = dispatch_call(extra.tool_name, extra.params, ran);
        turn.extra_calls.push_back(std::move(extra));
    }
    return turn;
}

// One call: mode policy, HITL, the checklist, the verification contract and the
// deliverable ledger -- all applied in ONE place (S9.3), so a batched call is governed
// exactly as a lone one is. `executed` answers "did this actually run?", which is what
// classification turns on (S9.1).
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
    if (decl != nullptr && decl->mutates_workspace && !policy_.allow_workspace_writes) {
        emit("tool_refused", {{"tool", name}, {"why", "mode policy"}});
        return tools::ToolResult::refused("this mode does not permit workspace writes");
    }

    // --- HITL: writes -------------------------------------------------------
    //
    // Separate from the command gate below because the question is different. A command
    // is asked about because of what it MIGHT do, inferred from a string; a write is
    // asked about because of what it definitely does, to a named path. There is no risk
    // score here and there should not be one -- the operator asked to see writes, so
    // every write is shown.
    // A tool DECLARED irreversible always asks, exactly as an irreversible command does,
    // and for the same reason -- but it has to be asked here, because a tool call has no
    // command string for the blast-radius classifier to read. `delete_file` destroys data
    // with no command in sight, so the entire risk-and-approval apparatus was watching
    // `shell` while a run wiped a workspace through a tool it never scored.
    const bool irreversible_tool = decl != nullptr && decl->irreversible;
    if (decl != nullptr && decl->mutates_workspace &&
        (!config_.auto_approve_writes || irreversible_tool)) {
        tools::RiskHint hint;
        // Declared, not inferred, and reported to the card as the fact it is.
        hint.caps.destroys_data = irreversible_tool;
        hint.status = blast_radius::ParseStatus::Parsed;
        const bool allowed =
            approver_ && approver_(name, "", preview_of(name, params), hint);
        if (!allowed) {
            emit("tool_denied", {{"tool", name},
                                 {"why", irreversible_tool ? "irreversible, not approved"
                                                           : "write not approved"}});
            return tools::ToolResult::refused(
                approver_ ? "denied by the operator"
                          : "this call needs a human decision and no approver is attached");
        }
    }

    // --- HITL: commands -----------------------------------------------------
    if (decl != nullptr && decl->executes_commands) {
        const std::string cmd = param_value(params, "command");
        const tools::RiskHint hint =
            !cmd.empty() ? tools::classify_command(cmd, "", "") : tools::RiskHint{};
        Approval route = route_approval(hint, config_.hitl);

        // Three checks, and the ORDER is the design.
        //
        //   1. The allowlist can only ever loosen, and only for ordinary commands.
        //   2. auto_approve_exec off can only ever tighten.
        //   3. Irreversibility overrides both, in the tightening direction, always.
        //
        // Written as three separate steps rather than one condition because each is a
        // different kind of claim -- "the operator said yes to this before", "the
        // operator wants to see everything", "this cannot be undone" -- and collapsing
        // them into one boolean is how the last one got lost.
        const bool allowlisted = is_allowlisted(cmd, config_.allowed_commands);
        if (allowlisted && route == Approval::Escalate) {
            route = Approval::AutoApprove;
        }
        if (!config_.auto_approve_exec && route == Approval::AutoApprove && !allowlisted) {
            route = Approval::Escalate;
        }
        if (is_irreversible(hint) && route == Approval::AutoApprove) {
            emit("irreversible", {{"tool", name},
                                  {"risk", std::to_string(risk_score(hint))},
                                  {"allowlisted", allowlisted ? "1" : "0"}});
            route = Approval::Escalate;
        }

        // An escalation with nobody to escalate TO is a denial, not a pass. This is the
        // unattended path: an eval, a script, a dead editor. Deny-by-default (S7.2).
        if (route == Approval::Escalate && !approver_) {
            emit("tool_denied", {{"tool", name}, {"why", "escalation with no approver"}});
            return tools::ToolResult::refused(
                "this call needs a human decision and no approver is attached");
        }

        bool allowed = route == Approval::AutoApprove;
        if (route == Approval::Escalate && approver_) {
            allowed = approver_(name, cmd, preview_of(name, params), hint);
        }
        if (!allowed) {
            emit("tool_denied",
                 {{"tool", name}, {"risk", std::to_string(risk_score(hint))}});
            return tools::ToolResult::refused(
                route == Approval::Reject ? "rejected: risk score above the reject threshold"
                                          : "denied by the operator");
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
        if (observer_.on_verification) {
            for (std::size_t i = before; i < ctx_.verifications().size(); ++i) {
                observer_.on_verification(ctx_.verifications()[i]);
            }
        }
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
            // Mechanism: drop the duplicate observation so the repeated result stops
            // occupying context and stops looking like progress.
            emit("corrective", {{"kind", "break_repeat"}, {"tool", turn.tool_name}});
            context::TurnRecord marker;
            marker.tool_name = turn.tool_name;
            marker.observation =
                "(repeat suppressed: this exact call already returned this result)";
            marker.observation_is_error = false;
            ctx_.add_turn(std::move(marker));
            return;
        }
        case Corrective::SynthesizeVerification: {
            // Mechanism: make the call the model described but did not make.
            emit("corrective", {{"kind", "synthesize_verification"}});
            Verifier verifier(registry_, ctx_);
            (void)verifier.run_and_record("cmake --build build", policy_.sandbox_tier);
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
        apply_corrective(choose_corrective(turn, repeats_, report.iterations,
                                           config_.budget, out_of_time),
                         turn);

        // Any verification a corrective produced flows to the UI from the ledger --
        // the one choke point, so nothing can report a result that was not recorded.
        if (observer_.on_verification) {
            for (std::size_t i = before; i < ctx_.verifications().size(); ++i) {
                observer_.on_verification(ctx_.verifications()[i]);
            }
        }

        const CompletionVerdict verdict = evaluate_completion(ctx_);
        if (verdict.complete) {
            report.completed = true;
            report.termination_reason = "completed";
            emit("completion", {{"reason", verdict.reason},
                                {"open_items", std::to_string(verdict.open_items)}});
            break;
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
