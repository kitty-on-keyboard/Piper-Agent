#include "src/loop/agent.hpp"

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
      clock_(clock), config_(config), policy_(ModePolicy::for_mode(config.mode)) {
    tools_guidance_ = registry_.tools_json();
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
    task.sampling.seed = config_.seed;

    // Every harness->model append is an event. This invariant is what makes "did the
    // model receive this?" answerable (S8.1, S14).
    emit("prompt", {{"tokens", std::to_string(task.prompt.size())},
                    {"messages", std::to_string(messages.size())},
                    {"compactions", std::to_string(ctx_.compaction_count())}});

    // --- constrained generation --------------------------------------------
    model::TurnGrammar grammar(tok_, registry_.guard_specs());
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

    // --- the call ----------------------------------------------------------
    turn.tool_name = grammar.tool_name();
    for (const auto& p : grammar.tool_params()) {
        turn.tool_params.push_back({p.name, p.value});
    }

    const tools::ToolDecl* decl = registry_.find(turn.tool_name);
    // Mode policy is applied HERE, in one place (S9.3).
    if (decl != nullptr && decl->mutates_workspace && !policy_.allow_workspace_writes) {
        turn.tool_result = tools::ToolResult::refused(
            "this mode does not permit workspace writes");
        turn.outcome = classify_turn(turn.generation, grammar, false, true);
        emit("tool_refused", {{"tool", turn.tool_name}, {"why", "mode policy"}});
        return turn;
    }

    // --- HITL --------------------------------------------------------------
    if (decl != nullptr && decl->executes_commands) {
        const std::string* cmd = nullptr;
        for (const auto& p : turn.tool_params) {
            if (p.name == "command") {
                cmd = &p.value;
            }
        }
        const tools::RiskHint hint =
            cmd != nullptr ? tools::classify_command(*cmd, "", "") : tools::RiskHint{};
        const Approval route = route_approval(hint, config_.hitl);
        bool allowed = route == Approval::AutoApprove;
        if (route == Approval::Escalate && approver_) {
            allowed = approver_(turn.tool_name,
                                preview_of(turn.tool_name, turn.tool_params), hint);
        }
        if (!allowed) {
            turn.tool_result = tools::ToolResult::refused(
                route == Approval::Reject
                    ? "rejected: risk score above the reject threshold"
                    : "denied by the operator");
            turn.outcome = classify_turn(turn.generation, grammar, false, true);
            emit("tool_denied", {{"tool", turn.tool_name},
                                 {"risk", std::to_string(risk_score(hint))}});
            return turn;
        }
    }

    turn.tool_result =
        registry_.execute(turn.tool_name, turn.tool_params, policy_.sandbox_tier);
    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    const bool executed = turn.tool_result.status != tools::Status::Refused;
    turn.outcome = classify_turn(turn.generation, grammar, executed, !executed);

    emit("tool_result", {{"tool", turn.tool_name},
                         {"status", std::string(tools::to_string(turn.tool_result.status))},
                         {"summary", turn.tool_result.summary}});
    return turn;
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
        ctx_.add_turn(std::move(rec));

        if (turn.outcome == Outcome::ToolCallExecuted) {
            repeats_.record(turn.tool_name, turn.tool_params);
        }

        // Compaction, not eviction (S8.3).
        (void)ctx_.compact_oldest(config_.keep_recent_turns);

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
            break;
        }
        if (turn.outcome == Outcome::TextOnly && report.iterations > 1 &&
            ctx_.checklist().empty()) {
            // A text-only turn with no checklist cannot progress; ending is honest.
            report.termination_reason = "text_only_no_plan";
            break;
        }
    }
    if (report.termination_reason.empty()) {
        report.termination_reason = halted_ ? halt_reason_ : "loop_exit";
    }
    report.compactions = ctx_.compaction_count();
    emit("run_end", {{"termination_reason", report.termination_reason},
                     {"iterations", std::to_string(report.iterations)},
                     {"completed", report.completed ? "true" : "false"}});
    return report;
}

} // namespace lmp::loop
