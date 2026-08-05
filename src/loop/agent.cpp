#include "src/loop/agent.hpp"

#include <algorithm>
#include <cassert> // the emit() field-name guard
#include <chrono>
#include <cstdlib> // getenv/atoi, for the LMP_TRACE_TEXT gate
#include <memory>

#include "src/loop/parallel_calls.hpp"
#include "src/loop/token_stream.hpp"
#include "src/platform/fs.hpp"

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

// How repetitive a generation is, measured on the text rather than guessed from its
// length. Reported ALWAYS, unlike the text itself, because these three numbers are small
// and a degenerate turn is invisible without them.
//
// MEASURED: one turn of a real run emitted "I'll fix all compilation errors
// systematically. Let me read all source files first." roughly two hundred times and then
// hit the token cap. In the event log that turn is `generation tokens=4096 status=1` and
// nothing else -- indistinguishable from a legitimately long write_file. The run had 66
// turns and several like it; the trace could not tell them apart, so nothing in the
// harness could either.
struct TextShape {
    std::size_t lines = 0;
    std::size_t distinct = 0;
    std::size_t worst_line_repeats = 0; // how often the most-repeated non-blank line occurs
};

TextShape shape_of(const std::string& text) {
    TextShape s;
    std::vector<std::pair<std::string_view, std::size_t>> counts;
    std::size_t at = 0;
    while (at <= text.size()) {
        std::size_t nl = text.find('\n', at);
        if (nl == std::string::npos) {
            nl = text.size();
        }
        const std::string_view line(text.data() + at, nl - at);
        at = nl + 1;
        // Blank and near-blank lines repeat in every healthy generation (indentation,
        // paragraph breaks) and would dominate the count without saying anything.
        if (line.find_first_not_of(" \t\r") == std::string_view::npos) {
            continue;
        }
        ++s.lines;
        bool seen = false;
        for (auto& [text_seen, n] : counts) {
            if (text_seen == line) {
                ++n;
                s.worst_line_repeats = std::max(s.worst_line_repeats, n);
                seen = true;
                break;
            }
        }
        if (!seen) {
            counts.emplace_back(line, 1);
            s.worst_line_repeats = std::max<std::size_t>(s.worst_line_repeats, 1);
        }
    }
    s.distinct = counts.size();
    return s;
}

// When a generation is worth flagging as degenerate rather than merely long.
//
// The discriminator is HOW MUCH DISTINCT CONTENT there is, not how often the commonest
// line recurs. That distinction was found by testing rather than reasoning: a repeat-count
// threshold flags a perfectly good `write_file` of a source file, because real code
// repeats `    }` a hundred times in two hundred lines. Measured on four inputs --
//
//   the real failure  lines=200 distinct=  1 worst=200   <- 0.5% distinct
//   a source file     lines=244 distinct=125 worst=120   <- 51% distinct
//   a brace-heavy file lines=200 distinct=101 worst=100  <- 50% distinct
//   a short answer    lines=  2 distinct=  2 worst=  1
//
// -- the repeat counts of the last three are indistinguishable from the first's, and the
// distinct ratio separates them by an order of magnitude. Both halves still have to hold:
// the floor keeps short answers out, since two identical lines out of two is 100% repeat
// and no evidence of anything.
constexpr std::size_t kRepeatFloor = 8;
constexpr std::size_t kDistinctCeilingPercent = 25;

bool looks_degenerate(const TextShape& s) {
    return s.lines >= kRepeatFloor && s.worst_line_repeats >= kRepeatFloor &&
           s.distinct * 100 <= s.lines * kDistinctCeilingPercent;
}

// A JSON array of strings, rewritten as one element per line. Empty when the text is not
// one, which is the signal to parse it as the newline-separated list it claims to be.
//
// THE TEST IS DELIBERATELY NARROW: first non-space character `[`, next non-space character
// `"`, and a `]` at the end. That is a JSON array and cannot be anything else -- in
// particular it cannot be the checklist item `[ ] ship the parser`, which is the one
// string a looser test would eat. A malformed array (an unterminated quote) flattens to
// nothing and falls through to the line parser, which is the behaviour that was there
// before this function existed.
std::string flatten_json_array(const std::string& raw) {
    const std::size_t open = raw.find_first_not_of(" \t\r\n");
    if (open == std::string::npos || raw[open] != '[') {
        return {};
    }
    const std::size_t first = raw.find_first_not_of(" \t\r\n", open + 1);
    if (first == std::string::npos || raw[first] != '"') {
        return {};
    }
    const std::size_t close = raw.find_last_not_of(" \t\r\n");
    if (close == std::string::npos || raw[close] != ']') {
        return {};
    }

    std::string out;
    bool in_string = false;
    std::string current;
    for (std::size_t i = first; i < close; ++i) {
        const char c = raw[i];
        if (!in_string) {
            if (c == '"') {
                in_string = true;
            }
            continue;
        }
        if (c == '\\' && i + 1 < close) {
            // Only the escapes that can appear in a checklist item. Anything else keeps
            // its literal character, which is what the line parser would have seen anyway.
            const char next = raw[++i];
            switch (next) {
                case 'n': current += '\n'; break;
                case 't': current += '\t'; break;
                case 'r': break;
                default: current += next; break;
            }
            continue;
        }
        if (c == '"') {
            in_string = false;
            out += current;
            out += '\n';
            current.clear();
            continue;
        }
        current += c;
    }
    // An unterminated final string means the text was not the array it looked like.
    return in_string ? std::string{} : out;
}

} // namespace

Agent::Agent(const model::QwenTokenizer& tok, model::InferenceBackend& backend,
             tools::Registry& registry, context::ContextStore& ctx,
             platform::EventLogWriter& log, const platform::Clock& clock,
             AgentConfig config)
    : tok_(tok), backend_(backend), registry_(registry), ctx_(ctx), log_(log),
      clock_(clock), config_(config), policy_(ModePolicy::for_mode(config.mode)),
      verifier_(registry, ctx) {
    // The operator's tier, when they named one. Plan mode is exempt in the one direction
    // that matters: it pins T0, so "no execution" cannot be undone by a settings field.
    if (config_.sandbox_tier_override >= 0 && config_.mode != Mode::Plan) {
        policy_.sandbox_tier = config_.sandbox_tier_override;
    }
    tools_guidance_ =
        registry_.tools_json([this](const tools::ToolDecl& d) { return tool_allowed(d); });
    std::string withheld_by_mode;
    for (const parsephony::ToolSpec& s : registry_.guard_specs()) {
        const tools::ToolDecl* d = registry_.find(s.name);
        if (d != nullptr && !tool_allowed(*d)) {
            withheld_by_mode += withheld_by_mode.empty() ? "" : ",";
            withheld_by_mode += s.name;
            continue;
        }
        mode_specs_.push_back(s);
    }
    // In the Agent rather than in the sidecar, so every client gets it -- the eval harness
    // and scripts/drive.py send a mode too, and a brief only the editor's runs received
    // would make the two disagree about what plan mode even is.
    ctx_.set_mode_brief(std::string(mode_brief(config_.mode)));

    // WHAT THIS MODE TOOK AWAY, once, at the top of the run. The per-turn `grammar` event
    // measures against this set rather than the registry's, so the two answer different
    // questions and neither drowns the other: this one is "what can this run never do",
    // that one is "what could this TURN not do".
    if (!withheld_by_mode.empty()) {
        emit("mode_tools", {{"withheld", withheld_by_mode},
                            {"samplable", std::to_string(mode_specs_.size())},
                            {"of", std::to_string(registry_.guard_specs().size())}});
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
    // SEEDED FROM THE SESSION, not started empty.
    //
    // `run_wrote_` answers "is this file the run's own output or the operator's data", and
    // a follow-up builds a fresh Agent over the SAME ContextStore -- so an empty set told
    // the second run that everything the first one produced belonged to someone else. A
    // run resuming its own work raised an overwrite card on its own files, which is the
    // gate firing on exactly the case it was built to let through.
    //
    // The deliverable ledger is the durable copy of the same fact and needs no second
    // store: it is written from the one place run_wrote_ is (a successful mutating call
    // with a path), and it outlives the run because the context does.
    for (const std::string& path : ctx_.deliverables()) {
        run_wrote_.insert(platform::lexically_normal(path));
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
    // ONE ITEM PER LINE -- unless the model sent a JSON array, which it does, because
    // `items` is a list-shaped parameter and that is what a list looks like to a model
    // that has spent its training on JSON tool calls.
    //
    // This parser split on '\n' and nothing else, so an array arrived as ONE line, the
    // `[ ]` test failed against the leading `["`, and the entire array became a single
    // checklist item whose text was its own JSON source.
    //
    // MEASURED, and it is what the operator saw on the surface: a run replanned mid-flight
    // with a seven-element array and the log recorded `items=1 open=1`, the checklist read
    // `0/1`, and the item's text was `["[ ] Task 1: ...", "[ ] Task 2: ...", ...]`. It also
    // destroyed a healthy six-item checklist from turn 1 and left one item that could never
    // be ticked, so the completion gate spent the rest of the run guarding a syntax error.
    const std::string flattened = flatten_json_array(*raw);
    const std::string& source = flattened.empty() ? *raw : flattened;
    std::size_t at = 0;
    while (at < source.size()) {
        std::size_t nl = source.find('\n', at);
        if (nl == std::string::npos) {
            nl = source.size();
        }
        std::string line = source.substr(at, nl - at);
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
    // THE PARSED ITEMS, not just how many. `plan` reports a count, and a count is exactly
    // the wrong thing to trust here: this parser tolerates prose-ish markdown, so a
    // checklist that arrives on one line, or with the marker but no text, or with nesting
    // it cannot see, still produces a plausible number. A run was observed showing five
    // items with no text at all in the surface, and the log said `items=5 open=5` -- which
    // is what a healthy plan looks like. The text is the only way to tell them apart.
    {
        std::string joined;
        for (const context::ChecklistItem& item : ctx_.checklist()) {
            joined += joined.empty() ? "" : " | ";
            joined += (item.done ? "[x] " : "[ ] ") +
                      (item.text.empty() ? std::string("<EMPTY>") : item.text);
        }
        emit("checklist", {{"count", std::to_string(total)},
                           {"open", std::to_string(open)},
                           {"items", capped(joined)}});
    }
    if (observer_.on_checklist) {
        observer_.on_checklist(ctx_.checklist());
    }
    std::string s = "checklist set: " + std::to_string(total - open) + "/" +
                    std::to_string(total) + " done";
    if (!ctx_.verify_contract().empty()) {
        s += "; completion requires '" + ctx_.verify_contract() + "' to pass";
        // Named on every plan call, with the current count, because the two halves of the
        // gate are easy to hold as one. A run that reads "completion requires the command
        // to pass" and nothing else will let its list rot -- which is the whole failure
        // ReconcileChecklist exists to catch, and catching it costs a turn that saying so
        // here does not.
        if (open != 0) {
            s += " and all " + std::to_string(open) + " open item(s) to be closed";
        }
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
    // ONE BASELINE PER ATOMIC CHECK, because each is a criterion in its own right and each
    // needs its own red-before-the-work to ever become evidence. A single reading filed
    // under the compound string proved nothing about either half.
    const std::vector<std::string> checks = contract_checks(ctx_.verify_contract());
    std::vector<std::string> unread;
    for (const std::string& check : checks) {
        bool seen = false;
        for (const context::VerificationRecord& v : ctx_.verifications()) {
            seen = seen || v.contract == check;
        }
        if (!seen) {
            unread.push_back(check);
        }
    }
    if (unread.empty()) {
        return {}; // already have a reading for every check in this contract
    }
    // The contract AS DECLARED is what runs -- `swift test && swift build` short-circuits,
    // and running the halves separately here would be a different command from the one the
    // operator asked for. One execution, recorded against each half that still needs a
    // baseline.
    const bool passed = verifier_.run_and_record_as(ctx_.verify_contract(),
                                                    policy_.sandbox_tier, unread);
    emit("baseline_check", {{"contract", canon},
                            {"checks", std::to_string(checks.size())},
                            {"passed", passed ? "1" : "0"}});
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
    if (!passed) {
        return "\nBaseline: that command currently FAILS, as expected. Making it pass is "
               "now provable evidence rather than an unproven green.";
    }
    // A GREEN BASELINE IS A BROKEN CRITERION, NOT A DEBT TO WORK OFF.
    //
    // This used to say: "break the behaviour it covers, run the command and watch it go
    // red, then restore what you broke". The harness was instructing the run to damage
    // working code, before any other work, on turn one.
    //
    // MEASURED, and it did exactly as it was told: a run wrote
    // `syntax_error_here!` into HostStatsService.swift to manufacture the red, then spent
    // its entire budget failing to climb back out. That is the harness fighting the agent,
    // and it is worse than useless here -- breaking a build to watch the build break
    // establishes only that a compiler compiles. It says nothing about the mission.
    //
    // is_proven() already documents the right rule: a red observed in the NATURAL order of
    // work is the proof, and manufacturing one is mutation testing, a QA activity, not an
    // inline agent step. So the honest reading of a green baseline is that the check does
    // not exercise the mission -- it passed before the work started -- and the fix is a
    // better criterion, which costs one `plan` call and no damage.
    const std::string_view broken = unfalsifiable_reason(executable_form(canon));
    if (!broken.empty()) {
        return "\nBaseline: that command PASSES, but it cannot be a criterion at all: " +
               std::string(broken) +
               ". This is a problem with the command, not with the code. Re-declare "
               "verify_with using the command's own exit status -- for a build or a test "
               "run, that is the tool invoked plainly, with no `| grep` and no "
               "`|| echo` after it.";
    }
    return "\nBaseline: that command already PASSES, before any work has been done -- so "
           "it does not measure this mission and cannot finish the run. Re-declare "
           "verify_with with a check that is RED right now BECAUSE the work is not done "
           "yet: usually the tests for the thing you are about to build. Making that go "
           "green is the proof, and you get it for free by working in the normal order. "
           "Do NOT break working code to manufacture a failure.";
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

// NO FIELD MAY BE CALLED `kind` OR `seq`. The writer stamps both onto every event, so a
// payload field with either name emits the key TWICE in one JSON object -- and every JSON
// parser in existence keeps the last one silently.
//
// MEASURED on the shipped log: 291 `corrective` events whose own kind field overwrote the
// event kind, so a parser saw 291 events of kind "break_repeat" and ZERO of kind
// "corrective"; and 63 `verification` events whose ledger position overwrote the writer's
// sequence number. Nothing consumed either field, which is why it survived -- the only
// readers are diagnosis and the ratchets, and diagnosis was reading the trace with grep.
// The log exists to make "what did the harness do" answerable; a field that deletes the
// answer to "what KIND of thing was this" is worse than no field.
void Agent::emit(const std::string& kind, std::vector<platform::EventField> fields) {
    for (const platform::EventField& f : fields) {
        assert(f.key != "kind" && f.key != "seq" &&
               "event payload may not shadow the writer's own kind/seq fields");
    }
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
                              // `at_seq`, not `seq`: the writer stamps its own `seq` on
                              // every event, and a second one silently replaced it in 63
                              // events of the shipped log. See Agent::emit.
                              {"at_seq", std::to_string(vs[i].seq)},
                              {"workspace_writes",
                               std::to_string(vs[i].workspace_writes)}});
        if (observer_.on_verification) {
            observer_.on_verification(vs[i]);
        }
    }
}

TurnResult Agent::step(const model::CancelToken& cancel) {
    TurnResult turn;
    // Per-turn, so the no-progress counter can ask "did this turn add any bytes the model
    // was not already holding?" -- see the `made_no_move` test in run(). Counted at
    // dispatch because that is the only moment the answer exists: the duplicate collapse
    // rewrites the earlier copy immediately afterwards.
    turn_reads_ = 0;
    turn_reads_redundant_ = 0;
    turn_calls_ = 0;
    turn_inert_calls_ = 0;

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
    // The real size of the prompt this turn, free of charge -- it has just been tokenized.
    // Read by collapse_duplicate_read, which must not call prompt_tokens() per read: that
    // re-renders and re-tokenizes the whole context, and a batched turn would do it four
    // times to answer a question this number already answers.
    last_prompt_tokens_ = task.prompt.size();
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
    //
    // RederiveContract narrows the same way and for the same reason, and it is spent below
    // like a suppression rather than latched -- see that corrective for why a latched flag
    // deadlocks.
    const bool must_replan = ctx_.plan_is_stale() || replan_turns_ > 0;

    // NOT IN A CONVERSATIONAL MODE. The checklist gate exists to stop a working run from
    // doing work before it has said what it is doing; plan mode's deliverable is agreement
    // with a human, and it owes no checklist. The condition as it stood narrowed the
    // grammar to `plan` and nothing else whenever the checklist was empty -- which is
    // every plan-mode run, on turn 1 -- so a mode whose entire job is to go and read the
    // code could not call read_file until it had first filed a checklist for a mission it
    // had not been allowed to look at yet.
    const bool plan_only =
        (ctx_.checklist().empty() || must_replan) && !policy_.conversational;
    std::vector<parsephony::ToolSpec> specs;
    if (plan_only) {
        for (const parsephony::ToolSpec& s : mode_specs()) {
            if (s.name == "plan") {
                specs.push_back(s);
            }
        }
    } else {
        specs = mode_specs();
    }
    specs = without_blocked(specs, refusals_, registry_.guard_specs());
    // The one-turn narrowing BreakRepeat asked for. Taken (and cleared) here rather than
    // held across turns, so it costs exactly the next turn and cannot accumulate into a
    // run that has quietly lost half its tools.
    //
    // Applied AFTER the plan gate, so a run that owes a checklist still gets `plan` even
    // if `plan` is what repeated -- otherwise a repeated plan would leave nothing callable
    // and the fallback below would hand `plan` straight back.
    specs = without_suppressed(specs, suppressed_tools_);
    // Spent AFTER the grammar is narrowed, so each suppression covers exactly the number
    // of turns it was given. Entries that have run out are dropped rather than left at
    // zero, so a tool that repeats again later starts a fresh, longer window instead of
    // inheriting a stale one.
    for (auto& [name, turns_left] : suppressed_tools_) {
        if (turns_left > 0) {
            --turns_left;
        }
    }
    std::erase_if(suppressed_tools_,
                  [](const auto& e) { return e.second <= 0; });
    // THE FLOOR. A turn that is meant to do work must have SOME way to change a file.
    //
    // Nothing above intends to take every writer away -- the holds are built to exclude
    // mutators now -- and this is here because the run that motivated it lost its editor to a
    // mechanism whose author did not think it could. Four separate narrowings feed this
    // grammar and they compose without any of them knowing about the others; the cheapest
    // durable answer is one invariant checked where the final set is known.
    //
    // NOT in plan-only mode: a replan turn is deliberately `plan` and nothing else, and it
    // lasts one turn. Restoring writers there would defeat the one escalation that replaced
    // the editor bans.
    //
    // AND NOT IN A MODE THAT GRANTS NO WRITES, which is the trap this floor sets for any
    // attempt to take the writers away on purpose. It restores every mutating tool the
    // moment none is samplable -- so mode filtering, which removes exactly those tools,
    // would have been undone on the first turn of every plan-mode run, with a single
    // `write_floor_restored` event as the only trace. A floor whose whole premise is "a
    // turn meant to do work must be able to change a file" has nothing to say about a turn
    // that is not meant to change files.
    if (!plan_only && policy_.allow_workspace_writes) {
        bool have_writer = false;
        for (const parsephony::ToolSpec& s : specs) {
            have_writer = have_writer || mutates_workspace(s.name);
        }
        if (!have_writer) {
            for (const parsephony::ToolSpec& s : mode_specs()) {
                // Restored only if the operator has not refused it: a refusal is the
                // operator's decision and outranks any floor the harness sets for itself.
                if (mutates_workspace(s.name) && !refusals_.is_blocked(s.name)) {
                    specs.push_back(s);
                }
            }
            for (auto& [name, turns_left] : suppressed_tools_) {
                if (mutates_workspace(name)) {
                    turns_left = 0;
                }
            }
            emit("corrective", {{"corrective", "write_floor_restored"},
                                {"why", "no_samplable_mutation_tool"}});
        }
    }
    // WHAT THIS TURN WAS ALLOWED TO DO. Emitted only when something was withheld, so a
    // healthy turn costs nothing and a constrained one is impossible to miss.
    //
    // The gap this closes: a run flailing between two tools looks identical in the trace
    // whether the model is confused or the harness has quietly removed the tool it was
    // reaching for. Four separate mechanisms narrow this grammar -- the plan gate,
    // RederiveContract, BreakRepeat's suppressions and BlockRefusedTool -- and none of
    // them said so in the log. A run measured at 66 turns had `read_file` suppressed
    // thirteen separate times and the trace showed only that it kept calling `read_slice`.
    //
    // Against the MODE's set, not the registry's. What the mode withholds is a constant of
    // the run, emitted once as `mode_tools` at construction; repeating it on every turn
    // would bury the per-turn narrowings this event exists to make visible under a list
    // that never changes.
    if (specs.size() != mode_specs().size()) {
        std::string withheld;
        for (const parsephony::ToolSpec& s : mode_specs()) {
            bool present = false;
            for (const parsephony::ToolSpec& kept : specs) {
                if (kept.name == s.name) {
                    present = true;
                    break;
                }
            }
            if (!present) {
                withheld += withheld.empty() ? "" : ",";
                withheld += s.name;
            }
        }
        emit("grammar", {{"samplable", std::to_string(specs.size())},
                         {"of", std::to_string(registry_.guard_specs().size())},
                         {"withheld", withheld},
                         {"why_plan_only", must_replan               ? "1"
                                           : ctx_.checklist().empty() ? "no_checklist"
                                                                      : "0"},
                         {"replan_turns_left", std::to_string(replan_turns_)}});
    }
    // Spent here for the same reason, and only after the grammar has been built from it:
    // the turn this narrowing was bought for is the one being assembled right now.
    if (replan_turns_ > 0) {
        --replan_turns_;
    }
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
        // AGAINST THE BUDGET THE RUN IS ACTUALLY MANAGED BY, which is the only denominator
        // that means anything to the person watching the meter.
        //
        // It used to report max = max_new_tokens + prompt tokens. That is not a capacity:
        // it grows with its own numerator, so the meter read prompt/(prompt + 4096) and
        // climbed toward 100% no matter how much room was left. A real run showed
        // "90% of context" at 36,864 prompt tokens against a 96,000-token budget -- 38%.
        // The meter said the run was nearly out of context while compaction, which watches
        // the real budget, correctly saw no reason to trim; the two were measuring
        // different things and only one of them was measuring anything.
        observer_.on_perf(turn.generation, task.prompt.size(),
                          static_cast<std::size_t>(std::max(1, config_.context_budget_tokens)),
                          ctx_.compaction_count());
    }

    emit("generation", {{"status", std::to_string(static_cast<int>(turn.generation.status))},
                        {"tokens", std::to_string(turn.generation.tokens_generated)},
                        {"ttft_ms", std::to_string(turn.generation.ttft_ms)},
                        {"decode_tok_per_s", std::to_string(turn.generation.decode_tok_per_s)}});

    // THE SHAPE OF WHAT WAS SAID, always, even when the text itself is not traced. Three
    // integers per turn, and they separate the two failures that `tokens=4096 status=1`
    // cannot: a long legitimate write, and a model stuck emitting one sentence until the
    // cap. A `degenerate` line in the log is the run saying the model went into a loop --
    // which is not a harness bug, and until now looked exactly like one.
    {
        const TextShape said = shape_of(turn.reasoning + "\n" + turn.assistant_text);
        const bool degenerate = looks_degenerate(said);
        if (degenerate || sink.looped ||
            turn.generation.status == model::GenStatus::LengthCapped) {
            emit("degenerate_text",
                 {{"lines", std::to_string(said.lines)},
                  {"distinct", std::to_string(said.distinct)},
                  {"worst_line_repeats", std::to_string(said.worst_line_repeats)},
                  {"length_capped",
                   turn.generation.status == model::GenStatus::LengthCapped ? "1" : "0"},
                  {"degenerate", degenerate ? "1" : "0"},
                  // Whether the harness CUT it, as against merely noticing afterwards. The
                  // detector above has always been able to see a loop; until the breaker
                  // existed it saw it only once the run had already spent 4096 tokens and
                  // ~60 seconds on it.
                  {"cut_for_looping", sink.looped ? "1" : "0"},
                  {"loop_repeats", std::to_string(sink.loop_repeats)},
                  {"tokens", std::to_string(turn.generation.tokens_generated)}});
        }
    }

    // A turn the breaker cut is TextOnly, which is accurate -- no tool ran -- so it feeds
    // the no-progress counter and three of them end the run, which is the right ending for
    // a model that has started looping.
    //
    // What must NOT survive is the text. Carrying fifty copies of one paragraph into the
    // next prompt is how a loop seeds its own successor: the next turn renders a context
    // whose most recent content is the cycle, at a fixed seed, and draws it again. So the
    // reasoning and the answer are dropped and replaced by the FACT of the loop, which is
    // an observed property of this run and the one thing about it worth remembering.
    //
    // Deliberately NOT reclassified as LengthCapped. It is a different failure with a
    // different fix, and this repo has already paid once for a word that meant two limits
    // (see Agent::halt_reason_).
    if (sink.looped) {
        turn.reasoning.clear();
        turn.assistant_text =
            "(this turn was cut: the same " + std::to_string(loop::LoopBreaker::kWindow) +
            " tokens were emitted " + std::to_string(sink.loop_repeats) +
            " times over, so it had stopped making progress. Nothing was produced and "
            "nothing ran. Do not restate the plan or re-describe the situation -- take the "
            "next concrete action instead, as one tool call.)";
    }

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
        // Every read is eligible now. There used to be a second condition here excluding a
        // read the ledger would refuse -- when reads could be refused, running one in
        // parallel would have raced the decision. Reads always run, so the only question
        // left is the one this predicate was named for.
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
            ran = adopt_readonly_result(calls[i].name, params[i], result);
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
bool Agent::adopt_readonly_result(const std::string& name,
                                  const std::vector<tools::ToolParamValue>& params,
                                  const tools::ToolResult& result) {
    // Same duplicate collapse as the serial path. Safe here for the same reason this path
    // exists at all: a call is only eligible for it if it mutates nothing and executes
    // nothing, and the collapse touches only records already in the context.
    collapse_duplicate_read(name, params, result);
    emit("tool_result", {{"tool", name},
                         {"status", std::string(tools::to_string(result.status))},
                         {"summary", result.summary}});
    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    return result.status != tools::Status::Refused;
}

namespace {

// The range a read call covers, or empty for a whole file. Empty is not "no range" -- it
// is the WIDEST range, and already_in_context() relies on that ordering.
std::string read_range(const std::string& tool,
                       const std::vector<tools::ToolParamValue>& params) {
    if (tool != "read_slice") {
        return {};
    }
    return param_value(params, "start_line") + "-" + param_value(params, "end_line");
}

// Whether this tool answers with FILE CONTENT keyed on a path.
//
// `list_dir` is deliberately absent, and it is the third most repeated call in the run
// this was built from. A directory's answer changes whenever anything under it is created
// or removed, which for a working agent is most turns -- so the invalidation rule would
// have to model the whole tree to be correct, and a wrong one suppresses a listing the run
// needed. read_file and read_slice were 34 of the 44 read turns; they are keyed on one
// path each, and a single write is the entire invalidation rule.
bool is_content_read(const std::string& tool) {
    return tool == "read_file" || tool == "read_slice";
}

// Every tool that answers the SAME question as this one, including itself.
//
// BreakRepeat's mechanism is to take the repeated tool out of the grammar, and it was
// taking exactly one name. `read_file` and `read_slice` return the same bytes about the
// same path, so holding one down routes the next turn straight to the other -- and the
// window is sized from seen_count(), which is keyed on (tool, params), so the substitute
// arrives with a fresh count and the smallest possible window. The suppression never got
// ahead of the ping-pong.
//
// MEASURED: 15 BreakRepeat firings in 38 turns, alternating `read_file` and `read_slice`
// over the same four files, with `corrective_ineffective` firing ten times and the run
// making two writes in total.
//
// The list is deliberately tiny and explicit. A family is a claim that two tools are
// interchangeable ANSWERS, which is true of these two and is not something to infer from a
// schema -- `list_dir` and `search` overlap but neither substitutes for the other.
std::vector<std::string> answer_family(const std::string& tool) {
    if (is_content_read(tool)) {
        return {"read_file", "read_slice"};
    }
    // The same argument, one tool class over, and it went unmade because the read
    // ping-pong was the one in front of us at the time.
    //
    // `write_file` and `replace_in_file` are interchangeable ANSWERS to "make this file say
    // that": anything one can express the other can, and the model picks between them on
    // taste. So holding one hands the next turn the other, on the same path, immediately.
    //
    // MEASURED, in the trace that prompted this: seq 111 suppressed `write_file`, seq 117
    // was a `replace_in_file` on the same file, seq 144 suppressed `replace_in_file`, seq
    // 151 was a `write_file`. Four events, two suppressions, zero turns of delay bought.
    //
    // `append_file` is NOT in this family and the omission is deliberate. It cannot remove
    // or correct anything already in the file, so it does not answer the same question --
    // and a family is a claim of interchangeability, not a grouping by subsystem.
    if (tool == "write_file" || tool == "replace_in_file") {
        return {"write_file", "replace_in_file"};
    }
    return {tool};
}

// WHAT USED TO BE HERE: `turn_is_reads_only`, the guard on the re-read stall test -- one
// write, one build, one search anywhere in the turn and the turn counted as work.
//
// It was the reason the stall test could only ever see a pure read loop, and a model that
// alternates reading with rewriting is not a pure read loop. The test now counts inert
// CALLS instead of classifying the turn by tool name (turn_inert_calls_), which needs no
// predicate at all: a call that added nothing is counted where it happens, and a call that
// might have added something simply is not.

std::string join_names(const std::vector<std::string>& names) {
    std::string out;
    for (const std::string& n : names) {
        out += out.empty() ? "" : ",";
        out += n;
    }
    return out;
}

} // namespace

// Whether the declared contract has ever been observed to pass in this run.
//
// Not "is currently green": one green ever is enough to say the criterion measures
// something reachable, which is the only question the near-miss finding turns on.
bool Agent::mutates_workspace(const std::string& tool) const {
    const tools::ToolDecl* d = registry_.find(tool);
    return d != nullptr && d->mutates_workspace;
}

bool Agent::contract_has_passed() const {
    const std::string declared = canonicalize_check(ctx_.verify_contract());
    for (const context::VerificationRecord& v : ctx_.verifications()) {
        if (v.ran && v.passed && v.contract == declared) {
            return true;
        }
    }
    return false;
}

// Workspace-changing writes since the last verification of any kind ran.
//
// Subtraction against a number the ledger already keeps, rather than a counter of its own.
// VerificationRecord stores workspace_writes at the moment it ran (verification.cpp), and
// ctx_.workspace_writes() only advances on writes that actually changed a file, so the
// difference is exactly "edits this run has not checked". Nothing to reset and nothing to
// forget to reset.
//
// No verifications yet means every write so far is unverified, which is the honest reading
// and also the one that gets a fresh run its first build early instead of late.
std::size_t Agent::writes_since_verification() const {
    const std::vector<context::VerificationRecord>& v = ctx_.verifications();
    // THE LAST READING THAT ACTUALLY RAN, not simply the last one filed.
    //
    // A refusal, a command the shell could not execute, and a command whose exit status
    // cannot mean pass-or-fail are all "not evidence either way" (S6.2) -- and a record that
    // checked nothing must not be able to reset the count of edits that nothing has checked.
    // Reading `v.back()` let it: one `| grep`-terminated build filed a ran=false record,
    // this returned 0, and ForceVerification stopped asking. The run would then have been
    // free to edit forever without ever taking a reading, which is the opposite of what the
    // inverted-verdict guard is for.
    std::size_t at_last = 0;
    for (const context::VerificationRecord& r : v) {
        if (r.ran) {
            at_last = r.workspace_writes;
        }
    }
    const std::size_t now = ctx_.workspace_writes();
    return now > at_last ? now - at_last : 0;
}

// Whether this turn re-sent a workspace mutation the run had already made.
//
// Asked of every call in the turn, because a turn that batches four edits behind one
// primary is the shape this misses otherwise -- the same hole batching opened in the
// repeat detector itself.
//
// Read by run() into the ForceVerification gate. The corrective this feeds runs the check
// rather than withholding the editor, and that choice is the finding: a model that has
// just re-sent an edit is not short of tools, it is short of evidence.
bool Agent::turn_repeated_a_mutation(const TurnResult& turn) const {
    const auto repeated = [this](const std::string& name,
                                 const std::vector<tools::ToolParamValue>& params) {
        const tools::ToolDecl* d = registry_.find(name);
        return d != nullptr && d->mutates_workspace &&
               repeats_.seen_count(name, params) > 1;
    };
    if (turn.outcome != Outcome::ToolCallExecuted) {
        return false;
    }
    if (repeated(turn.tool_name, turn.tool_params)) {
        return true;
    }
    return std::any_of(turn.extra_calls.begin(), turn.extra_calls.end(),
                       [&repeated](const TurnResult::ExtraCall& e) {
                           return repeated(e.tool_name, e.params);
                       });
}

// A RE-READ IS ANSWERED, ALWAYS. What it costs is charged to the context, not to the model.
//
// What this replaced: a ledger of (path, range) notes that REFUSED a read whose bytes it
// believed were still in the prompt, and told the model to "scroll up". Two things were
// wrong with it, and only one of them was a coding error.
//
// The coding error: a whole-file note answered ANY later slice of that path, on the
// reasoning that whole covers part. It does not. The note records that a read HAPPENED, never
// how much of the answer survived the tool's own byte cap into the prompt -- so a whole-file
// note for a 12 KB file elided at the cap would answer "give me lines 96-103" with a refusal
// and no lines. And invalidation keyed on the raw path string, so a write to
// `ResMon/Sources/Swift/HostStatsService.swift` never cleared the note for
// `Sources/Swift/HostStatsService.swift` -- the same file, two spellings, in a workspace that
// really did have both.
//
// MEASURED: one file refused SEVENTEEN times in a 66-turn run, almost all of them slice
// requests answered by a single whole-file note from turn 4. BreakRepeat then fired 13 times
// and took `read_file` away, and the run was cancelled having written nothing.
//
// The design error, which is the one that matters: "scroll up" is not an instruction a model
// can follow. It has a context window, not a viewport. Withholding a tool result to save
// tokens trades a bounded cost (a few thousand tokens) for an unbounded one (a turn, and then
// another turn, because the model asks again). No agent worth shipping refuses to read a file.
//
// So the read runs, and the DUPLICATE is collapsed instead: the newest copy stays verbatim,
// the copy the model was already holding becomes one line, and the prompt ends the turn the
// size it started. That keeps the only thing the ledger was actually built for -- a 12.5 KB
// file read eleven times must not cost eleven copies of 12.5 KB -- and it keeps it on a test
// that cannot be wrong, because identical bytes are identical bytes.
void Agent::collapse_duplicate_read(const std::string& name,
                                    const std::vector<tools::ToolParamValue>& params,
                                    const tools::ToolResult& result) {
    if (!is_content_read(name) || !result.ok()) {
        return;
    }
    // Counted BEFORE the collapse, and for every read regardless of size: once the earlier
    // copy has been rewritten to a pointer, "were these bytes already here?" can no longer
    // be answered. This is the measurement the run that prompted all of this did not have
    // -- thirty turns of re-reading four unchanged files, and `no_progress_streak=0` on
    // every line of the trace, because a turn that calls a tool looked like a turn that did
    // something.
    ++turn_reads_;
    if (ctx_.has_observation(result.summary)) {
        ++turn_reads_redundant_;
        // The same fact, in the units the widened no-progress test counts in. Kept as two
        // counters rather than one because the log reports the breakdown -- "three
        // redundant reads" and "three no-op writes" are different findings about a run,
        // and collapsing them into "three inert calls" would lose the one that names the
        // bug.
        ++turn_inert_calls_;
    }

    // ONLY UNDER CONTEXT PRESSURE -- because this rewrites HISTORY, and rewritten history
    // is a KV cache thrown away.
    //
    // The collapse edits a turn record that sits INSIDE the stable prefix the backend
    // checkpoints. plan_turn_reuse() compares the cache against the new prompt token by
    // token (kv_cache.cpp), finds the divergence at the rewritten message, and correctly
    // returns ReuseMode::Reset -- a full re-prefill from token zero. Nothing is stale and
    // nothing is wrong; the saving is simply bought with the entire prefill.
    //
    // MEASURED on the run this came from, and the separation is total:
    //
    //     turns where a collapse fired (n=15)   median TTFT  21,011 ms
    //     turns where none fired      (n=22)    median TTFT     940 ms
    //
    // -- a 22x penalty, with no overlap between the two groups. The run peaked at 34,096
    // tokens against a 96,000-token budget, so all 33 collapses were reclaiming a few KB
    // of a context that was two thirds empty, and each one cost twenty seconds of a
    // wall-clock-bounded run. Roughly five minutes went to re-prefilling for a saving
    // nothing was short of.
    //
    // So the rule is the one this file already applies to compaction (S8.3: "since a trim
    // pays a full re-prefill anyway, it spends that cost on a summary"): pay the prefill
    // only when buying something with it. Below the mark a compaction would trim TO, the
    // bytes are free and the cache is worth more. Above it, a collapse may spare the run a
    // compaction that costs the same prefill AND destroys information the collapse keeps --
    // so at that point it is strictly the better of the two.
    const auto budget = static_cast<std::size_t>(std::max(1, config_.context_budget_tokens));
    if (last_prompt_tokens_ <= budget * kCollapseAtPercent / 100) {
        return;
    }

    // Below this, the pointer costs more than the bytes it replaces and the collapse is
    // pure loss -- and short results are things like "(empty file)", where two identical
    // observations are not duplication worth touching.
    static constexpr std::size_t kMinDuplicateBytes = 512;
    if (result.summary.size() < kMinDuplicateBytes) {
        return;
    }
    const std::string path = param_value(params, "path");
    const std::string range = read_range(name, params);
    const std::size_t collapsed = ctx_.supersede_duplicate_observation(
        result.summary,
        "(" + path + (range.empty() ? "" : " lines " + range) +
            " was read again below and is unchanged; this earlier identical copy is "
            "collapsed to keep one copy in context)");
    if (collapsed == 0) {
        return;
    }
    // The saving, named. This is the number that justifies the mechanism, and if it stops
    // being large the mechanism should go rather than be tuned.
    emit("duplicate_read_collapsed",
         {{"tool", name},
          {"path", path},
          {"range", range.empty() ? "<whole-file>" : range},
          {"copies_collapsed", std::to_string(collapsed)},
          {"bytes_reclaimed", std::to_string(collapsed * result.summary.size())}});
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

    // The two that END THE RUN. Same reason `plan` is here -- the registry cannot stop the
    // loop -- and they set the halt directly rather than returning a status the loop would
    // have to interpret, because every other way of ending a run is already a named
    // termination_reason and these are two more of them.
    //
    // Both are refused rather than silently ignored outside a conversational mode. They
    // are filtered out of the grammar there, so this is unreachable by sampling; it is
    // reachable by a synthesized call or a future corrective, and "the loop quietly
    // stopped" is not an outcome worth leaving a path to.
    if (name == "ask_user" || name == "exit_plan_mode") {
        if (!policy_.conversational) {
            return tools::ToolResult::refused(
                "'" + name + "' is only available in a mode that yields to the operator");
        }
        const std::string body =
            param_value(params, name == "ask_user" ? "question" : "plan");
        if (body.empty()) {
            return tools::ToolResult::error(
                tools::ErrorClass::Malformed, true,
                "'" + name + "' needs a non-empty " +
                    (name == "ask_user" ? "'question'" : "'plan'"));
        }
        executed = true;
        halted_ = true;
        if (name == "ask_user") {
            halt_reason_ = "awaiting_user";
            emit("ask_user", {{"question", body}});
            // Down the ANSWER channel rather than a notification of its own. The question
            // is prose the model wrote for a human to read, and the surface already knows
            // how to render that -- a second path would be a second thing to keep in step
            // for no difference on screen. Safe here: the token streamer was joined before
            // the turn's text was read, so this is single-threaded and lands ahead of the
            // tool row, which still names `ask_user` and keeps the trace honest.
            if (observer_.on_token) {
                observer_.on_token("answer", body);
            }
            return tools::ToolResult::okay("asked the operator; the run stops here");
        }
        halt_reason_ = "plan_ready";
        emit("plan_ready", {{"chars", std::to_string(body.size())}});
        if (observer_.on_plan_ready) {
            observer_.on_plan_ready(body);
        }
        return tools::ToolResult::okay("presented the plan; the run stops here");
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
    // AGAINST EACH ATOMIC CHECK, not against the contract as one string. `swift test &&
    // swift build` is two criteria, and a run satisfies it with two commands -- neither of
    // which contains the whole declared string, so neither used to be recognised. See
    // contract_checks(): this is not a wider match, it is a match against the right unit.
    // A command that happens to run both halves at once matches both and records both.
    std::vector<std::string> matched;
    if (name == "shell" && !canon_cmd.empty()) {
        for (const std::string& check : contract_checks(ctx_.verify_contract())) {
            if (canon_cmd.find(check) != std::string::npos) {
                matched.push_back(check);
            }
        }
    }
    const bool is_the_check = !matched.empty();
    if (is_the_check) {
        const std::size_t before = ctx_.verifications().size();
        // Filed under each atomic check it satisfies, so every spelling accumulates history
        // on one identity per criterion instead of minting a fresh, historyless one.
        //
        // The command runs ONCE and its one result is recorded against each check it
        // covers. Running it per check would execute the operator's verification n times
        // for one request, and two readings of the same execution are not two observations.
        (void)verifier_.run_and_record_as(cmd, policy_.sandbox_tier, matched);
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
            // TWO CAUSES, ONE FLAG, AND OPPOSITE FIXES. `ran == false` means "not evidence",
            // and until now it always got the "your contract is unrunnable, re-declare it"
            // message. That is right when the command could not execute and actively
            // misleading when it executed fine and merely reported the wrong thing: the
            // contract may be perfect, and the fix is to the pipeline the model appended to
            // it, not to `verify_with`. Sending a run off to re-declare a contract that was
            // never the problem is how it loses turns to the harness's own advice.
            const std::string_view inverted =
                unfalsifiable_reason(executable_form(param_value(params, "command")));
            result.summary +=
                inverted.empty()
                    ? std::string(
                          "\n[contract] This is the run's declared verification contract, "
                          "and it could not be executed at all -- so it can never pass, and "
                          "the run cannot finish while it stands. Fix the command itself and "
                          "re-declare it with plan(verify_with=...): use a command you have "
                          "already watched run in this workspace.")
                    : "\n[contract] That ran, and it is not a reading of the contract: " +
                          std::string(inverted) +
                          ". Nothing is wrong with the contract and nothing needs "
                          "re-declaring -- the output above is still worth reading, but to "
                          "get a verdict run the check on its own, with nothing piped after "
                          "it. Its own exit status is the answer.";
            result.retryable = false;
        }
        emit_verifications(before);
    } else {
        result = registry_.execute(name, params, policy_.sandbox_tier);
        // A NEAR MISS THAT PASSED, while the declared contract never has.
        //
        // The contract is matched by containment, so a command that runs the same program a
        // different way is not the check and never reaches the ledger. Keeping it that way
        // is deliberate -- widening the match would let a WEAKER command be recorded as the
        // contract passing -- but the silence is what loses runs: a run that CORRECTS its
        // own command stops being watched at the moment it starts being right.
        //
        // Both halves of the condition matter. A near miss that fails says nothing. A near
        // miss that passes while the contract has already passed is just a run doing extra
        // work. A near miss that passes while the contract has NEVER passed is the run
        // telling us, with evidence, that it is measuring something the criterion does not.
        if (name == "shell" && result.ok() && !canon_contract.empty() &&
            is_near_miss(cmd, ctx_.verify_contract()) && !contract_has_passed()) {
            passing_near_miss_ = canonicalize_check(cmd);
            emit("near_miss", {{"ran", passing_near_miss_},
                               {"contract", canon_contract}});
        }
    }

    // Refused means the tool NEVER RAN, so it is not an execution (S9.1).
    executed = result.status != tools::Status::Refused;

    // The denominator of the no-progress test, counted at the only place that sees every
    // call the turn makes -- primary and batched alike, since both come through here.
    //
    // A no-op mutation is inert by the same standard a redundant read is: it succeeded, and
    // the state of the world after it is byte-for-byte the state before. A FAILED call is
    // not inert -- an error the model has not seen yet is information, and if it sees the
    // same one twice that is BreakRepeat's finding, not this one.
    if (executed) {
        ++turn_calls_;
        if (result.ok() && result.mutation_was_noop) {
            ++turn_inert_calls_;
        }
    }

    // A successful write IS the deliverable. Nothing recorded these before, so the
    // completion check's "no deliverable was recorded" gate could never be satisfied.
    if (decl != nullptr && decl->mutates_workspace && result.ok()) {
        const std::string path = param_value(params, "path");
        if (!path.empty()) {
            // EVERY WRITE, with the path as the ledger will key it. Writes are the only
            // events that change the workspace, and the deliverable list only reports
            // DISTINCT paths -- so a run editing one file eight times and a run scattering
            // eight files look the same in every other event.
            //
            // `first_touch` is the one that catches the accident this was built for: a run
            // whose workspace root was already `.../ResMon` wrote to `ResMon/Sources/...`,
            // creating a second copy of the tree one level down and then reading the
            // ORIGINAL back and wondering why its edit was missing. Both paths are legal,
            // both writes succeeded, and nothing in the trace marked the moment a second
            // tree appeared.
            const std::string norm = platform::lexically_normal(path);
            // `changed` separates the two things this event used to conflate: a call that
            // moved the workspace, and a call that asked to and found the move already
            // made. Both were logged identically, so the trace of a run rewriting one file
            // twenty times was indistinguishable from a run building twenty files.
            emit("write", {{"path", path},
                           {"normalised", norm},
                           {"tool", name},
                           {"changed", result.mutation_was_noop ? "0" : "1"},
                           {"first_touch", run_wrote_.count(norm) == 0 ? "1" : "0"},
                           {"distinct_files", std::to_string(ctx_.deliverables().size())}});
            if (!result.mutation_was_noop) {
                ctx_.record_deliverable(path);
                // From here on, a whole-file rewrite of this path is the run editing its
                // own output rather than destroying the operator's data -- which is the
                // difference the write gate needs and did not have. Recorded only on
                // SUCCESS: a write that was refused or failed left the file as the operator
                // had it. Normalised to match the gate's lookup key.
                //
                // A no-op does not claim the path either, and that direction is the safe
                // one: nothing was written, so nothing of the operator's was replaced, and
                // the next rewrite of this file should still stop for a human.
                run_wrote_.insert(norm);
            }
            // The post-edit check goes on the SAME observation rather than becoming a
            // turn of its own: it is a consequence of this edit, not a separate action,
            // and a turn would violate one-turn-one-outcome (S9.1) and burn an iteration.
            annotate_with_syntax_check(path, result);
        }
    }

    // The duplicate collapse, on the SAME success condition as everything above it: a call
    // that was refused or failed read nothing, so there is nothing to have duplicated.
    collapse_duplicate_read(name, params, result);

    emit("tool_result", {{"tool", name},
                         {"status", std::string(tools::to_string(result.status))},
                         {"summary", result.summary}});
    return result;
}

// The prompt this run would send right now, in REAL tokens rather than an estimate: the
// prompt is rendered and tokenized anyway, so asking the tokenizer costs nothing extra and
// a character heuristic would be wrong exactly where it matters (code and diffs tokenize
// badly).
//
// Rendered the way step() renders it, and that is the whole point of having one function
// for it. The budget check used to build a DIFFERENT prompt: it passed tools_guidance_ to
// ctx_.render() AND to the template, so the entire <tools> block -- every schema for every
// registered tool -- was counted twice, while step() sends it once. The number the run was
// managed against was never the number it was sending.
std::size_t Agent::prompt_tokens() const {
    const model::ChatTemplate tmpl(tok_);
    return tmpl.render(ctx_.render(""), tools_guidance_).size();
}

void Agent::compact_to_budget() {
    // WITH HEADROOM, AND WITH HYSTERESIS. This used to trim only once the prompt had
    // already passed the whole budget, and then dropped exactly one turn -- so a run that
    // crossed the line sat on it, paying a compaction every single turn thereafter and
    // never getting back any slack.
    //
    // Now it starts at kCompactAtPercent and trims down to kCompactToPercent, so a
    // compaction buys enough room for several turns before the next one is due. The gap
    // between the two is what stops the thrash; the low mark is not aggressive because
    // context that is thrown away is re-read later at the cost of a turn (S8.3).
    const auto budget =
        static_cast<std::size_t>(std::max(1, config_.context_budget_tokens));
    const std::size_t high_water = budget * kCompactAtPercent / 100;
    const std::size_t low_water = budget * kCompactToPercent / 100;

    std::size_t tokens = prompt_tokens();
    if (tokens <= high_water) {
        return;
    }
    const std::size_t before = tokens;
    const std::size_t turns_before = ctx_.recent().size();
    while (ctx_.recent().size() > kMinRecentTurns && tokens > low_water) {
        if (ctx_.compact_oldest(ctx_.recent().size() - 1) == 0) {
            break;
        }
        tokens = prompt_tokens();
    }
    // One event per compaction EVENT, carrying both ends of it. Emitting per dropped turn
    // said how often the trim looped and never said whether it achieved anything.
    emit("compaction", {{"tokens_before", std::to_string(before)},
                        {"tokens_after", std::to_string(tokens)},
                        {"budget_tokens", std::to_string(budget)},
                        {"turns_dropped", std::to_string(turns_before - ctx_.recent().size())},
                        {"recent_turns", std::to_string(ctx_.recent().size())}});
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
        // The reconcile question was asked about a checklist this instruction predates,
        // and the waiver was granted over items that are no longer the open ones. Both
        // are answers to a question about DIFFERENT scope, so they are not carried into
        // it -- otherwise the first follow-up of a run would inherit a standing permission
        // to finish over its own open list.
        reconcile_asked_ = false;
        checklist_waived_ = false;
    }
    return messages.size();
}

// How many firings of one corrective against one target before the log says it is not
// working. Three is the smallest number that cannot be a coincidence: the first is the
// diagnosis, the second is the model not taking it, and the third means the mechanism has
// been applied and re-applied to a situation it does not move.
constexpr std::size_t kIneffectiveAfter = 3;

void Agent::run_contract_now(const char* why) {
    const std::string contract = ctx_.verify_contract();
    const std::size_t unverified = writes_since_verification();
    emit("corrective", {{"corrective", "force_verification"},
                        {"contract", contract},
                        {"why", why},
                        {"unverified_writes", std::to_string(unverified)}});
    const std::size_t before = ctx_.verifications().size();
    // Through verifier_, for the reasons SynthesizeVerification lists: proven_ is the
    // falsifiability cache and a fresh Verifier starts empty, and filing under the
    // canonical contract keeps every spelling of the check on one identity (S10.1).
    (void)verifier_.run_and_record_as(contract, policy_.sandbox_tier,
                                      canonicalize_check(contract));

    // THE LEDGER RENDERS A VERDICT AND NOTHING ELSE -- one `- FAIL swift build` line, no
    // output (ContextStore::render). A run handed that has been told what it already
    // assumed. What changes the next edit is the compiler's actual complaint, and this
    // record is the only route it has into the prompt.
    //
    // Same reasoning as RederiveContract carrying `stuck.failure` rather than pointing at
    // the ledger, and the same rule: the sentence describes a state change that HAS
    // happened -- a command ran, a record was filed -- which is what separates a mechanism
    // from the prose S9.2 forbids.
    std::string detail;
    bool passed = false;
    if (ctx_.verifications().size() > before) {
        const context::VerificationRecord& v = ctx_.verifications().back();
        detail = v.detail;
        passed = v.passed;
    }
    const std::string edits = std::to_string(unverified) +
                              (unverified == 1 ? " edit" : " edits");
    context::TurnRecord marker;
    marker.tool_name = "verify";
    marker.tool_args_summary = contract;
    marker.observation =
        passed ? ("(" + edits + " had gone unchecked, so `" + contract +
                  "` was run for you, and it PASSES. Whatever you were about to change in "
                  "that file does not need changing -- go to your open checklist items.)")
               : ("(" + edits + " had gone unchecked, so `" + contract +
                  "` was run for you. It still fails, and what follows is the CURRENT "
                  "output, with every edit you have made since the last run in it:\n\n" +
                  detail +
                  "\n\nThis, not your last guess, is what your edits did. Read it before "
                  "writing that file again.)");
    marker.observation_is_error = !passed;
    ctx_.add_turn(std::move(marker));
}

bool Agent::escalate_ineffective(Corrective c, const std::string& target,
                                 std::size_t hits) {
    // RUNG ONE: GET THE RUN NEW EVIDENCE.
    //
    // A corrective that keeps firing is a run going round a loop, and a loop is held in
    // place by a fixed input. The model keeps re-deriving the same move because nothing it
    // has done since has been measured -- so the escalation is to measure it. This is the
    // only rung with anything to offer a READ loop too: fresh compiler output changes which
    // file is worth opening, which is the decision the loop is stuck on.
    //
    // Gated on there being unchecked edits. Re-running a check across zero writes returns
    // the same bytes and teaches nothing, and the case where the CONTRACT is the problem
    // belongs to RederiveContract, which outranks everything here.
    //
    // AND NOT WHEN RUNNING THE CHECK IS ALREADY THE MECHANISM THAT FAILED. A ladder whose
    // first rung is the corrective being escalated is not a ladder, it is the tally this
    // pass replaced -- and a run that has been handed three fresh readings of its build
    // and re-sent the same edit each time is not short of evidence, it is not using it.
    const bool already_verifying = c == Corrective::ForceVerification ||
                                   c == Corrective::SynthesizeVerification;
    if (!already_verifying && !ctx_.verify_contract().empty() &&
        writes_since_verification() > 0) {
        run_contract_now("corrective_ineffective");
        return true;
    }
    // RUNG TWO, WHEN THE REPEATED TOOL IS NOT HOW THE WORK GETS DONE.
    //
    // kMaxSuppressTurns is capped because a repeated tool is usually the RIGHT tool with
    // wrong arguments, and holding `read_file` down for a dozen turns would end a healthy
    // run. Three ineffective firings against this one target is that assumption being
    // falsified for THIS run by THIS run, so the cap loosens -- for this target only,
    // scaled by the evidence, and now still bounded, see kMaxEscalatedSuppressTurns.
    //
    // MEASURED: `break_repeat` against `read_file` reported `suppressed_turns` of 4 on its
    // sixth, seventh, eighth, ninth and tenth firings. The window had stopped growing five
    // firings before anybody looked.
    //
    // WHAT USED TO BE HERE, AND WHY IT IS GONE: ForceVerification escalated into this branch
    // too, on the reasoning that taking the editor away is "the right last answer for a run
    // that has been given three fresh builds and written the same thing anyway".
    //
    // MEASURED, on the run this pass came from: it was the single most destructive thing the
    // harness did. `escalated_hold` withheld `write_file,replace_in_file` for 12 turns and
    // `read_file,read_slice` for 20. Across 85 turns the write family was unsamplable on at
    // least 34, and on at least 15 of those `shell` was gone with it -- so the model was
    // asked to fix a build with no way to change a file. It did the only thing left and
    // wrote its files through `shell` heredocs, which are worse in every way that matters:
    // no path is recorded, no syntax check runs, `mutation_was_noop` cannot fire, and
    // because `shell` is not `mutates_workspace` those writes never reach
    // ctx_.workspace_writes() -- which is the input failure_is_unmoved() needs, so the one
    // detector that would have said "this red has not moved" went blind as well.
    //
    // And it was self-sustaining. Withholding the editor made progress impossible, the
    // absence of progress re-fired the corrective, and the corrective widened the hold:
    // 14 `corrective_ineffective` events in that run, 41 across the log.
    //
    // The premise was also false. The run had NOT been given three fresh builds -- it had
    // been given three false greens off a `| grep`-terminated command (see
    // Verifier::run_and_record_as). It was not ignoring evidence; it was acting on evidence
    // the harness had corrupted.
    //
    // So a run that keeps editing blind is escalated by being made to REPLAN, below. That is
    // a real state change, it lasts one turn, and it asks for the thing that is actually
    // missing.
    const bool holds_a_tool = c == Corrective::BreakRepeat && !mutates_workspace(target);
    if (holds_a_tool && !target.empty() && target != "-") {
        const int window =
            std::min(static_cast<int>(hits) * kMaxSuppressTurns, kMaxEscalatedSuppressTurns);
        for (const std::string& held : answer_family(target)) {
            bool updated = false;
            for (auto& [name, turns_left] : suppressed_tools_) {
                if (name == held) {
                    turns_left = std::max(turns_left, window);
                    updated = true;
                    break;
                }
            }
            if (!updated) {
                suppressed_tools_.emplace_back(held, window);
            }
        }
        emit("corrective", {{"corrective", "escalated_hold"},
                            {"after", std::string(to_string(c))},
                            {"tool", target},
                            {"family", join_names(answer_family(target))},
                            {"fired", std::to_string(hits)},
                            {"suppressed_turns", std::to_string(window)}});
        context::TurnRecord marker;
        marker.tool_name = target;
        marker.observation =
            "(`" + target + "` and everything that answers the same question are held for " +
            std::to_string(window) + " turns. This run has been through " +
            std::to_string(hits) +
            " rounds of the same correction against this same call and has come straight "
            "back to it every time. Repeating it will not produce a different result -- "
            "take a different action, or stop and say what is blocking you.)";
        marker.observation_is_error = true;
        ctx_.add_turn(std::move(marker));
        return true;
    }

    // RUNG TWO FOR EVERYTHING THAT EDITS: MAKE IT REPLAN, DO NOT TAKE THE EDITOR AWAY.
    //
    // The state change is real (the next turn's grammar is plan-only, so S9.2 is satisfied by
    // a mechanism rather than a sentence), it lasts exactly one turn, and it cannot strand the
    // run -- restating the checklist is what clears it. Compare the twelve- and twenty-turn
    // holds this replaced: those removed the only tools that can finish the work and left the
    // model to discover `shell` heredocs.
    //
    // It asks for the thing that is actually missing. A run that has re-sent the same edit
    // three times after three fresh readings does not need fewer tools; its plan is wrong,
    // and `plan` is the one call that can say so.
    const bool edits = c == Corrective::ForceVerification ||
                       (c == Corrective::BreakRepeat && mutates_workspace(target));
    if (edits) {
        replan_turns_ = 1;
        emit("corrective", {{"corrective", "escalated_replan"},
                            {"after", std::string(to_string(c))},
                            {"tool", target},
                            {"fired", std::to_string(hits)}});
        context::TurnRecord marker;
        marker.tool_name = target;
        marker.observation =
            "(This run has been through " + std::to_string(hits) +
            " rounds of the same correction against `" + target +
            "` and has come straight back to it every time, so the next turn is a REPLAN: "
            "only `plan` can be called, and restating your checklist is what releases it. "
            "Your editing tools are untouched and will be back immediately afterwards. What "
            "is being asked for is a different approach, not a smaller one -- say what you "
            "now believe is wrong, which checklist items that changes, and what you will do "
            "instead. If something is blocking you that you cannot get past, say that "
            "instead of restating the same plan.)";
        marker.observation_is_error = true;
        ctx_.add_turn(std::move(marker));
        return true;
    }
    return false;
}

void Agent::apply_corrective(Corrective c, const TurnResult& turn) {
    // A CORRECTIVE THAT KEEPS FIRING IS NOT WORKING. Nothing in the harness measured this,
    // and it is the single loudest signal a bad run produces.
    //
    // MEASURED: 66 turns in which BreakRepeat fired thirteen times against `read_file` and
    // six against `read_slice`, always correctly -- the calls WERE repeats -- while the
    // suppression window is capped at kMaxSuppressTurns, so the model waited it out and
    // came straight back. Every individual `corrective` line in that trace looks like the
    // system working. Only the count says otherwise, and nobody was counting.
    //
    // AND THEN NOTHING HAPPENED. The measurement landed and the response did not: this
    // block emitted its event and fell straight through into the same switch, which
    // re-applied the identical mechanism. The seventeen `corrective_ineffective` lines in
    // the run that prompted this pass were followed, every one of them, by a
    // byte-identical `corrective` line. Counting a failure and repeating it is worse than
    // not counting it, because the log now says the harness noticed.
    //
    // So the count is a LADDER, not a tally. Below the rung, apply the mechanism as
    // chosen. At or above it, the mechanism has been given three chances against this
    // exact target and has not moved the run, and the correct move is a different
    // mechanism -- see escalate_ineffective() for which one and why.
    bool escalated = false;
    if (c != Corrective::None) {
        const std::string target = turn.tool_name.empty() ? std::string("-") : turn.tool_name;
        const std::string key = std::string(to_string(c)) + ":" + target;
        const std::size_t hits = ++corrective_hits_[key];
        if (hits >= kIneffectiveAfter) {
            emit("corrective_ineffective",
                 {{"which", std::string(to_string(c))},
                  {"target", target},
                  {"fired", std::to_string(hits)},
                  {"iterations", std::to_string(consecutive_no_progress_)},
                  {"workspace_writes", std::to_string(ctx_.workspace_writes())}});
            escalated = escalate_ineffective(c, target, hits);
        }
    }
    if (escalated) {
        return;
    }
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
            // The window grows with how many times this exact call has come back, so a
            // tool the run keeps returning to is held longer each time. One turn was not
            // enough: with two tools that answer the same question, a one-turn hold on
            // either just routes the next turn to the other one.
            const int seen =
                static_cast<int>(repeats_.seen_count(turn.tool_name, turn.tool_params));
            const int window = std::min(std::max(seen - 1, 1), kMaxSuppressTurns);
            // THE WHOLE FAMILY, not the one name that happened to be at the front of the
            // turn. Suppressing `read_file` alone hands the next turn `read_slice` on the
            // same path, which is the same answer bought for one turn of delay -- see
            // answer_family().
            //
            // MINUS ANYTHING THAT WRITES. This is the suppression floor, applied at the point
            // the hold is built rather than filtered out of the grammar later, so
            // `suppressed_tools_` never contains an entry that would not be honoured.
            //
            // A repeated READ is a run going in circles and the tool is not the deliverable,
            // so holding it forces a different move at no cost to the work. A repeated WRITE
            // is a different animal: the tool is exactly how the work gets done, and the
            // model is short of EVIDENCE, not of tools. That case already has its own and
            // better corrective -- turn_repeated_a_mutation() feeds ForceVerification, which
            // runs the check and hands back the compiler's current complaint.
            //
            // MEASURED: holding the write family produced 34 of 85 turns on which the model
            // could not edit a file, and it responded by writing its sources through `shell`
            // heredocs -- untracked, unchecked, and invisible to workspace_writes().
            std::vector<std::string> family;
            for (const std::string& name : answer_family(turn.tool_name)) {
                if (!mutates_workspace(name)) {
                    family.push_back(name);
                }
            }
            emit("corrective", {{"corrective", "break_repeat"},
                                {"tool", turn.tool_name},
                                {"family", join_names(family)},
                                {"seen", std::to_string(seen)},
                                {"suppressed_turns", std::to_string(window)}});
            for (const std::string& held : family) {
                bool updated = false;
                for (auto& [name, turns_left] : suppressed_tools_) {
                    if (name == held) {
                        // The LONGER of the two, never the newer. A substitute arriving
                        // with a fresh seen_count would otherwise shorten the window its
                        // twin had already earned, which is how the ping-pong outlasted
                        // every suppression aimed at it.
                        turns_left = std::max(turns_left, window);
                        updated = true;
                        break;
                    }
                }
                if (!updated) {
                    suppressed_tools_.emplace_back(held, window);
                }
            }
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
                std::string("`") + turn.tool_name + "` cannot be called for the next " +
                std::to_string(window) +
                (window == 1 ? " turn" : " turns") + " -- take a different action.)";
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
            emit("corrective", {{"corrective", "synthesize_verification"}, {"contract", contract}});
            (void)verifier_.run_and_record_as(contract, policy_.sandbox_tier,
                                              canonicalize_check(contract));
            return;
        }
        case Corrective::ForceVerification: {
            // Mechanism: run the contract, and hand back its output.
            //
            // Note what this deliberately does NOT do: it does not take the editor away.
            // The run is not editing too much, it is editing without looking, and a model
            // denied `write_file` for a turn comes back with the same guess and the other
            // tool. Giving it the compiler's current answer is the only move that changes
            // what it writes next.
            run_contract_now("writes_unverified");
            return;
        }
        case Corrective::BlockRefusedTool: {
            // Mechanism: take the tool off the grammar for the rest of the run, so the
            // next turn cannot sample it at all. Recording the reason matters as much as
            // the block -- a call that silently stops being available is indistinguishable
            // from a model that forgot the tool exists.
            emit("corrective", {{"corrective", "block_refused_tool"}, {"tool", turn.tool_name}});
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
        case Corrective::RederiveContract: {
            // Mechanism: pin the next turn's grammar to `plan`, and record the contract so
            // this fires at most once for it.
            //
            // ONE turn, spent the way BreakRepeat's suppression is spent, and never
            // latched. A flag that only `plan` could clear would re-arm on the next
            // identical red, and a model that re-declares the same broken contract would
            // then get `plan` as its only legal move forever -- fourteen consecutive plan
            // turns and no work, which is exactly how the deleted `must_reconcile`
            // restriction died. Once per contract instead: say it once, with the evidence,
            // and let the run proceed. If it re-declares the same command the run will fail,
            // and `contract_unmoved` in the log is why.
            const UnmovedContract stuck = unmoved_contract(ctx_);
            const std::string declared =
                stuck.unmoved ? stuck.contract : canonicalize_check(ctx_.verify_contract());
            emit("corrective", {{"corrective", "rederive_contract"},
                                {"contract", declared},
                                {"why", stuck.unmoved ? "unmoved" : "passing_near_miss"},
                                {"failure", stuck.unmoved ? stuck.failure
                                                          : passing_near_miss_}});
            disputed_contracts_.insert(declared);
            replan_turns_ = 1;
            context::TurnRecord marker;
            // Two findings, two sentences, because they are different facts and the next
            // move differs: one says the criterion cannot be reached, the other says the
            // run has already reached a better one.
            marker.observation =
                stuck.unmoved
                    ? "(the declared verification contract `" + declared +
                          "` has now failed twice with the same error across your own edits "
                          "to this workspace -- the work moved and the failure did not, so "
                          "this failure is not about the code and no further edit will clear "
                          "it. The failing output is:\n\n" + stuck.failure +
                          "\n\nThe next turn can only call `plan`. Re-derive verify_with "
                          "from what you have actually observed in this workspace -- a "
                          "command you have already watched run here -- rather than "
                          "restating the one above. If the criterion is right and the "
                          "environment is wrong, say so in the checklist instead.)"
                    : "(`" + passing_near_miss_ +
                          "` just PASSED, and it is not the contract this run is measured "
                          "by. The declared contract is `" + declared +
                          "`, which has never passed, and only that one can finish the run "
                          "-- so the work you just proved is not being counted. The next "
                          "turn can only call `plan`: if the command that passed is the "
                          "real proof of this mission, declare it with verify_with. If it "
                          "is not, leave verify_with alone and run the declared contract "
                          "itself.)";
            passing_near_miss_.clear();
            marker.observation_is_error = true;
            marker.first_event_seq = log_.events_written();
            marker.last_event_seq = marker.first_event_seq;
            ctx_.add_turn(std::move(marker));
            return;
        }
        case Corrective::ReconcileChecklist: {
            // Mechanism: pin the next turn's grammar to `plan`, and hand the run the two
            // facts it is being asked to reconcile -- the green it earned, and the items
            // it is still claiming.
            //
            // ONE turn, spent like every other narrowing, and armed once per run. The
            // observation names both legal answers because they are genuinely different
            // findings and the run is the only thing that knows which one is true: either
            // the work is done and the list is stale, or the list is right and the contract
            // is narrower than the mission. The second is the `rename_across_files` failure
            // -- declare `pytest -q`, make it pass, stop, leave the rest of the mission
            // undone -- and it is the reason the ask offers verify_with as an answer rather
            // than only a checkbox.
            const std::size_t open = ctx_.open_checklist_items();
            emit("corrective", {{"corrective", "reconcile_checklist"},
                                {"open_items", std::to_string(open)},
                                {"contract", ctx_.verify_contract()}});
            reconcile_asked_ = true;
            replan_turns_ = 1;
            std::string items;
            for (const context::ChecklistItem& item : ctx_.checklist()) {
                if (!item.done) {
                    items += "\n  - " + item.text;
                }
            }
            context::TurnRecord marker;
            marker.observation =
                "(`" + ctx_.verify_contract() +
                "` is green and proven, which is everything this run needs to report the "
                "mission complete -- except that your own checklist still lists " +
                std::to_string(open) + " open item(s):" + items +
                "\n\nThe run cannot finish while its evidence and its checklist disagree, "
                "so the next turn can only call `plan`. If that work is in fact done, "
                "restate the checklist with those items ticked and the run completes. If it "
                "is not done, leave them open and declare a verify_with that would go red "
                "until it is -- the contract above is passing without covering them, which "
                "means it is measuring less than the mission.)";
            marker.observation_is_error = true;
            marker.first_event_seq = log_.events_written();
            marker.last_event_seq = marker.first_event_seq;
            ctx_.add_turn(std::move(marker));
            return;
        }
        case Corrective::BudgetNearlyGone: {
            // Mechanism: an observation stating the remaining turn count, injected into
            // the history like any other. Not a request to hurry -- a fact the run cannot
            // otherwise obtain, because nothing else in the prompt says how many turns are
            // left, and a model that cannot see the edge cannot avoid stopping on the
            // wrong side of it.
            emit("corrective", {{"corrective", "budget_nearly_gone"},
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
            //
            // NAMED FOR THE LIMIT THAT ACTUALLY FIRED. Both budgets used to end a run as
            // "budget_exhausted", and that word reads as the turn cap -- so a run the
            // CLOCK killed at 45 of its 80 turns looked like proof that 80 was too few,
            // and the fix people reach for is the dial that was never the constraint.
            // Two limits, two names, and the numbers beside them so the trace shows how
            // much of each was left.
            emit("corrective", {{"corrective", "halt_on_budget"},
                  {"limit", out_of_time_ ? "wall_clock" : "turns"},
                  {"max_iterations", std::to_string(config_.budget.max_iterations)},
                  {"wall_clock_seconds",
                   std::to_string(config_.budget.wall_clock_seconds)}});
            halted_ = true;
            halt_reason_ = out_of_time_ ? "wall_clock_exhausted" : "turn_budget_exhausted";
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
        // Carried to the corrective, which is where the run's ending is named.
        out_of_time_ = out_of_time;

        const TurnResult turn = step(cancel);
        ++report.iterations;

        // ONE LINE PER TURN, always. The log had every ingredient of a turn and no record
        // of the turn itself, so reconstructing "what did iteration 41 actually do" meant
        // correlating four event kinds by sequence number and guessing at the boundaries.
        //
        // Everything here is an OBSERVED fact about the turn that just happened, and the
        // three counters are the ones that answer "is this run healthy": how many turns in
        // a row produced no action, how many files exist, and how much of the context is
        // gone. A run that is thrashing shows it in these three before it shows it
        // anywhere else.
        emit("turn", {{"n", std::to_string(report.iterations)},
                      {"outcome", std::string(to_string(turn.outcome))},
                      {"tool", turn.tool_name.empty() ? "-" : turn.tool_name},
                      {"batched", std::to_string(turn.extra_calls.size())},
                      {"ok", turn.tool_result.ok() ? "1" : "0"},
                      {"no_progress_streak", std::to_string(consecutive_no_progress_)},
                      {"deliverables", std::to_string(ctx_.deliverables().size())},
                      {"workspace_writes", std::to_string(ctx_.workspace_writes())},
                      {"open_items", std::to_string(ctx_.open_checklist_items())},
                      {"compactions", std::to_string(ctx_.compaction_count())}});

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
        // EVERY CALL THE TURN MADE, not just the one at the front of it.
        //
        // This recorded the primary call only, and batching made that a hole big enough to
        // drive the whole failure through: a turn that reads four files records ONE of
        // them, so three reads per turn were invisible to the detector no matter how often
        // they came back. seen_count() for those three never left zero, BreakRepeat could
        // not fire on them, and the model re-read the same set turn after turn with the
        // ledger agreeing it had never seen them.
        //
        // MEASURED: a 38-turn run whose turns 34, 35 and 37 were each `read_file` plus
        // three batched reads of the SAME four files. The primary reached seen=2; the other
        // three sat at 0, and the run was cancelled having written two files.
        //
        // A refusal is not an execution (S9.1), so it is not recorded here either.
        for (const TurnResult::ExtraCall& extra : turn.extra_calls) {
            if (extra.result.status != tools::Status::Refused) {
                repeats_.record(extra.tool_name, extra.params);
            }
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

        // A RUN THAT HAS ENDED ITSELF IS NOT A RUN TO CORRECT.
        //
        // `halted_` is only tested at the top of the while, so a turn that sets it during
        // dispatch still falls through everything below it -- the corrective, the
        // completion verdict and the stall detectors. That is right for the budget halt,
        // which is chosen down there and wants the full accounting first. It is wrong for
        // ask_user and exit_plan_mode: that turn produced no deliverable, ticked no
        // checklist item and ran no contract, so the verdict reads incomplete and the
        // streak reads no-progress, and a run that did precisely what it was asked would
        // be handed a corrective on its way out and filed under a stall reason it had
        // nothing to do with.
        //
        // Placed AFTER every record above and before the first judgement below: the turn
        // and its batched calls are in the context the operator's reply continues from,
        // and nothing gets to grade them.
        if (halted_) {
            report.termination_reason = halt_reason_;
            break;
        }

        // Compaction, not eviction (S8.3) -- and only when the BUDGET says so.
        //
        // This used to run unconditionally every turn against a turn-count limit, so a
        // run compacted 18 times in 29 turns while holding ~3k tokens against a 96k
        // budget: it discarded its own history at 3% of capacity, then re-read the same
        // files because it no longer remembered reading them. The budget was never
        // consulted at all. Now a trim happens when the prompt actually needs one.
        compact_to_budget();

        // Asked AFTER the turn's verification (if any) has joined the ledger, because an
        // unmoved failure is a property of two readings and the second one is this turn's.
        //
        // The once-per-contract policy lives here rather than in choose_corrective, which
        // is pure and has no memory. An operator contract is exempt: the operator's
        // criterion is not the model's to re-derive, and telling a run to restate a
        // contract that `plan` is forbidden to change would pin it to a tool that cannot
        // help it (ContextStore::set_verify_contract ignores the call).
        //
        // TWO WAYS a criterion is wrong, one corrective. Either no work moves its failure,
        // or the run has demonstrated a passing command that runs the same program and the
        // contract has never passed. Both mean "the thing being measured is not the thing
        // being built", and both are answered the same way: one turn to re-derive it.
        const UnmovedContract stuck = unmoved_contract(ctx_);
        const std::string wrong_contract =
            stuck.unmoved ? stuck.contract
                          : (passing_near_miss_.empty()
                                 ? std::string()
                                 : canonicalize_check(ctx_.verify_contract()));
        const bool contract_unmoved =
            !wrong_contract.empty() &&
            ctx_.verify_contract_source() != context::ContextStore::ContractSource::Operator &&
            disputed_contracts_.find(wrong_contract) == disputed_contracts_.end();

        // Evaluated TWICE, and the two answers are to different questions.
        //
        // This one is "what did this turn observe", which is what a corrective is chosen
        // from. The one below is "where does the run stand now", after any state a
        // corrective changed -- SynthesizeVerification can run the contract and turn the
        // ledger green, and computing the verdict before that would spend a whole extra
        // turn noticing.
        //
        // ReconcileChecklist needs the first: it fires on the turn the evidence arrives,
        // not a turn later.
        const CompletionVerdict observed = evaluate_completion(ctx_, checklist_waived_);
        const bool checklist_unreconciled =
            observed.evidence_complete && observed.open_items > 0 && !reconcile_asked_;

        // TWO WAYS a run's edits get ahead of its evidence, one corrective.
        //
        // The first is arithmetic: enough writes have piled up since the last check that
        // the next edit cannot be informed by anything observed. The second is a repeat --
        // the run has re-sent an edit it already made, which is the same condition arriving
        // early and loudly. Both mean the model is working from stale output, and both are
        // answered by running the check rather than by narrowing its tools, which is why
        // the repeated mutation is routed HERE and not into BreakRepeat.
        const bool writes_unverified =
            writes_since_verification() >=
                static_cast<std::size_t>(kMaxUnverifiedWrites) ||
            turn_repeated_a_mutation(turn);

        // At most ONE corrective per turn, chosen by rank (S9.2).
        apply_corrective(choose_corrective(turn, repeats_, refusals_, report.iterations,
                                           config_.budget, out_of_time,
                                           !ctx_.verify_contract().empty(),
                                           contract_unmoved, checklist_unreconciled,
                                           writes_unverified),
                         turn);

        // Any verification a corrective produced flows to the UI from the ledger --
        // the one choke point, so nothing can report a result that was not recorded.
        emit_verifications(before);

        // NOT IN A CONVERSATIONAL MODE, for either of the two judgements below.
        //
        // Completion here means "the declared command has been seen to pass and no item is
        // open". A plan-mode run declares no command and writes no code, so it can never
        // satisfy that -- and it should not have to: it is finished when the human says the
        // plan is right, which is what exit_plan_mode is for. Left in, the verdict spends a
        // `not_complete` event every turn saying the run has no contract, which is true and
        // is not news.
        //
        // The stall detectors below are the half that actually broke it. A conversational
        // turn produces text and calls nothing, which is `made_no_move` by definition, so
        // three of them in a row ended the run `text_only_no_progress` -- a stall reason,
        // for a mode whose entire output is text. That is what plan mode looked like from
        // the outside: it talked, and the harness scored talking as failing.
        if (policy_.conversational) {
            // AND A TURN THAT CALLED NOTHING HAS SAID ITS PIECE.
            //
            // Removing the stall detectors removes the only bound this loop had, and a
            // mode that never stalls and never completes would run to the budget narrating
            // -- 200 turns of it. But the ending is not a stall, it is the shape of a
            // conversation: the model read what it needed, wrote an answer and made no
            // call, and there is nobody but the operator who can say what happens next.
            //
            // This is also what makes the mode work without the model having learned
            // `ask_user`. That tool is worth having -- it is explicit, and it tells the
            // surface a question is waiting rather than a statement -- but a mode that
            // depends on the model reaching for a new tool to avoid spinning is a mode
            // that spins.
            if (turn.outcome == Outcome::TextOnly) {
                halted_ = true;
                halt_reason_ = "awaiting_user";
                report.termination_reason = halt_reason_;
                emit("yielded", {{"why", "text_only_turn"}});
                break;
            }
            continue;
        }

        const CompletionVerdict verdict = evaluate_completion(ctx_, checklist_waived_);
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
        // A turn that spent every one of its calls re-reading bytes the prompt ALREADY
        // HELD made no move either, and this is the case that mattered most and was
        // invisible. Calling a tool was treated as progress by definition, so a run could
        // re-read the same four files for thirty turns with the streak pinned at zero, and
        // the only thing left to end it was the budget or the operator.
        //
        // The test is byte identity on every read the turn made, so it cannot misfire on
        // real work: a read after a write returns different bytes, a first read has nothing
        // to match, and any non-read call in the turn -- a write, a build, a search -- takes
        // the turn out of this branch entirely.
        //
        // MEASURED: turns 34, 35 and 37 of a 38-turn run each read the same four unchanged
        // files, all twelve reads collapsed as duplicates, and `no_progress_streak` read 0.
        //
        // AND A WRITE STILL CANCELLED IT, which is how the same failure came back wearing
        // different tools. The test above asks whether every READ was redundant and hands
        // the turn a pass the moment it also writes -- so a model that alternates read and
        // rewrite is inert on every turn and stalled on none of them. That is not a corner
        // case; it is what the read suppression pushed the next run into.
        //
        // MEASURED, the run that prompted this pass: 73 turns, 39 writes, 13 of them
        // rewriting bytes already on disk, build red throughout, `no_progress_streak` at 0
        // on 68 turns and never past 1 against a cap of 3. Not one stall was detectable.
        //
        // So the question widens from "were all the reads redundant" to "did ANY call this
        // turn add something" -- counted over every call, with a write the file did not
        // need counting exactly as a read the prompt already held. It cannot misfire on
        // real work: a changing write, a build, a search, a listing, or a failed call all
        // increment the denominator and never the numerator, so one of them anywhere in the
        // turn takes it out of this branch, exactly as before.
        const bool inert_turn = turn.outcome == Outcome::ToolCallExecuted &&
                                turn_calls_ > 0 && turn_inert_calls_ == turn_calls_;
        const bool made_no_move = turn.outcome == Outcome::TextOnly ||
                                  turn.outcome == Outcome::LengthCapped || inert_turn;
        consecutive_no_progress_ = made_no_move ? consecutive_no_progress_ + 1 : 0;
        if (inert_turn) {
            // The breakdown, not just the verdict: "four redundant reads" and "two writes
            // that changed nothing" are different bugs with the same streak, and the
            // triage that reads this log needs to tell them apart.
            emit("inert_turn",
                 {{"calls", std::to_string(turn_calls_)},
                  {"redundant_reads", std::to_string(turn_reads_redundant_)},
                  {"unchanged_writes",
                   std::to_string(turn_inert_calls_ - turn_reads_redundant_)},
                  {"streak", std::to_string(consecutive_no_progress_)}});
        }

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
            // THE ONE PLACE THE CHECKLIST STOPS GATING. A run that has been asked to
            // reconcile a green ledger against its own open items, has answered without
            // clearing them, and has now stopped making any move at all, is not going to
            // tick them. Ending it `text_only_no_progress` would throw away a proven green
            // over bookkeeping -- which is the failure that made the seventh pass drop the
            // checklist from the gate in the first place, and it is worth not rebuilding.
            //
            // The waiver costs an ask, and the ask is what makes this honest: the run was
            // given the turn, the mechanism and both legal answers, and said nothing. The
            // ending still reports the open items, so the disagreement reaches the human
            // rather than being resolved silently in the harness's favour.
            if (reconcile_asked_ && !checklist_waived_ && verdict.evidence_complete) {
                checklist_waived_ = true;
                const CompletionVerdict waived = evaluate_completion(ctx_, true);
                if (waived.complete) {
                    report.completed = true;
                    report.self_declared = waived.self_declared();
                    report.termination_reason = "completed";
                    emit("completion",
                         {{"reason", waived.reason},
                          {"open_items", std::to_string(waived.open_items)},
                          {"checklist_waived", "1"},
                          {"self_declared", waived.self_declared() ? "1" : "0"}});
                    break;
                }
            }
            // Named for what actually happened: "it narrated" and "it thought until
            // the token cap" are different failures and want different responses.
            //
            // And so is the third one, which is why it gets its own name rather than
            // borrowing the narration one. A run that called tools on every turn and
            // changed nothing is not a run that stopped working -- it is a run that kept
            // working on nothing, and reporting it as `text_only_no_progress` sends the
            // reader looking for narration that is not in the trace. The same mistake
            // `budget_exhausted` made when it meant two different limits.
            report.termination_reason =
                stalled_without_plan ? "text_only_no_plan"
                : turn.outcome == Outcome::LengthCapped ? "length_capped_no_progress"
                : inert_turn                            ? "inert_calls_no_progress"
                                                        : "text_only_no_progress";
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
