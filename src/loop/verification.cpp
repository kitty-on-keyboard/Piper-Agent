#include "src/loop/verification.hpp"

#include <algorithm>
#include <cctype>

namespace lmp::loop {
namespace {

std::string trim(std::string s) {
    const auto not_space = [](unsigned char c) { return std::isspace(c) == 0; };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    return s;
}

// A trailing pipe into a pager or a truncator: `| tail -20`, `| head -n 5`, `| cat`.
//
// These are not neutral, and that is the point of removing them rather than tolerating
// them. A shell pipeline exits with the status of its LAST element, so
// `pytest tests/ | tail -20` exits 0 when pytest fails, when pytest errors, and when
// there is no pytest at all -- `tail` succeeded either way.
//
// MEASURED: a run declared `python -m pytest tests/ -v --tb=short 2>&1 | tail -20` as its
// contract in an EMPTY workspace. `python` does not exist on this host, so the baseline
// was `command not found` -- and it was recorded as PASSING, because tail exited 0. The
// agent was told its tests already passed before it had written a line, and no failing
// test it wrote afterwards could ever be seen to fail.
//
// Only formatters are stripped. `| grep -q FAIL` and `| wc -l` are predicates whose
// status is load-bearing, and removing one would change what is being checked.
std::size_t formatter_pipe_at(const std::string& s) {
    static constexpr std::string_view kFormatters[] = {"cat", "tail", "head",
                                                       "tee", "less", "more"};
    const std::size_t bar = s.find_last_of('|');
    // `||` is an or-list, not a pipe, and its right side is a real command.
    if (bar == std::string::npos || bar == 0 || s[bar - 1] == '|' ||
        (bar + 1 < s.size() && s[bar + 1] == '|')) {
        return std::string::npos;
    }
    const std::size_t word = s.find_first_not_of(" \t", bar + 1);
    if (word == std::string::npos) {
        return std::string::npos;
    }
    const std::size_t end = s.find_first_of(" \t", word);
    const std::string name =
        s.substr(word, end == std::string::npos ? std::string::npos : end - word);
    for (std::string_view f : kFormatters) {
        if (name == f) {
            return bar;
        }
    }
    return std::string::npos;
}

// The trailing `|| <something>` of an or-list, or npos. Only when what follows is a
// STATUS SWALLOWER -- a command that always succeeds and produces no verdict of its own.
// `make || make clean` is a real fallback and its status still means something.
std::size_t swallowing_or_at(const std::string& s) {
    const std::size_t bar = s.rfind("||");
    if (bar == std::string::npos) {
        return std::string::npos;
    }
    const std::size_t word = s.find_first_not_of(" \t", bar + 2);
    if (word == std::string::npos) {
        return std::string::npos;
    }
    const std::size_t end = s.find_first_of(" \t", word);
    const std::string name =
        s.substr(word, end == std::string::npos ? std::string::npos : end - word);
    return name == "echo" || name == "true" || name == ":" ? bar : std::string::npos;
}

} // namespace

std::string_view unfalsifiable_reason(std::string_view command) {
    const std::string s = trim(std::string(command));
    if (s.empty()) {
        return {};
    }
    // AFTER executable_form has taken the swallowers off: what is left is the pipeline
    // whose last stage decides the exit status. A trailing `grep` makes the check mean
    // "the pattern was FOUND" -- which for the error-pattern spelling every model reaches
    // for is exactly backwards: it goes green on a broken build and red on a clean one.
    // Naming that is worth far more to the run than any amount of proving.
    //
    // QUOTE-AWARE, because the pattern being grepped for is very often an alternation:
    // in `grep -E "(error:|warning:)"` the last `|` in the string is INSIDE the regex and
    // is not a pipe at all. Reading it as one made this return "nothing wrong here" for
    // the exact contract it exists to catch.
    std::size_t bar = std::string::npos;
    char quote = '\0';
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (quote != '\0') {
            if (c == quote) {
                quote = '\0';
            }
            continue;
        }
        if (c == '\'' || c == '"') {
            quote = c;
        } else if (c == '|') {
            // `||` is an or-list; step over both halves so neither is taken for a pipe.
            if (i + 1 < s.size() && s[i + 1] == '|') {
                ++i;
                bar = std::string::npos;
            } else {
                bar = i;
            }
        }
    }
    if (bar == std::string::npos || bar == 0) {
        return {};
    }
    const std::size_t word = s.find_first_not_of(" \t", bar + 1);
    if (word == std::string::npos) {
        return {};
    }
    const std::size_t end = s.find_first_of(" \t", word);
    const std::string name =
        s.substr(word, end == std::string::npos ? std::string::npos : end - word);
    if (name == "grep" || name == "egrep" || name == "rg") {
        return "its exit status is the final `grep`'s, so it reports whether the PATTERN "
               "WAS FOUND, not whether the work succeeded -- with an error pattern that is "
               "backwards: green while the build is broken, red once it is clean";
    }
    return {};
}

std::string executable_form(std::string_view command) {
    std::string s(command);

    // Strip reporting wrappers that change nothing about what is being verified. This
    // is why: a proof of falsifiability is expensive (it breaks and restores the
    // workspace), and if `cmake --build build` and `cmake --build build; echo $?` had
    // separate identities the agent would pay for the proof twice and, worse, could
    // present the unproven variant as evidence.
    // `|| echo ...`, `|| true` and `|| :` are the same class as `; echo $?` and by far the
    // most common: a model writes `swift build | grep error: || echo "Build successful"`
    // meaning it to read nicely, and the or-list makes the whole command exit 0 ALWAYS.
    // The check is then structurally incapable of red, so it can never become evidence,
    // and the run is told forever that its green is unproven.
    //
    // MEASURED: exactly that contract, on a workspace whose build was full of errors,
    // exited 0 every time. Stripping the swallower is what lets the command's real status
    // through -- see unfalsifiable_reason() for the ones that survive stripping.
    static constexpr std::string_view kTrailers[] = {"; echo $?", "&& echo $?",
                                                     "; echo done", "2>&1", "| cat",
                                                     "|| true",    "|| :"};
    bool changed = true;
    while (changed) {
        changed = false;
        s = trim(s);
        for (std::string_view t : kTrailers) {
            if (s.size() > t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0) {
                s.resize(s.size() - t.size());
                changed = true;
            }
        }
        // Before the formatter pipe, because `... | tail -5 || echo ok` hides the pipe
        // behind the or-list and neither comes off until the other has.
        if (const std::size_t bar = swallowing_or_at(s); bar != std::string::npos) {
            s.resize(bar);
            changed = true;
        }
        // After the fixed trailers, so `... | tail -20` and then the `2>&1` it was hiding
        // both come off in one pass.
        if (const std::size_t bar = formatter_pipe_at(s); bar != std::string::npos) {
            s.resize(bar);
            changed = true;
        }
    }
    return s;
}

namespace {

std::string collapse_spaces_for_identity(const std::string& s) {
    std::string out;
    bool in_space = false;
    for (char c : s) {
        const bool space = std::isspace(static_cast<unsigned char>(c)) != 0;
        if (space) {
            in_space = true;
            continue;
        }
        if (in_space && !out.empty()) {
            out.push_back(' ');
        }
        in_space = false;
        out.push_back(c);
    }
    return out;
}

// The harness's own T1 compatibility flag, removed for IDENTITY ONLY.
//
// `xcrun swift build` and `xcrun swift build --disable-sandbox` are the SAME CHECK. The flag
// asserts nothing about the code: it stops SwiftPM starting a second sandbox inside ours, and
// at T1 the harness adds it whether the model typed it or not (tools::t1_compat_rewrite), so
// which spelling reaches the ledger is not even the model's choice.
//
// MEASURED: a run declared `xcrun swift build` and took nine readings of it. It then
// corrected itself to `xcrun swift build --disable-sandbox`, which minted a SECOND contract
// with an empty history -- `falsifiable` dropped back to 0, a second baseline_check ran, and
// failure_is_unmoved (which requires two readings to share a contract string) went blind
// across the fork at the exact moment the model started getting the command right. Twelve
// readings of one build, split into two ledgers, neither able to see the other.
//
// DELIBERATELY NOT in executable_form(): that form is what runs, and taking the flag off it
// would hand the nesting EPERM straight back.
std::string strip_compat_flag(std::string s) {
    static constexpr std::string_view kFlag = "--disable-sandbox";
    std::size_t at = 0;
    while ((at = s.find(kFlag, at)) != std::string::npos) {
        const std::size_t end = at + kFlag.size();
        // A whole token, so a longer flag that merely starts with these bytes survives.
        const bool bounded_left =
            at == 0 || std::isspace(static_cast<unsigned char>(s[at - 1])) != 0;
        const bool bounded_right =
            end >= s.size() || std::isspace(static_cast<unsigned char>(s[end])) != 0;
        if (!bounded_left || !bounded_right) {
            at = end;
            continue;
        }
        s.erase(at, kFlag.size());
    }
    return s;
}

} // namespace

// IDENTITY, not something to run. The whitespace collapse is exactly why this and
// executable_form() are two functions: `pytest -k "a  or  b"` is the same CHECK as its
// single-spaced spelling and must share one ledger entry, but it is NOT the same command,
// and running this form would silently rewrite the model's own quoted arguments.
std::string canonicalize_check(std::string_view command) {
    // Flag stripped BEFORE the collapse, so the space it leaves behind is cleaned up here
    // rather than becoming part of the identity.
    return collapse_spaces_for_identity(strip_compat_flag(executable_form(command)));
}

namespace {

// Top-level `&&` positions -- outside quotes, and outside $(...) / `...` substitution.
//
// A naive find() splits `sh -c "a && b"` into two checks, neither of which is a command,
// and files two ledger identities that can never run. The quoting rules here are the shell's
// own, kept to the three forms that actually appear in a verification contract.
std::vector<std::size_t> top_level_and_positions(std::string_view s) {
    std::vector<std::size_t> out;
    char quote = '\0';
    int depth = 0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (quote != '\0') {
            if (c == '\\' && quote == '"' && i + 1 < s.size()) {
                ++i;
            } else if (c == quote) {
                quote = '\0';
            }
            continue;
        }
        if (c == '\'' || c == '"' || c == '`') {
            quote = c;
            continue;
        }
        if (c == '$' && i + 1 < s.size() && s[i + 1] == '(') {
            ++depth;
            ++i;
            continue;
        }
        if (c == ')' && depth > 0) {
            --depth;
            continue;
        }
        if (depth == 0 && c == '&' && i + 1 < s.size() && s[i + 1] == '&') {
            out.push_back(i);
            ++i;
        }
    }
    return out;
}

} // namespace

std::vector<std::string> contract_checks(std::string_view contract) {
    const std::vector<std::size_t> ands = top_level_and_positions(contract);
    if (ands.empty()) {
        std::string one = canonicalize_check(contract);
        if (one.empty()) {
            return {};
        }
        return {std::move(one)};
    }

    std::vector<std::string> checks;
    std::size_t at = 0;
    const auto take = [&](std::size_t from, std::size_t to) {
        const std::string_view seg = contract.substr(from, to - from);
        // A segment that names no program asserts nothing: `cd build`, `export CC=clang`.
        // check_program() already knows how to see past both, and it is the same function
        // the near-miss finding uses -- so "is this a command?" has one answer here.
        if (check_program(seg).empty()) {
            return;
        }
        std::string canon = canonicalize_check(seg);
        if (canon.empty()) {
            return;
        }
        // Deduplicated: a contract that names the same check twice is one criterion, and
        // two entries would demand two independent falsifiability proofs of one command.
        if (std::find(checks.begin(), checks.end(), canon) == checks.end()) {
            checks.push_back(std::move(canon));
        }
    };
    for (const std::size_t pos : ands) {
        take(at, pos);
        at = pos + 2;
    }
    take(at, contract.size());

    // Everything was navigation or nothing parsed: fall back to the whole string, so a
    // contract this cannot decompose behaves exactly as it did before decomposition existed
    // rather than silently becoming no criterion at all.
    if (checks.empty()) {
        std::string one = canonicalize_check(contract);
        if (one.empty()) {
            return {};
        }
        return {std::move(one)};
    }
    return checks;
}

std::string failure_signature(std::string_view detail) {
    std::string out;
    out.reserve(detail.size());
    bool in_digits = false;
    bool in_space = false;
    for (const char c : detail) {
        const auto uc = static_cast<unsigned char>(c);
        if (std::isdigit(uc) != 0) {
            if (!in_digits) {
                out.push_back('#');
                in_digits = true;
            }
            in_space = false;
            continue;
        }
        in_digits = false;
        if (std::isspace(uc) != 0) {
            in_space = true;
            continue;
        }
        if (in_space && !out.empty()) {
            out.push_back(' ');
        }
        in_space = false;
        out.push_back(c);
    }
    return out;
}

bool failure_is_unmoved(const std::vector<context::VerificationRecord>& ledger,
                        std::size_t index) {
    if (index >= ledger.size()) {
        return false;
    }
    const context::VerificationRecord& me = ledger[index];
    // Only an executed red can be unmoved. A refusal and a command the shell could not run
    // are not evidence in either direction (S6.2), and a green is not a failure at all.
    if (!me.ran || me.passed) {
        return false;
    }
    const std::string sig = failure_signature(me.detail);
    for (std::size_t i = 0; i < ledger.size(); ++i) {
        if (i == index) {
            continue;
        }
        const context::VerificationRecord& other = ledger[i];
        if (!other.ran || other.passed || other.contract != me.contract) {
            continue;
        }
        if (other.workspace_writes != me.workspace_writes &&
            failure_signature(other.detail) == sig) {
            return true;
        }
    }
    return false;
}

UnmovedContract unmoved_contract(const context::ContextStore& ctx) {
    const std::string declared = canonicalize_check(ctx.verify_contract());
    if (declared.empty()) {
        return {};
    }
    const auto& vs = ctx.verifications();
    // The LATEST executed reading of the declared contract, the same one
    // evaluate_completion() gates on -- so this answers a question about the reading that
    // is actually deciding the run, not about some earlier one it has moved past.
    std::size_t latest = vs.size();
    for (std::size_t i = 0; i < vs.size(); ++i) {
        if (vs[i].ran && vs[i].contract == declared) {
            latest = i;
        }
    }
    if (latest == vs.size() || !failure_is_unmoved(vs, latest)) {
        return {};
    }
    return {true, declared, vs[latest].detail};
}

std::string_view check_program(std::string_view command) {
    std::size_t at = 0;
    while (at < command.size()) {
        while (at < command.size() && (command[at] == ' ' || command[at] == '\t')) {
            ++at;
        }
        std::size_t end = at;
        while (end < command.size() && command[end] != ' ' && command[end] != '\t') {
            ++end;
        }
        if (end == at) {
            return {};
        }
        const std::string_view word = command.substr(at, end - at);
        // `cd DIR &&` -- skip the word AND its argument; the connective is handled below.
        if (word == "cd") {
            at = end;
            while (at < command.size() && (command[at] == ' ' || command[at] == '\t')) {
                ++at;
            }
            while (at < command.size() && command[at] != ' ' && command[at] != '\t') {
                ++at;
            }
            continue;
        }
        // A connective, or an inline `VAR=value` assignment: neither is the program.
        if (word == "&&" || word == ";" || word == "||" || word == "|" ||
            word.find('=') != std::string_view::npos) {
            at = end;
            continue;
        }
        return word;
    }
    return {};
}

bool is_near_miss(std::string_view command, std::string_view contract) {
    const std::string canon_cmd = canonicalize_check(command);
    const std::string canon_contract = canonicalize_check(contract);
    if (canon_cmd.empty() || canon_contract.empty()) {
        return false;
    }
    // Being the check wins: containment is what dispatch_call routes on, and a command
    // that IS the contract is not a miss of any kind.
    if (canon_cmd.find(canon_contract) != std::string::npos) {
        return false;
    }
    // Nor is a command that IS one of the contract's atomic checks. dispatch_call routes
    // per check now, so running `swift test` against `swift test && swift build` is
    // recorded evidence, not a miss -- and reporting it as one told a run that its correct
    // move was not being counted while the ledger was in fact counting it.
    for (const std::string& check : contract_checks(contract)) {
        if (!check.empty() && canon_cmd.find(check) != std::string::npos) {
            return false;
        }
    }
    const std::string_view program = check_program(canon_contract);
    return !program.empty() && check_program(canon_cmd) == program;
}

bool Verifier::proven_by(const std::string& contract,
                         const context::VerificationRecord& pending) const {
    if (std::find(proven_.begin(), proven_.end(), contract) != proven_.end()) {
        return true;
    }
    std::vector<context::VerificationRecord> ledger = ctx_.verifications();
    ledger.push_back(pending);
    const std::size_t self = ledger.size() - 1;
    for (std::size_t i = 0; i < ledger.size(); ++i) {
        if (i == self) {
            continue; // present so it can disqualify; never its own proof
        }
        if (ledger[i].ran && !ledger[i].passed && ledger[i].contract == contract &&
            !failure_is_unmoved(ledger, i)) {
            return true;
        }
    }
    return false;
}

bool Verifier::is_proven(const std::string& command) const {
    const std::string canon = canonicalize_check(command);
    if (std::find(proven_.begin(), proven_.end(), canon) != proven_.end()) {
        return true;
    }
    // A red OBSERVED earlier in this run is the proof, and it is free.
    //
    // This is the FAIL_TO_PASS discipline: run the check before the fix, see it red, fix,
    // see it green. That sequence is exactly "this check has been shown capable of
    // failing", and it is what the surrounding industry actually does -- reverting a
    // patch to manufacture a red is mutation testing, a QA activity, not an inline agent
    // step. prove_falsifiable() remains for checks a run wants to prove deliberately.
    //
    // A refusal is skipped: the command never ran, so it is not evidence (S6.2).
    //
    // AND SO IS AN UNMOVED RED. A red proves a check capable of failing only if it is a red
    // ABOUT THE THING BEING CHECKED, and this is the same hole never_executed() was added
    // for, one level up: exit 127 is a red that proves nothing because the command never
    // ran, and a red that is identical across a changed workspace proves nothing because it
    // is not reading the workspace. Both certify a contract as evidence on the strength of
    // a failure that has nothing to do with the code -- see failure_is_unmoved().
    const auto& vs = ctx_.verifications();
    for (std::size_t i = 0; i < vs.size(); ++i) {
        if (vs[i].ran && !vs[i].passed && vs[i].contract == canon &&
            !failure_is_unmoved(vs, i)) {
            return true;
        }
    }
    return false;
}

bool Verifier::run_and_record(const std::string& command, int approved_tier) {
    return run_and_record_as(command, approved_tier, canonicalize_check(command));
}

bool Verifier::run_and_record_as(const std::string& command, int approved_tier,
                                 const std::string& contract_id) {
    return run_and_record_as(command, approved_tier,
                             std::vector<std::string>{contract_id});
}

bool Verifier::run_and_record_as(const std::string& command, int approved_tier,
                                 const std::vector<std::string>& contract_ids) {
    if (contract_ids.empty()) {
        return false;
    }
    // What RUNS has the status-swallowing wrappers removed and nothing else changed.
    // A contract came to be recorded green on the strength of `tail`'s exit status; this
    // is the fix. It cannot run something weaker than what was asked for -- it removes
    // only the part that was hiding the answer. Deliberately NOT canonicalize_check(),
    // which also collapses whitespace: that is right for a ledger key and wrong for a
    // command line.
    const std::string to_run = executable_form(command);
    // AN INVERTED EXIT STATUS IS NOT EVIDENCE, IN EITHER DIRECTION.
    //
    // unfalsifiable_reason() has always known that a check ending in `grep` reports whether
    // the PATTERN WAS FOUND rather than whether the work succeeded. It was only ever asked at
    // DECLARATION time, about the contract as declared -- never about the command actually
    // being recorded against it. dispatch_call routes by containment, so
    // `swift build 2>&1 | grep "error:" | head -30` contains the declared `swift build` and
    // was accepted as a reading of it; executable_form() strips the `| head` formatter and
    // leaves `| grep`, whose status is the pipeline's; and grep exits 0 when it FINDS errors.
    //
    // MEASURED, and this is the run this whole pass came from: three records with
    // `passed: 1, falsifiable: 1` against `xcrun swift build` on a tree the same command was
    // printing sixteen compiler errors from. Every `| tail` spelling in that trace is a FAIL
    // and every `| grep` spelling is a PASS, on the same broken workspace. The model then
    // said "the build has been passing consistently (12 successful runs)" and began marking
    // tasks complete -- which was not a hallucination, it was an accurate reading of the
    // ledger it had been handed.
    //
    // The command still RUNS: `swift build 2>&1 | grep error:` is a perfectly reasonable way
    // to read a build's errors, and refusing it would take a normal move away. What it must
    // not do is become a VERDICT. So this is recorded exactly as a refusal and an
    // unexecutable command are -- ran=false, passed=false, not evidence either way (S6.2) --
    // and the model is told which part of its own command line destroyed the answer.
    const std::string_view inverted = unfalsifiable_reason(to_run);
    // ONCE, above the loop. See the header: n ids are n readings of ONE execution.
    const tools::ToolResult r =
        registry_.execute("shell", {{"command", to_run}}, approved_tier);

    bool passed = false;
    for (const std::string& contract_id : contract_ids) {
        context::VerificationRecord rec;
        rec.contract = contract_id;
        // Refused is NOT failed (S6.2): the command never ran, so it is not evidence in
        // either direction, and recording it as a failure would send the agent off fixing
        // a build that was never attempted.
        //
        // A command the shell could not execute is the same case wearing a different exit
        // code -- see ToolResult::never_executed(). `python: command not found` is not the
        // test suite failing, and counting it as a red would hand the run a falsifiability
        // proof for a check that has never once been executed.
        rec.passed = r.status == tools::Status::Ok && inverted.empty();
        rec.ran = r.status != tools::Status::Refused && !r.never_executed() &&
                  inverted.empty();
        // Stamped at the moment of the reading, because the question it answers is "how much
        // work had happened when this was observed" -- a number read later would be the run's
        // final total and every record would carry the same one.
        rec.workspace_writes = ctx_.workspace_writes();
        rec.detail =
            r.status == tools::Status::Refused ? "REFUSED (never ran): " + r.summary
            : r.never_executed()
                ? "NEVER RAN (the command could not be executed, so this is not "
                  "evidence either way): " + r.summary
            : !inverted.empty()
                ? "NOT A VERDICT (the command ran, but " + std::string(inverted) +
                      "). Its output is below and is still worth reading; what it cannot do "
                      "is say whether the check passed. Run the check itself, with nothing "
                      "piped after it, to get an answer.\n" + r.summary
                : r.summary;
        // Asked with this record VISIBLE but not eligible to be the proof. Both halves matter
        // and they pull in opposite directions: a check still must not prove itself (S10.2),
        // and an unmoved pair is only visible once both of its readings exist -- so asking
        // against the ledger as it stands would let the second identical red be certified by
        // the first, which is precisely the run this was built from. `detail` is therefore set
        // ABOVE this line: the signature is computed from it.
        rec.falsifiable = proven_by(contract_id, rec);
        passed = rec.passed;
        ctx_.record_verification(rec);
    }
    return passed;
}

bool Verifier::prove_falsifiable(const std::string& command, int approved_tier,
                                 const std::function<bool()>& breaker,
                                 const std::function<bool()>& restore) {
    const std::string canon = canonicalize_check(command);
    if (is_proven(command)) {
        return true; // paid for once
    }

    const auto run = [&]() {
        // Same reason as run_and_record_as: a green/red/green sequence read off a
        // formatter's exit status would prove nothing at all.
        return registry_.execute("shell", {{"command", executable_form(command)}},
                                 approved_tier)
                   .status == tools::Status::Ok;
    };

    // Green first: a check that is already red proves nothing by being broken.
    if (!run()) {
        return false;
    }
    if (!breaker()) {
        return false;
    }
    const bool went_red = !run();
    if (!restore()) {
        return false;
    }
    const bool back_green = run();

    // All three must hold: green -> red under intervention -> green again. Anything
    // less and the check is not measuring what it claims.
    if (!went_red || !back_green) {
        return false;
    }
    proven_.push_back(canon);
    return true;
}

} // namespace lmp::loop
