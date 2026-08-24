#pragma once
//
// The turn machine (spec S9).
//
// ONE TURN, ONE OUTCOME (S9.1). TurnClassifier maps a completed generation to exactly
// one Outcome -- not two, not zero. It asks "did this call actually EXECUTE?" before
// anything else, because in v1 a second observation silently erased the first.
//
// WHAT IS DELIBERATELY NOT HERE ANY MORE: the Corrective enum, the completion gate, the
// falsifiability ledger and the refusal blocklist. The harness they steered adjudicated
// whether the run's work was CORRECT, and the eighth pass removed the whole apparatus
// after a measured run deadlocked inside it: a contract that could never execute was
// scored as a contract that was failing, the one corrective built for the case reset
// itself on a no-op, and the completion gate rejected the honest check the model finally
// wrote ("passed but never seen to fail") -- so the model went back to the broken one and
// ended the run "no progress" with the build red. The harness now surfaces evidence
// (observations, exit statuses, repeat counts) and enforces only operator-set policy
// (modes, approvals, budgets). Whether the work is done is the model's claim and the
// operator's judgement, informed by the operator's own check when one is configured.
//
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "src/context/context.hpp"
#include "src/model/backend.hpp"
#include "src/model/grammar.hpp"
#include "src/tools/registry.hpp"

namespace lmp::loop {

enum class Outcome : std::uint8_t {
    ToolCallExecuted,   // the call ran; the result is the observation
    ToolCallRefused,    // policy or HITL said no; the tool NEVER RAN
    TextOnly,           // the model answered without acting
    LengthCapped,       // generation hit the cap -- NOT completion
    Cancelled,
    BackendError,
};

[[nodiscard]] std::string_view to_string(Outcome o) noexcept;

struct TurnResult {
    // What applying a `plan` call did. Reported back to the model as the tool result, so
    // a malformed checklist is a correctable observation rather than a silent no-op.
    struct PlanOutcome {
        bool ok = false;
        std::string detail;
    };

    // A call batched into the same turn behind the first one. Each still gets its own
    // history record and its own UI row -- batching changes how many prefills a turn
    // costs, not how honestly its calls are reported.
    struct ExtraCall {
        std::string tool_name;
        std::vector<tools::ToolParamValue> params;
        tools::ToolResult result;
        // See TurnResult::produced_new_information. Batched calls carry it too, because a
        // turn is only inert when EVERY call in it was.
        bool produced_new_information = false;
    };

    Outcome outcome = Outcome::BackendError;
    // The harness cut this generation because it was emitting the same tokens over and
    // over (see LoopBreaker). The turn classifies TextOnly -- no tool ran -- but the text
    // is a cut-off cycle, NOT the model's answer, and the loop must not end the run on
    // it as though the model had concluded. The distinction is a fact about what the
    // harness did, which is exactly what a TurnResult is for.
    bool cut_for_looping = false;
    // Did this call's observation tell the run something it did not already have?
    //
    // False when the bytes are identical to a previous call's -- the same fact the
    // `repeat_reread` event reports as `unchanged` -- and false by default, so a turn that
    // executed nothing (TextOnly, LengthCapped) is not credited with information. It is
    // the loop's measure of PROGRESS, and the reason a run can now end on "produced
    // nothing new for N turns" rather than only on "narrated twice in a row".
    bool produced_new_information = false;
    std::string assistant_text;
    std::string reasoning;      // peeled off, surfaced separately, never in the answer
    std::string tool_name;
    std::vector<tools::ToolParamValue> tool_params;
    tools::ToolResult tool_result;
    std::vector<ExtraCall> extra_calls;
    model::GenResult generation;
    // Exact grammar-channel token counts for the generation. tool_tokens is everything
    // not routed to reasoning or answer prose: tool XML plus structural delimiters.
    std::size_t think_tokens = 0;
    std::size_t text_tokens = 0;
    std::size_t tool_tokens = 0;
    // Position in one model-emitted call batch. Text-only turns use count 0.
    std::size_t batch_index = 0;
    std::size_t batch_count = 0;
    // Non-empty only when generation hit its cap; names the grammar phase at the cap.
    std::string cap_phase;
};

// Value of a named param, or empty. The grammar guarantees required params are present,
// so an empty return means "not supplied" rather than "lost".
[[nodiscard]] std::string param_value(const std::vector<tools::ToolParamValue>& params,
                                      std::string_view name);

// Mode policy is applied in ONE place (S9.3); the loop does not apply it itself, so a
// config that skips the policy cannot run with writes enabled anyway.
enum class Mode : std::uint8_t { Plan, Debug, Agent };

struct ModePolicy {
    int sandbox_tier = 0;
    bool allow_workspace_writes = false;
    // May this mode destroy something the workspace cannot get back? Separate from the
    // write bit because "fix this bug" and "delete this file" are different powers, and a
    // mode that needs the first has no need of the second. Debug mode is the case that
    // forced the split: it is useless without writes and has no business deleting.
    bool allow_destructive = false;
    // Does this mode run commands at all? Distinct from `sandbox_tier == 0`, which is
    // where this lived at first and is the wrong thing to read: the tier is an operator
    // setting and can be moved, and mode policy must not be a function of a knob. It also
    // made the HITL gate untestable, because the tests that pin the real, measured
    // approval bugs set tier 0 precisely so `rm -rf Sources` reaches the gate and never
    // the shell.
    bool allow_execution = false;
    // This mode YIELDS rather than loops: a text-only turn means "your move, operator"
    // and the run stops there as `awaiting_user`. In a working mode the same turn is the
    // model's final answer and the run ends `ended`. The two are different relationships
    // with the human, not different amounts of machinery.
    bool conversational = false;

    [[nodiscard]] static ModePolicy for_mode(Mode m) noexcept;
};

// What this mode is, in words the model reads.
//
// Debug and Agent share one block of working discipline -- verifying, editing, getting
// unstuck -- under their own openings, which is why this returns a std::string rather than
// a view. Both modes write code and both fail the same ways when they are not told how, so
// the text has ONE home: a lesson measured in either run reaches both. Plan mode does not
// take it; nothing there can build or edit.
[[nodiscard]] std::string mode_brief(Mode m);

// Exactly one repeat detector, and it is a DETECTOR / annotator -- not an authority for
// mutable workspace state.
//
// A call is a repeat when the same tool is invoked with the same arguments as a previous
// turn. The old response was grammar surgery -- suppress the tool for a few turns -- and
// it measurably made runs worse: a model with two ways to ask the same question asks the
// other way (a real run alternated `read_file`/`read_slice` for 27 turns with the
// suppression firing on nearly every one), and a model denied its editor wrote source
// files through shell heredocs. Serving a cached observation without revalidation was the
// next failure mode: shell, MCP, editor, and external edits can change files without the
// detector noticing. So repeats are counted and annotated; file/search/directory/git
// reads always revalidate current state. Under context pressure, an older verbatim
// duplicate still sitting in rendered history may be collapsed -- never replaced by a
// pointer to content that compaction already dropped.
//
// `cached()` remains for diagnostics and tests: it reports the prior successful
// observation only while the workspace freshness epoch is unchanged. The agent does not
// serve that string as a substitute for re-execution.
class RepeatDetector {
  public:
    struct SeenCall {
        std::size_t count = 0;
        bool last_ok = false;
        std::string last_summary;
        // ctx.workspace_writes() (freshness epoch) when the call last ran. One write,
        // successful shell, remote call, or external invalidation bumps the epoch.
        std::size_t writes_at = 0;
    };

    // Returns the number of times this exact call has already been made.
    [[nodiscard]] std::size_t seen_count(const std::string& tool,
                                         const std::vector<tools::ToolParamValue>& params) const;
    void record(const std::string& tool, const std::vector<tools::ToolParamValue>& params,
                bool ok, const std::string& summary, std::size_t writes_now);
    // Prior successful observation for this exact call while the freshness epoch is
    // unchanged. Null otherwise. Diagnostic only -- not a substitute for revalidation.
    [[nodiscard]] const SeenCall* cached(const std::string& tool,
                                         const std::vector<tools::ToolParamValue>& params,
                                         std::size_t writes_now) const;
    // Last recorded call for this key, regardless of freshness. Used to measure whether a
    // revalidated re-read returned the same bytes after a shell/remote bump.
    [[nodiscard]] const SeenCall* previous(
        const std::string& tool, const std::vector<tools::ToolParamValue>& params) const;
    void clear() noexcept { seen_.clear(); }

  private:
    [[nodiscard]] static std::string key(const std::string& tool,
                                         const std::vector<tools::ToolParamValue>& params);
    std::vector<std::pair<std::string, SeenCall>> seen_;
};

// What a run is allowed to spend. These are ceilings on a RUNAWAY, not a target for a
// healthy run, and they were set too low to finish real work.
//
// MEASURED on the KV-store-with-TTL-and-transactions mission: the run that got all the
// way through declared completion on turn 41, so the old ceiling of 40 cut its
// predecessors off one turn short. 80 was then measured too low IN TURN on a five-task
// macOS app mission -- not thrashing, just still writing the sixth of ten files. A
// multi-task mission spends turns on ORDINARY work: each file is a write, each build is a
// shell call, each compile error is a read and an edit.
//
// 200 turns is the ceiling for a mission of that size with room to iterate, and it is
// still a ceiling: a run that hangs is caught by stall_seconds long before, and the repeat
// cache starves the cheap loops of anything to do.
//
// "HUNG" AND "LONG" ARE DIFFERENT QUESTIONS AND GET DIFFERENT DIALS. They shared one for a
// while, and the sharing is what broke: wall_clock_seconds was calibrated at 12.1-12.7 s
// per turn on the KV mission, so 4800 s was ~1.9x the room 200 turns needed. Then turns
// got expensive for a reason that has nothing to do with hanging. On a 2026-08-17 run
// against a vision-enabled 27B, the context budget clamped to 46,797 tokens, compaction
// fired every ~5 turns, and every compaction costs one full re-prefill -- 13 of 13,
// median 84 s of time-to-first-token. 40% of the clock went to re-prefill and the run
// ended `wall_clock` on turn 48 of 200, having done nothing wrong. The operator's only
// move was to retype `continue`, repeatedly, which is the harness asking a human to do
// its accounting.
//
// So the clock now measures the thing it was always FOR. A run that has completed a turn
// recently is working, however long it has been working; a run that has not completed one
// in stall_seconds is stuck, however briefly it has been running.
//
//   max_iterations   how much work a run may do        -- the ceiling that should bind
//   stall_seconds    how long it may do nothing        -- the hang detector
//   wall_clock_seconds  how long it may exist at all   -- runaway stop of last resort
//
// stall_seconds = 1200 is sized off the longest LEGITIMATE turn, not off an average. The
// worst observed time-to-first-token is 129 s; a 4096-token generation at the measured
// 16 tok/s adds ~256 s; a shell tool can hold its own 300 s (session.cpp). That is ~700 s
// of honest single turn, so 1200 s clears it with room and still ends a wedged run inside
// twenty minutes.
//
// wall_clock_seconds = 14400 restores the property the old comment was right to insist on:
// the turn ceiling, not the clock, decides. 200 turns at the ~60 s/turn this workload
// actually costs is ~12,000 s, so four hours sits above a full-budget run rather than
// cutting it at a quarter.
//
// All three are operator-settable -- lmPipe.maxIterations, lmPipe.stallSeconds and
// lmPipe.wallClockSeconds, live in the sidebar drawer -- and the drawer says which will
// stop the run first, because a raised turn limit under an unraised clock does nothing.
struct Budget {
    int max_iterations = 200;
    int wall_clock_seconds = 14400;
    int stall_seconds = 1200;
};

// How many iterations before the limit the run is told, once, how much room is left.
// A fact the model cannot observe any other way -- the budget is the harness's -- and a
// run mid-way through a multi-file change deserves the chance to land it. Eight rather
// than five because five was measured too short for a restore that is several edits long.
inline constexpr int kBudgetWarningTurns = 8;

// --- classification ---------------------------------------------------------

// `executed` answers "did this call actually EXECUTE?" -- asked first, and the reason
// the classifier takes it as an input rather than inferring it.
[[nodiscard]] Outcome classify_turn(const model::GenResult& gen,
                                    const model::TurnGrammar& grammar, bool executed,
                                    bool refused);

} // namespace lmp::loop
