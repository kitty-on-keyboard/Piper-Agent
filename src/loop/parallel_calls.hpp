#pragma once
//
// Concurrent execution of a turn's READ-ONLY batched tool calls (S9.1 amended).
//
// A turn may carry up to TurnGrammar::kMaxCallsPerTurn calls, and the model batches them
// precisely BECAUSE they are independent -- "reading four files" is the case the batching
// exists for. They ran strictly in order anyway, so four reads cost four times what one
// did for no reason anyone chose.
//
// WHAT IS NOT HERE, AND WHY. There is no queue and no pool. The cap is four calls per
// turn, decided by the grammar, and each call is a filesystem read measured in
// milliseconds; a thread costs tens of microseconds to start. A work queue would be the
// right structure for many small items and unknown parallelism, and the wrong one here --
// it would add a shared consumer to coordinate for a bounded set of four that is already
// known up front.
//
// WHAT MAY RUN HERE. Only calls that reach `Registry::execute` and nothing else: no
// workspace mutation, no command execution, no `plan`. Every other path in
// Agent::dispatch_call touches shared state -- the approver (a human cannot answer four
// cards at once), the verification ledger, the deliverable ledger, the checklist, the
// event log -- and none of it is safe to enter from four threads. The caller decides
// eligibility; this only runs what it is given.
//
// ORDER IS PRESERVED. Results come back indexed to the call they came from, so the
// transcript, the history records and the UI rows are byte-for-byte what the serial path
// produced. Parallelism is an execution detail and must not be observable in the output.
//
#include <cstddef>
#include <functional>
#include <vector>

#include "src/tools/tool_result.hpp"

namespace lmp::loop {

// Runs `work(i)` for every i in `indices`, each on its own thread, and returns the results
// in the SAME ORDER as `indices`. `work` must be safe to call concurrently; that is the
// caller's claim to make, and the header above says which calls can make it.
[[nodiscard]] std::vector<tools::ToolResult> run_calls_concurrently(
    const std::vector<std::size_t>& indices,
    const std::function<tools::ToolResult(std::size_t)>& work);

} // namespace lmp::loop
