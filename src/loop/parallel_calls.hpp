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
// Implementation lives in tools::run_calls_concurrently so `read_many` can reuse it
// without creating a tools→loop include edge.
//
#include "src/tools/concurrent_calls.hpp"

namespace lmp::loop {

using tools::run_calls_concurrently;

} // namespace lmp::loop
