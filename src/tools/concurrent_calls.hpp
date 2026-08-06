#pragma once
//
// Concurrent execution of independent read-only tool work.
//
// Lives in tools so Registry::read_many can fan out inner reads with the same primitive
// Agent::step uses for multi-call batches. Order is preserved: results are indexed to the
// input indices, never to completion order.
//
#include <cstddef>
#include <functional>
#include <vector>

#include "src/tools/tool_result.hpp"

namespace lmp::tools {

[[nodiscard]] std::vector<ToolResult> run_calls_concurrently(
    const std::vector<std::size_t>& indices,
    const std::function<ToolResult(std::size_t)>& work);

} // namespace lmp::tools
