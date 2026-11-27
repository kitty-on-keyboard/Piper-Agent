#include "src/tools/concurrent_calls.hpp"

#include <algorithm>
#include <atomic>
#include <exception>
#include <thread>

namespace lmp::tools {

std::vector<ToolResult> run_calls_concurrently(
    const std::vector<std::size_t>& indices,
    const std::function<ToolResult(std::size_t)>& work) {
    std::vector<ToolResult> results(indices.size());
    if (indices.empty()) {
        return results;
    }
    auto safe_work = [&work](std::size_t idx) -> ToolResult {
        try {
            return work(idx);
        } catch (const std::exception& e) {
            return ToolResult::error(ErrorClass::Transient, false,
                                     std::string("parallel call failed: ") + e.what());
        } catch (...) {
            return ToolResult::error(ErrorClass::Transient, false,
                                     "parallel call threw an unknown exception");
        }
    };
    // One call is not worth a thread; run it here and skip the spawn entirely.
    if (indices.size() == 1) {
        results[0] = safe_work(indices[0]);
        return results;
    }

    // Bounded worker pool + atomic index claiming so large batches don't exhaust
    // RLIMIT_NPROC and exceptions never call std::terminate on a worker thread.
    const std::size_t workers = std::min<std::size_t>(indices.size(), 8);
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t w = 0; w < workers; ++w) {
        threads.emplace_back([&results, &indices, &next, &safe_work] {
            while (true) {
                const std::size_t k = next.fetch_add(1, std::memory_order_relaxed);
                if (k >= indices.size()) {
                    break;
                }
                results[k] = safe_work(indices[k]);
            }
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    return results;
}

} // namespace lmp::tools
