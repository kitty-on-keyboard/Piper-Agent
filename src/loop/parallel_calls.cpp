#include "src/loop/parallel_calls.hpp"

#include <thread>

namespace lmp::loop {

std::vector<tools::ToolResult> run_calls_concurrently(
    const std::vector<std::size_t>& indices,
    const std::function<tools::ToolResult(std::size_t)>& work) {
    std::vector<tools::ToolResult> results(indices.size());
    if (indices.empty()) {
        return results;
    }
    // One call is not worth a thread; run it here and skip the spawn entirely.
    if (indices.size() == 1) {
        results[0] = work(indices[0]);
        return results;
    }

    // Each thread writes its OWN element of a vector sized up front. No element is ever
    // resized, moved or read while the threads run, so the writes cannot race and the
    // join is the only synchronisation needed.
    std::vector<std::thread> threads;
    threads.reserve(indices.size());
    for (std::size_t k = 0; k < indices.size(); ++k) {
        threads.emplace_back([&results, &indices, &work, k] {
            results[k] = work(indices[k]);
        });
    }
    for (std::thread& t : threads) {
        t.join();
    }
    return results;
}

} // namespace lmp::loop
