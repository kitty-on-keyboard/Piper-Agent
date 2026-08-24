#pragma once
//
// The identity of one run, minted by the SIDECAR.
//
// WHY THIS EXISTS. `run_id` used to be the JSON-RPC request id -- `private nextId = 1` in
// extension/src/client.ts, incremented per request and reset whenever the extension host
// restarts. It is a correlation token for one request/response pair and was never an
// identity for anything. Measured on the real event log before this was written:
//
//     97 runs, 6 distinct run_id values, and run_id="2" used 63 times.
//
// Everything keyed on it therefore collided. src/context/resume.cpp filters the event log
// by run_id to rebuild a context, so resuming "2" would have spliced 63 unrelated runs
// into one conversation; ContextJournal opens a PCC session under the same string, so 63
// runs shared one session's durable memory. Neither failure is visible in a test, because
// a test builds a log containing exactly one run.
//
// WHAT IT MUST GUARANTEE, in the order the guarantees matter:
//
//   1. Unique across processes and across restarts. Two sidecars started in the same
//      millisecond on the same machine must not collide, so time alone is not enough.
//   2. Sortable by creation time as a plain string, which makes "the most recent
//      unfinished run" a max() over ids rather than a join against the log's timestamps.
//   3. Stable in length and alphabet, because it lands in a JSON field, a SQLite session
//      key and a UI list, and each of those is easier when it cannot contain a surprise.
//
// Wall time gives 2, randomness gives 1, and fixed-width hex gives 3. The wall clock going
// backwards (NTP, a laptop waking) breaks ordering but not uniqueness -- the random half
// still separates the ids, which is the guarantee that actually matters.
//
#include <array>
#include <chrono>
#include <cstdint>
#include <random>
#include <string>

#include "src/platform/clock.hpp"

namespace lmp::surface {

// 16 hex digits of nanoseconds since the epoch, then 8 hex digits of entropy.
// `r-` prefixes it so a run id is recognisable on sight in a log line or a UI row, and so
// it can never be mistaken for the bare integer it used to be.
[[nodiscard]] inline std::string mint_run_id(const platform::Clock& clock) {
    const auto ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock.wall().time_since_epoch())
            .count());
    // One generator per process, seeded once from the system source. A fresh
    // random_device per call is both slow and, on some implementations, identical across
    // rapid calls -- which would defeat the only guarantee this half provides.
    static std::mt19937_64 gen{std::random_device{}()};
    static std::uniform_int_distribution<std::uint32_t> dist;
    const std::uint32_t salt = dist(gen);

    constexpr char kHex[] = "0123456789abcdef";
    std::string out = "r-";
    out.reserve(2 + 16 + 1 + 8);
    for (int shift = 60; shift >= 0; shift -= 4) {
        out.push_back(kHex[(ns >> shift) & 0xFU]);
    }
    out.push_back('-');
    for (int shift = 28; shift >= 0; shift -= 4) {
        out.push_back(kHex[(salt >> shift) & 0xFU]);
    }
    return out;
}

// Whether `s` has the shape mint_run_id produces. Used to tell a real id from the legacy
// integers still sitting in old event logs, which must never be offered as resumable --
// they are the colliding ones.
[[nodiscard]] inline bool is_minted_run_id(const std::string& s) {
    if (s.size() != 27 || s[0] != 'r' || s[1] != '-' || s[18] != '-') {
        return false;
    }
    for (std::size_t i = 2; i < s.size(); ++i) {
        if (i == 18) {
            continue;
        }
        const char c = s[i];
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        if (!hex) {
            return false;
        }
    }
    return true;
}

} // namespace lmp::surface
