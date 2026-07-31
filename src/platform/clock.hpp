#pragma once
//
// Clock -- the time seam (spec S3, L0).
//
// Time is injected, never read from a global. Every layer exposes a pure core with "no
// I/O, no globals, no clock"; this is the interface that lets the impure edge be one
// constructor argument instead of a call to std::chrono scattered through the loop.
//
// The reason this exists on day one rather than when something needs it: the phase
// timings in S14 (queue, prefill, decode, parse, tool, HITL, validate, policy,
// prompt-build) are always-on instrumentation, and a timing test that cannot control
// the clock has to sleep. A gate that sleeps is a gate that either flakes or blows its
// five-minute budget.
//
#include <chrono>
#include <cstdint>

namespace lmp::platform {

// Monotonic, for durations. Never for timestamps -- it has no epoch.
using MonoTime = std::chrono::steady_clock::time_point;
// Wall, for timestamps in the event log. Never for durations -- it can step backwards.
using WallTime = std::chrono::system_clock::time_point;

class Clock {
  public:
    Clock() = default;
    Clock(const Clock&) = delete;
    Clock& operator=(const Clock&) = delete;
    Clock(Clock&&) = delete;
    Clock& operator=(Clock&&) = delete;
    virtual ~Clock() = default;

    [[nodiscard]] virtual MonoTime mono() const noexcept = 0;
    [[nodiscard]] virtual WallTime wall() const noexcept = 0;
};

class SystemClock final : public Clock {
  public:
    [[nodiscard]] MonoTime mono() const noexcept override {
        return std::chrono::steady_clock::now();
    }
    [[nodiscard]] WallTime wall() const noexcept override {
        return std::chrono::system_clock::now();
    }
};

// Test double. Advances only when told to, so a duration assertion is exact rather
// than a tolerance band.
class ManualClock final : public Clock {
  public:
    ManualClock() = default;

    // system_clock ticks in microseconds on libc++ and steady_clock in nanoseconds, so
    // the cast is explicit rather than left to common_type. Storing ns internally and
    // narrowing here keeps advance() exact for every duration this project measures --
    // the finest phase timing in S14 is sub-millisecond, not sub-microsecond.
    [[nodiscard]] MonoTime mono() const noexcept override {
        return MonoTime{} +
               std::chrono::duration_cast<MonoTime::duration>(std::chrono::nanoseconds{mono_ns_});
    }
    [[nodiscard]] WallTime wall() const noexcept override {
        return WallTime{} +
               std::chrono::duration_cast<WallTime::duration>(std::chrono::nanoseconds{wall_ns_});
    }

    // Advances both clocks together, which is the normal case.
    void advance(std::chrono::nanoseconds d) noexcept {
        mono_ns_ += d.count();
        wall_ns_ += d.count();
    }

    // Moves wall time alone -- including backwards, which real wall clocks do under
    // NTP correction. Any code that computes a duration from wall() is a bug, and this
    // is how a test proves it.
    void set_wall(std::chrono::nanoseconds since_epoch) noexcept {
        wall_ns_ = since_epoch.count();
    }

  private:
    std::int64_t mono_ns_ = 0;
    std::int64_t wall_ns_ = 0;
};

[[nodiscard]] inline std::int64_t to_ns(WallTime t) noexcept {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch())
        .count();
}

[[nodiscard]] inline std::int64_t to_us(MonoTime t) noexcept {
    return std::chrono::duration_cast<std::chrono::microseconds>(t.time_since_epoch())
        .count();
}

} // namespace lmp::platform
