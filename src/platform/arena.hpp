#pragma once
//
// Arena -- a measured PMR scratch allocator (spec S3, L0).
//
// One arena per bounded scope (a turn, a prefill, a tool call). `reset()` rewinds the
// whole thing in O(1) instead of running N destructors, which is the point.
//
// It MEASURES rather than enforces. An arena that aborts a 300-second run because a
// budget was set too low is worse than the allocation it prevented, so `cap_bytes` sets
// a high-water line that `overflowed()` reports and the event log records -- it does
// not refuse. If a component needs a hard refusal, that refusal belongs at the
// component's own admission point where it can return a typed error, not down here
// where the only vocabulary is std::bad_alloc.
//
#include <cstddef>
#include <memory_resource>

namespace lmp::platform {

class Arena {
  public:
    // Both parameters are required. `reserve_bytes` is allocated up front and reused
    // across reset(); `cap_bytes` is the high-water line reported by overflowed().
    Arena(std::size_t reserve_bytes, std::size_t cap_bytes);

    Arena(const Arena&) = delete;
    Arena& operator=(const Arena&) = delete;
    Arena(Arena&&) = delete;
    Arena& operator=(Arena&&) = delete;
    ~Arena() = default;

    [[nodiscard]] std::pmr::memory_resource* resource() noexcept { return &counter_; }

    // Bytes requested since the last reset(). Counted as REQUESTED, before the
    // monotonic resource's alignment padding -- so this is a floor on true footprint,
    // never an overstatement. Documented rather than corrected, because a number whose
    // direction of error is known is more useful than one that is merely closer.
    [[nodiscard]] std::size_t used() const noexcept { return counter_.used; }

    // Largest `used()` seen across all resets over this arena's lifetime.
    [[nodiscard]] std::size_t high_water() const noexcept { return counter_.high_water; }

    [[nodiscard]] std::size_t cap() const noexcept { return counter_.cap; }

    // True if used() ever exceeded cap(). Sticky across reset() -- a run that blew the
    // budget once did blow the budget, and clearing that on rewind would hide it.
    [[nodiscard]] bool overflowed() const noexcept { return counter_.overflowed; }

    // Rewinds allocation to the reserve block. Does NOT run destructors: only place
    // trivially-destructible types, or types whose destructor you have proven is a
    // no-op, in an arena.
    void reset() noexcept;

  private:
    class Counting final : public std::pmr::memory_resource {
      public:
        std::pmr::memory_resource* downstream = nullptr;
        std::size_t used = 0;
        std::size_t high_water = 0;
        std::size_t cap = 0;
        bool overflowed = false;

      private:
        void* do_allocate(std::size_t bytes, std::size_t align) override;
        void do_deallocate(void* p, std::size_t bytes, std::size_t align) override;
        [[nodiscard]] bool do_is_equal(
            const std::pmr::memory_resource& other) const noexcept override;
    };

    std::pmr::monotonic_buffer_resource monotonic_;
    Counting counter_;
};

inline Arena::Arena(std::size_t reserve_bytes, std::size_t cap_bytes)
    : monotonic_(reserve_bytes, std::pmr::new_delete_resource()) {
    counter_.downstream = &monotonic_;
    counter_.cap = cap_bytes;
}

inline void Arena::reset() noexcept {
    monotonic_.release();
    counter_.used = 0;
}

inline void* Arena::Counting::do_allocate(std::size_t bytes, std::size_t align) {
    void* p = downstream->allocate(bytes, align);
    used += bytes;
    if (used > high_water) {
        high_water = used;
    }
    if (used > cap) {
        overflowed = true;
    }
    return p;
}

inline void Arena::Counting::do_deallocate(void* p, std::size_t bytes, std::size_t align) {
    // Monotonic deallocation is a no-op by design; `used` deliberately does not shrink,
    // so used() reads as "bytes handed out this scope", which is the number that
    // predicts the reserve size you actually need.
    downstream->deallocate(p, bytes, align);
}

inline bool Arena::Counting::do_is_equal(
    const std::pmr::memory_resource& other) const noexcept {
    return this == &other;
}

} // namespace lmp::platform
