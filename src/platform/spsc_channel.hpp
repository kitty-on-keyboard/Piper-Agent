#pragma once
//
// SpscChannel -- a bounded single-producer/single-consumer queue of OWNED values
// (spec S3 L0, consumed by S4.2).
//
// This exists to make a read boundary semantically invisible. v1's stdin reader pulled
// 127 bytes at a time into a fixed-chunk ring and reassembled in the consumer, so the
// reader's control-message check ran against an arbitrary byte window and could not see
// a message that straddled two reads. The fix is structural, not a bigger buffer: the
// producer frames, and what crosses this queue is always exactly one complete message.
// A partial message is therefore not representable here, so the class that could have
// mishandled one does not exist.
//
// The values are owned (moved in, moved out). Views into a shared buffer would put the
// read boundary back.
//
#include <atomic>
#include <cstddef>
#include <new>
#include <utility>
#include <vector>

namespace lmp::platform {

// Apple Silicon: 128-byte L2 stride. Padding is measured against false sharing, not
// against std::hardware_destructive_interference_size, which libc++ reports as 64.
inline constexpr std::size_t kCacheLinePad = 128;

template <class T>
class SpscChannel {
  public:
    // `capacity` is rounded up to a power of two so the index wrap is a mask rather
    // than a modulo in the hot path. capacity() reports the rounded value, not the
    // requested one -- callers that care must read it back rather than assume.
    explicit SpscChannel(std::size_t capacity)
        : mask_(round_up_pow2(capacity) - 1), slots_(round_up_pow2(capacity)) {}

    SpscChannel(const SpscChannel&) = delete;
    SpscChannel& operator=(const SpscChannel&) = delete;

    // Producer thread only. Returns false if full; the caller decides whether that is
    // backpressure or an error, because only the caller knows.
    [[nodiscard]] bool try_push(T&& value) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t head = head_.load(std::memory_order_acquire);
        if (tail - head >= slots_.size()) {
            return false;
        }
        slots_[tail & mask_] = std::move(value);
        tail_.store(tail + 1, std::memory_order_release);
        return true;
    }

    // Consumer thread only. Returns false if empty.
    [[nodiscard]] bool try_pop(T& out) noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        if (head == tail) {
            return false;
        }
        out = std::move(slots_[head & mask_]);
        slots_[head & mask_] = T{};
        head_.store(head + 1, std::memory_order_release);
        return true;
    }

    // Producer signals end-of-stream (stdin EOF). The consumer must still drain: a
    // closed channel can hold messages, and treating closed as empty is how the last
    // request before EOF gets dropped.
    void close() noexcept { closed_.store(true, std::memory_order_release); }

    [[nodiscard]] bool closed() const noexcept {
        return closed_.load(std::memory_order_acquire);
    }

    // True only when the producer has closed AND nothing remains. This is the
    // consumer's real exit condition.
    [[nodiscard]] bool drained() const noexcept { return closed() && empty(); }

    [[nodiscard]] bool empty() const noexcept {
        return head_.load(std::memory_order_acquire) ==
               tail_.load(std::memory_order_acquire);
    }

    // Approximate under concurrency, exact when only one side is running. Named so no
    // call site can mistake it for a synchronisation point.
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::size_t tail = tail_.load(std::memory_order_acquire);
        const std::size_t head = head_.load(std::memory_order_acquire);
        return tail - head;
    }

    [[nodiscard]] std::size_t capacity() const noexcept { return slots_.size(); }

  private:
    static constexpr std::size_t round_up_pow2(std::size_t n) noexcept {
        std::size_t p = 1;
        while (p < n) {
            p <<= 1U;
        }
        return p;
    }

    const std::size_t mask_;
    std::vector<T> slots_;
    alignas(kCacheLinePad) std::atomic<std::size_t> head_{0};
    alignas(kCacheLinePad) std::atomic<std::size_t> tail_{0};
    alignas(kCacheLinePad) std::atomic<bool> closed_{false};
};

} // namespace lmp::platform
