#pragma once
//
// MlxBackend -- in-process Qwen 3.6 MoE inference on MLX (spec S5.11, S5.12).
//
// Compiled only when the build finds the MLX C++ SDK (the pip package ships it;
// src/model/CMakeLists.txt probes `python3 -m mlx --cmake-dir`, the same discovery v1
// used). Without MLX the class still exists and REFUSES at construction with
// instructions -- loud and typed, never a silent fallback (S13).
//
// The numerics under it (src/model/mlx/) are adapted from v1: the hybrid
// linear-attention/full-attention schedule, 256-expert switch MoE with shared expert,
// gated q-projection, partial RoPE -- debugged against this exact checkpoint
// (Qwen3.6-35B-A3B-MLX-4bit). Model math passes S2.2's asset test; only the seam
// around it was rebuilt.
//
// S5.11 discipline, where each item lives:
//   one model load per process          -- load() refuses a second call
//   no .item() in the hot decode loop   -- one logits row is pulled per step, into the
//                                          CPU Sampler (mask-first, S5.6)
//   chunked prefill                     -- kPrefillChunk tokens per eval
//   prefix reuse verified, never assumed-- KvCacheLedger id-by-id (S5.10); divergence
//                                          resets and honestly re-prefills
//   TTFT / prefill tok/s / decode tok/s -- filled on every GenResult
//   speculative decoding                -- model-free path in speculative.hpp; OFF by
//                                          default (LMP_SPECULATIVE=1 or
//                                          MlxBackendConfig::speculative). draft_model_dir
//                                          is a separate draft-model seam and stays empty;
//                                          EAGLE / second-checkpoint speculation is out of
//                                          scope. Do not enable speculation by default.
//
#include <cstddef>
#include <memory>
#include <string>

#include "src/model/backend.hpp"
#include "src/model/speculative.hpp"
#include "src/model/kv_cache.hpp"
#include "src/platform/clock.hpp"

namespace lmp::model {

// What MLX is holding right now, in bytes. `active` is what live arrays use; `cache` is
// what its allocator has taken from the OS and NOT given back, which is the number that
// makes "unloaded" a lie -- see ~MlxBackend. All zero in a build without MLX.
struct MemoryReport {
    std::size_t active = 0;
    std::size_t cache = 0;
    std::size_t peak = 0;
};

[[nodiscard]] MemoryReport mlx_memory_report();

struct MlxBackendConfig {
    std::string model_dir; // required (S7.5: no defaultable security-relevant input)
    std::string draft_model_dir; // a DRAFT MODEL seam; empty = off. Unrelated to the
                                 // model-free speculative path below, which needs no
                                 // second checkpoint (and could not have one: 19 GB
                                 // already, on a 48 GB host).
    // Model-free speculative decoding (src/model/speculative.hpp). Off by default; the
    // plain decode path stays the reference. LMP_SPECULATIVE=1 turns it on without a
    // rebuild so the two can be measured against each other on the same binary.
    SpecConfig speculative{};
};

class MlxBackend final : public InferenceBackend {
  public:
    explicit MlxBackend(const platform::Clock& clock);
    ~MlxBackend() override;

    // Loads config + weights once. A second call refuses (S5.11: one load per process).
    [[nodiscard]] LoadStatus load(const MlxBackendConfig& config);
    [[nodiscard]] bool available() const noexcept { return loaded_; }

    [[nodiscard]] GenResult generate(const InferenceTask& task, TokenSink& sink,
                                     const CancelToken& cancel) override;

    // The verified-reuse ledger, exposed for tests and for the loop's fresh-window
    // restart (S8.3).
    [[nodiscard]] const KvCacheLedger& ledger() const noexcept { return ledger_; }
    void reset_cache();

  private:
    struct Impl; // holds the mx graph objects; keeps mlx headers out of this header
    std::unique_ptr<Impl> impl_;
    const platform::Clock& clock_;
    KvCacheLedger ledger_;
    SpecConfig spec_{};
    bool loaded_ = false;
};

} // namespace lmp::model
