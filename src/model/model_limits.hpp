#pragma once
//
// Checkpoint sequence limits readable without loading weights.
//
#include <cstddef>
#include <string>

namespace lmp::model {

// text_config.max_position_embeddings (or root), or 0 when absent / unreadable.
[[nodiscard]] int load_max_position_embeddings(const std::string& model_dir);

// Bytes of KV cache one token of context costs this checkpoint. 0 when unreadable.
// Counts only full-attention layers -- a hybrid checkpoint's linear layers hold a
// fixed-size state that does not grow with the sequence. See the definition for the
// measurement that confirms the arithmetic.
[[nodiscard]] std::size_t kv_bytes_per_token(const std::string& model_dir);

// The largest context_budget_tokens whose KV cache can actually be held, given the
// checkpoint's per-token cost, the bytes the weights already occupy, and the device
// working set. 0 means "cannot tell" -- no MLX, or an unreadable config -- and the
// caller must then leave the operator's budget alone.
//
// context_budget_tokens has only ever been checked against the checkpoint's
// max_position_embeddings, which is a statement about what the MODEL can address and says
// nothing about what the MACHINE can hold. A budget can pass that check and be physically
// impossible: 245,760 tokens against 262,144 is legal and needs ~18 GB of KV on top of
// 16.3 GB of weights on a 48 GB machine.
//
// THE FAILURE IT PREVENTS IS UNCATCHABLE, which is why it has to be answered up front.
// MLX's own valve sits at the device working set (~40.2 GB here) and never fired: the run
// died at 38.0 GB because the OS reached ITS limit first and killed the process. No
// exception, no run_end, no crash report -- events.jsonl and sidecar-stderr.log simply
// stop at the same second. See mlx_backend.cpp's set_memory_ceiling comment, which
// predicted exactly this.
//
// The budget is also what schedules compaction (kCompactAtPercent), so an oversized one
// does not merely risk the ceiling -- it disables the mechanism that would have kept the
// prompt away from it. The run this came from reached 112,088 tokens with compactions=0
// because 75% of 245,760 is 184,320 and it never got there.
[[nodiscard]] int max_affordable_context_tokens(std::size_t kv_bytes_per_token,
                                                std::size_t weights_bytes,
                                                std::size_t device_working_set_bytes);

// Does this checkpoint declare a vision tower? Read from config.json, without touching
// the weights, because the answer decides HOW to load them.
//
// Asking is the point. MlxBackendConfig::with_vision REFUSES against a text-only
// checkpoint rather than falling back silently, which is the right contract for an
// explicit request and the wrong one for a product that must load whatever it is
// pointed at. The caller asks first and then requests exactly what is there.
[[nodiscard]] bool checkpoint_declares_vision(const std::string& model_dir);

} // namespace lmp::model
