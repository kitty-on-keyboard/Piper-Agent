#pragma once

#include <vector>
#include <cstdint>
#include <string_view>

#ifdef MLX_QWEN_TOKENIZER_ENABLE_MLX
#include "mlx/mlx.h"
#endif

namespace mlx_qwen_tokenizer {
class Tokenizer;

#ifdef MLX_QWEN_TOKENIZER_ENABLE_MLX
mlx::core::array encode_to_mlx(const Tokenizer& tokenizer, std::string_view text);
#endif

} // namespace mlx_qwen_tokenizer
