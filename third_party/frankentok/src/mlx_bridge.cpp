#include "mlx_qwen_tokenizer/mlx_bridge.h"
#include "mlx_qwen_tokenizer/tokenizer.h"

namespace mlx_qwen_tokenizer {

#ifdef MLX_QWEN_TOKENIZER_ENABLE_MLX
mlx::core::array encode_to_mlx(const Tokenizer& tokenizer, std::string_view text) {
    std::vector<int32_t> ids = tokenizer.encode(text);
    return mlx::core::array(ids.data(), {1, static_cast<int>(ids.size())}, mlx::core::int32);
}
#endif

} // namespace mlx_qwen_tokenizer
