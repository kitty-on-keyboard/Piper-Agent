#include "src/model/model_limits.hpp"

#include "src/model/mlx/qwen35_moe_config.hpp"

namespace lmp::model {

int load_max_position_embeddings(const std::string& model_dir) {
    return mlxl::load_max_position_embeddings(model_dir);
}

} // namespace lmp::model
