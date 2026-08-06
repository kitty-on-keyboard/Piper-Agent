#pragma once
//
// Checkpoint sequence limits readable without loading weights.
//
#include <string>

namespace lmp::model {

// text_config.max_position_embeddings (or root), or 0 when absent / unreadable.
[[nodiscard]] int load_max_position_embeddings(const std::string& model_dir);

} // namespace lmp::model
