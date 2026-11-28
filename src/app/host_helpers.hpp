#pragma once
// Native host helpers for Piper.app (C++20, testable without AppKit/WebKit).

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::app {

struct ResolvedAsset {
    bool ok = false;
    std::filesystem::path path;
    std::string mime_type;
    std::string error;
};

// Resolves a request path under `ui_root` cleanly, rejecting any traversal outside it.
// Falls back to `index.html` only for extensionless SPA routes.
[[nodiscard]] ResolvedAsset resolve_ui_asset(const std::filesystem::path& ui_root,
                                             std::string_view url_path);

// Only local Vite dev servers on localhost / 127.0.0.1 are permitted in --dev-url.
[[nodiscard]] bool is_allowed_dev_url(std::string_view url);

[[nodiscard]] std::filesystem::path default_settings_path();
[[nodiscard]] std::string load_settings_json(const std::filesystem::path& path);
[[nodiscard]] bool save_settings_json(const std::filesystem::path& path,
                                      std::string_view json_str,
                                      std::string& error_out);

enum class CheckpointKind : std::uint8_t {
    Unknown = 0,
    Dense,
    Moe,
    Mtp,
};

[[nodiscard]] std::string_view to_string(CheckpointKind kind) noexcept;
[[nodiscard]] CheckpointKind classify_checkpoint(const std::filesystem::path& dir);
[[nodiscard]] std::filesystem::path find_sibling_mtp(const std::filesystem::path& dense_dir);
[[nodiscard]] std::string scan_models_json(const std::vector<std::filesystem::path>& roots);

// Strictly read-only workspace inspection helpers; never mutate git or tracked files.
[[nodiscard]] std::string workspace_diff_json(const std::filesystem::path& workspace_root);
[[nodiscard]] std::string workspace_files_json(const std::filesystem::path& workspace_root,
                                               int limit = 2000);
[[nodiscard]] std::string read_file_preview_json(const std::filesystem::path& workspace_root,
                                                 std::string_view rel_path,
                                                 std::size_t max_bytes = 262144);
[[nodiscard]] std::string save_image_attachment_json(const std::filesystem::path& workspace_root,
                                                     std::string_view ext,
                                                     std::string_view base64_data);

// Serializes `raw` as a JSON string literal with U+2028/U+2029 escaped for evaluateJavaScript.
[[nodiscard]] std::string escape_js_string_arg(std::string_view raw);

} // namespace lmp::app
