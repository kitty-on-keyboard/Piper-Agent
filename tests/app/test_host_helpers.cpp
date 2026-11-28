#include "src/app/host_helpers.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "src/platform/event_log.hpp"
#include "tests/check.hpp"

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct TempDir {
    fs::path path;
    explicit TempDir(const char* name) {
        static int seq = 0;
        path = fs::temp_directory_path() /
               ("lmp_test_host_" + std::string(name) + "_" + std::to_string(++seq));
        std::error_code ec;
        fs::remove_all(path, ec);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void write_text(const fs::path& p, const std::string& content) {
    fs::create_directories(p.parent_path());
    std::ofstream out(p, std::ios::binary);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
}

} // namespace

TEST(resolve_ui_asset_serves_and_rejects_traversal) {
    TempDir dir("assets");
    write_text(dir.path / "index.html", "<!doctype html>");
    write_text(dir.path / "assets" / "app.js", "console.log(1);");
    write_text(dir.path / "assets" / "logo.svg", "<svg/>");

    const auto root_hit = lmp::app::resolve_ui_asset(dir.path, "/");
    CHECK(root_hit.ok);
    CHECK_EQ(root_hit.path.filename().string(), "index.html");
    CHECK_EQ(root_hit.mime_type, "text/html; charset=utf-8");

    const auto js_hit = lmp::app::resolve_ui_asset(dir.path, "/assets/app.js?v=123#hash");
    CHECK(js_hit.ok);
    CHECK_EQ(js_hit.path.filename().string(), "app.js");
    CHECK_EQ(js_hit.mime_type, "application/javascript; charset=utf-8");

    const auto svg_hit = lmp::app::resolve_ui_asset(dir.path, "/assets/logo.svg");
    CHECK(svg_hit.ok);
    CHECK_EQ(svg_hit.mime_type, "image/svg+xml");

    const auto spa_hit = lmp::app::resolve_ui_asset(dir.path, "/sessions/abc");
    CHECK(spa_hit.ok);
    CHECK_EQ(spa_hit.path.filename().string(), "index.html");

    const auto missing_ext = lmp::app::resolve_ui_asset(dir.path, "/missing.js");
    CHECK(!missing_ext.ok);

    const auto dotdot = lmp::app::resolve_ui_asset(dir.path, "/assets/../../etc/passwd");
    CHECK(!dotdot.ok);

    const auto encoded_dotdot = lmp::app::resolve_ui_asset(dir.path, "/assets/%2e%2e/%2e%2e/etc/passwd");
    CHECK(!encoded_dotdot.ok);
}

TEST(allowed_dev_url_accepts_only_loopback_http) {
    CHECK(lmp::app::is_allowed_dev_url("http://localhost:5173"));
    CHECK(lmp::app::is_allowed_dev_url("http://127.0.0.1:5173/"));
    CHECK(lmp::app::is_allowed_dev_url("http://localhost"));
    CHECK(!lmp::app::is_allowed_dev_url("https://localhost:5173"));
    CHECK(!lmp::app::is_allowed_dev_url("http://localhost.evil.com:5173"));
    CHECK(!lmp::app::is_allowed_dev_url("http://localhost@evil.com:5173"));
    CHECK(!lmp::app::is_allowed_dev_url("javascript:alert(1)"));
    CHECK(!lmp::app::is_allowed_dev_url("file:///etc/passwd"));
}

TEST(settings_atomic_roundtrip_and_default_path) {
    CHECK(!lmp::app::default_settings_path().empty());

    TempDir dir("settings");
    const fs::path cfg = dir.path / "nested" / "settings.json";
    CHECK_EQ(lmp::app::load_settings_json(cfg), "{}");

    std::string err;
    CHECK(!lmp::app::save_settings_json(cfg, "{bad json", err));
    CHECK(!err.empty());

    CHECK(lmp::app::save_settings_json(cfg, R"({"mode":"agent","theme":"dark"})", err));
    CHECK_EQ(err, "");
    const json reloaded = json::parse(lmp::app::load_settings_json(cfg));
    CHECK_EQ(reloaded.value("mode", ""), "agent");
    CHECK_EQ(reloaded.value("theme", ""), "dark");
}

TEST(checkpoint_classification_and_model_scanning) {
    TempDir dir("models");
    const fs::path dense = dir.path / "Qwen3.8-27B-MLX-4bit";
    const fs::path mtp = dir.path / "Qwen3.8-27B-MTP-4bit";
    const fs::path moe = dir.path / "Qwen3.6-35B-A3B-MLX-4bit";
    write_text(dense / "config.json", R"({"text_config":{"model_type":"qwen3_5_text"}})");
    write_text(dense / "model.safetensors", "abcd");
    write_text(mtp / "config.json", R"({"model_type":"qwen3_5_mtp","block_size":4})");
    write_text(moe / "config.json", R"({"model_type":"qwen3_5_moe"})");

    CHECK_EQ(std::string(lmp::app::to_string(lmp::app::classify_checkpoint(dense))), "dense");
    CHECK_EQ(std::string(lmp::app::to_string(lmp::app::classify_checkpoint(moe))), "moe");
    CHECK_EQ(std::string(lmp::app::to_string(lmp::app::classify_checkpoint(mtp))), "mtp");
    CHECK_EQ(std::string(lmp::app::to_string(lmp::app::classify_checkpoint(dir.path / "nope"))),
             "unknown");

    CHECK_EQ(lmp::app::find_sibling_mtp(dense).filename().string(), "Qwen3.8-27B-MTP-4bit");

    const json scanned = json::parse(lmp::app::scan_models_json({dir.path}));
    REQUIRE(scanned.contains("models") && scanned["models"].is_array());
    CHECK_EQ(scanned["models"].size(), static_cast<std::size_t>(2));
}

TEST(workspace_inspection_and_containment) {
    TempDir outside("outside");
    write_text(outside.path / "secret.txt", "top-secret");

    TempDir ws("ws");
    write_text(ws.path / "src" / "main.cpp", "int main() { return 0; }\n");
    write_text(ws.path / "bin.dat", std::string("AB\0CD", 5));

    std::error_code ec;
    fs::create_symlink(outside.path / "secret.txt", ws.path / "escape.txt", ec);

    const json listed = json::parse(lmp::app::workspace_files_json(ws.path, 50));
    CHECK(listed["files"].is_array());
    CHECK(listed["files"].size() >= static_cast<std::size_t>(1));

    const json diffed = json::parse(lmp::app::workspace_diff_json(ws.path));
    CHECK_EQ(diffed.value("is_git", true), false);

    const json good = json::parse(lmp::app::read_file_preview_json(ws.path, "src/main.cpp", 1024));
    CHECK(good.value("ok", false));
    CHECK_EQ(good.value("binary", true), false);
    CHECK(good.value("content", "").find("int main()") != std::string::npos);

    const json bin = json::parse(lmp::app::read_file_preview_json(ws.path, "bin.dat", 1024));
    CHECK(bin.value("ok", false));
    CHECK_EQ(bin.value("binary", false), true);

    const json trav = json::parse(lmp::app::read_file_preview_json(ws.path, "../outside/secret.txt", 1024));
    CHECK(!trav.value("ok", true));

    if (fs::is_symlink(ws.path / "escape.txt", ec)) {
        const json sym = json::parse(lmp::app::read_file_preview_json(ws.path, "escape.txt", 1024));
        CHECK(!sym.value("ok", true));
    }

    const std::string png_b64 = lmp::platform::base64_encode("PNGDATA");
    const json attached = json::parse(lmp::app::save_image_attachment_json(ws.path, "png", png_b64));
    CHECK(attached.value("ok", false));
    CHECK(!attached.value("rel_path", "").empty());
    CHECK(fs::is_regular_file(ws.path / attached.value("rel_path", "")));

    const json bad_ext = json::parse(lmp::app::save_image_attachment_json(ws.path, "exe", png_b64));
    CHECK(!bad_ext.value("ok", true));
}

TEST(escape_js_string_arg_escapes_line_separators) {
    const std::string raw = std::string("a\"b\\c\nd") + "\xE2\x80\xA8" + "e" + "\xE2\x80\xA9";
    const std::string escaped = lmp::app::escape_js_string_arg(raw);
    CHECK(escaped.find("\\u2028") != std::string::npos);
    CHECK(escaped.find("\\u2029") != std::string::npos);
    CHECK(escaped.find('\n') == std::string::npos);
}
