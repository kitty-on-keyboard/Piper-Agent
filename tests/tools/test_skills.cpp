#include "src/tools/skills.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "tests/check.hpp"

using namespace lmp::tools;

namespace {

std::string create_test_temp_dir() {
    const std::filesystem::path p = std::filesystem::temp_directory_path() /
        ("test_skills_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(p);
    return p.string();
}

void write_test_file(const std::string& path, const std::string& content) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path);
    out << content;
}

} // namespace

TEST(skills_valid_id) {
    CHECK(is_valid_skill_id("godoer"));
    CHECK(is_valid_skill_id("swift-6-best-practices"));
    CHECK(is_valid_skill_id("c_synth_123"));
    CHECK(!is_valid_skill_id(""));
    CHECK(!is_valid_skill_id("../traversal"));
    CHECK(!is_valid_skill_id("a/b"));
    CHECK(!is_valid_skill_id("a\\b"));
    CHECK(!is_valid_skill_id("skill.name"));
}

TEST(skills_parse_frontmatter) {
    std::string name, desc, body, err;

    // Valid
    const std::string valid =
        "---\nname: my-skill\ndescription: \"Use this for testing\"\n---\n# Recipe Body\nSteps here.";
    CHECK(parse_skill_frontmatter(valid, name, desc, body, err));
    CHECK_EQ(name, "my-skill");
    CHECK_EQ(desc, "Use this for testing");
    CHECK_EQ(body, "# Recipe Body\nSteps here.");

    // Missing opening delimiter
    CHECK(!parse_skill_frontmatter("name: no-delim\ndescription: x\n---\nbody", name, desc, body, err));
    CHECK(err.find("start with '---'") != std::string::npos);

    // Missing closing delimiter
    CHECK(!parse_skill_frontmatter("---\nname: no-close\ndescription: x\nbody", name, desc, body, err));
    CHECK(err.find("missing closing '---'") != std::string::npos);

    // Missing name
    CHECK(!parse_skill_frontmatter("---\ndescription: only desc\n---\nbody", name, desc, body, err));
    CHECK(err.find("missing required frontmatter 'name'") != std::string::npos);

    // Missing description
    CHECK(!parse_skill_frontmatter("---\nname: only name\n---\nbody", name, desc, body, err));
    CHECK(err.find("missing required frontmatter 'description'") != std::string::npos);
}

TEST(skills_discovery_and_priority) {
    const std::string ws = create_test_temp_dir();
    const std::string home = create_test_temp_dir();

    // Priority 1: .piper
    write_test_file(ws + "/.piper/skills/common-skill/SKILL.md",
                    "---\nname: Common Piper\ndescription: From .piper\n---\nPiper body");
    // Priority 2: .cursor (should be shadowed for common-skill)
    write_test_file(ws + "/.cursor/skills/common-skill/SKILL.md",
                    "---\nname: Common Cursor\ndescription: From .cursor\n---\nCursor body");
    // Cursor only
    write_test_file(ws + "/.cursor/skills/cursor-only/SKILL.md",
                    "---\nname: Cursor Only\ndescription: From .cursor\n---\nBody");
    // User only
    write_test_file(home + "/.piper/skills/user-skill/SKILL.md",
                    "---\nname: User Skill\ndescription: From user library\n---\nUser body");

    const auto skills = discover_skills(ws, home);
    CHECK_EQ(skills.size(), std::size_t{3});

    auto find_skill = [&](const std::string& id) -> const SkillSummary* {
        for (const auto& s : skills) {
            if (s.id == id) return &s;
        }
        return nullptr;
    };

    const auto* common = find_skill("common-skill");
    REQUIRE(common != nullptr);
    CHECK_EQ(common->name, "Common Piper");
    CHECK_EQ(common->source_root, ".piper");

    const auto* cur = find_skill("cursor-only");
    REQUIRE(cur != nullptr);
    CHECK_EQ(cur->source_root, ".cursor");

    const auto* usr = find_skill("user-skill");
    REQUIRE(usr != nullptr);
    CHECK_EQ(usr->source_root, "user");

    // Clean up
    std::filesystem::remove_all(ws);
    std::filesystem::remove_all(home);
}

TEST(skills_load_and_truncation) {
    const std::string ws = create_test_temp_dir();
    std::string large_body(70 * 1024, 'A');
    write_test_file(ws + "/.piper/skills/large-skill/SKILL.md",
                    "---\nname: Large\ndescription: Tests truncation\n---\n" + large_body);

    std::string err;
    auto detail = load_skill(ws, "large-skill", "", &err);
    REQUIRE(detail.has_value());
    CHECK(detail->truncated);
    CHECK_EQ(detail->original_bytes, large_body.size());
    CHECK(detail->body.find("[TRUNCATED: Skill body exceeded 64 KiB cap]") != std::string::npos);

    // Traversal rejection
    CHECK(!load_skill(ws, "../large-skill", "", &err).has_value());
    CHECK(err.find("invalid skill id") != std::string::npos);

    // Non-existent skill
    CHECK(!load_skill(ws, "no-such-skill", "", &err).has_value());
    CHECK(err.find("skill not found") != std::string::npos);

    std::filesystem::remove_all(ws);
}
