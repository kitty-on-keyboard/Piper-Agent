#pragma once
//
// Cursor/Claude-style skills: reusable recipes an agent discovers and loads.
//
// On-disk format: SKILL.md with YAML frontmatter (name + description) and markdown body.
// Allowlisted roots (priority order):
//   1. <cwd>/.piper/skills/<id>/SKILL.md
//   2. <cwd>/.cursor/skills/<id>/SKILL.md
//   3. <cwd>/.agents/skills/<id>/SKILL.md
//   4. ~/.piper/skills/<id>/SKILL.md
//
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lmp::tools {

inline constexpr std::size_t kSkillMaxBytes = 64U * 1024;

struct SkillSummary {
    std::string id;          // kebab-case folder name
    std::string name;        // frontmatter name
    std::string description; // frontmatter description
    std::string path;        // absolute path to SKILL.md
    std::string source_root; // ".piper", ".cursor", ".agents", or "user"
};

struct SkillDetail {
    SkillSummary summary;
    std::string body;
    bool truncated = false;
    std::size_t original_bytes = 0;
};

// Returns true if id consists only of [a-zA-Z0-9_-], is non-empty, and has no path separators.
[[nodiscard]] bool is_valid_skill_id(std::string_view id) noexcept;

// Parses YAML frontmatter between leading '---' and second '---'.
// Extracts name and description. Body is everything after the second '---'.
[[nodiscard]] bool parse_skill_frontmatter(std::string_view content,
                                          std::string& name,
                                          std::string& description,
                                          std::string& body,
                                          std::string& error);

// Scans allowlisted roots in priority order, returning discovered skills. Earlier roots win duplicate ids.
[[nodiscard]] std::vector<SkillSummary> discover_skills(
    const std::string& workspace_root,
    const std::string& home_dir = "");

// Loads a skill by id from allowlisted roots. Caps body at kSkillMaxBytes with truncation notice if needed.
[[nodiscard]] std::optional<SkillDetail> load_skill(
    const std::string& workspace_root,
    std::string_view skill_id,
    const std::string& home_dir = "",
    std::string* error = nullptr);

} // namespace lmp::tools
