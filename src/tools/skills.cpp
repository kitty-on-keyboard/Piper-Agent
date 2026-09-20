#include "src/tools/skills.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <unordered_set>

#include "src/platform/fs.hpp"

namespace lmp::tools {
namespace {

std::string trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t' || s.front() == '\r' || s.front() == '\n')) {
        s.remove_prefix(1);
    }
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n')) {
        s.remove_suffix(1);
    }
    return std::string(s);
}

std::string unquote(std::string_view s) {
    std::string t = trim(s);
    if (t.size() >= 2) {
        if ((t.front() == '"' && t.back() == '"') || (t.front() == '\'' && t.back() == '\'')) {
            return t.substr(1, t.size() - 2);
        }
    }
    return t;
}

std::string resolve_home(const std::string& home_dir) {
    if (!home_dir.empty()) return home_dir;
    const char* h = std::getenv("HOME");
    return h ? std::string(h) : "";
}

struct RootEntry {
    std::string path;
    std::string source_root;
};

std::vector<RootEntry> candidate_roots(const std::string& workspace_root, const std::string& home_dir) {
    std::vector<RootEntry> roots;
    if (!workspace_root.empty()) {
        roots.push_back({workspace_root + "/.piper/skills", ".piper"});
        roots.push_back({workspace_root + "/.cursor/skills", ".cursor"});
        roots.push_back({workspace_root + "/.agents/skills", ".agents"});
    }
    const std::string home = resolve_home(home_dir);
    if (!home.empty()) {
        roots.push_back({home + "/.piper/skills", "user"});
    }
    return roots;
}

} // namespace

bool is_valid_skill_id(std::string_view id) noexcept {
    if (id.empty()) return false;
    for (char c : id) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
            return false;
        }
    }
    return true;
}

bool parse_skill_frontmatter(std::string_view content,
                              std::string& name,
                              std::string& description,
                              std::string& body,
                              std::string& error) {
    name.clear();
    description.clear();
    body.clear();
    error.clear();

    // Check for leading '---'
    std::size_t start = 0;
    while (start < content.size() && (content[start] == ' ' || content[start] == '\t' ||
                                     content[start] == '\r' || content[start] == '\n')) {
        start++;
    }
    if (content.substr(start, 3) != "---") {
        error = "SKILL.md must start with '---' YAML frontmatter";
        return false;
    }
    start += 3;
    if (start < content.size() && content[start] == '\r') start++;
    if (start < content.size() && content[start] == '\n') start++;

    // Find closing '---'
    std::size_t end_fm = std::string_view::npos;
    std::size_t search_pos = start;
    while (search_pos < content.size()) {
        std::size_t newline = content.find('\n', search_pos);
        std::size_t line_end = (newline == std::string_view::npos) ? content.size() : newline;
        std::string_view line = content.substr(search_pos, line_end - search_pos);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        if (line == "---") {
            end_fm = search_pos;
            search_pos = (newline == std::string_view::npos) ? content.size() : newline + 1;
            break;
        }
        if (newline == std::string_view::npos) break;
        search_pos = newline + 1;
    }

    if (end_fm == std::string_view::npos) {
        error = "SKILL.md missing closing '---' for YAML frontmatter";
        return false;
    }

    std::string_view fm_content = content.substr(start, end_fm - start);
    body = std::string(content.substr(search_pos));
    while (!body.empty() && (body.front() == '\r' || body.front() == '\n')) {
        body.erase(0, 1);
    }

    // Parse lines in fm_content
    std::istringstream stream{std::string(fm_content)};
    std::string line;
    std::string current_key;
    std::string current_val;

    auto flush_key = [&](const std::string& k, const std::string& v) {
        if (k == "name") {
            name = unquote(v);
        } else if (k == "description") {
            description = unquote(v);
        }
    };

    while (std::getline(stream, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;

        auto colon = line.find(':');
        if (colon != std::string::npos && (line[0] != ' ' && line[0] != '\t')) {
            if (!current_key.empty()) {
                flush_key(current_key, current_val);
            }
            current_key = trim(line.substr(0, colon));
            current_val = trim(line.substr(colon + 1));
        } else if (!current_key.empty() && (line[0] == ' ' || line[0] == '\t')) {
            // Continuation line
            current_val += " " + trim(line);
        }
    }
    if (!current_key.empty()) {
        flush_key(current_key, current_val);
    }

    if (name.empty()) {
        error = "SKILL.md missing required frontmatter 'name'";
        return false;
    }
    if (description.empty()) {
        error = "SKILL.md missing required frontmatter 'description'";
        return false;
    }

    return true;
}

std::vector<SkillSummary> discover_skills(const std::string& workspace_root,
                                         const std::string& home_dir) {
    std::vector<SkillSummary> results;
    std::unordered_set<std::string> seen_ids;
    std::error_code ec;

    const auto roots = candidate_roots(workspace_root, home_dir);
    for (const auto& r : roots) {
        if (!std::filesystem::is_directory(r.path, ec)) continue;

        for (const auto& entry : std::filesystem::directory_iterator(r.path, ec)) {
            if (!entry.is_directory(ec)) continue;

            const std::string id = entry.path().filename().string();
            if (!is_valid_skill_id(id) || seen_ids.count(id) > 0) continue;

            const std::filesystem::path skill_file = entry.path() / "SKILL.md";
            if (!std::filesystem::is_regular_file(skill_file, ec)) continue;

            const platform::FileContents f = platform::read_file_whole(skill_file.string(), 128U * 1024);
            if (!f.ok()) continue;

            std::string name, desc, body, err;
            if (parse_skill_frontmatter(f.bytes, name, desc, body, err)) {
                SkillSummary sum;
                sum.id = id;
                sum.name = std::move(name);
                sum.description = std::move(desc);
                sum.path = skill_file.string();
                sum.source_root = r.source_root;
                results.push_back(std::move(sum));
                seen_ids.insert(id);
            }
        }
    }
    return results;
}

std::optional<SkillDetail> load_skill(const std::string& workspace_root,
                                      std::string_view skill_id,
                                      const std::string& home_dir,
                                      std::string* error) {
    if (error) error->clear();
    if (!is_valid_skill_id(skill_id)) {
        if (error) *error = "invalid skill id: " + std::string(skill_id);
        return std::nullopt;
    }

    std::error_code ec;
    const auto roots = candidate_roots(workspace_root, home_dir);
    for (const auto& r : roots) {
        const std::filesystem::path skill_file = std::filesystem::path(r.path) / std::string(skill_id) / "SKILL.md";
        if (!std::filesystem::is_regular_file(skill_file, ec)) continue;

        const platform::FileContents f = platform::read_file_whole(skill_file.string(), 2U * 1024 * 1024);
        if (!f.ok()) {
            if (error) *error = "cannot read " + skill_file.string() + ": " + f.error;
            return std::nullopt;
        }

        std::string name, desc, body, err;
        if (!parse_skill_frontmatter(f.bytes, name, desc, body, err)) {
            if (error) *error = "failed parsing " + skill_file.string() + ": " + err;
            return std::nullopt;
        }

        SkillDetail detail;
        detail.summary.id = std::string(skill_id);
        detail.summary.name = std::move(name);
        detail.summary.description = std::move(desc);
        detail.summary.path = skill_file.string();
        detail.summary.source_root = r.source_root;
        detail.original_bytes = body.size();

        if (body.size() > kSkillMaxBytes) {
            body.resize(kSkillMaxBytes);
            body += "\n\n[TRUNCATED: Skill body exceeded 64 KiB cap]";
            detail.truncated = true;
        }
        detail.body = std::move(body);
        return detail;
    }

    if (error) *error = "skill not found: " + std::string(skill_id);
    return std::nullopt;
}

} // namespace lmp::tools
