// Child-env construction for MCP spawn, without spawning.
//
// Subprocess used to copy the sidecar's full environ. These checks are the ones that
// would have been green under that policy and are red under the allowlist: parent
// secrets must not appear, config env must, and a config PATH must replace the parent's
// rather than sitting behind it where getenv never sees it.

#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/mcp/spawn_env.hpp"

#include "tests/check.hpp"

using lmp::mcp::build_child_environ;

namespace {

struct Lookup {
    int count = 0;
    std::string value;
};

Lookup find_key(const std::vector<std::string>& env, std::string_view key) {
    Lookup out;
    const std::string prefix = std::string(key) + "=";
    for (const std::string& kv : env) {
        if (kv.size() >= prefix.size() &&
            kv.compare(0, prefix.size(), prefix) == 0) {
            ++out.count;
            out.value = kv.substr(prefix.size());
        }
    }
    return out;
}

bool has_key(const std::vector<std::string>& env, std::string_view key) {
    return find_key(env, key).count > 0;
}

} // namespace

TEST(allowlisted_parent_keys_are_inherited) {
    const char* parent[] = {
        "PATH=/usr/bin",
        "HOME=/Users/dev",
        "LC_ALL=en_US.UTF-8",
        "USER=dev",
        nullptr,
    };
    const auto env = build_child_environ(parent, {});
    CHECK_EQ(find_key(env, "PATH").value, std::string("/usr/bin"));
    CHECK_EQ(find_key(env, "HOME").value, std::string("/Users/dev"));
    CHECK_EQ(find_key(env, "LC_ALL").value, std::string("en_US.UTF-8"));
    CHECK_EQ(find_key(env, "USER").value, std::string("dev"));
}

TEST(parent_secrets_are_dropped) {
    const char* parent[] = {
        "PATH=/usr/bin",
        "AWS_SECRET_ACCESS_KEY=wJalr",
        "SSH_AUTH_SOCK=/tmp/ssh.sock",
        "LMP_EVENT_LOG=/tmp/events.jsonl",
        "GITHUB_TOKEN=ghp_xxx",
        "DYLD_INSERT_LIBRARIES=/tmp/evil.dylib",
        nullptr,
    };
    const auto env = build_child_environ(parent, {});
    CHECK(has_key(env, "PATH"));
    CHECK(!has_key(env, "AWS_SECRET_ACCESS_KEY"));
    CHECK(!has_key(env, "SSH_AUTH_SOCK"));
    CHECK(!has_key(env, "LMP_EVENT_LOG"));
    CHECK(!has_key(env, "GITHUB_TOKEN"));
    CHECK(!has_key(env, "DYLD_INSERT_LIBRARIES"));
}

TEST(config_env_appears_and_replaces_path) {
    const char* parent[] = {"PATH=/usr/bin", "HOME=/home/dev", nullptr};
    const std::vector<std::pair<std::string, std::string>> extra = {
        {"PATH", "/opt/godoer/bin:/usr/bin"},
        {"GODOT_BIN", "/Applications/Godot.app/Contents/MacOS/Godot"},
    };
    const auto env = build_child_environ(parent, extra);
    const Lookup path = find_key(env, "PATH");
    CHECK_EQ(path.count, 1);
    CHECK_EQ(path.value, std::string("/opt/godoer/bin:/usr/bin"));
    CHECK_EQ(find_key(env, "GODOT_BIN").value,
             std::string("/Applications/Godot.app/Contents/MacOS/Godot"));
    CHECK_EQ(find_key(env, "HOME").value, std::string("/home/dev"));
}

TEST(config_may_set_a_secret_the_parent_held) {
    // Explicit opt-in. The leak is silent inheritance, not an operator-named key.
    const char* parent[] = {"PATH=/usr/bin", "AWS_SECRET_ACCESS_KEY=from-parent", nullptr};
    const std::vector<std::pair<std::string, std::string>> extra = {
        {"AWS_SECRET_ACCESS_KEY", "from-config"},
    };
    const auto env = build_child_environ(parent, extra);
    const Lookup aws = find_key(env, "AWS_SECRET_ACCESS_KEY");
    CHECK_EQ(aws.count, 1);
    CHECK_EQ(aws.value, std::string("from-config"));
}

TEST(deny_beats_a_coincidental_allow_prefix) {
    // LC_* is inherited; LC_TOKEN looks like a secret sitting under that prefix.
    const char* parent[] = {"LC_ALL=C", "LC_TOKEN=nope", "XDG_SECRET=nope", nullptr};
    const auto env = build_child_environ(parent, {});
    CHECK_EQ(find_key(env, "LC_ALL").value, std::string("C"));
    CHECK(!has_key(env, "LC_TOKEN"));
    CHECK(!has_key(env, "XDG_SECRET"));
}

TEST(empty_extra_still_keeps_allowlisted_parent_keys) {
    const char* parent[] = {"PATH=/bin", "HOME=/tmp", "PYTHONPATH=/evil", nullptr};
    const auto env = build_child_environ(parent, {});
    CHECK(has_key(env, "PATH"));
    CHECK(has_key(env, "HOME"));
    CHECK(!has_key(env, "PYTHONPATH"));
}

TEST(invalid_parent_entries_are_skipped) {
    const char* parent[] = {"NOEQUALS", "=emptykey", "PATH=/usr/bin", nullptr};
    const auto env = build_child_environ(parent, {});
    CHECK_EQ(env.size(), std::size_t{1});
    CHECK_EQ(find_key(env, "PATH").value, std::string("/usr/bin"));
}

TEST(null_parent_is_only_the_config_env) {
    const std::vector<std::pair<std::string, std::string>> extra = {{"FOO", "bar"}};
    const auto env = build_child_environ(nullptr, extra);
    CHECK_EQ(env.size(), std::size_t{1});
    CHECK_EQ(find_key(env, "FOO").value, std::string("bar"));
}
