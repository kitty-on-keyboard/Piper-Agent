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
        "GITHUB_PAT=pat_123",
        "BEARER_AUTH=bearer_xyz",
        "SESSION_JWT=jwt_abc",
        "PRIVKEY_SECRET=priv_123",
        "AUTH_PASSCODE=code_456",
        "SERVICE_TICKET=tkt_789",
        "LC_CRED=secret_cred",
        "XDG_CREDS=secret_creds",
        "LC_TOKEN_ID=secret_id",
        "XDG_SECRET_KEY=secret_key",
        "LC_AUTH_TOKEN=secret_token",
        "XDG_ACCESS_TOKEN=secret_access",
        "LC_SECRET_ID=secret_id",
        "XDG_SESSION_ID=session_id",
        "LC_SESS_ID=sess_id",
        "LC_PASS=secret_pass",
        "XDG_KEYFILE=/tmp/keyfile",
        "LC_KEY_FILE=/tmp/keyfile",
        "XDG_SECRET_FILE=/tmp/secretfile",
        "LC_TOKEN_FILE=/tmp/tokenfile",
        "XDG_AUTH_FILE=/tmp/authfile",
        "LC_CERT_FILE=/tmp/certfile",
        "XDG_CERT_PATH=/tmp/certpath",
        "LC_KEY_PATH=/tmp/keypath",
        "XDG_SECRET_PATH=/tmp/secretpath",
        "LC_TOKEN_PATH=/tmp/tokenpath",
        "XDG_AUTH_PATH=/tmp/authpath",
        "LC_CRED_FILE=/tmp/credfile",
        "XDG_CRED_PATH=/tmp/credpath",
        "LC_PEM=/tmp/cert.pem",
        "XDG_PEM_FILE=/tmp/key.pem",
        nullptr,
    };
    const auto env = build_child_environ(parent, {});
    CHECK(has_key(env, "PATH"));
    CHECK(!has_key(env, "AWS_SECRET_ACCESS_KEY"));
    CHECK(!has_key(env, "SSH_AUTH_SOCK"));
    CHECK(!has_key(env, "LMP_EVENT_LOG"));
    CHECK(!has_key(env, "GITHUB_TOKEN"));
    CHECK(!has_key(env, "DYLD_INSERT_LIBRARIES"));
    CHECK(!has_key(env, "GITHUB_PAT"));
    CHECK(!has_key(env, "BEARER_AUTH"));
    CHECK(!has_key(env, "SESSION_JWT"));
    CHECK(!has_key(env, "PRIVKEY_SECRET"));
    CHECK(!has_key(env, "AUTH_PASSCODE"));
    CHECK(!has_key(env, "SERVICE_TICKET"));
    CHECK(!has_key(env, "LC_CRED"));
    CHECK(!has_key(env, "XDG_CREDS"));
    CHECK(!has_key(env, "LC_TOKEN_ID"));
    CHECK(!has_key(env, "XDG_SECRET_KEY"));
    CHECK(!has_key(env, "LC_AUTH_TOKEN"));
    CHECK(!has_key(env, "XDG_ACCESS_TOKEN"));
    CHECK(!has_key(env, "LC_SECRET_ID"));
    CHECK(!has_key(env, "XDG_SESSION_ID"));
    CHECK(!has_key(env, "LC_SESS_ID"));
    CHECK(!has_key(env, "LC_PASS"));
    CHECK(!has_key(env, "XDG_KEYFILE"));
    CHECK(!has_key(env, "LC_KEY_FILE"));
    CHECK(!has_key(env, "XDG_SECRET_FILE"));
    CHECK(!has_key(env, "LC_TOKEN_FILE"));
    CHECK(!has_key(env, "XDG_AUTH_FILE"));
    CHECK(!has_key(env, "LC_CERT_FILE"));
    CHECK(!has_key(env, "XDG_CERT_PATH"));
    CHECK(!has_key(env, "LC_KEY_PATH"));
    CHECK(!has_key(env, "XDG_SECRET_PATH"));
    CHECK(!has_key(env, "LC_TOKEN_PATH"));
    CHECK(!has_key(env, "XDG_AUTH_PATH"));
    CHECK(!has_key(env, "LC_CRED_FILE"));
    CHECK(!has_key(env, "XDG_CRED_PATH"));
    CHECK(!has_key(env, "LC_PEM"));
    CHECK(!has_key(env, "XDG_PEM_FILE"));
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
    // LC_* and XDG_* are inherited; secrets with those prefixes must still be denied.
    const char* parent[] = {
        "LC_ALL=C",
        "LC_TOKEN=nope",
        "XDG_SECRET=nope",
        "LC_APIKEY=secret",
        "XDG_PASSPHRASE=secret",
        "LC_PRIVATEKEY=secret",
        "XDG_COOKIE=secret",
        "LC_SESSID=secret",
        "LC_BEARER=secret",
        "XDG_CERT=secret",
        "LC_CERTIFICATE=secret",
        "XDG_SIGNATURE=secret",
        "LC_PRIVKEY=secret",
        "XDG_PASSCODE=secret",
        "LC_TICKET=secret",
        "LC_CRED=secret",
        "XDG_CREDS=secret",
        "LC_TOKEN_ID=secret",
        "XDG_SECRET_KEY=secret",
        "LC_AUTH_TOKEN=secret",
        "XDG_ACCESS_TOKEN=secret",
        "LC_SECRET_ID=secret",
        "XDG_SECRET_ID=secret",
        "LC_SESSION_ID=secret",
        "XDG_SESSION_ID=secret",
        "LC_SESS_ID=secret",
        "XDG_SESS_ID=secret",
        "LC_PASS=secret",
        "XDG_KEYFILE=secret",
        "LC_KEY_FILE=secret",
        "XDG_SECRET_FILE=secret",
        "LC_TOKEN_FILE=secret",
        "XDG_AUTH_FILE=secret",
        "LC_CERT_FILE=secret",
        "XDG_CERT_PATH=secret",
        "LC_KEY_PATH=secret",
        "XDG_SECRET_PATH=secret",
        "LC_TOKEN_PATH=secret",
        "XDG_AUTH_PATH=secret",
        "LC_CRED_FILE=secret",
        "XDG_CRED_PATH=secret",
        "LC_PEM=secret",
        "XDG_PEM_FILE=secret",
        nullptr,
    };
    const auto env = build_child_environ(parent, {});
    CHECK_EQ(find_key(env, "LC_ALL").value, std::string("C"));
    CHECK(!has_key(env, "LC_TOKEN"));
    CHECK(!has_key(env, "XDG_SECRET"));
    CHECK(!has_key(env, "LC_APIKEY"));
    CHECK(!has_key(env, "XDG_PASSPHRASE"));
    CHECK(!has_key(env, "LC_PRIVATEKEY"));
    CHECK(!has_key(env, "XDG_COOKIE"));
    CHECK(!has_key(env, "LC_SESSID"));
    CHECK(!has_key(env, "LC_BEARER"));
    CHECK(!has_key(env, "XDG_CERT"));
    CHECK(!has_key(env, "LC_CERTIFICATE"));
    CHECK(!has_key(env, "XDG_SIGNATURE"));
    CHECK(!has_key(env, "LC_PRIVKEY"));
    CHECK(!has_key(env, "XDG_PASSCODE"));
    CHECK(!has_key(env, "LC_TICKET"));
    CHECK(!has_key(env, "LC_CRED"));
    CHECK(!has_key(env, "XDG_CREDS"));
    CHECK(!has_key(env, "LC_TOKEN_ID"));
    CHECK(!has_key(env, "XDG_SECRET_KEY"));
    CHECK(!has_key(env, "LC_AUTH_TOKEN"));
    CHECK(!has_key(env, "XDG_ACCESS_TOKEN"));
    CHECK(!has_key(env, "LC_SECRET_ID"));
    CHECK(!has_key(env, "XDG_SECRET_ID"));
    CHECK(!has_key(env, "LC_SESSION_ID"));
    CHECK(!has_key(env, "XDG_SESSION_ID"));
    CHECK(!has_key(env, "LC_SESS_ID"));
    CHECK(!has_key(env, "XDG_SESS_ID"));
    CHECK(!has_key(env, "LC_PASS"));
    CHECK(!has_key(env, "XDG_KEYFILE"));
    CHECK(!has_key(env, "LC_KEY_FILE"));
    CHECK(!has_key(env, "XDG_SECRET_FILE"));
    CHECK(!has_key(env, "LC_TOKEN_FILE"));
    CHECK(!has_key(env, "XDG_AUTH_FILE"));
    CHECK(!has_key(env, "LC_CERT_FILE"));
    CHECK(!has_key(env, "XDG_CERT_PATH"));
    CHECK(!has_key(env, "LC_KEY_PATH"));
    CHECK(!has_key(env, "XDG_SECRET_PATH"));
    CHECK(!has_key(env, "LC_TOKEN_PATH"));
    CHECK(!has_key(env, "XDG_AUTH_PATH"));
    CHECK(!has_key(env, "LC_CRED_FILE"));
    CHECK(!has_key(env, "XDG_CRED_PATH"));
    CHECK(!has_key(env, "LC_PEM"));
    CHECK(!has_key(env, "XDG_PEM_FILE"));
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
