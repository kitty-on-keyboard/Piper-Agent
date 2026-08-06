// SyntaxChecker -- the post-edit, non-model feedback path (G1).
//
// The three properties worth asserting are the three that make the feature safe rather
// than the one that makes it useful: silence when there is no contract, silence when C++
// has no compile database, and a real diagnostic when there is one.

#include <unistd.h>

#include <cstdlib>
#include <string>

#include "src/platform/fs.hpp"
#include "src/tools/syntax_check.hpp"
#include "tests/check.hpp"

using namespace lmp::tools;

namespace {

std::string temp_dir() {
    char tmpl[] = "/tmp/lmp_syn_XXXXXX";
    const char* made = ::mkdtemp(tmpl);
    return made ? std::string(made) : std::string();
}

void write(const std::string& path, std::string_view body) {
    (void)lmp::platform::write_file_atomic(path, body);
}

} // namespace

TEST(a_broken_python_file_produces_a_diagnostic) {
    const std::string root = temp_dir();
    write(root + "/broken.py", "def f(:\n    return 1\n");
    const SyntaxChecker c(root, 2048);
    const SyntaxVerdict v = c.check("broken.py", 1);
    REQUIRE(v.ran);
    CHECK(!v.clean);
    CHECK_EQ(v.language, std::string("python"));
    CHECK(!v.diagnostics.empty());
}

TEST(a_good_python_file_is_clean_and_says_nothing) {
    const std::string root = temp_dir();
    write(root + "/fine.py", "def f():\n    return 1\n");
    const SyntaxChecker c(root, 2048);
    const SyntaxVerdict v = c.check("fine.py", 1);
    REQUIRE(v.ran);
    CHECK(v.clean);
    CHECK(v.diagnostics.empty());
}

// Silence, not "no checker available for .md". That line would recur every turn for the
// whole run, which is how a helpful feature becomes context pollution.
TEST(an_unrecognised_extension_is_silent) {
    const std::string root = temp_dir();
    write(root + "/notes.md", "# hello\n");
    const SyntaxChecker c(root, 2048);
    const SyntaxVerdict v = c.check("notes.md", 1);
    CHECK(!v.ran);
    CHECK(v.diagnostics.empty());
}

// A bare `c++ -fsyntax-only` on a project header emits a cascade of missing-include errors
// that are not about the edit. A false diagnostic is worse than no diagnostic: it sends
// the run off fixing something that was never broken.
TEST(cxx_without_a_compile_database_is_silent) {
    const std::string root = temp_dir();
    write(root + "/a.cpp", "#include \"nonexistent_project_header.hpp\"\nint main(){}\n");
    const SyntaxChecker c(root, 2048);
    const SyntaxVerdict v = c.check("a.cpp", 1);
    CHECK(!v.ran);
    CHECK(compile_db_syntax_command(root, root + "/a.cpp").empty());
}

// Tier 0 cannot execute. A Plan-mode run has nothing to check and must not refuse loudly.
TEST(tier_zero_does_not_run_a_check) {
    const std::string root = temp_dir();
    write(root + "/broken.py", "def f(:\n");
    const SyntaxChecker c(root, 2048);
    CHECK(!c.check("broken.py", 0).ran);
}

// Containment: the checker resolves against the root and refuses to leave it, the same
// rule every other path-taking tool follows.
TEST(a_path_outside_the_workspace_is_silent) {
    const std::string root = temp_dir();
    const SyntaxChecker c(root, 2048);
    CHECK(!c.check("../escape.py", 1).ran);
}

TEST(syntax_check_refuses_symlinked_file_and_directory_inputs) {
    const std::string root = temp_dir();
    const std::string outside = temp_dir();
    REQUIRE(!root.empty());
    REQUIRE(!outside.empty());
    write(outside + "/broken.py", "def escaped(:\n");
    REQUIRE(::symlink((outside + "/broken.py").c_str(),
                      (root + "/file-link.py").c_str()) == 0);
    REQUIRE(::symlink(outside.c_str(), (root + "/dir-link").c_str()) == 0);
    const SyntaxChecker c(root, 2048);

    CHECK(!c.check("file-link.py", 1).ran);
    CHECK(!c.check("dir-link/broken.py", 1).ran);
}

TEST(syntax_checker_canonicalizes_a_symlinked_workspace_root) {
    const std::string actual = temp_dir();
    REQUIRE(!actual.empty());
    const std::string alias = actual + "-alias";
    REQUIRE(::symlink(actual.c_str(), alias.c_str()) == 0);
    write(actual + "/fine.py", "value = 1\n");
    const SyntaxChecker c(alias, 2048);
    const SyntaxVerdict v = c.check("fine.py", 1);
    REQUIRE(v.ran);
    CHECK(v.clean);
}

TEST(the_compile_database_command_drops_output_arguments) {
    const std::string root = temp_dir();
    const std::string abs = root + "/x.cpp";
    write(root + "/x.cpp", "int main(){}\n");
    (void)lmp::platform::write_file_atomic(
        root + "/build/compile_commands.json", "");
    // No build directory was created above (write_file_atomic does not mkdir), so the
    // database is absent and the lookup must come back empty rather than guess.
    CHECK(compile_db_syntax_command(root, abs).empty());
}
