#!/usr/bin/env python3
"""Generate the log-triage answer key by running the real toolchain.

THE COMPILER WRITES THE KEY. Nothing here is hand-labelled. Every case is a real
source tree with a real defect, built by the real cmake/make/clang/swiftc/ctest/python
on this machine, and the noise around the diagnostic is real build output rather than
simulated build output.

Each case is compiled TWICE:

  * the FULL run -- a real `cmake --build . -- VERBOSE=1`, or a real swiftc/python/ctest
    invocation, with every rendering flag on. Its combined stdout+stderr is the corpus
    input, byte for byte: it is what SubprocessVerifier actually captures.

  * the KEY run -- the same source through the compiler's own noise-suppression flags
    (-fno-caret-diagnostics, -ftemplate-backtrace-limit=1, ...). What comes back is the
    compiler's own answer to "what are the errors".

The one judgement we make is which of the compiler's diagnostics an AGENT needs, and
it is made mechanically from two facts the generator already knows:

  primary  the FIRST error the compiler emitted. Later errors are usually cascade.
  local    every error, fatal error or NOTE whose location is a file inside the project.
           This is what says WHICH LINE OF ITS OWN CODE the agent must edit. In a
           template blow-up there is no error in the project at all, and the only local
           diagnostic is a `note: in instantiation of ... requested here`.

  Warnings are deliberately NOT in the key. A failing build emits hundreds of them and
  they are precisely what a compactor should drop; scoring their retention would pay an
  entrant for crowding the error out.

  python3 bakeoff/log_triage/generate_corpus.py [--out DIR] [--only ID,ID]

Writes logs/<id>.log (committed, frozen) and corpus.jsonl. Re-running on a different
Xcode WILL change the logs and break the pinned scores. That is intended: regenerating
the corpus is a deliberate act, not a side effect of upgrading a laptop.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

# Project root as it appears in the COMMITTED logs. The build really happens in a temp
# directory; every occurrence of that path is rewritten to this one in both the log and
# the key, identically, so the corpus carries no machine-specific path.
STABLE_ROOT = "/work/proj"

CXX = "clang++"
SWIFTC = "swiftc"
CXX_STD = "-std=c++20"

# The compiler's own "just tell me the errors" switches. Passing these is what makes the
# key the toolchain's opinion rather than a regex we wrote.
CLANG_KEY_FLAGS = [
    "-fno-caret-diagnostics",
    "-fno-color-diagnostics",
    "-fno-diagnostics-show-note-include-stack",
    "-fno-diagnostics-fixit-info",
    "-fno-elide-type",
]

DIAG_RE = re.compile(r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): "
                     r"(?P<level>error|fatal error|warning|note): (?P<msg>.*)$")
BARE_FATAL_RE = re.compile(r"^(?P<level>fatal error|error): (?P<msg>.*)$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
RANDOM_SCRATCH_RE = re.compile(r"(TemporaryDirectory\.)[A-Za-z0-9]{6}")

# Levels that go in the key. Warnings are excluded on purpose -- see the module docstring.
KEY_LEVELS = ("error", "fatal error", "note")


def strip_ansi(s):
    return ANSI_RE.sub("", s)


def run(cmd, cwd, env=None):
    """Run a command, combining stdout and stderr exactly as SubprocessVerifier does."""
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    proc = subprocess.run(cmd, cwd=cwd, env=full_env, shell=isinstance(cmd, str),
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Source generation
# ---------------------------------------------------------------------------

def filler_cpp(name, n_funcs=8):
    body = "\n".join(f"int {name}_f{i}(int x) {{ return x * {i + 1}; }}"
                     for i in range(n_funcs))
    return f"#include <string>\n#include <vector>\n{body}\n"


def cmake_project(defect_src, defect_index, n_files=48, extra_flags=(), defect_name=None):
    """A real cmake project of n_files translation units with the defect in one of them.

    Returns (files, broken_relpath). The build is driven by real cmake and real make with
    VERBOSE=1, so the tens of kilobytes of noise around the diagnostic are genuine build
    output -- full compiler command lines, include paths, progress percentages.
    """
    files = {}
    names = []
    broken = None
    for i in range(n_files):
        rel = f"src/mod_{i:02d}.cpp"
        names.append(rel)
        if i == defect_index:
            broken = rel
            files[rel] = defect_src
        else:
            files[rel] = filler_cpp(f"mod_{i:02d}")
    if defect_name:
        # Give the broken file a name a human would recognise in a log.
        new_rel = f"src/{defect_name}"
        files[new_rel] = files.pop(broken)
        names[names.index(broken)] = new_rel
        broken = new_rel

    flags = " ".join(["-Wall", "-Wextra", *extra_flags])
    files["CMakeLists.txt"] = (
        "cmake_minimum_required(VERSION 3.20)\n"
        "project(triage_bench CXX)\n"
        "set(CMAKE_CXX_STANDARD 20)\n"
        "set(CMAKE_CXX_STANDARD_REQUIRED ON)\n"
        f"add_compile_options({flags})\n"
        "include_directories(include include/engine include/backend third_party/inc)\n"
        "add_library(triage_bench STATIC\n    " + "\n    ".join(names) + ")\n")
    files["include/.keep"] = ""
    files["include/engine/.keep"] = ""
    files["include/backend/.keep"] = ""
    files["third_party/inc/.keep"] = ""
    return files, broken


def long_source_two_errors():
    lines = ["#include <string>", "#include <vector>", ""]
    for i in range(300):
        if i == 8:
            lines.append("int early(int x) { return x + undefined_thing_one; }")
        elif i == 280:
            lines.append("int late(int x) { return x + undefined_thing_two; }")
        else:
            lines.append(f"int pad_{i}(int x) {{ return x + {i}; }}")
    return "\n".join(lines) + "\n"


def warning_flood_source(n_warnings=160):
    lines = ["#include <cstdio>", ""]
    for i in range(n_warnings):
        lines.append(f"int warn_{i}(int unused_{i}) {{ int s = -1; unsigned u = 2u;"
                     f" return (s < u) ? {i} : 0; }}")
    lines.append("int broken(void) { return not_a_symbol_at_all; }")
    return "\n".join(lines) + "\n"


CASES = []
HOLDOUT = []
_target = CASES


def case(**kw):
    _target.append(kw)
    return kw


def holdout(**kw):
    HOLDOUT.append(kw)
    return kw


MISSING_SEMI = ("#include <string>\n"
                "struct Config {\n"
                "    int retries;\n"
                "    std::string host\n"
                "    bool verbose;\n"
                "};\n"
                "int use_config(const Config& c) { return c.retries; }\n")

UNDECLARED = ("#include <cstdio>\n"
              "int report(int n) {\n"
              "    return n + retryCount;\n"
              "}\n")

TEMPLATE_DEEP = ("#include <vector>\n#include <algorithm>\n"
                 "struct Point { int x; int y; };\n"
                 "void sort_points(std::vector<Point>& pts) {\n"
                 "    std::sort(pts.begin(), pts.end());\n"
                 "}\n")

NO_MATCHING_CTOR = ("#include <string>\n"
                    "struct Conn {\n"
                    "    Conn(const std::string& host, int port);\n"
                    "    Conn(int fd);\n"
                    "    Conn(const Conn&) = delete;\n"
                    "};\n"
                    "Conn* open_conn() { return new Conn(\"localhost\", \"8080\"); }\n")

ABSTRACT = ("struct Backend {\n"
            "    virtual ~Backend() = default;\n"
            "    virtual int generate(int) = 0;\n"
            "};\n"
            "int run_backend() { Backend b; return 0; }\n")

CONST_ASSIGN = ("#include <string>\n"
                + "\n".join(f"// padding line {i}" for i in range(60)) + "\n"
                "struct Session {\n"
                "    const std::string id;\n"
                "    void rename(const std::string& v) { id = v; }\n"
                "};\n")

MISSING_HEADER = ("#include \"engine/pipeline_config.hpp\"\n"
                  "int configure() { return 0; }\n")


# --- A. real cmake builds, diagnostic buried in real build noise -------------

_f, _b = cmake_project(MISSING_SEMI, 3, defect_name="config.cpp")
case(id="build_missing_semi_early", family="build", files=_f, broken=_b, cmake=True,
     why="A missing semicolon in the 4th of 48 translation units. `make -k` keeps "
         "going, so the diagnostic sits near the HEAD and is followed by tens of "
         "kilobytes of successful compiles. Tail-biased truncation loses it entirely.")

_f, _b = cmake_project(UNDECLARED, 42, defect_name="report.cpp")
case(id="build_undeclared_late", family="build", files=_f, broken=_b, cmake=True,
     why="The mirror image: the 43rd of 48 units fails, so the diagnostic is deep in "
         "the tail half. Head-biased truncation loses it.")

_f, _b = cmake_project(TEMPLATE_DEEP, 24, defect_name="sort_points.cpp")
case(id="build_template_deep", family="build", files=_f, broken=_b, cmake=True,
     why="40KB of template backtrace for ONE real error, inside a 48-unit build. Every "
         "error line the compiler emits is inside libc++; the ONLY line naming a file "
         "the agent can edit is a `note: in instantiation of ... requested here`. An "
         "errors-only matcher keeps the message and throws away the address.")

_f, _b = cmake_project(NO_MATCHING_CTOR, 18, defect_name="conn.cpp")
case(id="build_no_matching_ctor", family="build", files=_f, broken=_b, cmake=True,
     why="Overload resolution. The error line is short and the NOTES carry the "
         "candidate signatures; dropping every note leaves the agent guessing.")

_f, _b = cmake_project(ABSTRACT, 30, defect_name="backend.cpp")
case(id="build_abstract_instantiate", family="build", files=_f, broken=_b, cmake=True,
     why="The error names the class and a note names the unimplemented method. Two "
         "diagnostics at different levels, both needed.")

_f, _b = cmake_project(CONST_ASSIGN, 11, defect_name="session.cpp")
case(id="build_const_assign", family="build", files=_f, broken=_b, cmake=True,
     why="The offending assignment is 60 lines below the class it belongs to. The "
         "caret block is the only thing that says which assignment.")

_f, _b = cmake_project(MISSING_HEADER, 7, defect_name="pipeline.cpp")
case(id="build_missing_header", family="build", files=_f, broken=_b, cmake=True,
     why="`fatal error: file not found`. That unit stops immediately, so the log is a "
         "normal-sized build with one short diagnostic in it.")

_f, _b = cmake_project(long_source_two_errors(), 20, defect_name="pipeline_stage.cpp")
case(id="build_two_errors_one_file", family="build", files=_f, broken=_b, cmake=True,
     why="Two independent errors 270 source lines apart in one unit. Keeping only the "
         "first costs the agent a whole extra turn.")

_f, _b = cmake_project(warning_flood_source(), 15, defect_name="legacy_math.cpp")
case(id="build_warning_flood", family="build", files=_f, broken=_b, cmake=True,
     why="160 warnings and one error, the error emitted LAST in that unit. If warnings "
         "compete for the budget the error is what gets dropped. This is the case that "
         "justifies line_is_diagnostic being deliberately errors-only.")

_f, _b = cmake_project(UNDECLARED, 5, defect_name="report.cpp",
                       extra_flags=["-fcolor-diagnostics"])
case(id="build_color_diagnostics", family="build", files=_f, broken=_b, cmake=True,
     why="The same defect with -fcolor-diagnostics forced, which is what happens "
         "whenever a build runs under a pty. Every marker is now wrapped in ANSI "
         "escapes and every byte of them is charged against the budget.")

_f, _b = cmake_project("int main_unused() { return 0; }\n", 0, n_files=48)
case(id="build_ok_verbose", family="build", files=_f, broken=None, cmake=True,
     why="A large build that SUCCEEDS. There is no diagnostic to keep, so what is "
         "being measured is budget compliance and whether the compactor invents "
         "importance where there is none.")


def multi_defect_project():
    files, _ = cmake_project(UNDECLARED, 4, defect_name="alpha.cpp")
    files["src/beta.cpp"] = MISSING_SEMI
    files["src/gamma.cpp"] = ABSTRACT
    txt = files["CMakeLists.txt"]
    files["CMakeLists.txt"] = txt.replace(
        "add_library(triage_bench STATIC\n    ",
        "add_library(triage_bench STATIC\n    src/beta.cpp\n    src/gamma.cpp\n    ")
    return files


case(id="build_three_files_fail", family="build", files=multi_defect_project(),
     broken=None, cmake=True,
     why="Three separate units fail in one -k build, spread across the log. Keeping "
         "only the first error means the agent fixes one, re-runs, and pays again.")


# --- B. link ----------------------------------------------------------------

def link_project():
    files, _ = cmake_project("int compute_checksum(int);\n"
                             "int use_checksum() { return compute_checksum(7); }\n",
                             9, n_files=24, defect_name="checksum_user.cpp")
    files["src/main.cpp"] = "int use_checksum();\nint main() { return use_checksum(); }\n"
    txt = files["CMakeLists.txt"]
    files["CMakeLists.txt"] = (
        txt.replace("add_library(triage_bench STATIC\n    ",
                    "add_library(triage_bench STATIC\n    ")
        + "add_executable(app src/main.cpp)\ntarget_link_libraries(app triage_bench)\n")
    return files


case(id="build_link_undefined", family="link", files=link_project(), broken=None,
     cmake=True,
     why="Every compile succeeds and the LINK fails. There is no `error:` anywhere: "
         "the markers are `Undefined symbols` and `ld: symbol(s) not found`, the "
         "symbol is mangled, and the diagnostic is the last thing in the log.")


# --- C. bare single-unit cases (deliberately small) -------------------------

case(id="bare_undeclared_id", family="bare", files={"main.cpp": UNDECLARED},
     compile=["main.cpp"],
     why="171 bytes: UNDER every budget we score. The only correct answer is to return "
         "the input unchanged. A compactor that rewrites this is destroying information "
         "for nothing.")

case(id="bare_type_mismatch", family="bare",
     files={"main.cpp": "#include <string>\n"
                        "int main() {\n"
                        "    int port = \"8080\";\n"
                        "    std::string host = 42;\n"
                        "    return port;\n"
                        "}\n"},
     compile=["main.cpp"],
     why="Two independent errors in a short file, still under the 8K budget. Also a "
         "pass-through, and it catches a compactor that compacts unconditionally.")

case(id="bare_error_limit", family="bare",
     files={"main.cpp": "int main() {\n"
                        + "\n".join(f"    int v{i} = missing_{i};" for i in range(260))
                        + "\n    return 0;\n}\n"},
     compile=["main.cpp"],
     why="Enough errors to trip -ferror-limit. The log ends in a location-free "
         "`fatal error: too many errors emitted, stopping now`, a real diagnostic with "
         "no path in it at all.")


# --- D. Swift ---------------------------------------------------------------

def swift_project(broken_name, broken_src, n_pad=24):
    """A real SwiftPM package. Built with `swift build -v`, which is what a Swift build
    log actually looks like: full frontend command lines, and every diagnostic emitted
    TWICE -- once by emit-module and once by the compile job."""
    files = {f"Sources/App/Pad{i:02d}.swift":
             (f"struct Pad{i:02d} {{\n"
              f"    let value: Int = {i}\n"
              f"    func doubled() -> Int {{ return value * 2 }}\n"
              f"}}\n") for i in range(n_pad)}
    files[f"Sources/App/{broken_name}"] = broken_src
    files["Sources/App/main.swift"] = "print(Pad00().doubled())\n"
    files["Package.swift"] = (
        "// swift-tools-version:5.9\n"
        "import PackageDescription\n"
        "let package = Package(name: \"App\", targets: [\n"
        "    .executableTarget(name: \"App\", path: \"Sources/App\")\n])\n")
    return files


case(id="swift_missing_member", family="swift",
     files=swift_project("Hunt.swift",
                         "struct Hunt {\n    let id: Int\n    let title: String\n}\n"
                         "func describe(_ h: Hunt) -> String {\n"
                         "    return h.name\n"
                         "}\n"),
     swift_all=True,
     why="The shape the incumbent's markers were tuned on, so it should do well here. "
         "Present precisely so the clang/swift-tuned rows are visible next to the rest.")

case(id="swift_conformance", family="swift",
     files=swift_project("MLX.swift",
                         "protocol Backend {\n    func generate(_ n: Int) -> String\n}\n"
                         "struct MLX: Backend {\n    let name = \"mlx\"\n}\n"),
     swift_all=True,
     why="The error names the type and a NOTE names the missing requirement -- Swift's "
         "version of the abstract-class case.")

case(id="swift_type_mismatch", family="swift",
     files=swift_project("Port.swift",
                         "func port() -> Int {\n"
                         "    let raw: String = \"8080\"\n"
                         "    return raw\n"
                         "}\n"),
     swift_all=True,
     why="Swift renders a multi-line caret block with surrounding numbered source "
         "lines, so its diagnostic block is structurally unlike clang's.")


# --- E. tools that are not clang or swift -----------------------------------
# The incumbent is seven substrings tuned for clang/swift. An agent also runs python,
# pytest and ctest. These cases are in the headline on purpose.

case(id="python_traceback", family="other",
     files={"pipeline.py": "def load(rows):\n"
                           "    return [r['value'] for r in rows]\n"
                           "\n"
                           "def summarise(rows):\n"
                           "    values = load(rows)\n"
                           "    return sum(values) / len(values)\n"
                           "\n"
                           "if __name__ == '__main__':\n"
                           "    print(summarise([{'value': 1}, {'v': 2}]))\n"},
     python="pipeline.py",
     why="A real Python traceback. There is no `error:` marker anywhere in it: the "
         "locator is `File \"...\", line N` and the message is the last line. Seven "
         "clang substrings score zero here.")

case(id="python_traceback_noisy", family="other",
     files={"pipeline.py": "import sys\n"
                           "for i in range(2000):\n"
                           "    print('processing shard %04d ... ok' % i)\n"
                           "rows = [{'value': 1}, {'v': 2}]\n"
                           "print(sum(r['value'] for r in rows))\n"},
     python="pipeline.py",
     why="The same failure after 2000 lines of progress output -- what a real data "
         "script looks like. The traceback is the last 400 bytes of a 60KB log, so a "
         "compactor with any tail at all should get it. The question is whether it "
         "spends the rest of the budget on 'processing shard' lines.")

case(id="python_assert_midway", family="other",
     files={"pipeline.py": "import sys, traceback\n"
                           "for i in range(900):\n"
                           "    print('processing shard %04d ... ok' % i)\n"
                           "rows = [{'value': 1}, {'v': 2}]\n"
                           "try:\n"
                           "    print(sum(r['value'] for r in rows))\n"
                           "except KeyError:\n"
                           "    traceback.print_exc()\n"
                           "    print('shard 0900 unrecoverable, continuing with rest')\n"
                           "for i in range(900, 1800):\n"
                           "    print('processing shard %04d ... ok' % i)\n"
                           "print('done: 1799 shards ok, 1 failed')\n"},
     python="pipeline.py",
     why="The traceback is in the MIDDLE and the run CONTINUES -- 900 lines of progress "
         "before it and 900 after, ending in a cheerful summary. Head+tail truncation "
         "returns 100% noise. This is the case that separates a compactor that searches "
         "from one that keeps the ends, and the reason the family exists: a Python "
         "traceback carries no `error:` marker at all.")

def cargo_project():
    """A real cargo package. rustc's rendering is structurally unlike clang's: the
    message is on one line (`error[E0308]: mismatched types`) and the LOCATOR is on the
    NEXT (` --> src/shard.rs:4:21`). Anything that keeps only the line its matcher fired
    on keeps the message and loses the address."""
    mods = {f"m{i:02d}" for i in range(18)}
    files = {"Cargo.toml": "[package]\nname = \"shardpipe\"\nversion = \"0.1.0\"\n"
                           "edition = \"2021\"\n\n[dependencies]\n"}
    lib = "\n".join(f"pub mod {m};" for m in sorted(mods)) + "\npub mod shard;\n"
    files["src/lib.rs"] = lib
    for m in sorted(mods):
        n = int(m[1:]) + 1
        # Each module also earns a fistful of warnings, so the log looks like a real
        # crate rather than a two-error toy. rustc reports every warning in the crate
        # before it aborts, so they all land in front of the errors.
        files[f"src/{m}.rs"] = (
            f"pub fn value() -> i32 {{ {n} }}\n"
            f"pub fn doubled() -> i32 {{ value() * 2 }}\n"
            f"pub fn unusedCamel_{n}() -> i32 {{\n"
            f"    let unused_local = {n};\n"
            f"    let mut never_mutated = {n};\n"
            f"    never_mutated\n"
            f"}}\n"
            f"fn dead_{n}() -> i32 {{ {n} }}\n")
    files["src/shard.rs"] = ("pub fn label(ids: &Vec<i32>) -> String {\n"
                             "    let first: String = ids[0];\n"
                             "    first\n"
                             "}\n"
                             "pub fn count(ids: &Vec<i32>) -> usize {\n"
                             "    ids.len(missing_arg)\n"
                             "}\n")
    return files


case(id="cargo_two_errors", family="other", files=cargo_project(), cargo=True,
     why="Real `cargo build` on an 19-module crate with two errors. rustc splits a "
         "diagnostic across lines -- `error[E0308]: mismatched types` and then "
         "` --> src/shard.rs:2:25` -- so the message and the locator are never on the "
         "same line. A compactor that keeps matched lines keeps the message and drops "
         "the address.")


def pytest_project():
    files = {}
    for i in range(16):
        body = "\n".join(f"def test_ok_{i}_{j}():\n    assert {j} + 1 == {j + 1}\n"
                         for j in range(8))
        files[f"tests/test_mod_{i:02d}.py"] = body
    files["tests/test_pipeline.py"] = (
        "import pytest\n\n"
        "def summarise(rows):\n"
        "    return sum(r['value'] for r in rows) / len(rows)\n\n"
        "def test_summarise_average():\n"
        "    assert summarise([{'value': 2}, {'value': 4}]) == 4\n\n"
        "def test_summarise_missing_key():\n"
        "    assert summarise([{'value': 2}, {'v': 4}]) == 3\n\n"
        "def test_summarise_empty():\n"
        "    assert summarise([]) == 0\n")
    return files


case(id="pytest_failures", family="other", files=pytest_project(), pytest=True,
     why="42 passing tests and 3 failing ones. pytest renders each failure as a source "
         "excerpt, an `E   AssertionError` line, and a `path:line:` footer, then repeats "
         "the whole set in a short summary at the end. Nothing in it contains `error:`.")


case(id="ctest_failure", family="other", ctest=True,
     why="Real ctest output: twelve passing tests, one failing, and a summary block. "
         "The assertion text is in the middle and the failing test's name is in the "
         "tail summary.")



# ===========================================================================
# THE HELD-OUT SET
#
# Written AFTER src/tools/log_triage.hpp was finished, scored ONCE, never iterated
# against. The engine's score on corpus.jsonl is a memory test -- it was written with that
# corpus open, tuning until the scorer went quiet. This is the number that is blind on the
# engine's side, and it is the one to quote.
#
# Deliberately different from the corpus in shape, not just in content: a Makefile build
# rather than cmake, a header shared by many failing units, a C compile, an interleaved
# parallel build, a unittest runner rather than pytest, a rustc run without cargo.
#
# The limit, stated: these come from the SAME generator, so they inherit its blind spots.
# A truly independent set would come from a different author. The blast-radius holdout
# shared exactly one such blind spot with its corpus and it cost us a real bug.
# ===========================================================================

def makefile_project(defect_src, defect_name, n_files=90):
    """A hand-written Makefile rather than cmake -- different noise entirely: no progress
    percentages, no `Building CXX object`, just echoed compiler command lines."""
    files, objs = {}, []
    names = []
    for i in range(n_files):
        rel = f"src/unit_{i:02d}.c"
        names.append(rel)
        objs.append(f"src/unit_{i:02d}.o")
        # An unused parameter per unit: -Wextra makes every compile emit a
        # warning, which is what a real C build of this age looks like.
        files[rel] = ("#include <stdio.h>\n"
                      f"int unit_{i:02d}_value(int unused) {{ return {i}; }}\n"
                      f"int unit_{i:02d}_twice(void) {{ return unit_{i:02d}_value(0) * 2; }}\n")
    files[f"src/{defect_name}"] = defect_src
    objs.append(f"src/{defect_name[:-2]}.o")
    files["Makefile"] = (
        "CC = clang\n"
        "CFLAGS = -std=c17 -Wall -Wextra -Iinclude -Iinclude/core -c\n"
        "OBJS = " + " ".join(objs) + "\n\n"
        "all: $(OBJS)\n\n"
        "%.o: %.c\n"
        "\t$(CC) $(CFLAGS) $< -o $@\n")
    files["include/.keep"] = ""
    files["include/core/.keep"] = ""
    return files


holdout(id="ho_c_implicit_decl", family="build",
        files=makefile_project(
            "#include <stdio.h>\n"
            "int checksum(const char* s) {\n"
            "    return strlen(s) + missing_constant;\n"
            "}\n", "checksum.c"),
        make_plain=True,
        why="A C compile driven by a hand-written Makefile. Different toolchain, different "
            "noise: echoed command lines with no progress percentages at all.")

holdout(id="ho_header_breaks_many", family="build",
        files=(lambda: (lambda f: (f.update({
            "include/core/registry.hpp":
                "#pragma once\n#include <string>\nstruct Registry {\n"
                "    std::string name\n"
                "    int slots;\n};\n"}), f)[1])(
            cmake_project("#include \"core/registry.hpp\"\n"
                          "int use_registry() { Registry r; return r.slots; }\n",
                          4, n_files=36, defect_name="registry_user.cpp")[0]))(),
        broken="src/registry_user.cpp", cmake=True,
        why="ONE broken header included by many units, so the SAME diagnostic repeats "
            "across a dozen compiles with a different include stack each time. Directly "
            "targets the message-deduplication rule -- which must collapse the repeats "
            "without collapsing the distinct ones.")

holdout(id="ho_rustc_no_cargo", family="other",
        files={"shard.rs":
                   "".join(f"fn helper_{i}() -> i32 {{\n"
                           f"    let unusedCamel = {i};\n"
                           f"    let mut never_mutated = {i};\n"
                           f"    never_mutated\n}}\n" for i in range(120))
                   + "fn label(ids: &Vec<i32>) -> String {\n"
                     "    let first: String = ids[0];\n"
                     "    first\n}\n"
                     "fn main() {\n"
                     "    let v = vec![1, 2, 3];\n"
                     "    println!(\"{}\", label(v));\n}\n"},
        rustc="shard.rs",
        why="rustc directly, no cargo. Two errors, one of them a borrow/move diagnostic "
            "whose explanation spans several `|` gutter lines below the locator.")

holdout(id="ho_unittest_failures", family="other",
        files={"suite.py": "import unittest\n\n"
                           "def average(rows):\n"
                           "    return sum(r['v'] for r in rows) / len(rows)\n\n"
                           "class T(unittest.TestCase):\n"
                           + "".join(f"    def test_ok_{i}(self):\n"
                                     f"        self.assertEqual({i} + 1, {i + 1})\n"
                                     for i in range(260))
                           + "    def test_missing_key(self):\n"
                             "        self.assertEqual(average([{'v': 1}, {'x': 2}]), 1)\n"
                             "    def test_empty(self):\n"
                             "        self.assertEqual(average([]), 0)\n\n"
                           "if __name__ == '__main__':\n"
                           "    unittest.main(verbosity=2)\n"},
        python="suite.py",
        why="Python's unittest runner, not pytest: a different failure rendering again -- "
            "`FAIL: test_x (__main__.T.test_x)`, a dashed rule, then the traceback.")

holdout(id="ho_swift_two_defects", family="swift",
        files=swift_project("Broken.swift",
                            "struct Shard {\n    let id: Int\n}\n"
                            "func describe(_ s: Shard) -> String {\n"
                            "    return s.label\n}\n"
                            "func count(_ s: Shard) -> String {\n"
                            "    return s.id\n}\n"),
        swift_all=True,
        why="Two distinct Swift errors in one file, each printed twice by SwiftPM. Tests "
            "that de-duplication keeps BOTH distinct messages while collapsing the copies.")

holdout(id="ho_link_two_symbols", family="link",
        files=(lambda f: (f.update({
            "src/main.cpp": "int alpha();\nint beta();\n"
                            "int main() { return alpha() + beta(); }\n"}), f)[1])(
            cmake_project("int alpha();\nint beta();\n"
                          "int use_both() { return alpha() + beta(); }\n",
                          6, n_files=20, defect_name="user.cpp")[0]),
        cmake=True, link_exe=True,
        why="TWO undefined symbols at link time, each with its own referenced-from block. "
            "No `error:` marker anywhere and no file:line for either.")

holdout(id="ho_ok_large_build", family="build",
        files=cmake_project("int fine() { return 0; }\n", 0, n_files=60)[0],
        cmake=True,
        why="A 60-unit build that SUCCEEDS. Nothing to keep; measures budget compliance "
            "and whether the engine invents importance where there is none.")


# ---------------------------------------------------------------------------
# Key extraction
# ---------------------------------------------------------------------------

def parse_diagnostics(text):
    """Every rendered diagnostic header in `text`, in emission order."""
    out = []
    for raw in text.splitlines():
        line = strip_ansi(raw)
        m = DIAG_RE.match(line)
        if m:
            out.append({"path": m.group("path"), "line": int(m.group("line")),
                        "col": int(m.group("col")), "level": m.group("level"),
                        "message": m.group("msg"), "rendered": line,
                        # The locator must be a real SUBSTRING of the rendered line: the
                        # scorer searches a compacted log for it, so a locator we
                        # synthesised in a shape the tool never prints is a case nobody
                        # can answer. Taken from the match, never rebuilt from parts.
                        "locator": f"{m.group('path')}:{m.group('line')}:{m.group('col')}"})
            continue
        m = BARE_FATAL_RE.match(line)
        if m:
            out.append({"path": "", "line": 0, "col": 0, "level": m.group("level"),
                        "message": m.group("msg"), "rendered": line, "locator": ""})
    return out


def linker_key(full_log):
    """ld writes its own key. Its diagnostic has no `error:` and no file:line."""
    lines = [strip_ansi(l) for l in full_log.splitlines()]
    diags = []
    for l in lines:
        s = l.strip()
        if (s.startswith("Undefined symbols") or s.startswith("ld: ")
                or s.startswith("  \"") or "symbol(s) not found" in s):
            diags.append({"path": "", "line": 0, "col": 0, "level": "error",
                          "message": s, "rendered": l, "locator": ""})
    return diags


def python_key(full_log):
    """Python writes its own key: the traceback frames and the exception line."""
    lines = [strip_ansi(l) for l in full_log.splitlines()]
    primary, local = None, []
    frame_re = re.compile(r'^\s*File "(?P<path>[^"]+)", line (?P<line>\d+)')
    exc_re = re.compile(r"^(?P<exc>[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exit|Exception|Interrupt))"
                        r"(?::\s*(?P<msg>.*))?$")
    for l in lines:
        m = frame_re.match(l)
        if m and not m.group("path").startswith("<"):
            # Python does not print path:line:col. Its locator is the frame header it
            # actually emits, taken from the line verbatim.
            local.append({"path": m.group("path"), "line": int(m.group("line")),
                          "col": 0, "level": "note", "message": l.strip(),
                          "rendered": l, "locator": l[m.start():m.end()].strip()})
        m = exc_re.match(l)
        if m:
            primary = {"path": "", "line": 0, "col": 0, "level": "error",
                       "message": l, "rendered": l, "locator": ""}
    return primary, local


def rustc_key(key_text, full_text):
    """rustc writes its own key: --error-format=short renders each diagnostic as one
    `path:line:col: error[CODE]: message` line, so the SET of errors and their locations
    come from the compiler.

    The message text, though, is taken from the FULL log rather than from the short form,
    because the short form rewrites it -- it folds the label that the full rendering puts
    under the caret onto the end of the message ("cannot find value `x` in this scope: not
    found in this scope"). A key message in a shape the tool never printed is a case nobody
    can answer, so the short form is used to FIND the diagnostic and the log is used to
    QUOTE it.

    That split is also the point of the case: rustc prints `error[E0425]: message` on one
    line and ` --> path:line:col` on the NEXT, so a compactor that keeps the line its
    matcher fired on keeps the message and loses the address.
    """
    short_re = re.compile(r"^(?P<path>[^\s:][^:]*):(?P<line>\d+):(?P<col>\d+): "
                          r"(?P<level>error(?:\[[A-Z0-9]+\])?|warning): ")
    head_re = re.compile(r"^(?P<level>error(?:\[[A-Z0-9]+\])?|warning): (?P<msg>.+)$")
    full_lines = [strip_ansi(l) for l in full_text.splitlines()]
    diags = []
    for l in key_text.splitlines():
        m = short_re.match(strip_ansi(l))
        if not m or not m.group("level").startswith("error"):
            continue
        locator = f"{m.group('path')}:{m.group('line')}:{m.group('col')}"
        arrow = next((i for i, fl in enumerate(full_lines)
                      if fl.strip() == f"--> {locator}"), None)
        if arrow is None:
            continue
        head = next((full_lines[i] for i in range(arrow - 1, max(arrow - 4, -1), -1)
                     if head_re.match(full_lines[i])), None)
        if head is None:
            continue
        hm = head_re.match(head)
        diags.append({"path": m.group("path"), "line": int(m.group("line")),
                      "col": int(m.group("col")), "level": "error",
                      "message": hm.group("msg"), "rendered": head, "locator": locator,
                      # Anchor the caret block on the `-->` line: it is what the block
                      # hangs beneath, and the message line has the `-->` between it and
                      # the source.
                      "context_anchor": full_lines[arrow]})
    return diags


def pytest_key(key_text, full_text, proj):
    """pytest writes its own key: --tb=line renders each failure as one
    `path:line: message` footer.

    It prints that footer with an ABSOLUTE path while its normal output uses a path
    relative to the rootdir, so the locator is taken in whichever spelling the full log
    actually contains. A locator in a shape the tool never printed is a case nobody can
    answer, which is the defect the entrants found in our last corpus.
    """
    line_re = re.compile(r"^(?P<path>/[^\s:]+\.py):(?P<line>\d+): (?P<msg>.+)$")
    diags = []
    for l in key_text.splitlines():
        m = line_re.match(strip_ansi(l))
        if not m:
            continue
        # realpath on BOTH sides: on macOS /tmp is a symlink to /private/tmp, and
        # relpath between a resolved and an unresolved spelling of the same directory
        # returns a ../.. path that matches nothing.
        rel = os.path.relpath(os.path.realpath(m.group("path")), os.path.realpath(proj))
        for path in (m.group("path"), rel):
            locator = f"{path}:{m.group('line')}"
            if locator in full_text:
                diags.append({"path": path, "line": int(m.group("line")), "col": 0,
                              "level": "error", "message": m.group("msg"),
                              "rendered": f"{locator}: {m.group('msg')}",
                              "locator": locator})
                break
    return diags


def ctest_key(full_log):
    """ctest names its own failures, in the summary block it prints."""
    lines = [strip_ansi(l) for l in full_log.splitlines()]
    primary, local = None, []
    fail_re = re.compile(r"^\s*\d+\s*-\s*(?P<name>\S+)\s*\(Failed\)")
    assert_re = re.compile(r"^.*Assertion failed:.*$")
    for l in lines:
        s = strip_ansi(l)
        if fail_re.match(s) or assert_re.match(s):
            d = {"path": "", "line": 0, "col": 0, "level": "error",
                 "message": s.strip(), "rendered": s, "locator": ""}
            local.append(d)
            if primary is None:
                primary = d
    return primary, local


def relocate_in_log(diag, full_text):
    """Find the FULL LOG's rendering of a diagnostic the key run reported.

    Matched on (basename, line, col, level) -- the parts of a diagnostic that do not
    depend on which directory the compiler was invoked from. Returns None when the
    build genuinely did not emit it (a key-run-only diagnostic is a corpus defect and
    the caller reports it).
    """
    want = (os.path.basename(diag["path"]), diag["line"], diag["col"], diag["level"])
    for d in parse_diagnostics(full_text):
        if (os.path.basename(d["path"]), d["line"], d["col"], d["level"]) == want:
            return d
    return None


def is_local_path(path, proj, cwd):
    """True when a diagnostic's location is a source file the agent can edit.

    A diagnostic in the SDK or in libc++ tells the agent what is wrong; only one inside
    the project tells it where to type. Resolved against the directory the compiler
    actually ran in, because cmake compiles from build/ with absolute paths and a bare
    clang compiles from the project root with relative ones. Generated build artefacts
    are excluded -- the agent cannot usefully edit those either.
    """
    if not path:
        return False
    full = path if os.path.isabs(path) else os.path.join(cwd, path)
    real = os.path.realpath(full)
    root = os.path.realpath(proj)
    if not real.startswith(root + os.sep):
        return False
    if os.path.realpath(os.path.join(proj, "build")) + os.sep in real + os.sep:
        return False
    return os.path.exists(real)


# A line belonging to a caret block, in the compiler's own rendering. clang and swiftc
# both print `  123 | source text` and `      |      ^~~~`; clang without line numbers
# prints the bare source line followed by a line of only carets and tildes. Python's
# traceback prints the offending source indented under its `File "..."` frame.
CARET_RE = re.compile(r"^\s*(?:\d+\s*)?\|")
BARE_CARET_RE = re.compile(r"^\s*[~^`\- ]+$")
PY_SOURCE_RE = re.compile(r"^\s{4,}\S")


def context_for(full_log_lines, rendered):
    """The source + caret lines the compiler printed beneath a diagnostic header.

    Bounded by the compiler's OWN block rendering, not by a line count: a caret block is
    the run of `NNN | source` / `    | ^~~~` lines that follows the header. Stopping at
    "the next blank line" instead swept the build's next commands into the key, which
    made a compactor responsible for retaining text that has nothing to do with the
    diagnostic -- and every entrant would have failed it for the same wrong reason.
    """
    try:
        idx = next(i for i, l in enumerate(full_log_lines)
                   if strip_ansi(l).rstrip() == rendered.rstrip())
    except StopIteration:
        return []
    ctx = []
    for l in full_log_lines[idx + 1: idx + 12]:
        s = strip_ansi(l)
        if not s.strip():
            break
        if CARET_RE.match(s) or BARE_CARET_RE.match(s) or PY_SOURCE_RE.match(s):
            ctx.append(s)
            continue
        break
    return ctx


# ---------------------------------------------------------------------------
# Case execution
# ---------------------------------------------------------------------------

def write_files(proj, files):
    for rel, content in files.items():
        path = os.path.join(proj, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(content)


def build_ctest_case(proj):
    write_files(proj, {
        "CMakeLists.txt":
            "cmake_minimum_required(VERSION 3.20)\n"
            "project(triage_ctest CXX)\nenable_testing()\n"
            "add_executable(t_pass pass.cpp)\nadd_executable(t_fail fail.cpp)\n"
            "foreach(i RANGE 1 12)\n  add_test(NAME pass_${i} COMMAND t_pass)\n"
            "endforeach()\n"
            "add_test(NAME budget_enforced COMMAND t_fail)\n",
        "pass.cpp": "int main(){ return 0; }\n",
        "fail.cpp": "#include <cstdio>\n#include <cassert>\n"
                    "int main(){ std::printf(\"checking budget\\n\");"
                    " assert(1 == 2 && \"compacted output exceeded budget\");"
                    " return 0; }\n"})
    build = os.path.join(proj, "build")
    os.makedirs(build, exist_ok=True)
    run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Debug"], build)
    run(["cmake", "--build", "."], build)
    _, out = run(["ctest", "--output-on-failure"], build)
    return out


def build_case(spec, work_root, out_dir, log_subdir="logs"):
    cid = spec["id"]
    proj = os.path.join(work_root, cid)
    os.makedirs(proj, exist_ok=True)
    write_files(proj, spec.get("files", {}))

    tool, full_text, key_text = "clang", "", ""
    # The directory the compiler ran from. A relative path in a diagnostic resolves
    # against this, which is how we decide whether a diagnostic is one the agent can act on.
    cwd = proj

    if spec.get("cmake"):
        tool = "cmake"
        build = os.path.join(proj, "build")
        cwd = build
        os.makedirs(build, exist_ok=True)
        if spec.get("link_exe"):
            with open(os.path.join(proj, "CMakeLists.txt"), "a") as fh:
                fh.write("add_executable(app src/main.cpp)\n"
                         "target_link_libraries(app triage_bench)\n")
        _, conf = run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], build)
        _, out = run(["make", "-k", "VERBOSE=1"], build)
        full_text = conf + out
        broken = spec.get("broken")
        srcs = ([broken] if broken else
                sorted(f for f in spec["files"] if f.endswith(".cpp")))
        _, key_text = run([CXX, CXX_STD, "-fsyntax-only", *CLANG_KEY_FLAGS,
                           "-ftemplate-backtrace-limit=1", "-Iinclude", *srcs], proj)
    elif spec.get("make_plain"):
        tool = "make"
        _, full_text = run(["make", "-k"], proj)
        srcs = sorted(f for f in spec["files"] if f.endswith(".c"))
        _, key_text = run(["clang", "-std=c17", "-fsyntax-only", *CLANG_KEY_FLAGS,
                           "-Iinclude", *srcs], proj)
    elif spec.get("rustc"):
        tool = "cargo"  # same rendering, same key extractor
        _, full_text = run(["rustc", "--edition", "2021", spec["rustc"], "-o",
                            os.path.join(proj, "out")], proj)
        _, key_text = run(["rustc", "--edition", "2021", "--error-format=short",
                           spec["rustc"], "-o", os.path.join(proj, "out2")], proj)
    elif spec.get("compile"):
        _, full_text = run([CXX, CXX_STD, "-c", *spec["compile"], "-o", "/dev/null"], proj)
        _, key_text = run([CXX, CXX_STD, "-fsyntax-only", *CLANG_KEY_FLAGS,
                           "-ftemplate-backtrace-limit=1", *spec["compile"]], proj)
    elif spec.get("swift_all"):
        tool = "swift"
        # SwiftPM scratch dirs land in $TMPDIR and it prints their absolute paths.
        # Point TMPDIR inside the project so path normalisation catches them; without
        # this the committed log carries this machine's /var/folders path.
        swift_tmp = os.path.join(proj, "tmp")
        os.makedirs(swift_tmp, exist_ok=True)
        _, full_text = run(["swift", "build", "-v"], proj, env={"TMPDIR": swift_tmp})
        srcs = sorted(os.path.join(proj, f) for f in spec["files"]
                      if f.endswith(".swift") and f != "Package.swift")
        _, key_text = run([SWIFTC, "-typecheck", "-diagnostic-style=llvm", *srcs], proj)
        cwd = proj
    elif spec.get("python"):
        tool = "python"
        _, full_text = run([sys.executable, spec["python"]], proj)
        key_text = full_text
    elif spec.get("cargo"):
        tool = "cargo"
        env = {"CARGO_TARGET_DIR": os.path.join(proj, "target"), "CARGO_TERM_COLOR": "never"}
        _, full_text = run(["cargo", "build", "--offline", "-v"], proj, env=env)
        _, key_text = run(["cargo", "build", "--offline", "--message-format=short"],
                          proj, env=env)
    elif spec.get("pytest"):
        tool = "pytest"
        # -v prints one line per test, which is what a real suite's output looks
        # like and what buries the three failures among the 128 passes.
        _, full_text = run([sys.executable, "-m", "pytest", "tests", "-v", "-p",
                            "no:cacheprovider"], proj)
        _, key_text = run([sys.executable, "-m", "pytest", "tests", "-q", "--tb=line",
                           "-p", "no:cacheprovider"], proj)
    elif spec.get("ctest"):
        tool = "ctest"
        full_text = build_ctest_case(proj)
        key_text = full_text
    else:
        raise SystemExit(f"case {cid} declares no build")

    full_lines = full_text.splitlines()

    if tool == "python":
        primary, local = python_key(full_text)
        n_err = 1 if primary else 0
    elif tool == "ctest":
        primary, local = ctest_key(full_text)
        n_err = len(local)
    elif tool in ("cargo", "pytest"):
        diags = (rustc_key(key_text, full_text) if tool == "cargo"
                 else pytest_key(key_text, full_text, proj))
        n_err = len(diags)
        primary = diags[0] if diags else None
        local = diags
    else:
        key_diags = parse_diagnostics(key_text)
        errors = [d for d in key_diags if d["level"] in ("error", "fatal error")]
        n_err = len(errors)
        primary = errors[0] if errors else None
        # The key run and the build render the same diagnostic differently -- cmake
        # compiles from build/ with an absolute path, our quiet run from the project
        # root with a relative one. The scorer searches the LOG, so the key must hold
        # the log's rendering. Re-locate the primary by (basename, line, col, level).
        if primary:
            primary = relocate_in_log(primary, full_text) or primary
        # `local` comes from the FULL log, not the key run: the note backtrace that
        # reaches the project only exists when template backtraces are rendered.
        local = [d for d in parse_diagnostics(full_text)
                 if d["level"] in KEY_LEVELS and is_local_path(d["path"], proj, cwd)]
        if not primary and not local:
            ld = linker_key(full_text)
            if ld:
                primary, local, n_err = ld[0], ld, len(ld)

    seen, local_u = set(), []
    for d in local:
        k = (d["path"], d["line"], d["col"], d["level"], d["message"])
        if k not in seen:
            seen.add(k)
            local_u.append(d)
    local = local_u

    def freeze(d):
        return {"level": d["level"], "path": d["path"], "line": d["line"],
                "locator": d["locator"], "message": d["message"],
                "rendered": d["rendered"],
                "context": context_for(full_lines,
                                       d.get("context_anchor", d["rendered"]))}

    record = {"id": cid, "family": spec["family"], "tool": tool,
              "log": f"logs/{cid}.log", "bytes": len(full_text),
              "lines": len(full_lines),
              "primary": freeze(primary) if primary else None,
              "local": [freeze(d) for d in local],
              "error_count": n_err, "why": spec["why"]}

    os.makedirs(os.path.join(out_dir, log_subdir), exist_ok=True)
    with open(os.path.join(out_dir, log_subdir, f"{cid}.log"), "w") as fh:
        fh.write(full_text)
    return record, proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--only", default="")
    # The held-out set. Written AFTER the engine was finished and never iterated against:
    # the moment you tune against a corpus it stops being a benchmark and starts being a
    # memory test. Its cases use the same generator -- so they share the generator's blind
    # spots, which is a real limit and is stated in the README -- but different defects,
    # different tools and different shapes.
    ap.add_argument("--holdout", action="store_true")
    args = ap.parse_args()
    out_dir = args.out
    specs = HOLDOUT if args.holdout else CASES
    log_subdir = "holdout_logs" if args.holdout else "logs"
    manifest = "holdout.jsonl" if args.holdout else "corpus.jsonl"
    os.makedirs(os.path.join(out_dir, log_subdir), exist_ok=True)
    wanted = set(x for x in args.only.split(",") if x)

    work_root = tempfile.mkdtemp(prefix="log_triage_corpus_")
    records = []
    try:
        for spec in specs:
            if wanted and spec["id"] not in wanted:
                continue
            rec, proj = build_case(spec, work_root, out_dir, log_subdir)
            rec["log"] = f"{log_subdir}/{rec['id']}.log"
            log_path = os.path.join(out_dir, log_subdir, f"{rec['id']}.log")
            with open(log_path) as fh:
                text = fh.read()

            # Both spellings: on macOS /tmp is a symlink to /private/tmp, so some tools
            # print the resolved path and some print the one they were handed. Longest
            # first, or the shorter rewrite leaves a `/private` prefix behind.
            def normalise(s):
                for p in sorted({proj, os.path.realpath(proj)}, key=len, reverse=True):
                    s = s.replace(p, STABLE_ROOT)
                # SwiftPM names its scratch dirs randomly. Canonicalise the suffix so
                # a regenerated corpus diffs only where the toolchain actually changed.
                return RANDOM_SCRATCH_RE.sub(r"\1XXXXXX", s)

            text = normalise(text)
            with open(log_path, "w") as fh:
                fh.write(text)
            rec["bytes"] = len(text)
            rec = json.loads(normalise(json.dumps(rec)))
            records.append(rec)
            print(f"  {rec['id']:<28} {rec['bytes']:>7}B {rec['lines']:>5}L  "
                  f"{rec['error_count']:>3} err  {len(rec['local']):>3} local"
                  f"{'   NO PRIMARY' if rec['primary'] is None else ''}", flush=True)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    if not wanted:
        with open(os.path.join(out_dir, manifest), "w") as fh:
            for rec in records:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        print(f"\n{len(records)} cases -> {os.path.join(out_dir, manifest)}")


if __name__ == "__main__":
    main()
