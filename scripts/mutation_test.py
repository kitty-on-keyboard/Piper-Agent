#!/usr/bin/env python3
"""Mutation testing for the gate (S11.4): plant a defect, require the suite to notice.

  ./scripts/mutation_test.py                 run every mutation
  ./scripts/mutation_test.py --only NAME     run one
  ./scripts/mutation_test.py --list          show the catalogue and exit

WHY THIS FILE EXISTS AT ALL. docs/PHASES.md reported "3/8 killed, 5 survivors" and named
the survivors, but the harness that produced that number was never committed -- so the
figure was a report of a run nobody could reproduce, which is the precise thing
`ctest -E realmodel` is condemned for on the previous page of the same document. A
measurement with no instrument is not a measurement.

THE NULL MUTANT IS NOT OPTIONAL. The first version of this harness reported a perfect 6/8
and was measuring itself: three separate defects made every copy fail for reasons
unrelated to any mutation -- `ignore_patterns("build*")` also matched
`simdjson/builder.h`; excluding `node_modules` dropped test_extension_typecheck and so
broke the pinned gate manifest; and `copytree` dereferenced npm's `.bin` symlinks. Every
copy failing looks exactly like every mutation being caught. So an UNMUTATED copy must
build and pass the gate before any kill in this run is believed, and all three of those
defects are guarded against below by construction.

WHY MUTATIONS ARE STRINGS AND NOT LINE NUMBERS. PHASES.md names its survivors as
`grammar.cpp:105`, `agent.cpp:168`. Those line numbers no longer point at what they
described -- the files moved underneath them. An exact source string either still matches
or fails loudly, which is the behaviour worth having.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# `.git` is skipped for size. Nothing in the gate reads it; if that ever stops being true
# the null mutant fails and says so, which is the point of having one.
SKIP_AT_ROOT = {".git"}


def is_build_dir(path):
    """A top-level CMake build directory, identified by what it CONTAINS.

    Not by name. The original harness used `ignore_patterns("build*")`, which also matched
    `third_party/simdjson/include/simdjson/builder.h` and broke every copy. Replacing it
    with an exact list of {build, build-asan, build-bakeoff} rotted immediately in the
    other direction: this tree also holds `build-mcp` and `build-mcp-asan` from an earlier
    session, and the first run of this file copied 194 MB of stale object files because
    the list had never heard of them. A name is the wrong thing to test either way.
    """
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "CMakeCache.txt"))


# Each mutation is a defect a competent reviewer would call a bug. `expect` records what
# the last run measured, so a CHANGE in outcome is visible -- it is documentation, not a
# gate. A survivor is a finding about the suite (S11.4), not a thing to quietly delete.
MUTATIONS = [
    {
        "name": "grammar_close_ends_turn",
        "file": "src/model/grammar.cpp",
        "old": "        phase_ = TurnPhase::Text;\n        return Advance::Ok;\n    }\n"
               "    if (is_structural(id)) {\n        return Advance::Rejected;\n    }\n"
               "    if (guard_->feed(bytes) != parsephony::Error::Ok) {",
        "new": "        phase_ = TurnPhase::Done;\n        return Advance::Accepted;\n    }\n"
               "    if (is_structural(id)) {\n        return Advance::Rejected;\n    }\n"
               "    if (guard_->feed(bytes) != parsephony::Error::Ok) {",
        "note": "the real 4300a3c regression: a closed tool call ends the turn, so a "
                "second call in one turn becomes unrepresentable",
        "expect": "killed",
    },
    {
        "name": "grammar_mask_allows_structural_mid_call",
        "file": "src/model/grammar.cpp",
        "old": "        if (is_structural(id) && id != tok_.specials().tool_call_close) {\n"
               "            return false;\n        }",
        "new": "        if (false) {\n            return false;\n        }",
        "note": "permitted() stops rejecting structural ids inside a tool call, so the "
                "mask lies to the sampler where the grammar is strictest",
        "expect": "killed",
    },
    {
        "name": "grammar_think_accepts_structure",
        "file": "src/model/grammar.cpp",
        "old": "    if (is_structural(id)) {\n        // No nested <think>, no <|im_end|> "
               "mid-thought, no tool call inside reasoning.\n        return "
               "Advance::Rejected;\n    }",
        "new": "    if (false) {\n        return Advance::Rejected;\n    }",
        "note": "<|im_end|> and <tool_call> become legal mid-thought",
        "expect": "killed",
    },
    {
        "name": "grammar_call_cap_removed",
        "file": "src/model/grammar.cpp",
        "old": "        if (at_call_cap() || guard_ == nullptr) {",
        "new": "        if (guard_ == nullptr) {",
        "note": "a turn may batch unboundedly many tool calls",
        "expect": "killed",
    },
    {
        "name": "operator_check_everything_passes",
        "file": "src/loop/agent.cpp",
        "old": "    check.passed = r.ok();",
        "new": "    check.passed = true;",
        "note": "every operator check passes -- a failing build reports completed=true, "
                "which is the one lie the hook exists to make impossible (killed by "
                "the_operator_check_runs_after_a_write_turn_...)",
        "expect": "killed",
    },
    {
        "name": "repeat_cache_survives_a_write",
        "file": "src/loop/turn.cpp",
        "old": "        if (call.last_ok && call.writes_at == writes_now) {",
        "new": "        if (call.last_ok && call.writes_at <= writes_now) {",
        "note": "the cache serves stale bytes after a write -- manufactured stale "
                "evidence, the exact class of harness-corrupted feedback the rewrite "
                "removed (killed by a_repeat_after_a_workspace_write_executes_for_real)",
        "expect": "killed",
    },
    {
        "name": "agent_hides_pre_existing_syntax_failure",
        "file": "src/loop/agent.cpp",
        "old": '    result.summary += was_clean ? ": FAILED\\n" : ": still failing (it was '
               'already failing "\n                                                 "before '
               'this edit)\\n";',
        "new": '    result.summary += ": FAILED\\n";',
        "note": "the model is told its edit broke a file that arrived broken",
        "expect": "survived",
    },
    {
        "name": "sidecar_shutdown_does_not_exit",
        "file": "src/surface/sidecar.cpp",
        "old": "    log.close();\n    std::fflush(nullptr);\n    ::_exit(0);",
        "new": "    log.close();\n    std::fflush(nullptr);\n    ::_exit(1);",
        "note": "PHASES.md's sidecar.cpp survivor: the dispatch loop has no test, so the "
                "process exit status is unasserted",
        "expect": "survived",
    },
]


def copy_tree(dst, verbose):
    """Copy the worktree, preserving symlinks and keeping node_modules.

    Both of those are load-bearing. `symlinks=True` because copytree otherwise
    dereferences npm's `.bin` symlinks and fails; node_modules is kept because
    tests/gate/CMakeLists.txt only registers test_extension_typecheck when it exists, and
    dropping it makes the selected set disagree with the pinned manifest -- which fails
    test_gate_manifest by design, on every copy, for a reason that has nothing to do with
    any mutation.
    """
    def ignore(directory, names):
        if os.path.abspath(directory) != ROOT:
            return []
        return [n for n in names
                if n in SKIP_AT_ROOT or is_build_dir(os.path.join(directory, n))]

    started = time.monotonic()
    shutil.copytree(ROOT, dst, symlinks=True, ignore=ignore)
    if verbose:
        print(f"  copied the worktree in {time.monotonic() - started:.0f}s -> {dst}")


def run(cmd, cwd, log):
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.call(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)


def build_and_gate(work, log, jobs):
    """Returns (built, gate_passed)."""
    if run(["cmake", "--build", "build", f"-j{jobs}"], work, log) != 0:
        return False, False
    return True, run(["ctest", "--test-dir", "build", "-L", "gate"], work, log) == 0


def apply_mutation(work, mut):
    path = os.path.join(work, mut["file"])
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    if src.count(mut["old"]) != 1:
        return False, (f"the anchor matched {src.count(mut['old'])} times in "
                       f"{mut['file']} -- the file moved underneath this mutation")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(src.replace(mut["old"], mut["new"], 1))
    return True, ""


def restore(work, mut):
    shutil.copyfile(os.path.join(ROOT, mut["file"]), os.path.join(work, mut["file"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="run a single mutation by name")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--workdir", help="use this directory instead of a fresh temp copy; "
                                      "copied into if it does not already hold a tree, "
                                      "reused as-is if it does")
    ap.add_argument("--prepare", action="store_true",
                    help="copy and configure --workdir, then stop. The copy is the slow "
                         "part and does not need the machine quiet; the builds do.")
    ap.add_argument("--keep", action="store_true", help="do not delete the copy")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.list:
        for m in MUTATIONS:
            print(f"  {m['name']:<42} {m['expect']:<9} {m['file']}")
            print(f"      {m['note']}")
        return 0

    muts = MUTATIONS
    if args.only:
        muts = [m for m in MUTATIONS if m["name"] == args.only]
        if not muts:
            print(f"no mutation named {args.only!r}", file=sys.stderr)
            return 2

    work = os.path.abspath(args.workdir) if args.workdir \
        else tempfile.mkdtemp(prefix="lmp-mutation-")
    # Beside the workdir, not inside it: the directory may not exist yet, and a temp
    # workdir is deleted at the end -- taking the log with it exactly when it is wanted.
    log = work.rstrip(os.sep) + ".log"
    os.makedirs(os.path.dirname(log) or ".", exist_ok=True)
    open(log, "w", encoding="utf-8").close()
    print(f"  workdir {work}\n  log     {log}")

    try:
        if not os.path.exists(os.path.join(work, "CMakeLists.txt")):
            if args.workdir and os.path.isdir(work):
                os.rmdir(work)  # copytree needs to create it; refuses a non-empty one
            copy_tree(work, True)
        elif args.verbose:
            print("  reusing the tree already in the workdir")

        # Configured WITHOUT MLX on purpose: the gate is model-free, this is the half CI
        # compiles, and it keeps the harness clear of the one-MLX-process-at-a-time rule
        # so it can run while something else holds the model.
        print("\n  configuring (no MLX -- the gate is model-free)")
        if run(["cmake", "--preset", "dev", "-B", "build",
                "-DLMP_MLX_PYTHON=/usr/bin/false"], work, log) != 0:
            print(f"  FAIL: configure failed in the copy; see {log}")
            return 2

        if args.prepare:
            print(f"\n  prepared. Run the mutations with:\n"
                  f"    ./scripts/mutation_test.py --workdir {work}")
            return 0

        # --- the null mutant ------------------------------------------------------
        print("  null mutant: an unmutated copy must build and pass the gate")
        started = time.monotonic()
        built, passed = build_and_gate(work, log, args.jobs)
        if not (built and passed):
            what = "build" if not built else "gate"
            print(f"\n  FAIL: the null mutant's {what} failed. NOTHING in this run is a "
                  f"kill -- every mutation would 'die' of this instead. See {log}")
            return 2
        print(f"    null mutant clean in {time.monotonic() - started:.0f}s\n")

        # --- the mutations --------------------------------------------------------
        results = []
        for i, mut in enumerate(muts, 1):
            print(f"  [{i}/{len(muts)}] {mut['name']}", flush=True)
            ok, why = apply_mutation(work, mut)
            if not ok:
                print(f"      ANCHOR LOST: {why}")
                results.append((mut, "anchor-lost"))
                continue
            started = time.monotonic()
            built, passed = build_and_gate(work, log, args.jobs)
            restore(work, mut)
            if not built:
                outcome = "build-error"
            elif passed:
                outcome = "survived"
            else:
                outcome = "killed"
            results.append((mut, outcome))
            note = "" if outcome == mut["expect"] else f"  (expected {mut['expect']})"
            print(f"      {outcome}  {time.monotonic() - started:.0f}s{note}")

        # --- the report -----------------------------------------------------------
        killed = sum(1 for _, o in results if o == "killed")
        survived = [m for m, o in results if o == "survived"]
        broken = [(m, o) for m, o in results if o in ("build-error", "anchor-lost")]

        print(f"\n  {killed}/{len(results)} killed")
        if survived:
            print("\n  SURVIVORS -- each is a finding about the suite, not a bug to hide:")
            for m in survived:
                print(f"    {m['name']}  ({m['file']})")
                print(f"        {m['note']}")
        for m, o in broken:
            print(f"\n  {o}: {m['name']} -- neither killed nor survived; the mutation "
                  f"never ran as intended")

        changed = [(m, o) for m, o in results if o != m["expect"]]
        if changed:
            print("\n  CHANGED SINCE THE LAST RECORDED RUN:")
            for m, o in changed:
                print(f"    {m['name']}: {m['expect']} -> {o}")
        return 0
    finally:
        if not args.keep and not args.workdir and os.path.isdir(work):
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
