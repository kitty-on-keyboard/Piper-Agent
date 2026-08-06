#!/usr/bin/env python3
"""The CI gates (spec S15.3 - S15.7).

One entry point, five gates. Run with --self-test to prove each can go red: a
gate that has never been observed failing is a gate nobody has falsified, and
S2.1.2 does not make an exception for tooling.

  ./scripts/run_ratchets.py --root .
  ./scripts/run_ratchets.py --root . --self-test

WHAT BELONGS HERE, AND WHAT DOES NOT
  Every gate below checks a PROPERTY: the dependency graph points one way, a
  declared symbol is used, the two sides of the protocol are generated from one
  schema, a tool description names tools that exist, a corrective arm changes
  state rather than composing a sentence. Each is a thing that is either true of
  the code or false, and false is a defect a reviewer would also call a defect.

  There used to be a sixth gate here, and it counted lines: 800 per file, 80 per
  function, nesting depth 3. It is gone, and it is not coming back. A line count
  is not a property -- it is a proxy for one, and the proxy was optimized against
  instead of the thing. `ContextJournal::open()` dropped an out-parameter and
  swallowed its own failure reason to buy back one line; `mcp_settings` and two
  `tests/model` files were split on line count rather than on responsibility;
  `sidecar.cpp` sat at exactly 800/800, which is not where a well-factored file
  naturally lands. Every one of those is worse code that passed a green check.

  AND TWO OF THE THREE LIMITS NEVER RAN. Measured on 2026-08-02, after the fact.
  The function scanner only started a body at brace depth 0, and `namespace
  lmp::x {` puts a file at depth 1 from its first line and never returns -- so it
  saw zero function bodies in 107 of 149 source files, and the only four it would
  ever have flagged were all inside `blast_radius.hpp`, which is scan-exempt. The
  function-length and nesting limits fired on nothing, ever. Only the 800-line
  file cap was real, and it is the one that did the damage above.

  Its self-test passed the whole time, because the planted probe was a bare
  `inline void probe_fn() {` at depth 0 -- a shape no file in src/ has. That is
  the same defect as v1's `ctest -E realmodel`: a check that matched nothing while
  printing a number that looked fine, this time inside the falsifier itself. A
  probe has to have the shape of the code it stands for, or it proves only that
  the gate can fail on input the repo never produces.

  So: no gate here may enforce a number that stands in for a judgement. If a file
  is badly organised, that is a review comment about its organisation, and the
  argument has to be made about the code rather than about its length. And nobody
  gets to argue the function cap was harmless because nothing ever tripped it --
  nothing ever tripped it because it was not looking.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

CPP_EXT = (".hpp", ".cpp", ".h", ".cc")
PY_EXT = (".py",)
SCANNED_EXT = CPP_EXT + PY_EXT + (".cmake", ".txt", ".ts")


class Finding:
    def __init__(self, gate, path, line, message):
        self.gate, self.path, self.line, self.message = gate, path, line, message

    def __str__(self):
        loc = f"{self.path}:{self.line}" if self.line else self.path
        return f"  {loc}: {self.message}"


def load_config(root):
    with open(os.path.join(root, "scripts", "ratchets.json"), encoding="utf-8") as fh:
        return json.load(fh)


def walk_sources(root, cfg):
    """Yields (relpath, abspath) for every file a gate may inspect."""
    for rel_root in cfg["source_roots"]:
        base = os.path.join(root, rel_root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in (".git", "build", "__pycache__")]
            for name in sorted(filenames):
                if not name.endswith(SCANNED_EXT):
                    continue
                abspath = os.path.join(dirpath, name)
                yield os.path.relpath(abspath, root), abspath


def is_exempt(relpath, exemptions):
    return any(relpath.startswith(e["path"]) for e in exemptions)


# --- gate: layers ----------------------------------------------------------

INCLUDE_RE = re.compile(r'^\s*#\s*include\s+[<"]([^">]+)[">]')


def layer_of(path, layers):
    for prefix, level in layers.items():
        if path.startswith(prefix):
            return prefix, level
    return None, None


def gate_layers(root, cfg):
    """A layer may not include a header from a layer BELOW it (S3).

    L0 platform is the top of the list and the bottom of the dependency graph:
    src/loop may include src/platform, src/platform may never include src/loop.

    A file under src/ matching NO layer entry is a FAILURE, not a skip. `src/mcp` spent
    its entire life outside this gate because an unlisted directory read as "nothing to
    check": ~2500 lines that src/pcc and src/tools both depend on, with its include
    direction never once enforced. It was found by diffing the map against `ls src/`,
    which is not a thing anyone does on a schedule -- so absence is now the loud case.
    Only src/ is required to be layered; tests/ and scripts/ are not a dependency graph.
    """
    layers, findings, subjects = cfg["layers"], [], 0
    for rel, abspath in walk_sources(root, cfg):
        if not rel.endswith(CPP_EXT):
            continue
        src_prefix, src_level = layer_of(rel, layers)
        if src_level is None:
            if rel.startswith("src/") and not is_exempt(rel, cfg["scan_exempt"]):
                findings.append(Finding(
                    "layers", rel, 0,
                    "is under src/ but matches no entry in `layers` -- give it the level "
                    "its includes justify in scripts/ratchets.json, so its dependency "
                    "direction is checked like every other layer's"))
            continue
        subjects += 1
        with open(abspath, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                match = INCLUDE_RE.match(line)
                if not match:
                    continue
                dst_prefix, dst_level = layer_of(match.group(1), layers)
                if dst_level is not None and dst_level > src_level:
                    findings.append(Finding(
                        "layers", rel, lineno,
                        f"{src_prefix} (L{src_level}) includes {dst_prefix} "
                        f"(L{dst_level}) -- layers may only depend upward"))
    return subjects, findings


# --- gate: dead code -------------------------------------------------------

DECL_RE = re.compile(
    r"^\s*(?:\[\[nodiscard\]\]\s*)?(?:inline\s+|static\s+|constexpr\s+|virtual\s+)*"
    r"(?:class|struct|enum\s+class|enum)\s+([A-Za-z_][A-Za-z0-9_]*)")
FUNC_DECL_RE = re.compile(
    r"^(?:\[\[nodiscard\]\]\s*)?(?:inline\s+|constexpr\s+|static\s+)*"
    r"[A-Za-z_][A-Za-z0-9_:<>,\s*&]*?\b([a-z_][A-Za-z0-9_]*)\s*\(")


def collect_declarations(root, cfg):
    """Every hand-written .hpp declaration, as gate subjects.

    GENERATED files declare nothing a human chose, so their symbols are not subjects.
    They stay in the CORPUS -- references from them still count -- but a generator that
    emits one struct per request is not writing dead code when a request has no C++
    reader, and answering it symbol-by-symbol produced a 16-entry exemption list that
    grew by two every time the protocol did. Drift in these files is the protocol gate's
    job, and it diffs the whole file against protocol/schema.json.
    """
    decls = {}
    for rel, abspath in walk_sources(root, cfg):
        if not rel.endswith(".hpp") or is_exempt(rel, cfg["scan_exempt"]):
            continue
        if is_exempt(rel, cfg["generated"]):
            continue
        with open(abspath, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, start=1):
                if line.lstrip().startswith(("//", "*", "/*")):
                    continue
                match = DECL_RE.match(line) or FUNC_DECL_RE.match(line)
                if match and match.group(1) not in decls:
                    decls[match.group(1)] = (rel, lineno)
    return decls


def gate_dead_code(root, cfg):
    """An unreferenced non-API symbol fails the build (S15.4).

    v1 shipped ZeroTrustSanitizer -- 197 lines, zero callers -- because nobody
    deleted it. Exemptions live in ratchets.json and each states when it expires.
    """
    exempt = {e["symbol"] for e in cfg["dead_code_exempt"]}
    decls = collect_declarations(root, cfg)
    corpus = []
    for rel, abspath in walk_sources(root, cfg):
        if is_exempt(rel, cfg["scan_exempt"]):
            continue
        with open(abspath, encoding="utf-8", errors="replace") as fh:
            corpus.append((rel, fh.read()))

    findings = []
    for symbol, (decl_file, decl_line) in sorted(decls.items()):
        if symbol in exempt:
            continue
        pattern = re.compile(r"\b" + re.escape(symbol) + r"\b")
        uses = sum(len(pattern.findall(text)) for _, text in corpus)
        if uses <= 1:  # the declaration itself
            findings.append(Finding("dead_code", decl_file, decl_line,
                                    f"'{symbol}' is declared and never referenced"))
    return len(decls), findings


# --- dormant gates ---------------------------------------------------------

def gate_protocol(root, cfg):
    """S4.4: generated TS and C++ are regenerated and diffed; drift fails."""
    schema = os.path.join(root, "protocol", "schema.json")
    if not os.path.exists(schema):
        return 0, []
    gen = os.path.join(root, "scripts", "gen_protocol.py")
    result = subprocess.run([sys.executable, gen, "--root", root, "--check"],
                            capture_output=True, text=True, check=False)
    findings = []
    if result.returncode != 0:
        for line in result.stdout.splitlines():
            if line.startswith("DRIFT"):
                findings.append(Finding("protocol", line.split(":")[0].replace("DRIFT ", ""),
                                        0, "generated file differs from protocol/schema.json"
                                           " -- run scripts/gen_protocol.py"))
    return 1, findings


TOOL_DECL_RE = re.compile(r'\bd\.name = "([a-z_]+)"')
TOOL_MENTION_RE = re.compile(r"\b([a-z]+_[a-z_]+)\b")
# Words that look like tool names (letter_letter) but are ordinary prose/identifiers.
TOOL_MENTION_STOPLIST = {
    "wall_clock", "old_text", "new_text", "start_line", "end_line", "workspace_root",
    "max_result_bytes", "tool_name", "error_class", "exit_code",
    # blast-radius capability flags: model-facing vocabulary from the S7 contract,
    # deliberately shown to the model by job_status's advisory output.
    "write_out", "read_out",
    # The recall tools' parameters (src/tools/context_tools.cpp). Named in their own
    # descriptions because a budget the model cannot see is a budget it cannot choose,
    # and named in error text because the fix for a bad range is to re-read the two
    # numbers the prompt printed. Parameters, not tools.
    "token_budget", "this_session_only", "first_event", "last_event",
    # Optimistic-concurrency vocabulary: a parameter and the read-observation footer,
    # not tools. The harness tracks versions; the model rarely copies the digest.
    "expected_version", "content_version",
    # lmp/code_intel protocol op names (P2 §10), not tools. locate_symbol routes through
    # the editor using these ops; they must not be mistaken for ghost tool references.
    "workspace_symbols", "rename_preview", "code_intel", "provides_code_intel",
}


def gate_tool_honesty(root, cfg):
    """S6.3: any tool name a model-facing string mentions must resolve to a registered
    tool. v1 audited this once and 7 of 13 tools were lying -- describing follow-up
    tools that did not exist, teaching the model to call ghosts.

    Scope: the string literals of EVERY file that declares tools. That used to be
    src/tools/registry.cpp alone and is now registry.cpp plus context_tools.cpp, which
    declares the two that read the durable context store. The union matters in both
    directions: a description in the new file is checked, and a description in the old
    one may legally name a tool the new one declares. A gate whose scope silently stops
    covering half its subjects is the failure this file's own header is about.

    A mention is any snake_case token; declared names plus a stoplist of parameter-ish
    words are legal, and anything else that LOOKS like a tool reference fails."""
    rels = [os.path.join("src", "tools", name)
            for name in ("registry.cpp", "context_tools.cpp")]
    sources = [(rel, os.path.join(root, rel)) for rel in rels]
    sources = [(rel, path) for rel, path in sources if os.path.exists(path)]
    if not sources:
        return 0, []
    texts = {}
    for rel, path in sources:
        with open(path, encoding="utf-8") as fh:
            texts[rel] = fh.read()
    # Declared across ALL of them before any is checked: registry.cpp's `plan` mentions
    # no other file's tools today, but a description that names one must not fail merely
    # because the two declarations live apart.
    declared = set()
    for text in texts.values():
        declared |= set(TOOL_DECL_RE.findall(text))
    findings = []
    if not declared:
        findings.append(Finding("tool_honesty", sources[0][0], 0,
                                "registry file exists but declares no tools"))
        return 1, findings
    for rel, text in texts.items():
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line.lstrip().startswith("#include"):
                continue  # an include path is not a model-facing string
            for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', line):
                for mention in TOOL_MENTION_RE.findall(lit):
                    if mention in declared or mention in TOOL_MENTION_STOPLIST:
                        continue
                    findings.append(Finding(
                        "tool_honesty", rel, lineno,
                        f"model-facing string mentions '{mention}', which is not a "
                        f"registered tool (declared: {', '.join(sorted(declared))})"))
    return len(declared), findings


# WHAT USED TO BE HERE: gate_prose_correctives, which parsed the case arms of
# Agent::apply_corrective and required every one to contain a mechanism rather than a
# composed sentence. The corrective machinery it policed was deleted 2026-08-05 -- the
# loop no longer steers the model at all, so there are no corrective arms to inspect,
# and a live gate over zero subjects fails min_subjects by design. The gate goes with
# the mechanism it guarded; if steering ever comes back, so must a gate that keeps it
# honest, because prose-only correctives are how v1 got 11 of 34.

GATES = {
    "layers": gate_layers,
    "dead_code": gate_dead_code,
    "protocol": gate_protocol,
    "tool_honesty": gate_tool_honesty,
}


def run(root, cfg, quiet=False):
    failed, lines = False, []
    for name, fn in GATES.items():
        spec = cfg["gates"][name]
        subjects, findings = fn(root, cfg)
        if not spec["live"]:
            if subjects > 0:
                failed = True
                lines.append(
                    f"FAIL {name}: dormant gate found {subjects} subject(s). "
                    f"Phase {spec['activates_in_phase']} has landed -- set live:true in "
                    f"scripts/ratchets.json and implement the check. {spec['note']}")
            else:
                lines.append(f"DORMANT {name}: 0 subjects, activates in phase "
                             f"{spec['activates_in_phase']}")
            continue
        if subjects < spec["min_subjects"]:
            failed = True
            lines.append(f"FAIL {name}: inspected {subjects} subject(s), expected at "
                         f"least {spec['min_subjects']}. A gate that inspects nothing "
                         f"reports green for free.")
        if findings:
            failed = True
            lines.append(f"FAIL {name}: {len(findings)} violation(s) over {subjects} subject(s)")
            lines.extend(str(f) for f in findings)
        elif subjects >= spec["min_subjects"]:
            lines.append(f"PASS {name}: {subjects} subject(s) clean")
    if not quiet:
        print("\n".join(lines))
    return failed, lines


# --- falsification ---------------------------------------------------------

PROBE = "src/platform/_probe.hpp"


def violations():
    """The planted defects, for the gates a NEW FILE can violate.

    That is the honest scope, and it is narrower than it reads. `layers` and
    `dead_code` scan every source file, so dropping one probe file in is a real
    violation. The other three read one specific file each -- protocol/schema.json,
    src/tools/registry.cpp, src/loop/agent.cpp -- and a probe beside them proves
    nothing, so falsifying those needs a mutation mechanism rather than a new file.
    scripts/mutation_test.py is the tool for it.

    Until then this prints exactly which gates it proved and which it did not, rather
    than "all". It formerly claimed all six with three of them unprobed, which is the
    same unfalsified green S2.1.2 exists to forbid -- in the falsifier itself.

    The dead-code probe's symbol name is generated, not written as a literal. It was
    a literal at first, and the self-test reported MISS: `scripts/` is one of the
    scanned source roots, so the name appearing in THIS file counted as a reference
    and the planted dead symbol looked alive. The gate was correct; the probe was
    contaminated by the file that planted it. Generating the name removes the
    contamination instead of excluding scripts/ from the scan, which would have hidden
    a whole directory from every gate to fix one probe.
    """
    dead_symbol = "Probe" + uuid.uuid4().hex[:12]
    return {
        "layers": (PROBE, '#include "src/loop/turn_machine.hpp"\n'),
        "dead_code": (PROBE, "class " + dead_symbol + " {};\n"),
    }


def copy_tree_for_probe(root, dest, extra=()):
    """Copies the repo, excluding build output -- by DIRECTORY NAME, not by prefix.

    An earlier version passed shutil.ignore_patterns("build*"), which also matched
    third_party/simdjson/include/simdjson/builder.h. Every copy then failed to compile
    for a reason that had nothing to do with the mutation, and the mutation score came
    back a perfect 6/6 killed -- a number that was measuring this function. Caught by
    running the harness on an UNMUTATED copy and watching it "kill" that too (S16: no
    metric quoted without checking what it counts)."""
    skip_dirs = {".git", "node_modules", "out", "__pycache__"} | set(extra)

    def ignore(directory, names):
        drop = set()
        for name in names:
            full = os.path.join(directory, name)
            if not os.path.isdir(full):
                continue
            if name in skip_dirs or name == "build" or name.startswith("build-"):
                drop.add(name)
        return drop

    # symlinks=True: npm's node_modules/.bin entries are symlinks whose targets are
    # resolved RELATIVE to the link. Dereferencing them turns each into a real file
    # whose `require('../typescript/bin/tsc')` no longer resolves, and the extension
    # typecheck fails in the copy for a reason unrelated to any mutation. Third thing
    # the null mutant caught.
    shutil.copytree(root, dest, ignore=ignore, symlinks=True)


def self_test(root, cfg):
    """Proves each live gate can go red. A green from a gate that has never been
    seen red is not evidence (S2.1.2)."""
    ok = True
    for label, (relpath, content) in violations().items():
        gate = label.split(":")[0]
        with tempfile.TemporaryDirectory() as tmp:
            copy_tree_for_probe(root, os.path.join(tmp, "repo"),
                                extra=("third_party", "bakeoff"))
            repo = os.path.join(tmp, "repo")
            probe = os.path.join(repo, relpath)
            os.makedirs(os.path.dirname(probe), exist_ok=True)
            with open(probe, "w", encoding="utf-8") as fh:
                fh.write(content)
            subjects, findings = GATES[gate](repo, cfg)
            hit = any(f.path.endswith("_probe.hpp") for f in findings)
            print(f"{'RED  ' if hit else 'MISS '} {label}: "
                  f"{len(findings)} finding(s) over {subjects} subject(s)")
            ok = ok and hit
    probed = {label.split(":")[0] for label in violations()}
    unprobed = sorted(set(GATES) - probed)
    if ok:
        print("\nProven capable of failing: " + ", ".join(sorted(probed)) + ".")
    else:
        print("\nFAIL: a gate did not fire on a violation planted for it. "
              "Until it does, its greens mean nothing.")
    if unprobed:
        # Named rather than counted, because "3 unprobed" is the kind of summary a
        # reader skims past and a list of names is one they act on.
        print("NOT proven here (each reads one specific file, so a probe file cannot "
              "violate them; use scripts/mutation_test.py): " + ", ".join(unprobed) + ".")
    return not ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    cfg = load_config(root)
    failed = self_test(root, cfg) if args.self_test else run(root, cfg)[0]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
