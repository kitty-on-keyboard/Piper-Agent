#!/usr/bin/env python3
"""The CI ratchets (spec S15.3 - S15.7).

One entry point, six gates. Run with --self-test to prove each gate can go red:
a gate that has never been observed failing is a gate nobody has falsified, and
S2.1.2 does not make an exception for tooling.

  ./scripts/run_ratchets.py --root .
  ./scripts/run_ratchets.py --root . --self-test

The C++ scanning here is a brace-depth scanner, not a parser. That is a real
limitation and it is stated rather than hidden: it can be fooled by a brace inside
a string literal or a macro that opens a scope. It is not fooled by anything in
this repo, and the alternative -- a full frontend in the pre-commit path -- costs
more than the defect class is worth. If it ever misreports, fix the scanner; do
not raise the limit.
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


def strip_comments_and_strings(line):
    """Blanks out // comments and "..." literals so braces inside them do not count."""
    out, i, in_str = [], 0, False
    while i < len(line):
        ch = line[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            i += 1
            continue
        if ch == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        out.append(ch)
        i += 1
    return "".join(out)


# --- gate: size ------------------------------------------------------------

FUNC_RE = re.compile(r"^[A-Za-z_~].*\)\s*(const\s*)?(noexcept\s*)?(override\s*)?\{\s*$")


def scan_function_shapes(lines):
    """Yields (start_line, length, max_nesting) for each brace-balanced body."""
    depth, start, peak = 0, None, 0
    for idx, raw in enumerate(lines, start=1):
        line = strip_comments_and_strings(raw)
        opens, closes = line.count("{"), line.count("}")
        if depth == 0 and opens > 0 and FUNC_RE.match(raw.strip()):
            start, peak = idx, 0
        if start is not None:
            depth += opens - closes
            peak = max(peak, depth - 1)
            if depth <= 0:
                yield start, idx - start + 1, peak
                depth, start, peak = 0, None, 0
        else:
            depth = max(0, depth + opens - closes)


def gate_size(root, cfg):
    limits, findings, subjects = cfg["limits"], [], 0
    for rel, abspath in walk_sources(root, cfg):
        if is_exempt(rel, cfg["size_exempt"]):
            continue
        subjects += 1
        with open(abspath, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        if len(lines) > limits["max_file_lines"]:
            findings.append(Finding("size", rel, len(lines),
                                    f"{len(lines)} lines exceeds {limits['max_file_lines']}"))
        if not rel.endswith(CPP_EXT):
            continue
        for start, length, nesting in scan_function_shapes(lines):
            if length > limits["max_function_lines"]:
                findings.append(Finding("size", rel, start,
                                        f"function body is {length} lines, limit is "
                                        f"{limits['max_function_lines']}"))
            if nesting > limits["max_nesting_depth"]:
                findings.append(Finding("size", rel, start,
                                        f"nesting depth {nesting} exceeds "
                                        f"{limits['max_nesting_depth']}"))
    return subjects, findings


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
    """
    layers, findings, subjects = cfg["layers"], [], 0
    for rel, abspath in walk_sources(root, cfg):
        if not rel.endswith(CPP_EXT):
            continue
        src_prefix, src_level = layer_of(rel, layers)
        if src_level is None:
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
    decls = {}
    for rel, abspath in walk_sources(root, cfg):
        if not rel.endswith(".hpp") or is_exempt(rel, cfg["size_exempt"]):
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
        if is_exempt(rel, cfg["size_exempt"]):
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
    # `plan`'s parameter naming the command that proves the mission complete. A
    # parameter, not a tool -- the gate is here to catch descriptions that promise tools
    # which do not exist, and this promises nothing.
    "verify_with",
    # blast-radius capability flags: model-facing vocabulary from the S7 contract,
    # deliberately shown to the model by job_status's advisory output.
    "write_out", "read_out",
}


def gate_tool_honesty(root, cfg):
    """S6.3: any tool name a model-facing string mentions must resolve to a registered
    tool. v1 audited this once and 7 of 13 tools were lying -- describing follow-up
    tools that did not exist, teaching the model to call ghosts.

    Scope: the string literals of src/tools/registry.cpp (the declarations file). A
    mention is any snake_case token; declared names plus a stoplist of parameter-ish
    words are legal, and anything else that LOOKS like a tool reference fails."""
    path = os.path.join(root, "src", "tools", "registry.cpp")
    if not os.path.exists(path):
        return 0, []
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    declared = set(TOOL_DECL_RE.findall(text))
    findings = []
    if not declared:
        findings.append(Finding("tool_honesty", "src/tools/registry.cpp", 0,
                                "registry file exists but declares no tools"))
        return 1, findings
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#include"):
            continue  # an include path is not a model-facing string
        for lit in re.findall(r'"((?:[^"\\]|\\.)*)"', line):
            for mention in TOOL_MENTION_RE.findall(lit):
                if mention in declared or mention in TOOL_MENTION_STOPLIST:
                    continue
                findings.append(Finding(
                    "tool_honesty", "src/tools/registry.cpp", lineno,
                    f"model-facing string mentions '{mention}', which is not a "
                    f"registered tool (declared: {', '.join(sorted(declared))})"))
    return len(declared), findings


# A corrective site is a case arm of Agent::apply_corrective. Each must change state,
# synthesize a call, or alter control flow -- these are the shapes that count as
# mechanism.
MECHANISM_RE = re.compile(
    r"\b(ctx_\.|registry_\.|repeats_\.|halted_\s*=|policy_\.|log_\.|"
    r"verifier\.|\.run_and_record|\.add_turn|\.set_checklist|\.record_|"
    r"return;|break;|continue;)")


def gate_prose_correctives(root, cfg):
    """S9.2: the count of prose-only corrective sites must be 0.

    v1 had 11 of 34, including one named ForceWrite that forced nothing while logging
    that it was forcing something. The check does not count injection SITES -- that
    number is gameable by moving branches into one function -- it checks that every
    corrective arm contains a mechanism and not merely a composed sentence."""
    path = os.path.join(root, "src", "loop", "agent.cpp")
    if not os.path.exists(path):
        return 0, []
    with open(path, encoding="utf-8") as fh:
        lines = fh.read().splitlines()

    # `current` stays None until the first `case`, so the preamble between the
    # function header and the switch is not mistaken for an arm.
    body, arms, current, start = None, [], None, 0
    for idx, line in enumerate(lines, start=1):
        if "void Agent::apply_corrective" in line:
            body = True
            continue
        if body and line.startswith("}"):
            break
        if not body:
            continue
        if re.match(r"\s*case Corrective::", line):
            if current is not None:
                arms.append((start, current))
            current, start = [], idx
        elif current is not None:
            current.append(line)
    if current is not None:
        arms.append((start, current))

    findings = []
    for lineno, arm in arms:
        text = "\n".join(arm)
        code = "\n".join(l for l in arm if not l.strip().startswith("//"))
        if not MECHANISM_RE.search(code):
            findings.append(Finding(
                "prose_correctives", "src/loop/agent.cpp", lineno,
                "corrective arm changes no state and alters no control flow -- it is "
                "prose-only, which S9.2 forbids"))
    return len(arms), findings


GATES = {
    "size": gate_size,
    "layers": gate_layers,
    "dead_code": gate_dead_code,
    "protocol": gate_protocol,
    "tool_honesty": gate_tool_honesty,
    "prose_correctives": gate_prose_correctives,
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
    """The planted defects, one per live gate.

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
        "size:file": (PROBE, "// probe\n" + "int x_probe_pad;\n" * 900),
        "size:function": (PROBE,
                          "inline void probe_fn() {\n" + "    int a = 1;\n" * 120 + "}\n"),
        "size:nesting": (PROBE,
                         "inline void probe_fn() {\n    if (a) {\n        if (b) {\n"
                         "            if (c) {\n                if (d) {\n"
                         "                    x();\n                }\n            }\n"
                         "        }\n    }\n}\n"),
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
    if ok:
        print("\nAll live ratchets proven capable of failing.")
    else:
        print("\nFAIL: a ratchet did not fire on a violation planted for it. "
              "Until it does, its greens mean nothing.")
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
