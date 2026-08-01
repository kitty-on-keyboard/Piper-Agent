#!/bin/zsh
# Build the neutral scoreboard once per PrefixLedger entrant and print one row each.
#
#   ./score.sh <dir-of-entrant-trees>
#
# Each subdirectory is one entrant, laid out as Brief D specified: include/ and src/.
# One binary per entrant, because N implementations of one symbol must never meet in a
# translation unit.
#
# Entrant sources compile with -w. An entrant is EVIDENCE, and evidence reformatted to
# satisfy the consuming project's warning flags is no longer the thing that was measured
# -- the same exemption bakeoff/CMakeLists.txt records for round 1.
#
# The falsifiers/ trees are entrants too, and deliberately broken ones. Run them every
# time: a scoreboard on which every entrant passes everything is not evidence that the
# entrants are good until it has been shown capable of going red. Expected output:
#   broken_divergence  -- agree_fail, semantics_fail (both directions), fp_collide,
#                         p50 over budget, trunc_ratio ~100
#   broken_stale_hash  -- fp_path_fail only
set -u
here="${0:A:h}"
root="${1:-$here/entrants}"
board="$here/scoreboard.cpp"

for d in "$root"/*(/N) "$here"/amalgam(/N) "$here"/falsifiers/*(/N); do
  n="${d:t}"
  inc=()
  # Brief D specified include/prefix_ledger.hpp. One entrant shipped
  # include/kv/prefix_ledger.hpp instead; both are put on the search path rather than
  # being "corrected", so the deviation costs an include flag and not a rewrite.
  [[ -d "$d/include" ]] && inc+=("-I$d/include")
  [[ -d "$d/include/kv" ]] && inc+=("-I$d/include/kv")
  [[ -f "$d/prefix_ledger.hpp" ]] && inc+=("-I$d")
  srcs=("$d"/src/*.cpp(N) "$d"/prefix_ledger.cpp(N))
  out="${TMPDIR:-/tmp}/pl_score_$n"
  if ! clang++ -std=c++20 -O2 -w "${inc[@]}" "$board" "${srcs[@]}" -o "$out" 2>"$out.log"; then
    printf '%-20s BUILD FAILED\n' "$n"
    tail -4 "$out.log" | sed 's/^/       /'
    continue
  fi
  printf '%-20s ' "$n"
  "$out" || printf 'RUN FAILED\n'
done
