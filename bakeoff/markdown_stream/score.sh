#!/bin/zsh
# Build the neutral scoreboard once per MarkdownStream entrant and print one row each.
#
#   ./score.sh <dir-of-entrant-trees>
#
# Each subdirectory is one entrant, laid out as Brief E specified: include/ and src/.
# One binary per entrant, because N implementations of one symbol must never meet in a
# translation unit.
#
# Entrant sources compile with -w. An entrant is EVIDENCE, and evidence reformatted to
# satisfy the consuming project's warning flags is no longer the thing that was measured --
# the same exemption bakeoff/CMakeLists.txt records for round 1.
#
# The falsifiers/ trees are built every time. A scoreboard on which every entrant passes
# everything is not evidence until it has been shown capable of going red. Expected:
#   clean_base       -- all zero (the columns are satisfiable)
#   broken_holdback  -- hold only
#   broken_swallow   -- fin only
#   broken_split     -- split only
set -u
here="${0:A:h}"
root="${1:-$here/entrants}"
board="$here/scoreboard.cpp"

printf '%-18s %6s %6s %6s %6s %6s %8s %5s %5s %6s %5s %9s\n' \
  '' split strict skel code lose hold fin utf8 reset pend us/KB

for d in "$root"/*(/N) "$here"/amalgam(/N) "$here"/falsifiers/*(/N); do
  n="${d:t}"
  [[ "$n" == "common" ]] && continue
  inc=()
  [[ -d "$d/include" ]] && inc+=("-I$d/include")
  [[ -f "$d/markdown_stream.hpp" ]] && inc+=("-I$d")
  srcs=("$d"/src/*.cpp(N) "$d"/markdown_stream.cpp(N))
  # An entrant may ship a test main() in src/; the scoreboard owns main().
  srcs=(${srcs:#*test*})
  out="${TMPDIR:-/tmp}/ms_score_$n"
  if ! clang++ -std=c++20 -O2 -w "${inc[@]}" "$board" "${srcs[@]}" -o "$out" 2>"$out.log"; then
    printf '%-18s BUILD FAILED\n' "$n"
    tail -4 "$out.log" | sed 's/^/       /'
    continue
  fi
  printf '%-18s ' "$n"
  "$out" || printf 'RUN FAILED\n'
done
