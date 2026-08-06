#!/bin/sh
set -e
test -f found.txt
token=$(tr -d '[:space:]' < found.txt)
test "$token" = "NEEDLE_TOKEN_lmp_eval_7f3c"
# Haystack should still be present for trajectory inspection.
test -d haystack
test -f haystack/chunk_17.md
grep -q 'NEEDLE:NEEDLE_TOKEN_lmp_eval_7f3c' haystack/chunk_17.md
