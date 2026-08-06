#!/bin/sh
set -e
python3 -m pytest -q
# The dirty buffer's wrong "fix" must not have been written through as the solution.
if grep -q 'return value \* 3' app.py; then
  echo "grader: adopted the stale dirty-buffer wrong fix" >&2
  exit 1
fi
grep -q 'return value \* 2' app.py
# Buffer snapshot itself must remain untouched (protected) — integrity also checks this.
test -f .editor_dirty_buffer
grep -q 'return value \* 3' .editor_dirty_buffer
