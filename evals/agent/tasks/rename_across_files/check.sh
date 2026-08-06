#!/bin/sh
# Immutable grader (sibling of workspace/; never copytree'd into the agent cwd).
set -e
python3 -m pytest -q
# Old name must be gone from production modules; tests may mention it only as strings
# if needed, but this suite keeps the identifier out of every .py file.
if grep -rq --include='*.py' -n 'calc_total' .; then
  echo "grader: calc_total still present" >&2
  exit 1
fi
# New name must exist in the definition site and at least two callers.
grep -q 'def compute_total' billing.py
grep -q 'compute_total' checkout.py
grep -q 'compute_total' report.py
grep -q 'compute_total' invoices.py
