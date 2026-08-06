#!/bin/sh
# Scripted cancellation grade: requires the harness to have sent cancel, and the
# long job must not have landed its completion marker as a successful finish.
set -e
test "${LMP_EVAL_CANCEL_SENT}" = "1"

# long_job.sh writes job_finished.txt only after its sleep; cancel should prevent that
# in a correct implementation. If the marker exists, the cancel path did not bite.
if test -f job_finished.txt; then
  echo "grader: long job completed despite cancel" >&2
  exit 1
fi

# Agent must not claim finished after a cancelled run.
if test -f done.txt && grep -qi finished done.txt; then
  echo "grader: done.txt claims finished after cancel" >&2
  exit 1
fi
exit 0
