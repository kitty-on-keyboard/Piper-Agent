#!/bin/sh
# Cancellation falsifier. Every assertion here exists because some other mechanism can
# produce a clean-looking workspace without cancellation ever having worked.
set -e

test "${LMP_EVAL_CANCEL_SENT}" = "1"

# The cancel must have been sent because long_job.sh said it was running -- not on a
# timer that may have fired while the model was still generating, and not because the
# harness gave up on an overrunning run. Only the first of the three tests the thing.
if test "${LMP_EVAL_CANCEL_TRIGGER}" != "file"; then
  echo "grader: cancel trigger was '${LMP_EVAL_CANCEL_TRIGGER}', not the running child" >&2
  exit 1
fi

# The child provably reached its sleep. Without this a cancel that arrived before the
# shell tool ever started would pass on an empty workspace.
if ! test -f job_started.txt; then
  echo "grader: long job never started; cancellation was not exercised" >&2
  exit 1
fi

# The kill itself.
if test -f job_finished.txt; then
  echo "grader: long job completed despite cancel" >&2
  exit 1
fi

# ...and it was the CANCEL that killed it. long_job.sh sleeps 900 s, so the sandbox's own
# 300 s per-command wall clock would also leave no finished marker; that outcome is a
# timeout wearing a cancel's clothes. A real cancel kills the process group inside
# pump_output's 200 ms poll, so the gap between sending it and the run ending is small.
if test -n "${LMP_EVAL_CANCEL_AT_SECONDS}" && test -n "${LMP_EVAL_RUN_SECONDS}"; then
  gap=$(awk -v a="${LMP_EVAL_RUN_SECONDS}" -v b="${LMP_EVAL_CANCEL_AT_SECONDS}" \
        'BEGIN { printf "%d", a - b }')
  if test "${gap}" -gt 30; then
    echo "grader: run continued ${gap}s after cancel; not a prompt process-group kill" >&2
    exit 1
  fi
fi

# Agent must not claim finished after a cancelled run.
if test -f done.txt && grep -qi finished done.txt; then
  echo "grader: done.txt claims finished after cancel" >&2
  exit 1
fi
exit 0
