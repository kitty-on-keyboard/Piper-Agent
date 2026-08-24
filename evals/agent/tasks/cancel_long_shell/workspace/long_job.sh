#!/bin/sh
# Deliberately long; the eval harness cancels the run while this is mid-flight.
#
# The started marker is what makes this fixture a falsifier rather than a coincidence.
# It is written BEFORE the sleep, so the harness can wait for the child to provably be
# running and cancel exactly then -- a cancel that lands while the model is still
# generating ends the run just as tidily and tests nothing about killing a process.
echo started > job_started.txt

# Longer than every timeout that could otherwise end this job and be mistaken for a
# cancel: the run's own wall clock (120 s), the sandbox's per-command limit
# (shell_wall_clock_seconds = 300 s, session.cpp) and the harness deadline (300 s).
# If job_finished.txt exists, the process group genuinely survived the cancel.
sleep 900
echo finished > job_finished.txt
