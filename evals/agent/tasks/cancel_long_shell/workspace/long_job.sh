#!/bin/sh
# Deliberately long; the eval harness cancels the run before this should finish.
sleep 120
echo finished > job_finished.txt
