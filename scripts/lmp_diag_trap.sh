#!/usr/bin/env bash
# HS1 / lmp_diag death trap — parent records exit code / signal when the child is
# SIGKILL'd (jetsam / external kill). SIGKILL cannot be caught in-process; fflush
# breadcrumbs in mlx_backend + cmd_reuse are the other half of this dig.
#
# Usage:
#   scripts/lmp_diag_trap.sh ./lmp_diag reuse 2 2048 128 32
#   scripts/lmp_diag_trap.sh /path/to/lmp_diag reuse 3 2048 128 32
#
# Prints START / END lines with ec and decoded signal (137 => SIGKILL) so a master
# log is more than a bare START when the child dies early.

set -u

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <lmp_diag> [args...]" >&2
  exit 2
fi

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

cmd=("$@")
echo "START ts=$(stamp) cmd=${cmd[*]}"
# Line-buffer child stdout/stderr when stdbuf exists (Linux + some Macs with coreutils).
if command -v stdbuf >/dev/null 2>&1; then
  stdbuf -oL -eL "${cmd[@]}" &
else
  "${cmd[@]}" &
fi
pid=$!
echo "CHILD pid=$pid"
wait "$pid"
ec=$?

sig=""
# bash: 128+N when killed by signal N; 137 = 128+9 = SIGKILL
if (( ec > 128 && ec < 160 )); then
  n=$((ec - 128))
  case "$n" in
    9)  sig="SIGKILL" ;;
    6)  sig="SIGABRT" ;;
    11) sig="SIGSEGV" ;;
    15) sig="SIGTERM" ;;
    *)  sig="signal_$n" ;;
  esac
fi

if [[ -n "$sig" ]]; then
  echo "END ts=$(stamp) ec=$ec signal=$sig DIED_EARLY=1"
else
  echo "END ts=$(stamp) ec=$ec signal= none DIED_EARLY=0"
fi
exit "$ec"
