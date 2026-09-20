#!/usr/bin/env bash
# Refresh /opt/homebrew/bin/piper so bare `piper` resolves to THIS checkout's
# built binary (build/src/surface/piper -> lmp_sidecar).
#
# Why: a checkout rename (e.g. LM_Pipe_2 -> Piper) leaves a stale Homebrew
# symlink; orchestrators that call bare `piper` then get "command not found".
#
# Usage (after building lmp_sidecar):
#   ./scripts/install_piper_link.sh
#   ./scripts/install_piper_link.sh /path/to/build/src/surface/piper
#   cmake --build --preset dev --target install-piper-link
#
# Mac + writable Homebrew only. Missing /opt/homebrew/bin (Linux CI, non-Mac)
# is a clean no-op (exit 0). Permission failures print a copy-paste ln and exit 1.
set -euo pipefail

readonly HOMEBREW_BIN="/opt/homebrew/bin"
readonly LINK_NAME="piper"

repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${here}/.." && pwd
}

resolve_built_piper() {
  if [[ $# -ge 1 && -n "${1}" ]]; then
    printf '%s\n' "${1}"
    return 0
  fi
  if [[ -n "${LMP_PIPER_BIN:-}" ]]; then
    printf '%s\n' "${LMP_PIPER_BIN}"
    return 0
  fi

  local root candidate
  root="$(repo_root)"
  for candidate in \
      "${root}/build/src/surface/piper" \
      "${root}/build/src/surface/lmp_sidecar" \
      "${root}/build-asan/src/surface/piper" \
      "${root}/build-asan/src/surface/lmp_sidecar"; do
    if [[ -e "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  echo "install_piper_link: no built piper/lmp_sidecar under build/src/surface/" >&2
  echo "  build first: cmake --preset dev && cmake --build --preset dev --target lmp_sidecar -j8" >&2
  return 1
}

absolute_target() {
  local path="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath "${path}"
  else
    # macOS without coreutils: python is always available in this project's env.
    python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "${path}"
  fi
}

main() {
  local built target dest
  built="$(resolve_built_piper "${1:-}")"
  if [[ ! -e "${built}" ]]; then
    echo "install_piper_link: built binary not found: ${built}" >&2
    echo "  build first: cmake --preset dev && cmake --build --preset dev --target lmp_sidecar -j8" >&2
    exit 1
  fi
  target="$(absolute_target "${built}")"

  if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "install_piper_link: skipping Homebrew link (not Darwin); piper is at ${target}"
    exit 0
  fi

  if [[ ! -d "${HOMEBREW_BIN}" ]]; then
    echo "install_piper_link: ${HOMEBREW_BIN} missing; skipping (piper is at ${target})"
    exit 0
  fi

  dest="${HOMEBREW_BIN}/${LINK_NAME}"
  if [[ ! -w "${HOMEBREW_BIN}" ]]; then
    echo "install_piper_link: ${HOMEBREW_BIN} not writable; run:" >&2
    echo "  ln -sfn '${target}' '${dest}'" >&2
    exit 1
  fi

  if ! ln -sfn "${target}" "${dest}"; then
    echo "install_piper_link: ln failed; run:" >&2
    echo "  ln -sfn '${target}' '${dest}'" >&2
    exit 1
  fi

  echo "install_piper_link: ${dest} -> ${target}"
}

main "$@"
