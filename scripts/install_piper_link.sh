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
# Mac preferred (Homebrew). Non-Darwin is a clean no-op (exit 0).
# If Homebrew bin is missing/unwritable, falls back to ~/.local/bin or ~/bin
# automatically — no copy-paste ln.
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

  if [[ -d "${HOMEBREW_BIN}" ]]; then
    dest="${HOMEBREW_BIN}/${LINK_NAME}"
    if [[ -w "${HOMEBREW_BIN}" ]] && ln -sfn "${target}" "${dest}"; then
      echo "install_piper_link: ${dest} -> ${target}"
      return 0
    fi
  else
    echo "install_piper_link: ${HOMEBREW_BIN} missing; trying user bin" >&2
  fi

  # Homebrew missing/not writable (or ln failed): fall back to a user bin — no copy-paste ln.
  local fallback dest2
  for fallback in "${HOME}/.local/bin" "${HOME}/bin"; do
    mkdir -p "${fallback}" 2>/dev/null || true
    if [[ -d "${fallback}" && -w "${fallback}" ]]; then
      dest2="${fallback}/${LINK_NAME}"
      if ln -sfn "${target}" "${dest2}"; then
        echo "install_piper_link: ${dest2} -> ${target}"
        case ":${PATH}:" in
          *":${fallback}:"*) ;;
          *)
            echo "install_piper_link: note: ${fallback} is not on PATH; open a new shell or add it" >&2
            ;;
        esac
        return 0
      fi
    fi
  done

  echo "install_piper_link: could not link into ${HOMEBREW_BIN} or ~/bin ~/.local/bin" >&2
  echo "  built piper is at: ${target}" >&2
  echo "  call it by absolute path, or fix directory permissions and re-run" >&2
  exit 1
}

main "$@"
