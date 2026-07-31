# The falsifiers, ported verbatim from v1 (spec S18). Do not rewrite, do not "improve".
#
# These arrive BEFORE the components they grade: the blast-radius capability classifier
# is phase 5 and the log-triage compactor is phase 4, but both answer keys are in the
# repo and loading in the gate from phase 0. An answer key written after the thing it
# grades is not an answer key.
#
# The entrant headers and the two corpus loaders are exempt from lmp_warnings. That
# exemption is deliberate and narrow: an entrant header is EVIDENCE, and evidence you
# have reformatted to satisfy -Wold-style-cast is no longer the thing that was measured.
# scripts/ratchets.json records the same exemption for the size gates, by path, with
# this reason attached.

add_library(bakeoff_corpora INTERFACE)
target_include_directories(bakeoff_corpora SYSTEM INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius"
  "${CMAKE_CURRENT_SOURCE_DIR}/log_triage")
target_link_libraries(bakeoff_corpora INTERFACE simdjson)
target_compile_definitions(bakeoff_corpora INTERFACE
  LMP_BAKEOFF_BR_DIR="${CMAKE_CURRENT_SOURCE_DIR}/blast_radius"
  LMP_BAKEOFF_LT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/log_triage")

# --- scoreboards -----------------------------------------------------------
#
# One binary per entrant, because 11 implementations of one symbol must never meet in a
# translation unit. Off by default: 28 entrants x simdjson is minutes of compile, and the
# gate has a five-minute budget for the whole suite. `-D LMP_BUILD_SCOREBOARDS=ON`
# builds them all. Two are built unconditionally below so the harness cannot rot
# unnoticed -- a benchmark that stopped compiling six weeks ago is a benchmark you have
# already lost, which is exactly how v1 lost the graft cookoff harness.

option(LMP_BUILD_SCOREBOARDS "Build every per-entrant bake-off scoreboard" OFF)

function(_lmp_blast_radius_scoreboard name header)
  add_executable(blast_radius_score_${name}
    "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/score.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/entrant_tu.cpp")
  target_link_libraries(blast_radius_score_${name} PRIVATE bakeoff_corpora)
  target_compile_definitions(blast_radius_score_${name} PRIVATE
    ENTRANT_HEADER="${header}"
    ENTRANT_NAME="${name}"
    CORPUS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/corpus.jsonl")
  target_compile_options(blast_radius_score_${name} PRIVATE -w)
endfunction()

function(_lmp_log_triage_scoreboard name header adapter)
  add_executable(log_triage_score_${name}
    "${CMAKE_CURRENT_SOURCE_DIR}/log_triage/score.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/log_triage/entrant_tu.cpp")
  target_link_libraries(log_triage_score_${name} PRIVATE bakeoff_corpora)
  target_compile_definitions(log_triage_score_${name} PRIVATE
    ENTRANT_HEADER="${header}"
    ADAPTER_HEADER="${adapter}"
    ENTRANT_NAME="${name}"
    CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/log_triage")
  target_compile_options(log_triage_score_${name} PRIVATE -w)
endfunction()

# Two blast-radius entrants are NOT built, and the distinction is worth stating because
# it decides which numbers this repo can still reproduce.
#
#   entrants/incumbent.hpp   includes v1's "security/command_risk.hpp"
#   entrants/e00_merged.hpp  includes v1's "security/blast_radius.hpp"
#
# Both are thin adapters around v1 source, and v1 source is deliberately not carried over
# (S2: nothing carries except S18). The other eleven -- e01..e14, the actual cookoff
# submissions -- are self-contained by construction, which is exactly the property the
# README says was enforced so that an entrant's own copy of the contract could never
# become the copy it was scored against. Those eleven still build and still score.
#
# So: README.md's e00_merged row (0 weighted misses, 179/179) and its incumbent row (253)
# are NOT reproducible in this repo today. They are historical record, not measurements.
# Re-wire both in phase 5, when v2 has a classifier of its own, and re-measure.
_lmp_blast_radius_scoreboard(e12 "entrants/e12.hpp")
_lmp_log_triage_scoreboard(incumbent "entrants/incumbent.hpp" "adapters/direct.hpp")

if(LMP_BUILD_SCOREBOARDS)
  file(GLOB _br_entrants RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius"
       "${CMAKE_CURRENT_SOURCE_DIR}/blast_radius/entrants/e*.hpp")
  foreach(_h ${_br_entrants})
    get_filename_component(_n "${_h}" NAME_WE)
    # e00_merged matches the glob but needs v1 source; see the note above.
    if(NOT _n STREQUAL "e00_merged")
      _lmp_blast_radius_scoreboard("${_n}" "${_h}")
    endif()
  endforeach()

  file(GLOB _lt_entrants RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}/log_triage"
       "${CMAKE_CURRENT_SOURCE_DIR}/log_triage/entrants/e*.hpp")
  foreach(_h ${_lt_entrants})
    get_filename_component(_n "${_h}" NAME_WE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/log_triage/adapters/${_n}.hpp")
      _lmp_log_triage_scoreboard("${_n}" "${_h}" "adapters/${_n}.hpp")
    endif()
  endforeach()
endif()
