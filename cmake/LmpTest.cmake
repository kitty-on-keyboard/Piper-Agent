# lmp_add_test(<name> SOURCES <...> [LABELS <...>] [LINK <...>] [TIMEOUT <s>])
#
# Every test carries at least one label. The gate is selected by label, never by an
# -E name pattern: v1 ran `ctest -E realmodel` for months, `-E` excludes by NAME, it
# matched nothing, and every "48/48" number it printed was meaningless. Labels are
# assigned here at declaration, and tests/gate/gate_manifest.json pins both the count
# and the names of what `-L gate` selects.
#
# Known labels:
#   gate      -- must run with no model, no network, under 5 minutes total (S11.1)
#   realmodel -- loads a real model; excluded from the gate; NEVER run in parallel (S11.6)
#   bakeoff   -- scores a corpus; part of the gate when it needs no model

function(lmp_add_test name)
  cmake_parse_arguments(T "" "TIMEOUT" "SOURCES;LABELS;LINK" ${ARGN})

  if(NOT T_SOURCES)
    message(FATAL_ERROR "lmp_add_test(${name}): SOURCES is required")
  endif()
  if(NOT T_LABELS)
    message(FATAL_ERROR
      "lmp_add_test(${name}): LABELS is required. An unlabelled test is invisible to "
      "`ctest -L gate` and would silently never run.")
  endif()

  add_executable(${name} ${T_SOURCES})
  target_link_libraries(${name} PRIVATE lmp_warnings lmp_assertions lmp_testing ${T_LINK})
  target_include_directories(${name} PRIVATE "${PROJECT_SOURCE_DIR}")

  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES
    LABELS "${T_LABELS}"
    TIMEOUT "$<IF:$<BOOL:${T_TIMEOUT}>,${T_TIMEOUT},120>"
    ENVIRONMENT "LMP_REPO_ROOT=${PROJECT_SOURCE_DIR}")
endfunction()

# lmp_add_script_test(<name> COMMAND <...> LABELS <...> [TIMEOUT <s>])
#
# For gates that are scripts rather than binaries (the ratchets). Same label
# requirement, same manifest, so a script test cannot hide from `-L gate` either.
function(lmp_add_script_test name)
  cmake_parse_arguments(T "" "TIMEOUT" "COMMAND;LABELS" ${ARGN})
  if(NOT T_COMMAND OR NOT T_LABELS)
    message(FATAL_ERROR "lmp_add_script_test(${name}): COMMAND and LABELS are required")
  endif()
  add_test(NAME ${name} COMMAND ${T_COMMAND})
  set_tests_properties(${name} PROPERTIES
    LABELS "${T_LABELS}"
    TIMEOUT "$<IF:$<BOOL:${T_TIMEOUT}>,${T_TIMEOUT},300>"
    ENVIRONMENT "LMP_REPO_ROOT=${PROJECT_SOURCE_DIR}")
endfunction()
