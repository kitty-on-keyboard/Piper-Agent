lmp_add_test(test_check_framework    SOURCES test_check_framework.cpp    LABELS gate)
lmp_add_test(test_assertions_enabled SOURCES test_assertions_enabled.cpp LABELS gate)
lmp_add_test(test_gate_manifest      SOURCES test_gate_manifest.cpp      LABELS gate)

target_compile_definitions(test_gate_manifest PRIVATE
  LMP_CTEST_COMMAND="${CMAKE_CTEST_COMMAND}"
  LMP_BINARY_DIR="${CMAKE_BINARY_DIR}"
  LMP_GATE_MANIFEST="${CMAKE_CURRENT_SOURCE_DIR}/gate_manifest.txt")

# The ratchets, and the proof that they can fail. Both in the gate: a ratchet suite
# that only ever runs green is indistinguishable from one that does nothing.
lmp_add_script_test(test_ratchets
  COMMAND "${PROJECT_SOURCE_DIR}/scripts/run_ratchets.py" --root "${PROJECT_SOURCE_DIR}"
  LABELS gate)
lmp_add_script_test(test_ratchets_falsifiable
  COMMAND "${PROJECT_SOURCE_DIR}/scripts/run_ratchets.py" --root "${PROJECT_SOURCE_DIR}" --self-test
  LABELS gate)

# The bake-off pins, including the assertion that the holdout stays harder (S11.3).
lmp_add_script_test(test_bakeoff_pins
  COMMAND "${CMAKE_COMMAND}" -E env "LMP_BUILD_DIR=${CMAKE_BINARY_DIR}"
          "${PROJECT_SOURCE_DIR}/scripts/eval.py" --root "${PROJECT_SOURCE_DIR}" score
  LABELS gate)

# The extension is part of the build contract: a protocol change that breaks the TS
# side must fail here, not in someone's editor three days later (S4.4).
find_program(LMP_NPX npx)
if(LMP_NPX AND EXISTS "${PROJECT_SOURCE_DIR}/extension/node_modules")
  lmp_add_script_test(test_extension_typecheck
    COMMAND "${CMAKE_COMMAND}" -E chdir "${PROJECT_SOURCE_DIR}/extension"
            "${LMP_NPX}" tsc -p . --noEmit
    LABELS gate)
else()
  message(STATUS "extension typecheck skipped: run `npm install` in extension/")
endif()
