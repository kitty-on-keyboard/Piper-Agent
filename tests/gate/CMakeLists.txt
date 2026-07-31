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
