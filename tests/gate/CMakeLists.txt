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

# The agent-eval pin ATTRIBUTION (V2). Not the eval itself -- that loads a 19 GB model
# and takes half an hour -- but the logic that turns a drop into a task name. Its
# falsifier is "change one task's outcome by hand and confirm the diff names it", which
# by hand means once and never again; here it is every build.
lmp_add_script_test(test_pin_attribution
  COMMAND "${PROJECT_SOURCE_DIR}/scripts/agent_eval.py" self-test
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

# The transcript's ordering invariant. Type-checking cannot see this: the view script is a
# TypeScript template literal, so tsc reads the whole thing as a string, and the bug it
# pins -- a tool row on screen before the text that introduced it -- typechecked perfectly
# for as long as it existed. Needs only node, not node_modules, because it reads
# src/webview.ts directly and slices the block under test out of it.
find_program(LMP_NODE node)
if(LMP_NODE)
  lmp_add_script_test(test_transcript_order
    COMMAND "${LMP_NODE}"
            "${PROJECT_SOURCE_DIR}/extension/scripts/verify-transcript-order.js"
    LABELS gate)
  # The other half of the same invariant, and invisible to the one above: that stub records
  # the order nodes ARRIVE, and in this defect every node arrived in the right order and
  # was then parented in the wrong place. Inline code spans were swept out of the sentence
  # that wrote them and collected at the end of the bubble.
  lmp_add_script_test(test_inline_code_order
    COMMAND "${LMP_NODE}"
            "${PROJECT_SOURCE_DIR}/extension/scripts/verify-inline-code-order.js"
    LABELS gate)
  # THAT THE VIEW SCRIPT PARSES AT ALL, which nothing above checks. tsc reads it as one
  # string (it is a template literal), so a syntax error inside it typechecks clean,
  # packages, and installs a pane where every button is dead -- the script stops before a
  # single listener is attached. Shipped exactly once, 2026-08-15, over an escaped slash.
  # Reads the COMPILED out/webview.js, so it needs `npm run compile` to have run.
  if(EXISTS "${PROJECT_SOURCE_DIR}/extension/out/webview.js")
    lmp_add_script_test(test_view_script_parses
      COMMAND "${LMP_NODE}"
              "${PROJECT_SOURCE_DIR}/extension/scripts/verify-view-script.js"
      LABELS gate)
  else()
    message(STATUS "view script check skipped: run `npm run compile` in extension/")
  endif()
else()
  message(STATUS "transcript order check skipped: node not found")
endif()
