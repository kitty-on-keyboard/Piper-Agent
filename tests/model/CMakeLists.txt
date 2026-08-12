# --- the miniature vocabulary (G0) ------------------------------------------
#
# Generated rather than committed: QwenTokenizer::load enforces the Qwen3 size band, so an
# honest fixture is ~140k entries and a few MB of JSON. Generating it keeps the property
# that matters -- the fixture goes through the SAME family verification production does --
# without committing a blob nobody will read. See tests/fixtures/make_mini_vocab.py.
#
# Three shapes: the Qwen-shaped one every gate test decodes with, and the two load() must
# REFUSE. The refusals used to need a 31 MB Gemma checkpoint on this machine, which made
# them `realmodel` and so they never ran in CI; generating them moves the family guard
# into the gate and costs nothing on disk.
set(LMP_MINI_VOCAB "${CMAKE_CURRENT_BINARY_DIR}/mini_qwen/tokenizer.json")
set(LMP_FOREIGN_VOCAB "${CMAKE_CURRENT_BINARY_DIR}/mini_foreign/tokenizer.json")
set(LMP_UNMARKED_VOCAB "${CMAKE_CURRENT_BINARY_DIR}/mini_unmarked/tokenizer.json")

function(lmp_make_vocab shape out)
  add_custom_command(
    OUTPUT "${out}"
    COMMAND "${CMAKE_COMMAND}" -E env python3
            "${PROJECT_SOURCE_DIR}/tests/fixtures/make_mini_vocab.py" "${shape}" "${out}"
    DEPENDS "${PROJECT_SOURCE_DIR}/tests/fixtures/make_mini_vocab.py"
    COMMENT "Generating the ${shape} vocabulary fixture"
    VERBATIM)
endfunction()

lmp_make_vocab(qwen "${LMP_MINI_VOCAB}")
lmp_make_vocab(foreign "${LMP_FOREIGN_VOCAB}")
lmp_make_vocab(unmarked "${LMP_UNMARKED_VOCAB}")
add_custom_target(lmp_mini_vocab DEPENDS "${LMP_MINI_VOCAB}")
add_custom_target(lmp_refused_vocabs
                  DEPENDS "${LMP_FOREIGN_VOCAB}" "${LMP_UNMARKED_VOCAB}")

lmp_add_test(test_backend_seam SOURCES test_backend_seam.cpp LABELS gate LINK lmp_model)
# The speculative block algebra runs in the GATE: SpecForward is an interface precisely so
# that "which positions were committed, what does the cache hold" is testable with no GPU.
lmp_add_test(test_speculative SOURCES test_speculative.cpp LABELS gate LINK lmp_model)
# The cross-turn cache decision, for the same reason: an off-by-one there decodes fluent
# wrong text rather than crashing, so the algebra is pure and asserted with no GPU.
lmp_add_test(test_kv_reuse SOURCES test_kv_reuse.cpp LABELS gate LINK lmp_model)
lmp_add_test(test_mini_vocab SOURCES test_mini_vocab.cpp LABELS gate LINK lmp_model)
add_dependencies(test_mini_vocab lmp_mini_vocab lmp_refused_vocabs)
target_compile_definitions(test_mini_vocab PRIVATE
  LMP_MINI_VOCAB_JSON="${LMP_MINI_VOCAB}"
  LMP_FOREIGN_VOCAB_JSON="${LMP_FOREIGN_VOCAB}"
  LMP_UNMARKED_VOCAB_JSON="${LMP_UNMARKED_VOCAB}")

# The turn grammar runs in BOTH suites off one source (R1). It reads token bytes, so it
# needs a vocabulary -- but a vocabulary is not a GPU, and labelling it `realmodel` meant
# CI never ran it: commit 4300a3c changed what closing a tool call means and this file
# asserted the old contract, red and unseen, for two days. The gate variant on the
# miniature vocabulary is what closes that hole; the realmodel variant below keeps the
# real checkpoint's merges under test.
lmp_add_test(test_grammar SOURCES test_grammar.cpp LABELS gate LINK lmp_model TIMEOUT 300)
add_dependencies(test_grammar lmp_mini_vocab)
target_compile_definitions(test_grammar PRIVATE LMP_MINI_VOCAB_JSON="${LMP_MINI_VOCAB}")

# Model sequence ceiling parsing + think-budget sink (P0 §4). Needs the mini vocab for
# the GrammarSink walk; config.json parsing needs no checkpoint.
# simdjson is explicit here (as in test_spec_cache) because this TU includes
# qwen35_moe_config.hpp directly to assert the RoPE parse; lmp_model keeps it PRIVATE.
lmp_add_test(test_model_bounds SOURCES test_model_bounds.cpp LABELS gate LINK lmp_model lmp_loop
             simdjson TIMEOUT 120)
add_dependencies(test_model_bounds lmp_mini_vocab)
target_compile_definitions(test_model_bounds PRIVATE LMP_MINI_VOCAB_JSON="${LMP_MINI_VOCAB}")

# Real-model tests (S11.6): excluded from the gate, and `ctest --preset realmodel`
# pins jobs=1 -- concurrent large model loads exhaust unified memory.
lmp_add_test(test_grammar_realmodel SOURCES test_grammar.cpp LABELS realmodel LINK lmp_model TIMEOUT 300)
lmp_add_test(test_realmodel SOURCES test_realmodel.cpp LABELS realmodel LINK lmp_model TIMEOUT 1200)
# These are the tests that actually reach the GPU, so they need mlx.metallib beside them
# (see lmp_stage_metallib in src/model/CMakeLists.txt). No-op when MLX is off.
lmp_stage_metallib(test_realmodel)
# G3's falsifier: reuse must fire, and must be INVISIBLE in the output. A stale-cache bug
# decodes fluent wrong text rather than crashing, so byte-identity is the only assertion
# that catches it.
lmp_add_test(test_kv_reuse_realmodel SOURCES test_kv_reuse_realmodel.cpp LABELS realmodel LINK lmp_model TIMEOUT 1200)
lmp_stage_metallib(test_kv_reuse_realmodel)
lmp_add_test(test_spec_cache SOURCES test_spec_cache.cpp LABELS realmodel LINK lmp_model simdjson TIMEOUT 900)
lmp_stage_metallib(test_spec_cache)
# Same relaxation mlx_backend.cpp and the diag driver carry: this TU includes the vendored
# MLX numerics directly, to drive Qwen35MoeModel's cache API below the backend seam.
set_source_files_properties(test_spec_cache.cpp PROPERTIES COMPILE_OPTIONS
  "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")

# A diagnostic driver, not a test: prints the model's actual turn so a failure can be
# attributed by observation rather than by guessing (S19.3). Built on demand.
add_executable(lmp_diag EXCLUDE_FROM_ALL diag_main.cpp diag_moe.cpp)
target_link_libraries(lmp_diag PRIVATE lmp_warnings lmp_assertions lmp_model simdjson)
lmp_stage_metallib(lmp_diag)
# Same relaxation mlx_backend.cpp carries: these TUs include the vendored MLX numerics
# directly so the scan can be benchmarked against its own reference implementation.
set_source_files_properties(diag_main.cpp diag_moe.cpp PROPERTIES COMPILE_OPTIONS
  "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")
