lmp_add_test(test_backend_seam SOURCES test_backend_seam.cpp LABELS gate LINK lmp_model)
# The speculative block algebra runs in the GATE: SpecForward is an interface precisely so
# that "which positions were committed, what does the cache hold" is testable with no GPU.
lmp_add_test(test_speculative SOURCES test_speculative.cpp LABELS gate LINK lmp_model)

# Real-model tests (S11.6): excluded from the gate, and `ctest --preset realmodel`
# pins jobs=1 -- concurrent large model loads exhaust unified memory.
lmp_add_test(test_grammar_realmodel SOURCES test_grammar.cpp LABELS realmodel LINK lmp_model TIMEOUT 300)
lmp_add_test(test_realmodel SOURCES test_realmodel.cpp LABELS realmodel LINK lmp_model TIMEOUT 1200)
lmp_add_test(test_spec_cache SOURCES test_spec_cache.cpp LABELS realmodel LINK lmp_model simdjson TIMEOUT 900)
# Same relaxation mlx_backend.cpp and the diag driver carry: this TU includes the vendored
# MLX numerics directly, to drive Qwen35MoeModel's cache API below the backend seam.
set_source_files_properties(test_spec_cache.cpp PROPERTIES COMPILE_OPTIONS
  "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")

# A diagnostic driver, not a test: prints the model's actual turn so a failure can be
# attributed by observation rather than by guessing (S19.3). Built on demand.
add_executable(lmp_diag EXCLUDE_FROM_ALL diag_main.cpp diag_moe.cpp)
target_link_libraries(lmp_diag PRIVATE lmp_warnings lmp_assertions lmp_model simdjson)
# Same relaxation mlx_backend.cpp carries: these TUs include the vendored MLX numerics
# directly so the scan can be benchmarked against its own reference implementation.
set_source_files_properties(diag_main.cpp diag_moe.cpp PROPERTIES COMPILE_OPTIONS
  "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")
