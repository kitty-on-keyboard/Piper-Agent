lmp_add_test(test_backend_seam SOURCES test_backend_seam.cpp LABELS gate LINK lmp_model)

# Real-model tests (S11.6): excluded from the gate, and `ctest --preset realmodel`
# pins jobs=1 -- concurrent large model loads exhaust unified memory.
lmp_add_test(test_grammar_realmodel SOURCES test_grammar.cpp LABELS realmodel LINK lmp_model TIMEOUT 300)
lmp_add_test(test_realmodel SOURCES test_realmodel.cpp LABELS realmodel LINK lmp_model TIMEOUT 1200)

# A diagnostic driver, not a test: prints the model's actual turn so a failure can be
# attributed by observation rather than by guessing (S19.3). Built on demand.
add_executable(lmp_diag EXCLUDE_FROM_ALL diag_main.cpp)
target_link_libraries(lmp_diag PRIVATE lmp_warnings lmp_assertions lmp_model simdjson)
# Same relaxation mlx_backend.cpp carries: this TU includes the vendored MLX numerics
# directly so the scan can be benchmarked against its own reference implementation.
set_source_files_properties(diag_main.cpp PROPERTIES COMPILE_OPTIONS
  "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")
