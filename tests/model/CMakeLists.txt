lmp_add_test(test_backend_seam SOURCES test_backend_seam.cpp LABELS gate LINK lmp_model)

# Real-model tests (S11.6): excluded from the gate, and `ctest --preset realmodel`
# pins jobs=1 -- concurrent large model loads exhaust unified memory.
lmp_add_test(test_grammar_realmodel SOURCES test_grammar.cpp LABELS realmodel LINK lmp_model TIMEOUT 300)
lmp_add_test(test_realmodel SOURCES test_realmodel.cpp LABELS realmodel LINK lmp_model TIMEOUT 1200)
