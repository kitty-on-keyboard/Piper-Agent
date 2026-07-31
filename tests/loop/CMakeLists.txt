lmp_add_test(test_loop SOURCES test_loop.cpp LABELS gate LINK lmp_loop)
lmp_add_test(test_verifier SOURCES test_verifier.cpp LABELS gate LINK lmp_loop TIMEOUT 120)
