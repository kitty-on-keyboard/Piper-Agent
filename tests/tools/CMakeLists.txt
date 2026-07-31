lmp_add_test(test_sandbox  SOURCES test_sandbox.cpp  LABELS gate LINK lmp_tools TIMEOUT 120)
lmp_add_test(test_registry SOURCES test_registry.cpp LABELS gate LINK lmp_tools TIMEOUT 120)
