lmp_add_test(test_sandbox  SOURCES test_sandbox.cpp  LABELS gate LINK lmp_tools TIMEOUT 120)
lmp_add_test(test_registry SOURCES test_registry.cpp LABELS gate LINK lmp_tools TIMEOUT 120)
lmp_add_test(test_syntax_check SOURCES test_syntax_check.cpp LABELS gate LINK lmp_tools TIMEOUT 120)

# MCP tools in the Registry (M1). Drives the real demo server as a subprocess, so it needs
# the binary's path and a build-order dependency -- same reason tests/mcp/test_mcp_stdio
# does. In the GATE, not realmodel: MCP has nothing to do with the model.
lmp_add_test(test_mcp_host SOURCES test_mcp_host.cpp LABELS gate LINK lmp_tools TIMEOUT 180)
target_compile_definitions(test_mcp_host PRIVATE
  LMP_MCP_DEMO_SERVER="$<TARGET_FILE:mcp_demo_server>")
add_dependencies(test_mcp_host mcp_demo_server)
