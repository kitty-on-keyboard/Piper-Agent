lmp_add_test(test_mcp_framing SOURCES test_mcp_framing.cpp LABELS gate LINK lmp_mcp TIMEOUT 60)
lmp_add_test(test_mcp_session SOURCES test_mcp_session.cpp LABELS gate LINK lmp_mcp TIMEOUT 120)

# The stdio test drives the demo server as a real subprocess, so it needs the binary's
# path -- and a build-order dependency, or it races the thing it is trying to spawn.
lmp_add_test(test_mcp_stdio SOURCES test_mcp_stdio.cpp LABELS gate LINK lmp_mcp TIMEOUT 180)
target_compile_definitions(test_mcp_stdio PRIVATE
  LMP_MCP_DEMO_SERVER="$<TARGET_FILE:mcp_demo_server>")
add_dependencies(test_mcp_stdio mcp_demo_server)
