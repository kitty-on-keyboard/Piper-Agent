add_library(lmp_tools
  registry.cpp
  memory_file.cpp
  sandbox.cpp
  container.cpp
  syntax_check.cpp
  mcp_host.cpp
)
target_include_directories(lmp_tools PUBLIC "${PROJECT_SOURCE_DIR}")
# lmp_mcp is PUBLIC because mcp_host.hpp is part of this library's interface (M1: MCP is
# how LM_Pipe uses MCP, not a library beside it). nlohmann is PUBLIC for the same reason:
# tool_spec_from_schema takes a json, and its tests must be able to build one.
target_link_libraries(lmp_tools
  PRIVATE lmp_warnings lmp_assertions simdjson
  PUBLIC lmp_platform parsephony lmp_mcp nlohmann_json)

# A diagnostic driver, not a test: points McpHost at a real third-party MCP server and
# prints what the agent would get. The gate tests drive our own demo server, which cannot
# surprise us; this is for the ones we did not write. Built on demand.
add_executable(mcp_host_probe EXCLUDE_FROM_ALL mcp_host_probe.cpp)
target_link_libraries(mcp_host_probe PRIVATE lmp_warnings lmp_assertions lmp_tools)
