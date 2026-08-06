add_library(lmp_tools
  registry.cpp
  memory_file.cpp
  concurrent_calls.cpp
  context_tools.cpp
  sandbox.cpp
  container.cpp
  syntax_check.cpp
  mcp_host.cpp
)
target_include_directories(lmp_tools PUBLIC "${PROJECT_SOURCE_DIR}")
# lmp_mcp is PUBLIC because mcp_host.hpp is part of this library's interface (M1: MCP is
# how LM_Pipe uses MCP, not a library beside it). nlohmann is PUBLIC for the same reason:
# tool_spec_from_schema takes a json, and its tests must be able to build one.
# lmp_pcc is PRIVATE: context_tools.cpp needs the store, but registry.hpp only
# forward-declares pcc::Store, so no consumer of this library compiles against SQLite for
# the sake of two tools. src/pcc sits at L1 and src/tools at L2, so the dependency points
# the allowed way (scripts/ratchets.json, layers gate).
# lmp_model is PUBLIC: CancelToken lives in src/model/backend.hpp and is part of the
# sandbox / Registry execute surface (run-scoped cancellation). tools is L2 over model
# at L1, so the include direction is allowed; the link must follow.
target_link_libraries(lmp_tools
  PRIVATE lmp_warnings lmp_assertions simdjson lmp_pcc
  PUBLIC lmp_platform lmp_model parsephony lmp_mcp nlohmann_json)

# A diagnostic driver, not a test: points McpHost at a real third-party MCP server and
# prints what the agent would get. The gate tests drive our own demo server, which cannot
# surprise us; this is for the ones we did not write. Built on demand.
add_executable(mcp_host_probe EXCLUDE_FROM_ALL mcp_host_probe.cpp)
target_link_libraries(mcp_host_probe PRIVATE lmp_warnings lmp_assertions lmp_tools)
