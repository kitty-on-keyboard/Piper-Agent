# PCC -- the Persistent Context Core (spec S8).
#
# Sits at L1 beside the model layer: it depends on nothing in LM_Pipe above platform, so
# both src/context (L3) and src/tools (L2) can reach it without inverting anything.
#
# It deliberately does NOT link src/model. The one thing it would want from there is the
# tokenizer, and taking it would drag a 19 GB-adjacent dependency into a component whose
# tests must run on a CI box with no model at all. recall() takes a TokenCounter instead,
# and the sidecar injects the real one.
add_library(lmp_pcc
  sqlite.cpp
  diff.cpp
  cas.cpp
  store.cpp
  recall.cpp
)
target_include_directories(lmp_pcc PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_pcc PRIVATE lmp_warnings lmp_assertions)

# SQLite and zlib both ship in the macOS SDK, so neither is a new install step for a
# contributor or for CI. That is most of why they were chosen over the cook-off's
# LanceDB + zstd + sentence-transformers stack, which would have added a Python runtime
# to a C++ agent. FTS5 is compiled into Apple's libsqlite3; tests/pcc asserts that rather
# than assuming it, because a build where it is missing fails at the first query with an
# error that reads like a typo.
target_link_libraries(lmp_pcc PUBLIC sqlite3 z)

# --- the MCP server ---------------------------------------------------------
#
# PCC over MCP, so the store is reachable from the editors Sean actually drives (Cursor,
# Antigravity) and not only from our own loop. Same binary, same database: a fact the
# agent wrote in a sidecar run is one Cursor can read a minute later.
add_executable(pcc_mcp_server mcp_server.cpp)
target_link_libraries(pcc_mcp_server PRIVATE lmp_pcc lmp_mcp lmp_warnings lmp_assertions)
