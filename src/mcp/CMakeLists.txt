# A standalone MCP implementation: client and server, spec-compliant, C++20.
#
# It links nothing from LM_Pipe. That is deliberate and worth keeping -- the whole
# library is liftable into its own repo, and nothing above it can quietly make the
# protocol layer depend on a model or a tokenizer.
add_library(lmp_mcp
  message.cpp
  framing.cpp
  content.cpp
  subprocess.cpp
  transport.cpp
  inproc_transport.cpp
  server.cpp
  client.cpp
)
target_include_directories(lmp_mcp PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_mcp
  PRIVATE lmp_warnings lmp_assertions
  PUBLIC lmp_nlohmann)
target_compile_features(lmp_mcp PUBLIC cxx_std_20)

# WHICH SPELLING OF THE SPAWN-CHDIR CALL THIS PLATFORM HAS. POSIX.1-2024 standardised
# `posix_spawn_file_actions_addchdir`; before that everyone shipped it as `..._np`, and
# macOS 26 deprecates the `_np` name loudly enough that -Werror rejects it. glibc has had
# only the `_np` spelling since 2.29. Checked rather than guessed at from __APPLE__: the
# guess is wrong on exactly the two platforms this builds on, in opposite directions.
include(CheckSymbolExists)
check_symbol_exists(posix_spawn_file_actions_addchdir "spawn.h" LMP_HAVE_SPAWN_ADDCHDIR)
check_symbol_exists(posix_spawn_file_actions_addchdir_np "spawn.h" LMP_HAVE_SPAWN_ADDCHDIR_NP)
if(LMP_HAVE_SPAWN_ADDCHDIR)
  target_compile_definitions(lmp_mcp PRIVATE LMP_SPAWN_ADDCHDIR=posix_spawn_file_actions_addchdir)
elseif(LMP_HAVE_SPAWN_ADDCHDIR_NP)
  target_compile_definitions(lmp_mcp PRIVATE LMP_SPAWN_ADDCHDIR=posix_spawn_file_actions_addchdir_np)
else()
  # No way to chdir the child, and doing it in the parent would race every other thread.
  # Subprocess::Options::working_dir then refuses rather than silently starting the server
  # in the wrong directory, which is the failure it exists to prevent.
  target_compile_definitions(lmp_mcp PRIVATE LMP_SPAWN_ADDCHDIR_UNAVAILABLE=1)
endif()

# --- demo binaries ----------------------------------------------------------
#
# These are the two things every MCP implementation needs and most skip: a server a real
# client can drive, and a client that can drive a real server. They are built by default
# because tests/mcp drives both, and an example that stops compiling is an example
# nobody will notice is broken.

add_executable(mcp_demo_server demo_server.cpp)
target_link_libraries(mcp_demo_server PRIVATE lmp_mcp lmp_warnings lmp_assertions)

add_executable(mcp_probe demo_client.cpp)
target_link_libraries(mcp_probe PRIVATE lmp_mcp lmp_warnings lmp_assertions)
