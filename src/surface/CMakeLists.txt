add_library(lmp_surface transport.cpp protocol.cpp mcp_settings.cpp context_journal.cpp
  session.cpp wire.cpp)
target_include_directories(lmp_surface PUBLIC "${PROJECT_SOURCE_DIR}")
# lmp_tools is PUBLIC because mcp_settings.hpp hands back tools:: types, and lmp_context
# and lmp_pcc for the same reason from context_journal.hpp -- it is the adapter between
# them and names both in its interface. lmp_loop joins them for session.hpp, which holds
# an AgentConfig. Surface is the top layer (S3), so depending downward on any of them is
# the allowed direction.
target_link_libraries(lmp_surface PRIVATE lmp_warnings lmp_assertions
  PUBLIC lmp_platform lmp_model lmp_tools lmp_context lmp_pcc lmp_loop)

# The sidecar: one process, one model, one loop (S4.1).
add_executable(lmp_sidecar sidecar.cpp)
target_link_libraries(lmp_sidecar PRIVATE lmp_warnings lmp_assertions
  lmp_surface lmp_loop lmp_tools lmp_context lmp_model lmp_platform)

# `cmake --build build --target vsix` produces extension/lm-pipe.vsix with THIS binary
# inside it, verified by hash. Not part of `all`: packaging is a deliberate act, and a
# packaging step nobody asked for is a slow build for everyone who is not shipping.
add_custom_target(vsix
  DEPENDS lmp_sidecar
  COMMAND npm --prefix "${PROJECT_SOURCE_DIR}/extension" run compile
  COMMAND npm --prefix "${PROJECT_SOURCE_DIR}/extension" run stage-sidecar
  COMMAND npm --prefix "${PROJECT_SOURCE_DIR}/extension" run package
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}/extension"
  COMMENT "Packaging lm-pipe.vsix (sidecar staged and hash-verified)"
  VERBATIM)
