add_library(lmp_surface transport.cpp protocol.cpp mcp_settings.cpp)
target_include_directories(lmp_surface PUBLIC "${PROJECT_SOURCE_DIR}")
# lmp_tools is PUBLIC because mcp_settings.hpp hands back tools:: types. Surface is the
# top layer (S3), so depending downward on tools is the allowed direction.
target_link_libraries(lmp_surface PRIVATE lmp_warnings lmp_assertions
  PUBLIC lmp_platform lmp_model lmp_tools)

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
