add_library(lmp_surface transport.cpp protocol.cpp)
target_include_directories(lmp_surface PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_surface PRIVATE lmp_warnings lmp_assertions
  PUBLIC lmp_platform lmp_model)

# The sidecar: one process, one model, one loop (S4.1).
add_executable(lmp_sidecar sidecar.cpp)
target_link_libraries(lmp_sidecar PRIVATE lmp_warnings lmp_assertions
  lmp_surface lmp_loop lmp_tools lmp_context lmp_model lmp_platform)
