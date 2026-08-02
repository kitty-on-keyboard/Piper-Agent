add_library(lmp_tools
  registry.cpp
  memory_file.cpp
  sandbox.cpp
  container.cpp
  syntax_check.cpp
)
target_include_directories(lmp_tools PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_tools
  PRIVATE lmp_warnings lmp_assertions simdjson
  PUBLIC lmp_platform parsephony)
