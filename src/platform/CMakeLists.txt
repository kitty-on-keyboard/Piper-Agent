add_library(lmp_platform
  event_log.cpp
  fs.cpp
)
target_include_directories(lmp_platform PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_platform PRIVATE lmp_warnings lmp_assertions
  PUBLIC nlohmann_json)

# L0 links nothing from src/. If this line ever needs a sibling layer, the layering
# is wrong -- fix the layering, not this file.
