add_library(lmp_loop agent.cpp approval.cpp parallel_calls.cpp token_stream.cpp turn.cpp verification.cpp)
target_include_directories(lmp_loop PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_loop PRIVATE lmp_warnings lmp_assertions
  PUBLIC lmp_context lmp_tools lmp_model lmp_platform)
