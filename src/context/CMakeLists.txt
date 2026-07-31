add_library(lmp_context context.cpp)
target_include_directories(lmp_context PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_context PRIVATE lmp_warnings lmp_assertions PUBLIC lmp_model)
