cmake_minimum_required(VERSION 3.14)
project(MarkdownStream VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add the main library
add_library(markdown_stream src/markdown_stream.cpp)
target_include_directories(markdown_stream PUBLIC include)

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Enable testing
enable_testing()

# Add test executable
add_executable(test_markdown_stream tests/test_markdown_stream.cpp)
target_link_libraries(test_markdown_stream PRIVATE markdown_stream gtest_main)

include(GoogleTest)
gtest_discover_tests(test_markdown_stream)
add_executable(test_debug tests/test_debug.cpp)
target_link_libraries(test_debug PRIVATE markdown_stream gtest_main)
