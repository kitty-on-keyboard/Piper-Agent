cmake_minimum_required(VERSION 3.14)
project(MarkdownStream CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_library(markdown_stream src/markdown_stream.cpp)
target_include_directories(markdown_stream PUBLIC include)

add_executable(test_markdown_stream tests/test_markdown_stream.cpp)
target_link_libraries(test_markdown_stream markdown_stream gtest_main)

include(GoogleTest)
gtest_discover_tests(test_markdown_stream)
