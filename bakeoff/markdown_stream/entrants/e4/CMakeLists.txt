cmake_minimum_required(VERSION 3.14)
project(MarkdownStream CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(markdown_stream src/markdown_stream.cpp)
target_include_directories(markdown_stream PUBLIC include)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(test_markdown_stream tests/test_markdown_stream.cpp)
target_link_libraries(test_markdown_stream PRIVATE markdown_stream gtest_main)

include(GoogleTest)
gtest_discover_tests(test_markdown_stream)
