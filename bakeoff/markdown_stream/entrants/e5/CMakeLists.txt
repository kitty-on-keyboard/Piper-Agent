cmake_minimum_required(VERSION 3.20)
project(markdown_stream VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

enable_testing()

add_executable(markdown_stream_test tests/test_markdown_stream.cpp)
target_link_libraries(markdown_stream_test PRIVATE markdown_stream GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(markdown_stream_test)
