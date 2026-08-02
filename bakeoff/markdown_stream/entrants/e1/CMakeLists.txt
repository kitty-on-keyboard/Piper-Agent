cmake_minimum_required(VERSION 3.14)
project(markdown_stream)

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

add_executable(markdown_stream_test test/markdown_stream_test.cpp)
target_link_libraries(markdown_stream_test markdown_stream GTest::gtest_main)

enable_testing()
include(GoogleTest)
gtest_discover_tests(markdown_stream_test)
