cmake_minimum_required(VERSION 3.24)
project(lm_pipe2 CXX)

# LM_Pipe v2 -- Apple Silicon only. Unified memory is assumed, not abstracted over.
if(NOT APPLE OR NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
  message(FATAL_ERROR
    "LM_Pipe v2 targets Apple Silicon only (see spec S1). "
    "Detected: ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}. "
    "This is a refusal, not a fallback -- there is no portable path.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LmpWarnings)
include(LmpAssertions)
include(LmpTest)

option(LMP_BUILD_TESTING "Build the test suite" ON)
option(LMP_SANITIZE "Build with ASan+UBSan" OFF)

if(LMP_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

add_subdirectory(third_party)
add_subdirectory(src)
add_subdirectory(bakeoff)

if(LMP_BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

# The ratchets are part of the build contract, not a side script (spec S15.3-S15.7).
add_custom_target(lmp_ratchets
  COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_ratchets.py"
          --root "${CMAKE_CURRENT_SOURCE_DIR}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMENT "Running size / layer / dead-code / honesty / prose ratchets"
  VERBATIM)
