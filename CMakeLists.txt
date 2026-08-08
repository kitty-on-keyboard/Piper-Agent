cmake_minimum_required(VERSION 3.24)

# The oldest macOS a shipped build runs on. STATED, and stated BEFORE project(), because
# the alternative is not "no opinion" -- it is an accidental one. Left unset, MLX compiled
# its objects and its metallib for whatever this machine happens to run (26.5.2) while our
# own targets linked at 26.0: ~120 "built for newer macOS version" linker warnings, and
# the developer's OS silently becomes the product's minimum. A metallib will not load
# below the version it was compiled for, so that surfaces at a user's first GPU call
# rather than here.
#
# 26.2 IS A PRODUCT DECISION, made deliberately for performance (Sean, 2026-08-08), not a
# build tuning knob somebody flipped. It is the threshold at which MLX compiles its M5
# neural-accelerator kernels -- quantized_nax, steel_gemm_*_nax, steel_attention_nax --
# which target the matrix units on M5-class GPUs. Below it MLX defines MLX_METAL_NO_NAX
# and every matrix kernel takes the general path. Difusio measured 3.4x end-to-end on one
# M5 Pro transformer forward from that difference alone.
#
# WHAT IT COSTS: macOS 26.2 becomes the minimum this product runs on, full stop. A
# metallib will not load below the version it was compiled for, so an older machine fails
# at the first GPU call rather than degrading. Shipping both would mean two metallibs and
# a runtime choice, which MLX's build is not arranged to produce.
#
# WHAT IT DOES NOT COST: older HARDWARE. MLX checks is_nax_available() at runtime (M5-class
# GPU) and falls back on its own, so a NAX build on an M4 is identical, not slower.
#
# Dropping back to "15.0" is the entire rollback, and is also the thing to compare against
# if a NAX kernel is ever suspected of being wrong.
#
# The guard matters: CMake always has this cache entry (empty by default), so a plain
# set(... CACHE ...) would be silently ignored, which is how this came to be unset at all.
# Note it also means an EXISTING build dir keeps its old value -- pass
# -DCMAKE_OSX_DEPLOYMENT_TARGET=... explicitly to change one in place.
if(NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "26.2" CACHE STRING
      "Minimum macOS. >= 26.2 enables MLX's M5 accelerator kernels (NAX)." FORCE)
endif()

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
