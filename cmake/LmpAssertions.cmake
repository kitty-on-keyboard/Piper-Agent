# Spec S15.1: "Assertions enabled in test builds -- v1 compiled 317 asserts out with
# -DNDEBUG for two months."
#
# CMake injects -DNDEBUG into Release, RelWithDebInfo and MinSizeRel by default. That
# is the exact mechanism that silently disarmed v1's assertions: nobody wrote -DNDEBUG,
# the build type did. Strip it from every configuration rather than avoiding the build
# types, so an assertion cannot be disarmed by a preset choice.
#
# This is belt-and-braces with tests/platform/test_assertions_enabled.cpp, which is a
# #error -- if this module ever stops working, the build fails at compile time with a
# message naming this file, not at runtime with a silently absent check.

add_library(lmp_assertions INTERFACE)
target_compile_options(lmp_assertions INTERFACE -UNDEBUG)

foreach(_cfg RELEASE RELWITHDEBINFO MINSIZEREL)
  foreach(_lang C CXX)
    string(REPLACE "-DNDEBUG" "" _stripped "${CMAKE_${_lang}_FLAGS_${_cfg}}")
    string(REPLACE "/DNDEBUG" "" _stripped "${_stripped}")
    set(CMAKE_${_lang}_FLAGS_${_cfg} "${_stripped}" CACHE STRING "" FORCE)
  endforeach()
endforeach()
unset(_cfg)
unset(_lang)
unset(_stripped)
