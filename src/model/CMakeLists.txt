# The amalgamated draft proposer from the round-1 cook-off, compiled as a library rather
# than copied into src/. It is EVIDENCE as much as code -- bakeoff/draft_proposer/ carries
# its scoreboard and README -- and a copy here would drift from the thing that was
# measured. -w for the same reason bakeoff/CMakeLists.txt gives: an entrant reformatted to
# satisfy this project's warning flags is no longer what the scoreboard scored.
add_library(lmp_draft "${PROJECT_SOURCE_DIR}/bakeoff/draft_proposer/suffix_proposer.cpp")
target_include_directories(lmp_draft PUBLIC "${PROJECT_SOURCE_DIR}")
target_compile_options(lmp_draft PRIVATE -w)

add_library(lmp_model
  backend.cpp
  chat_template.cpp
  grammar.cpp
  kv_cache.cpp
  mlx_backend.cpp
  model_limits.cpp
  qwen_tokenizer.cpp
  sampler.cpp
  spec_verifier.cpp
  speculative.cpp
)
target_include_directories(lmp_model PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_model
  PRIVATE lmp_warnings lmp_assertions simdjson
  PUBLIC lmp_platform frankentok parsephony lmp_draft)

# --- MLX --------------------------------------------------------------------
#
# PINNED AND BUILT FROM SOURCE. No Python is involved at any stage: MLX's own
# CMakeLists only mentions an interpreter under MLX_BUILD_PYTHON_BINDINGS, which is off.
#
# This replaced a `<python> -m mlx --cmake-dir` probe that linked the libmlx.dylib inside
# a pip package. Two things were wrong with it, and only the second is about taste:
#
#   * The version depended on whichever interpreter configured the build. This machine's
#     `python3` is 3.9, MLX stopped shipping cp39 wheels after 0.29.3, and find_package
#     caches MLX_DIR -- so the build could pin 0.29.3 while printing 0.31.2, and did.
#     A tag cannot do that.
#   * Every binary came out carrying an absolute rpath into ~/.venvs/lmp-mlx. That is
#     blocker #1 on shipping the extension: on any machine but this one the packaged
#     sidecar spawns and fails. A static libmlx.a plus a colocated metallib (below) is
#     the whole fix.
#
# Difusio's Engine/CMakeLists.txt does the same thing for the same reasons; that is
# where this arrangement comes from.
option(LMP_WITH_MLX "Build the MLX backend (compiles MLX from source -- slow, once)" OFF)
set(LMP_MLX_TAG "v0.32.0" CACHE STRING "MLX release tag to build against")

# LMP_MLX_PYTHON is gone, and its absence has to be LOUD rather than ignored. Several
# handoff docs still say `-DLMP_MLX_PYTHON=/usr/bin/false` is how you reproduce what CI
# builds. CMake accepts unknown -D cache entries silently, so following those
# instructions today would configure with MLX *on* (the dev preset turns it on) while
# the author believed they were testing the MLX-off configuration CI uses -- the exact
# "a check that matched nothing while printing green" shape scripts/run_ratchets.py is
# about. Fail with the replacement named instead.
if(DEFINED LMP_MLX_PYTHON)
  message(FATAL_ERROR
    "LMP_MLX_PYTHON no longer exists: MLX is pinned and built from source rather than "
    "discovered from a pip package, so there is no interpreter to point at.\n"
    "  -DLMP_MLX_PYTHON=/usr/bin/false  ->  -DLMP_WITH_MLX=OFF   (what CI builds)\n"
    "  -DLMP_MLX_PYTHON=<venv>/python   ->  -DLMP_WITH_MLX=ON    (the dev preset)\n"
    "Remove it from this build directory's cache to continue.")
endif()

# DEFAULT OFF, and deliberately: this reproduces exactly what the venv probe used to do
# on every configuration EXCEPT the `dev` preset. The asan and mcp build dirs never had
# an interpreter with mlx installed, so the probe failed and they compiled the typed
# refusal -- and CI passed for that reason too, not by design. `dev` turns it ON; CI
# passes -DLMP_WITH_MLX=OFF to keep its 20-minute budget. Making it unconditional would
# make CI build every Metal kernel from source at -j3.
if(LMP_WITH_MLX)
  include(FetchContent)

  set(MLX_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
  set(MLX_BUILD_METAL ON CACHE BOOL "" FORCE)
  # JIT off means the kernels are precompiled into mlx.metallib rather than built from
  # source at runtime: no codegen inside a shipped extension, and no first-op stall.
  set(MLX_METAL_JIT OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(
    mlx
    GIT_REPOSITORY https://github.com/ml-explore/mlx.git
    GIT_TAG ${LMP_MLX_TAG}
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(mlx)

  # MLX's headers are not warning-clean under THIS project's regime, and they do not have
  # to be -- they are not ours. find_package(MLX) used to hand back an IMPORTED target
  # whose includes were already SYSTEM, so this never came up; FetchContent builds MLX as
  # a real subdirectory target instead, and its INTERFACE includes arrive unmarked. The
  # result is five -Werror failures inside mlx/allocator.h, mlx/types/bf16.h and
  # mlx/types/complex.h (-Wunused-parameter, -Wdeprecated-copy, -Wdouble-promotion)
  # before a line of our code is reached.
  #
  # Marking them SYSTEM restores exactly what find_package gave us. It is deliberately
  # not a pile of -Wno- flags on mlx_backend.cpp: the warnings are in MLX's headers, so
  # suppressing them per-TU would also blind us to the same mistakes in OUR code in the
  # same file, and would have to be repeated in every TU that includes MLX.
  get_target_property(_mlx_includes mlx INTERFACE_INCLUDE_DIRECTORIES)
  if(_mlx_includes)
    set_target_properties(mlx PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_mlx_includes}")
  endif()

  # Report whether the M5 accelerator kernels went in, by ASKING MLX rather than
  # re-deriving its condition: MLX defines MLX_METAL_NO_NAX on the target itself when it
  # takes the general path, and that property is the only answer that cannot drift from
  # what was actually compiled. (Its MLX_METAL_VERSION / MACOS_SDK_VERSION live in a
  # subdirectory scope and are not readable here, so a re-derived check would be a
  # different question wearing the same name.)
  #
  # This is a loud STATUS line and not a silent success because the failure mode is
  # invisible: a deployment target that drifts below 26.2 still builds, still passes every
  # test, and just runs several times slower. See the top-level CMakeLists for the
  # decision and what it costs.
  get_target_property(_mlx_defs mlx COMPILE_DEFINITIONS)
  if(_mlx_defs MATCHES "MLX_METAL_NO_NAX")
    message(STATUS "MLX: NAX kernels OFF (deployment target "
                   "${CMAKE_OSX_DEPLOYMENT_TARGET} < 26.2) -- matrix ops take the "
                   "general path. This is the slow configuration.")
  else()
    message(STATUS "MLX: NAX kernels ON (deployment target "
                   "${CMAKE_OSX_DEPLOYMENT_TARGET}) -- M5 matrix units in use, and "
                   "macOS ${CMAKE_OSX_DEPLOYMENT_TARGET} is this build's minimum.")
  endif()
endif()

if(TARGET mlx)
  message(STATUS "MLX: building ${LMP_MLX_TAG} from source (static, no interpreter)")
  target_compile_definitions(lmp_model PUBLIC LMP_HAVE_MLX=1)
  target_link_libraries(lmp_model PUBLIC mlx)
  # The vendored numerics (src/model/mlx/) are adapted v1 code kept close to their
  # debugged form for diffability; they predate this repo's -Wconversion regime. The
  # relaxation is scoped to the one TU that includes them.
  set_source_files_properties(mlx_backend.cpp PROPERTIES COMPILE_OPTIONS
    "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")

  # The metallib is not linked in -- it is opened at runtime, and MLX looks for it
  # COLOCATED WITH THE EXECUTABLE first (mlx/backend/metal/device.cpp). With the pip
  # dylib it sat next to that dylib in the venv and nobody had to think about it; with a
  # static archive there is no dylib, so every executable that touches the GPU needs its
  # own copy beside it. Missing => a runtime "failed to load the default metallib",
  # which reads like a driver problem rather than a packaging one.
  #
  # Exported so src/surface and tests/model can stage it; see lmp_stage_metallib below.
  set(LMP_MLX_METALLIB "${mlx_BINARY_DIR}/mlx/backend/metal/kernels/mlx.metallib"
      CACHE INTERNAL "Precompiled MLX Metal kernels, staged beside GPU executables")
else()
  message(STATUS "MLX backend OFF -- MlxBackend will refuse at load() with instructions. "
                 "Configure with -DLMP_WITH_MLX=ON to build it.")
endif()

# Stages mlx.metallib beside a target's binary. A no-op when MLX is off, so callers do
# not need their own guard.
function(lmp_stage_metallib target)
  if(NOT TARGET mlx)
    return()
  endif()
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${LMP_MLX_METALLIB}" "$<TARGET_FILE_DIR:${target}>/mlx.metallib"
    COMMENT "Staging mlx.metallib beside ${target}"
    VERBATIM)
endfunction()
