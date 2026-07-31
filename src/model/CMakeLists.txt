add_library(lmp_model
  backend.cpp
  chat_template.cpp
  grammar.cpp
  kv_cache.cpp
  mlx_backend.cpp
  qwen_tokenizer.cpp
  sampler.cpp
)
target_include_directories(lmp_model PUBLIC "${PROJECT_SOURCE_DIR}")
target_link_libraries(lmp_model
  PRIVATE lmp_warnings lmp_assertions simdjson
  PUBLIC lmp_platform frankentok parsephony)

# --- MLX discovery ----------------------------------------------------------
#
# The MLX C++ SDK ships inside the mlx pip package (headers, libmlx.dylib,
# mlx.metallib, MLXConfig.cmake). `<python> -m mlx --cmake-dir` prints its CMake
# package dir -- the same discovery v1 used. Not found => MlxBackend compiles as a
# typed refusal and everything else still builds and tests (gate is model-free).
#
# WHICH python matters, and it is not a preference. This machine's `python3` is
# /usr/bin/python3, which is 3.9, and MLX stopped shipping cp39 wheels after 0.29.3 --
# so the default probe silently pins the whole project to an MLX that is two minor
# versions behind the one LM Studio runs (0.31.2). Point this at an interpreter with a
# current mlx installed:
#
#   python3.12 -m venv ~/.venvs/lmp-mlx && ~/.venvs/lmp-mlx/bin/pip install mlx==0.31.2
#   cmake --preset dev -DLMP_MLX_PYTHON=$HOME/.venvs/lmp-mlx/bin/python
#
# The configure output prints the version it resolved; if you are chasing a
# performance number, read that line before believing anything else.
set(LMP_MLX_PYTHON "python3" CACHE STRING
    "Interpreter whose installed mlx package provides the C++ SDK")

execute_process(
  COMMAND "${LMP_MLX_PYTHON}" -m mlx --cmake-dir
  OUTPUT_VARIABLE _mlx_cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
  RESULT_VARIABLE _mlx_probe_rc)

if(_mlx_probe_rc EQUAL 0 AND EXISTS "${_mlx_cmake_dir}")
  # find_package(CONFIG) resolves via the cached MLX_DIR and ignores PATHS once that
  # cache entry exists. Changing LMP_MLX_PYTHON on an existing build dir therefore kept
  # linking the *previous* interpreter's MLX -- silently, because the version reported
  # below used to come from the interpreter rather than from the package that was found.
  # That combination pinned a build to 0.29.3 while printing 0.31.2. Re-point MLX_DIR
  # whenever the probe disagrees with it.
  if(DEFINED MLX_DIR AND NOT MLX_DIR STREQUAL "${_mlx_cmake_dir}")
    message(STATUS "MLX: LMP_MLX_PYTHON moved MLX_DIR ${MLX_DIR} -> ${_mlx_cmake_dir}")
    set(MLX_DIR "${_mlx_cmake_dir}" CACHE PATH
        "The directory containing a CMake configuration file for MLX." FORCE)
    # MLXConfig.cmake calls find_library(MLX_LIBRARY ...), which is a no-op once that
    # cache entry exists, so it would keep naming the old dylib in the "Found MLX" line.
    unset(MLX_LIBRARY CACHE)
  endif()
  find_package(MLX CONFIG PATHS "${_mlx_cmake_dir}" NO_DEFAULT_PATH)
endif()

if(MLX_FOUND)
  # MLX_VERSION comes from the MLXConfigVersion.cmake that was actually found, so it
  # describes the library that will be linked. Do not substitute the interpreter's
  # mx.__version__ here: the two can disagree, and when they do this line is the only
  # warning that a performance number belongs to a different MLX than you think.
  message(STATUS "MLX C++ SDK found: ${_mlx_cmake_dir} (version ${MLX_VERSION})")
  target_compile_definitions(lmp_model PUBLIC LMP_HAVE_MLX=1)
  target_link_libraries(lmp_model PUBLIC mlx)
  # The vendored numerics (src/model/mlx/) are adapted v1 code kept close to their
  # debugged form for diffability; they predate this repo's -Wconversion regime. The
  # relaxation is scoped to the one TU that includes them.
  set_source_files_properties(mlx_backend.cpp PROPERTIES COMPILE_OPTIONS
    "-Wno-conversion;-Wno-sign-conversion;-Wno-old-style-cast;-Wno-shadow")
else()
  message(STATUS "MLX C++ SDK not found -- MlxBackend will refuse at load() with instructions")
endif()
