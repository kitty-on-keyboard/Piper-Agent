# Spec S15.1: -Wall -Wextra -Wpedantic -Werror. Applied via an INTERFACE target so
# vendored third-party code can be excluded explicitly rather than by everyone
# remembering to opt out.

add_library(lmp_warnings INTERFACE)

target_compile_options(lmp_warnings INTERFACE
  -Wall
  -Wextra
  -Wpedantic
  -Werror
  # Narrower checks that catch the specific v1 defect classes:
  -Wshadow                # a shadowed name is how a "fixed" variable stays unfixed
  -Wconversion            # silent narrowing in token-id arithmetic
  -Wsign-conversion
  -Wold-style-cast
  -Wnon-virtual-dtor      # InferenceBackend is a virtual base (S5.12)
  -Wunused                # feeds the dead-code gate (S15.4) at TU scope
  -Wdouble-promotion
  -Wformat=2
)
