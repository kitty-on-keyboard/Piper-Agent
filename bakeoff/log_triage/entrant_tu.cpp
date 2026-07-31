// One entrant, compiled alone, with its namespace renamed out of the way.
//
// See entrant_bridge.hpp for why this file exists. The rename is a preprocessor
// substitution of the NAMESPACE TOKEN only. An `#include "..."` directive's header-name is
// not macro-expanded, so an entrant that does include the canonical header still works and
// gets the canonical declarations placed in the private namespace too.
//
// ENTRANT_HEADER is vendored byte for byte. ADAPTER_HEADER is ours: it holds the call, and
// nothing else, for entrants whose signature is not the contract's.

#ifndef ENTRANT_HEADER
#error "ENTRANT_HEADER must be defined"
#endif
#ifndef ADAPTER_HEADER
#error "ADAPTER_HEADER must be defined"
#endif

#define log_triage lt_entrant
#include ENTRANT_HEADER
#undef log_triage

#include "entrant_bridge.hpp"
#include ADAPTER_HEADER
