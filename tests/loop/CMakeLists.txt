lmp_add_test(test_loop SOURCES test_loop.cpp LABELS gate LINK lmp_loop)
lmp_add_test(test_verifier SOURCES test_verifier.cpp LABELS gate LINK lmp_loop TIMEOUT 120)

# Needs the real vocab (944 byte-fragment tokens are the whole point), so realmodel, not
# gate -- same rule as tests/model.
lmp_add_test(test_token_stream SOURCES test_token_stream.cpp LABELS realmodel LINK lmp_loop TIMEOUT 300)

# Pure timing + ordering over the dispatch primitive: no tokenizer, no model, so gate.
lmp_add_test(test_parallel_calls SOURCES test_parallel_calls.cpp LABELS gate LINK lmp_loop TIMEOUT 60)
