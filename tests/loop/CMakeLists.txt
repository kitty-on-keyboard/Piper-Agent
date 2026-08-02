lmp_add_test(test_loop SOURCES test_loop.cpp LABELS gate LINK lmp_loop)
lmp_add_test(test_verifier SOURCES test_verifier.cpp LABELS gate LINK lmp_loop TIMEOUT 120)

# Agent::step end to end IN THE GATE, on the miniature vocabulary. PHASES.md named
# agent.cpp:168 and :287 as surviving mutants with one shared blocker -- "not driven
# end-to-end in the gate, vocab-fixture blocker" -- so shipping the fixture without this
# would have cleared the blocker on paper only.
lmp_add_test(test_agent_step SOURCES test_agent_step.cpp LABELS gate LINK lmp_loop TIMEOUT 120)
add_dependencies(test_agent_step lmp_mini_vocab)
target_compile_definitions(test_agent_step PRIVATE
  LMP_MINI_VOCAB_JSON="${CMAKE_BINARY_DIR}/tests/model/mini_qwen/tokenizer.json")

# Needs the real vocab (944 byte-fragment tokens are the whole point), so realmodel, not
# gate -- same rule as tests/model.
lmp_add_test(test_token_stream SOURCES test_token_stream.cpp LABELS realmodel LINK lmp_loop TIMEOUT 300)

# Pure timing + ordering over the dispatch primitive: no tokenizer, no model, so gate.
lmp_add_test(test_parallel_calls SOURCES test_parallel_calls.cpp LABELS gate LINK lmp_loop TIMEOUT 60)
