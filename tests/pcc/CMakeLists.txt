lmp_add_test(test_pcc_cas    SOURCES test_pcc_cas.cpp    LABELS gate LINK lmp_pcc TIMEOUT 60)
lmp_add_test(test_pcc_store  SOURCES test_pcc_store.cpp  LABELS gate LINK lmp_pcc TIMEOUT 60)
lmp_add_test(test_pcc_recall SOURCES test_pcc_recall.cpp LABELS gate LINK lmp_pcc TIMEOUT 60)

# The one that spans two layers: context compacts, pcc keeps what compaction dropped.
lmp_add_test(test_pcc_context SOURCES test_pcc_context.cpp LABELS gate
             LINK lmp_pcc lmp_context TIMEOUT 60)
