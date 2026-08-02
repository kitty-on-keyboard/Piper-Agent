// The artifact store: deduplication, delta chains, and the bound on them.
//
// The cook-off this component came from (docs/BAKEOFF_PCC.md) had twelve entrants
// implement roughly this and not one of them tested the thing that actually breaks:
// what a delta chain costs after the fortieth revision. Two of the checks here exist
// specifically because no entrant had them.

#include <string>
#include <vector>

#include "src/pcc/cas.hpp"
#include "src/pcc/diff.hpp"
#include "tests/check.hpp"

namespace {

using namespace lmp::pcc;

// A plausible source file, so the compression numbers below are measured against
// something with real redundancy rather than against random bytes (which do not
// compress) or a repeated character (which compresses absurdly).
std::string source_file(int lines, int flavour) {
    std::string out;
    for (int i = 0; i < lines; ++i) {
        out += "    const std::string value_" + std::to_string(i) +
               " = compute(input, " + std::to_string(i * flavour) + ");\n";
    }
    return out;
}

Db open_db() {
    Db db(":memory:");
    Cas::migrate(db);
    return db;
}

TEST(cas_hashes_are_content_addressed) {
    // The published SHA-256 of the empty string. A hash function that is subtly wrong
    // still round-trips through its own store, so this is pinned against an external
    // constant rather than against ourselves.
    CHECK_EQ(sha256_hex(""),
             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK_EQ(sha256_hex("abc"),
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST(cas_deduplicates_identical_content) {
    Db db = open_db();
    Cas cas(db);
    const std::string content = source_file(50, 3);

    const std::string a = cas.put(content);
    const std::string b = cas.put(content);
    CHECK_EQ(a, b);
    CHECK_EQ(cas.stats().blobs, 1);

    // Even arriving with a base hint, which is the path that would otherwise write a
    // zero-length delta row alongside the original.
    const std::string c = cas.put(content, a);
    CHECK_EQ(c, a);
    CHECK_EQ(cas.stats().blobs, 1);
}

TEST(cas_round_trips_through_a_delta) {
    Db db = open_db();
    Cas cas(db);
    const std::string v1 = source_file(200, 3);
    std::string v2 = v1;
    v2 += "    // one appended line\n";

    const std::string h1 = cas.put(v1);
    const std::string h2 = cas.put(v2, h1);
    CHECK(h1 != h2);
    CHECK_EQ(cas.get(h1).value_or("<missing>"), v1);
    CHECK_EQ(cas.get(h2).value_or("<missing>"), v2);

    const BlobStats stats = cas.stats();
    CHECK_EQ(stats.blobs, 2);
    CHECK_EQ(stats.whole_blobs, 1);
    CHECK_EQ(stats.delta_blobs, 1);
}

TEST(cas_binary_content_survives_a_delta) {
    // The cook-off's other design stored deltas as diff-match-patch text patches, which
    // cannot represent this at all: entrant 6 caught the UnicodeDecodeError and silently
    // fell back to a whole copy, and entrant 3 decoded with errors='replace' and
    // corrupted the artifact. A dictionary-compressed delta is bytes in, bytes out.
    Db db = open_db();
    Cas cas(db);
    std::string blob;
    for (int i = 0; i < 4096; ++i) {
        blob += static_cast<char>(i * 7 % 256);
    }
    std::string changed = blob;
    changed[2000] = '\0';
    changed[2001] = static_cast<char>(0xFF);

    const std::string h1 = cas.put(blob);
    const std::string h2 = cas.put(changed, h1);
    CHECK_EQ(cas.get(h1).value_or("<missing>"), blob);
    CHECK_EQ(cas.get(h2).value_or("<missing>"), changed);
    CHECK(cas.get(h2)->size() == changed.size());
}

TEST(cas_bounds_the_delta_chain) {
    // THE test none of the twelve entrants had. Every one of them chained a delta onto
    // whatever base it was given, so a file edited forty times became a forty-link chain
    // and reading the newest revision meant reconstructing all forty.
    Db db = open_db();
    Cas cas(db);

    std::string content = source_file(120, 2);
    std::string base = cas.put(content);
    std::vector<std::string> hashes{base};
    for (int rev = 1; rev <= 40; ++rev) {
        content += "    // revision " + std::to_string(rev) + "\n";
        base = cas.put(content, base);
        hashes.push_back(base);
    }

    // Every revision still reads back exactly.
    std::string expected = source_file(120, 2);
    for (std::size_t i = 0; i < hashes.size(); ++i) {
        if (i > 0) {
            expected += "    // revision " + std::to_string(i) + "\n";
        }
        CHECK_EQ(cas.get(hashes[i]).value_or("<missing>"), expected);
    }

    // And the chains were re-based rather than allowed to grow. 41 revisions at a bound
    // of 8 means at least 41/9 whole copies.
    const BlobStats stats = cas.stats();
    CHECK_EQ(stats.blobs, 41);
    CHECK(stats.whole_blobs >= 41 / (kMaxChainDepth + 1));

    Stmt depth(db, "SELECT MAX(depth) FROM blob");
    REQUIRE(depth.step());
    CHECK(depth.column_int(0) <= kMaxChainDepth);
}

TEST(cas_delta_actually_saves_space) {
    // Measured, not asserted. A "delta" that quietly stores a whole copy would pass
    // every correctness check above.
    Db db = open_db();
    Cas cas(db);

    std::string content = source_file(400, 5);
    std::string base = cas.put(content);
    const std::int64_t after_first = cas.stats().stored_bytes;
    for (int rev = 0; rev < 8; ++rev) {
        content += "    // appended " + std::to_string(rev) + "\n";
        base = cas.put(content, base);
    }
    const BlobStats stats = cas.stats();
    const std::int64_t delta_cost = stats.stored_bytes - after_first;

    // Eight near-identical revisions of a ~20 KB file must not cost eight more copies.
    // The bar is deliberately loose -- this pins the ORDER, not a ratio that would
    // become a maintenance tax the first time zlib's tables change.
    CHECK(delta_cost < after_first * 2);
    CHECK(stats.logical_bytes > stats.stored_bytes * 3);
    std::fprintf(stderr,
                 "  [measured] 9 revisions of a %lld-byte file: %lld logical, %lld "
                 "stored (%.1fx), 8 deltas cost %lld bytes\n",
                 static_cast<long long>(content.size()),
                 static_cast<long long>(stats.logical_bytes),
                 static_cast<long long>(stats.stored_bytes),
                 static_cast<double>(stats.logical_bytes) /
                     static_cast<double>(stats.stored_bytes),
                 static_cast<long long>(delta_cost));
}

TEST(cas_reports_a_missing_blob_rather_than_guessing) {
    Db db = open_db();
    Cas cas(db);
    CHECK(!cas.get(std::string(64, 'a')).has_value());
    CHECK(!cas.contains(std::string(64, 'a')));
    CHECK(!cas.diff(std::string(64, 'a'), std::string(64, 'b')).has_value());
}

TEST(cas_detects_a_corrupted_blob) {
    // The integrity check earns its keep only if it can fire. Corrupt a stored row
    // behind the store's back and the next read must refuse rather than hand back
    // plausible bytes.
    Db db = open_db();
    Cas cas(db);
    const std::string hash = cas.put(source_file(20, 1));

    Stmt tamper(db, "UPDATE blob SET size = size + 1 WHERE hash = ?");
    tamper.bind(1, hash);
    tamper.run();

    bool threw = false;
    try {
        (void)cas.get(hash);
    } catch (const SqlError&) {
        threw = true;
    }
    CHECK(threw);
}

TEST(diff_is_empty_for_identical_content) {
    Db db = open_db();
    Cas cas(db);
    const std::string hash = cas.put(source_file(10, 1));
    CHECK_EQ(cas.diff(hash, hash).value_or("<missing>"), std::string());
}

TEST(diff_shows_the_changed_lines_only) {
    const std::string before = "alpha\nbravo\ncharlie\ndelta\necho\nfoxtrot\ngolf\n";
    const std::string after = "alpha\nbravo\nCHARLIE\ndelta\necho\nfoxtrot\ngolf\n";
    const std::string out = unified_diff(before, after, "a", "b", 1);

    CHECK(out.find("-charlie") != std::string::npos);
    CHECK(out.find("+CHARLIE") != std::string::npos);
    // Context is one line either side, so the far end of the file must not appear.
    CHECK(out.find("golf") == std::string::npos);
    CHECK(out.find("@@") != std::string::npos);
}

TEST(diff_handles_pure_insertion_and_deletion) {
    CHECK(unified_diff("", "one\ntwo\n", "a", "b").find("+one") != std::string::npos);
    CHECK(unified_diff("one\ntwo\n", "", "a", "b").find("-two") != std::string::npos);

    const std::vector<std::string> lines = split_lines("a\nb\nc");
    CHECK_EQ(lines.size(), std::size_t{3});
    // A trailing newline must not manufacture an empty final line.
    CHECK_EQ(split_lines("a\nb\n").size(), std::size_t{2});
}

TEST(check_framework_can_still_fail_here) {
    // S2.1.2: this file's greens count only if a red is reachable in it.
    EXPECT_FAILING_CHECKS(2, {
        CHECK(sha256_hex("abc") == sha256_hex("abd"));
        CHECK_EQ(split_lines("a\nb\n").size(), std::size_t{99});
    });
}

} // namespace
