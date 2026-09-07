#include "src/platform/fs.hpp"
#include "src/tools/think_blocks.hpp"

#include "tests/check.hpp"

using lmp::tools::extract_fenced_blocks;
using lmp::tools::kMaxThinkBlockChars;
using lmp::tools::kMaxThinkBlocksPerTurn;

TEST(one_fence_is_kept_with_exact_bytes) {
    const auto h = extract_fenced_blocks("```\nhello\n```");
    CHECK_EQ(h.blocks.size(), std::size_t{1});
    CHECK_EQ(h.blocks[0].block_id, 0);
    CHECK_EQ(h.blocks[0].language, std::string(""));
    CHECK_EQ(h.blocks[0].content, std::string("hello\n"));
    CHECK_EQ(h.blocks[0].sha256, lmp::platform::content_sha256_hex("hello\n"));
    CHECK_EQ(h.dropped_overflow, std::size_t{0});
}

TEST(lang_tag_is_stored_and_body_excludes_the_closing_line) {
    const auto h = extract_fenced_blocks("```cpp\nint x = 1;\n```\n");
    REQUIRE(h.blocks.size() == 1);
    CHECK_EQ(h.blocks[0].language, std::string("cpp"));
    CHECK_EQ(h.blocks[0].content, std::string("int x = 1;\n"));
}

TEST(multiple_closed_fences_keep_stable_ids) {
    const auto h = extract_fenced_blocks("```a\none\n```\ntext\n```b\ntwo\n```");
    REQUIRE(h.blocks.size() == 2);
    CHECK_EQ(h.blocks[0].block_id, 0);
    CHECK_EQ(h.blocks[0].language, std::string("a"));
    CHECK_EQ(h.blocks[0].content, std::string("one\n"));
    CHECK_EQ(h.blocks[1].block_id, 1);
    CHECK_EQ(h.blocks[1].language, std::string("b"));
    CHECK_EQ(h.blocks[1].content, std::string("two\n"));
}

TEST(unclosed_fence_is_dropped) {
    const auto h = extract_fenced_blocks("```py\nprint(1)\n");
    CHECK(h.blocks.empty());
}

TEST(empty_and_whitespace_bodies_are_dropped) {
    CHECK(extract_fenced_blocks("```\n```").blocks.empty());
    CHECK(extract_fenced_blocks("```txt\n   \n\t\n```").blocks.empty());
}

TEST(known_fixture_sha_is_stable) {
    CHECK_EQ(lmp::platform::content_sha256_hex(""),
             std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    const std::string body = "abc\n";
    const auto h = extract_fenced_blocks("```txt\n" + body + "```");
    REQUIRE(h.blocks.size() == 1);
    CHECK_EQ(h.blocks[0].content, body);
    CHECK_EQ(h.blocks[0].sha256, lmp::platform::content_sha256_hex(body));
}

TEST(inner_triple_backticks_on_their_own_line_close_the_fence) {
    const auto h = extract_fenced_blocks("```txt\nbefore\n```\nstill outside\n```");
    REQUIRE(h.blocks.size() == 1);
    CHECK_EQ(h.blocks[0].content, std::string("before\n"));
}

TEST(inline_backticks_do_not_close) {
    const auto h = extract_fenced_blocks("```txt\nuse `code` here\n```");
    REQUIRE(h.blocks.size() == 1);
    CHECK_EQ(h.blocks[0].content, std::string("use `code` here\n"));
}

TEST(overflow_past_three_is_dropped) {
    std::string t;
    for (int n = 0; n < 5; ++n) {
        t += "```\nblock";
        t += std::to_string(n);
        t += "\n```\n";
    }
    const auto h = extract_fenced_blocks(t);
    CHECK_EQ(h.blocks.size(), kMaxThinkBlocksPerTurn);
    CHECK_EQ(h.dropped_overflow, std::size_t{2});
    CHECK_EQ(h.blocks[0].content, std::string("block0\n"));
    CHECK_EQ(h.blocks[2].content, std::string("block2\n"));
}

TEST(a_too_large_closed_fence_is_dropped_without_repair) {
    std::string body(kMaxThinkBlockChars + 1, 'x');
    body.push_back('\n');
    const auto h = extract_fenced_blocks("```\n" + body + "```");
    CHECK(h.blocks.empty());
    CHECK_EQ(h.dropped_too_large, std::size_t{1});
}
