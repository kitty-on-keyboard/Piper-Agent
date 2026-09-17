#include <string>
#include "parsephony/parsephony.hpp"
#include "tests/check.hpp"

TEST(parsephony_small_object_key_lookup) {
    std::string json = R"({"name": "alice", "age": 30, "city": "paris"})";
    parsephony::Document doc;
    CHECK(parsephony::parse(json, doc) == parsephony::Error::Ok);
    parsephony::Value root = doc.root();
    REQUIRE(root.is_object());
    CHECK(!doc.node(0).has_duplicate_keys());

    CHECK_EQ(root["name"].get_string(), "alice");
    CHECK_EQ(root["age"].get_int().value_or(0), static_cast<int64_t>(30));
    CHECK_EQ(root["city"].get_string(), "paris");
    CHECK(!root["nonexistent"].valid());
}

TEST(parsephony_large_object_key_lookup_without_duplicate_keys) {
    std::string json = "{";
    for (int i = 0; i < 100; ++i) {
        if (i > 0) json += ",";
        json += "\"key_" + std::to_string(i) + "\":" + std::to_string(i * 10);
    }
    json += "}";

    parsephony::Document doc;
    CHECK(parsephony::parse(json, doc) == parsephony::Error::Ok);
    parsephony::Value root = doc.root();
    REQUIRE(root.is_object());
    // Large objects without duplicate keys must NOT be marked with kDuplicateKeys
    CHECK(!doc.node(0).has_duplicate_keys());

    // Verify first, middle, last and non-existent key lookups
    CHECK_EQ(root["key_0"].get_int().value_or(0), static_cast<int64_t>(0));
    CHECK_EQ(root["key_50"].get_int().value_or(0), static_cast<int64_t>(500));
    CHECK_EQ(root["key_99"].get_int().value_or(0), static_cast<int64_t>(990));
    CHECK(!root["key_100"].valid());
    CHECK(!root["k"].valid());
}

TEST(parsephony_duplicate_key_resolution) {
    std::string json = R"({"a": 1, "b": 2, "a": 3, "c": 4, "a": 5})";
    parsephony::Document doc;
    CHECK(parsephony::parse(json, doc) == parsephony::Error::Ok);
    parsephony::Value root = doc.root();
    REQUIRE(root.is_object());
    // Duplicate keys detected -> flag must be set
    CHECK(doc.node(0).has_duplicate_keys());

    // Duplicate keys resolve to the last occurrence
    CHECK_EQ(root["a"].get_int().value_or(0), static_cast<int64_t>(5));
    CHECK_EQ(root["b"].get_int().value_or(0), static_cast<int64_t>(2));
    CHECK_EQ(root["c"].get_int().value_or(0), static_cast<int64_t>(4));
    CHECK(!root["d"].valid());
}

TEST(parsephony_flat_array_indexing) {
    std::string json = R"([10, 20, 30, 40, 50])";
    parsephony::Document doc;
    CHECK(parsephony::parse(json, doc) == parsephony::Error::Ok);
    parsephony::Value root = doc.root();
    REQUIRE(root.is_array());
    CHECK_EQ(root.size(), static_cast<size_t>(5));

    CHECK_EQ(root[0].get_int().value_or(0), static_cast<int64_t>(10));
    CHECK_EQ(root[2].get_int().value_or(0), static_cast<int64_t>(30));
    CHECK_EQ(root[4].get_int().value_or(0), static_cast<int64_t>(50));
    CHECK(!root[5].valid());
}

TEST(parsephony_nonflat_array_indexing) {
    std::string json = R"([{"a": 1}, {"b": 2}, [10, 20], "string_val", 42])";
    parsephony::Document doc;
    CHECK(parsephony::parse(json, doc) == parsephony::Error::Ok);
    parsephony::Value root = doc.root();
    REQUIRE(root.is_array());
    CHECK_EQ(root.size(), static_cast<size_t>(5));

    CHECK_EQ(root[0]["a"].get_int().value_or(0), static_cast<int64_t>(1));
    CHECK_EQ(root[1]["b"].get_int().value_or(0), static_cast<int64_t>(2));
    CHECK_EQ(root[2][1].get_int().value_or(0), static_cast<int64_t>(20));
    CHECK_EQ(root[3].get_string(), "string_val");
    CHECK_EQ(root[4].get_int().value_or(0), static_cast<int64_t>(42));
    CHECK(!root[5].valid());

    // Random / reverse access
    CHECK_EQ(root[4].get_int().value_or(0), static_cast<int64_t>(42));
    CHECK_EQ(root[0]["a"].get_int().value_or(0), static_cast<int64_t>(1));
}

TEST(parsephony_multi_array_and_clear_reuse) {
    std::string json = "[";
    for (int i = 0; i < 12; ++i) {
        if (i > 0) json += ",";
        json += "[{\"val\":" + std::to_string(i) + "}]";
    }
    json += "]";

    parsephony::Document doc;
    CHECK(parsephony::parse(json, doc) == parsephony::Error::Ok);
    parsephony::Value root = doc.root();
    REQUIRE(root.is_array());
    CHECK_EQ(root.size(), static_cast<size_t>(12));

    for (size_t i = 0; i < 12; ++i) {
        CHECK_EQ(root[i][0]["val"].get_int().value_or(-1), static_cast<int64_t>(i));
    }

    // Clear and reuse document
    std::string new_json = R"([{"x": 100}, {"y": 200}])";
    CHECK(parsephony::parse(new_json, doc) == parsephony::Error::Ok);
    parsephony::Value new_root = doc.root();
    REQUIRE(new_root.is_array());
    CHECK_EQ(new_root.size(), static_cast<size_t>(2));
    CHECK_EQ(new_root[0]["x"].get_int().value_or(0), static_cast<int64_t>(100));
    CHECK_EQ(new_root[1]["y"].get_int().value_or(0), static_cast<int64_t>(200));
}

TEST(parsephony_document_move_semantics) {
    std::string json = R"([{"a": 10}, {"b": 20}])";
    parsephony::Document doc1;
    CHECK(parsephony::parse(json, doc1) == parsephony::Error::Ok);

    parsephony::Document doc2 = std::move(doc1);
    CHECK_EQ(doc2.root().size(), static_cast<size_t>(2));
    CHECK_EQ(doc2.root()[0]["a"].get_int().value_or(0), static_cast<int64_t>(10));

    parsephony::Document doc3;
    doc3 = std::move(doc2);
    CHECK_EQ(doc3.root().size(), static_cast<size_t>(2));
    CHECK_EQ(doc3.root()[1]["b"].get_int().value_or(0), static_cast<int64_t>(20));
}
