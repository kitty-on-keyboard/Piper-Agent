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
