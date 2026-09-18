// Unit tests for MCP JSON-RPC message model, Id representation, Error, and LogLevel conversions.

#include "src/mcp/message.hpp"

#include <string>

#include "tests/check.hpp"

using namespace lmp::mcp;

TEST(id_default_and_none) {
    const Id id_default;
    CHECK(id_default.is_none());
    CHECK(!id_default.is_number());
    CHECK(!id_default.is_string());
    CHECK(id_default.to_json().is_null());
    CHECK_EQ(id_default.debug(), std::string("<none>"));

    const Id id_none = Id::none();
    CHECK(id_none.is_none());
    CHECK(id_default == id_none);
}

TEST(id_number_behavior) {
    const Id id = Id::number(42);
    CHECK(!id.is_none());
    CHECK(id.is_number());
    CHECK(!id.is_string());
    CHECK_EQ(id.as_number(), std::int64_t(42));
    CHECK_EQ(id.to_json().get<std::int64_t>(), std::int64_t(42));
    CHECK_EQ(id.debug(), std::string("42"));
}

TEST(id_string_behavior) {
    const Id id = Id::string("req-100");
    CHECK(!id.is_none());
    CHECK(!id.is_number());
    CHECK(id.is_string());
    CHECK_EQ(id.as_string(), std::string("req-100"));
    CHECK_EQ(id.to_json().get<std::string>(), std::string("req-100"));
    CHECK_EQ(id.debug(), std::string("\"req-100\""));
}

TEST(id_from_json_parsing) {
    const Id id_num = Id::from_json(nlohmann::json(123));
    CHECK(id_num.is_number());
    CHECK_EQ(id_num.as_number(), std::int64_t(123));

    const Id id_uint = Id::from_json(nlohmann::json(456u));
    CHECK(id_uint.is_number());
    CHECK_EQ(id_uint.as_number(), std::int64_t(456));

    const Id id_str = Id::from_json(nlohmann::json("hello"));
    CHECK(id_str.is_string());
    CHECK_EQ(id_str.as_string(), std::string("hello"));

    // Fractional numbers, objects, arrays, and nulls must yield none()
    CHECK(Id::from_json(nlohmann::json(12.34)).is_none());
    CHECK(Id::from_json(nlohmann::json::object()).is_none());
    CHECK(Id::from_json(nlohmann::json::array()).is_none());
    CHECK(Id::from_json(nullptr).is_none());
}

TEST(id_equality_and_hash) {
    const Id n1 = Id::number(1);
    const Id n1_again = Id::number(1);
    const Id n2 = Id::number(2);
    const Id s1 = Id::string("1");
    const Id s1_again = Id::string("1");
    const Id none_id = Id::none();

    CHECK(n1 == n1_again);
    CHECK(!(n1 == n2));
    CHECK(s1 == s1_again);
    CHECK(!(n1 == s1));
    CHECK(none_id == Id::none());

    const Id::Hash hasher;
    CHECK_EQ(hasher(n1), hasher(n1_again));
    CHECK_EQ(hasher(s1), hasher(s1_again));
    CHECK(hasher(n1) != hasher(s1));
    CHECK_EQ(hasher(none_id), std::size_t(0));
}

TEST(error_serialization) {
    Error err;
    err.code = -32600;
    err.message = "Invalid Request";

    nlohmann::json j_no_data = err.to_json();
    CHECK_EQ(j_no_data["code"].get<int>(), -32600);
    CHECK_EQ(j_no_data["message"].get<std::string>(), std::string("Invalid Request"));
    CHECK(!j_no_data.contains("data"));

    err.data = nlohmann::json{{"details", "missing field"}};
    nlohmann::json j_data = err.to_json();
    CHECK(j_data.contains("data"));
    CHECK_EQ(j_data["data"]["details"].get<std::string>(), std::string("missing field"));
}

TEST(classify_valid_and_invalid_messages) {
    // Valid Request
    const auto req = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"method":"ping","params":{}})"));
    CHECK(req.is_request());
    CHECK(!req.is_notification());
    CHECK(!req.is_response());
    CHECK(!req.is_invalid());
    CHECK_EQ(req.id.as_number(), std::int64_t(1));
    CHECK_EQ(req.method, std::string("ping"));
    CHECK_EQ(req.params.dump(), std::string("{}"));

    // Request with invalid id (e.g., float or null)
    const auto req_invalid_id = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1.5,"method":"ping"})"));
    CHECK(req_invalid_id.is_invalid());
    CHECK_EQ(req_invalid_id.invalid_reason, std::string("request id must be a string or an integer"));

    // Valid Notification
    const auto notif = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","method":"initialized"})"));
    CHECK(notif.is_notification());
    CHECK(notif.id.is_none());
    CHECK_EQ(notif.method, std::string("initialized"));

    // Response with result
    const auto resp_res = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":"abc","result":{"status":"ok"}})" ));
    CHECK(resp_res.is_response());
    CHECK_EQ(resp_res.id.as_string(), std::string("abc"));
    CHECK_EQ(resp_res.result["status"].get<std::string>(), std::string("ok"));
    CHECK(!resp_res.error.has_value());

    // Response with error (including data)
    const auto resp_err = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":5,"error":{"code":-32601,"message":"Method not found","data":{"info":"more"}}})" ));
    CHECK(resp_err.is_response());
    CHECK(resp_err.error.has_value());
    CHECK_EQ(resp_err.error->code, -32601);
    CHECK_EQ(resp_err.error->message, std::string("Method not found"));
    CHECK(resp_err.error->data.has_value());
    CHECK_EQ((*resp_err.error->data)["info"].get<std::string>(), std::string("more"));

    // Response with partial error object missing code or message
    const auto resp_partial_err = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":6,"error":{}})" ));
    CHECK(resp_partial_err.is_response());
    CHECK(resp_partial_err.error.has_value());
    CHECK_EQ(resp_partial_err.error->code, -32603); // default kInternalError
    CHECK(resp_partial_err.error->message.empty());

    // Response with non-object error
    const auto resp_bad_err = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":5,"error":"bad_error_fmt"})" ));
    CHECK(resp_bad_err.is_response());
    CHECK(resp_bad_err.error.has_value());
    CHECK_EQ(resp_bad_err.error->message, std::string("malformed error object"));

    // Missing jsonrpc key is accepted (for servers/peers that omit it)
    const auto req_no_jsonrpc = classify(nlohmann::json::parse(R"({"id":"req-1","method":"ping"})"));
    CHECK(req_no_jsonrpc.is_request());
    CHECK_EQ(req_no_jsonrpc.id.as_string(), std::string("req-1"));
    CHECK(req_no_jsonrpc.params.is_null());

    // Invalid message: non-object
    const auto inv_array = classify(nlohmann::json::parse(R"([1, 2, 3])"));
    CHECK(inv_array.is_invalid());
    CHECK_EQ(inv_array.invalid_reason, std::string("message is not a JSON object"));

    // Invalid message: jsonrpc version not "2.0"
    const auto inv_version = classify(nlohmann::json::parse(R"({"jsonrpc":"1.0","id":1,"method":"ping"})"));
    CHECK(inv_version.is_invalid());
    CHECK_EQ(inv_version.invalid_reason, std::string("jsonrpc field is not \"2.0\""));

    // Invalid message: carries both result and error
    const auto inv_both = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1,"result":{},"error":{"code":1,"message":"x"}})" ));
    CHECK(inv_both.is_invalid());
    CHECK_EQ(inv_both.invalid_reason, std::string("response carries both result and error"));

    // Invalid message: no method, result, or error
    const auto inv_empty = classify(nlohmann::json::parse(R"({"jsonrpc":"2.0","id":1})"));
    CHECK(inv_empty.is_invalid());
    CHECK_EQ(inv_empty.invalid_reason, std::string("message has neither method nor result nor error"));
}

TEST(outbound_message_helpers) {
    // make_request
    const auto req = make_request(Id::number(10), "tools/call", nlohmann::json{{"name", "test"}});
    CHECK_EQ(req["jsonrpc"].get<std::string>(), std::string("2.0"));
    CHECK_EQ(req["id"].get<std::int64_t>(), std::int64_t(10));
    CHECK_EQ(req["method"].get<std::string>(), std::string("tools/call"));
    CHECK_EQ(req["params"]["name"].get<std::string>(), std::string("test"));

    const auto req_no_params = make_request(Id::string("s1"), "ping", nullptr);
    CHECK(!req_no_params.contains("params"));

    // make_notification
    const auto notif = make_notification("notifications/cancelled", nullptr);
    CHECK(!notif.contains("id"));
    CHECK(!notif.contains("params"));
    CHECK_EQ(notif["method"].get<std::string>(), std::string("notifications/cancelled"));

    const auto notif_with_params = make_notification("notifications/progress", nlohmann::json{{"progress", 50}});
    CHECK(!notif_with_params.contains("id"));
    CHECK(notif_with_params.contains("params"));
    CHECK_EQ(notif_with_params["params"]["progress"].get<int>(), 50);

    // make_response
    const auto resp = make_response(Id::number(1), nlohmann::json{{"value", 42}});
    CHECK_EQ(resp["id"].get<std::int64_t>(), std::int64_t(1));
    CHECK_EQ(resp["result"]["value"].get<int>(), 42);

    const auto resp_null_res = make_response(Id::number(2), nullptr);
    CHECK(resp_null_res["result"].is_object());

    // make_error (by code & message)
    const auto err_msg1 = make_error(Id::number(3), ErrorCode::kMethodNotFound, "Not found");
    CHECK_EQ(err_msg1["error"]["code"].get<int>(), to_int(ErrorCode::kMethodNotFound));
    CHECK_EQ(err_msg1["error"]["message"].get<std::string>(), std::string("Not found"));

    const auto err_msg_data = make_error(Id::number(4), ErrorCode::kInvalidParams, "Invalid", nlohmann::json{{"field", "id"}});
    CHECK_EQ(err_msg_data["error"]["code"].get<int>(), to_int(ErrorCode::kInvalidParams));
    CHECK_EQ(err_msg_data["error"]["data"]["field"].get<std::string>(), std::string("id"));

    // make_error (by Error struct)
    Error custom_err;
    custom_err.code = -32000;
    custom_err.message = "Custom error";
    const auto err_msg2 = make_error(Id::string("e1"), custom_err);
    CHECK_EQ(err_msg2["id"].get<std::string>(), std::string("e1"));
    CHECK_EQ(err_msg2["error"]["code"].get<int>(), -32000);
}

TEST(log_level_conversions) {
    CHECK_EQ(to_string(LogLevel::kDebug), std::string_view("debug"));
    CHECK_EQ(to_string(LogLevel::kInfo), std::string_view("info"));
    CHECK_EQ(to_string(LogLevel::kNotice), std::string_view("notice"));
    CHECK_EQ(to_string(LogLevel::kWarning), std::string_view("warning"));
    CHECK_EQ(to_string(LogLevel::kError), std::string_view("error"));
    CHECK_EQ(to_string(LogLevel::kCritical), std::string_view("critical"));
    CHECK_EQ(to_string(LogLevel::kAlert), std::string_view("alert"));
    CHECK_EQ(to_string(LogLevel::kEmergency), std::string_view("emergency"));

    LogLevel level = LogLevel::kDebug;
    CHECK(parse_log_level("warning", level));
    CHECK(level == LogLevel::kWarning);

    CHECK(parse_log_level("emergency", level));
    CHECK(level == LogLevel::kEmergency);

    CHECK(!parse_log_level("invalid_level", level));
    CHECK(level == LogLevel::kEmergency); // unchanged on failure
}

TEST(test_mcp_message_check_framework_can_fail) {
    EXPECT_FAILING_CHECKS(1, {
        CHECK_EQ(to_string(LogLevel::kDebug), std::string_view("invalid"));
    });
}
