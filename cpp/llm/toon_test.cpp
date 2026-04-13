#include "toon.hpp"
#include <gtest/gtest.h>
#include <cstring>

using namespace infergo;

// ── toon_to_json ────────────────────────────────────────────────────────────

TEST(TOON, SimpleKV) {
    const char* t = "name:John|age:30";
    auto j = toon_to_json(t, static_cast<int>(std::strlen(t)));
    EXPECT_EQ(j, R"({"name":"John","age":30})");
}

TEST(TOON, BoolAndNull) {
    const char* t = "a:true|b:false|c:null";
    auto j = toon_to_json(t, static_cast<int>(std::strlen(t)));
    EXPECT_EQ(j, R"({"a":true,"b":false,"c":null})");
}

TEST(TOON, NestedObject) {
    const char* t = "user:(name:Alice|age:25)";
    auto j = toon_to_json(t, static_cast<int>(std::strlen(t)));
    EXPECT_EQ(j, R"({"user":{"name":"Alice","age":25}})");
}

TEST(TOON, Array) {
    const char* t = "tags:[go,rust,python]";
    auto j = toon_to_json(t, static_cast<int>(std::strlen(t)));
    EXPECT_EQ(j, R"({"tags":["go","rust","python"]})");
}

TEST(TOON, Complex) {
    const char* t = "name:John|age:30|tags:[go,rust]|addr:(city:NYC|zip:10001)";
    auto j = toon_to_json(t, static_cast<int>(std::strlen(t)));
    EXPECT_EQ(j, R"({"name":"John","age":30,"tags":["go","rust"],"addr":{"city":"NYC","zip":10001}})");
}

TEST(TOON, Empty) {
    EXPECT_EQ(toon_to_json(nullptr, 0), "{}");
    EXPECT_EQ(toon_to_json("", 0), "{}");
}

TEST(TOON, NumberFloat) {
    const char* t = "pi:3.14|neg:-42";
    auto j = toon_to_json(t, static_cast<int>(std::strlen(t)));
    EXPECT_EQ(j, R"({"pi":3.14,"neg":-42})");
}

// ── json_to_toon ────────────────────────────────────────────────────────────

TEST(TOON, JsonToToonSimple) {
    const char* j = R"({"name":"John","age":30})";
    auto t = json_to_toon(j, static_cast<int>(std::strlen(j)));
    EXPECT_EQ(t, "name:John|age:30");
}

TEST(TOON, JsonToToonNested) {
    const char* j = R"({"user":{"name":"Alice","age":25}})";
    auto t = json_to_toon(j, static_cast<int>(std::strlen(j)));
    EXPECT_EQ(t, "user:(name:Alice|age:25)");
}

TEST(TOON, JsonToToonArray) {
    const char* j = R"({"tags":["go","rust"]})";
    auto t = json_to_toon(j, static_cast<int>(std::strlen(j)));
    EXPECT_EQ(t, "tags:[go,rust]");
}

TEST(TOON, JsonToToonEmpty) {
    EXPECT_EQ(json_to_toon(nullptr, 0), "");
    EXPECT_EQ(json_to_toon("", 0), "");
}

// ── Roundtrip ───────────────────────────────────────────────────────────────

TEST(TOON, RoundtripToonJsonToon) {
    const char* original = "name:John|age:30|active:true";
    auto json = toon_to_json(original, static_cast<int>(std::strlen(original)));
    auto back = json_to_toon(json.c_str(), static_cast<int>(json.size()));
    EXPECT_EQ(back, original);
}

// ── Grammar ─────────────────────────────────────────────────────────────────

TEST(TOON, GrammarNotNull) {
    EXPECT_NE(TOON_GRAMMAR, nullptr);
    EXPECT_GT(std::strlen(TOON_GRAMMAR), 10u);
}

// ── C API ───────────────────────────────────────────────────────────────────

#include "infer_api.h"

TEST(TOON, CApiGrammar) {
    const char* g = infer_toon_grammar();
    EXPECT_NE(g, nullptr);
    EXPECT_NE(std::strstr(g, "root"), nullptr);
}

TEST(TOON, CApiToonToJson) {
    const char* t = "name:John|age:30";
    char buf[256];
    int n = infer_toon_to_json(t, static_cast<int>(std::strlen(t)), buf, 256);
    EXPECT_GT(n, 0);
    EXPECT_STREQ(buf, R"({"name":"John","age":30})");
}

TEST(TOON, CApiJsonToToon) {
    const char* j = R"({"name":"John","age":30})";
    char buf[256];
    int n = infer_json_to_toon(j, static_cast<int>(std::strlen(j)), buf, 256);
    EXPECT_GT(n, 0);
    EXPECT_STREQ(buf, "name:John|age:30");
}

TEST(TOON, CApiNullSafe) {
    EXPECT_EQ(infer_toon_to_json(nullptr, 0, nullptr, 0), -1);
    EXPECT_EQ(infer_json_to_toon(nullptr, 0, nullptr, 0), -1);
}
