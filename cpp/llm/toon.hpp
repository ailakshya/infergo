#pragma once

#include <cstdint>
#include <string>

namespace infergo {

/// TOON — Token-Oriented Object Notation
///
/// Compact format for LLM structured output. 60-70% fewer tokens than JSON.
///
///   JSON:  {"name":"John","age":30,"tags":["go","rust"]}  = ~20 tokens
///   TOON:  name:John|age:30|tags:[go,rust]                = ~8 tokens
///
/// Fewer tokens = fewer GPU forward passes = faster output.

/// GBNF grammar for TOON — much simpler than JSON grammar.
/// Simpler grammar = fewer parse states = faster grammar checking.
extern const char* const TOON_GRAMMAR;

/// Parse TOON string to JSON string (C++, zero-copy where possible).
/// Returns JSON string. Empty string on parse error.
std::string toon_to_json(const char* toon, int len);

/// Parse JSON string to TOON string.
/// Returns TOON string. Empty string on parse error.
std::string json_to_toon(const char* json_str, int len);

} // namespace infergo
