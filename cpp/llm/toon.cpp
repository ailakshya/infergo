#include "toon.hpp"
#include <cstring>
#include <vector>

namespace infergo {

const char* const TOON_GRAMMAR = R"GBNF(
root   ::= pair ("|" pair)*
pair   ::= key ":" value
key    ::= [a-zA-Z_] [a-zA-Z0-9_]*
value  ::= object | array | atom
object ::= "(" pair ("|" pair)* ")"
array  ::= "[" value ("," value)* "]"
atom   ::= [^|,()\[\]\n]+
)GBNF";

// ── TOON → JSON converter (all in C++) ──────────────────────────────

static bool is_number(const char* s, int len) {
    if (len == 0) return false;
    int i = 0;
    if (s[0] == '-') i++;
    bool has_dot = false;
    for (; i < len; i++) {
        if (s[i] == '.') { if (has_dot) return false; has_dot = true; }
        else if (s[i] < '0' || s[i] > '9') return false;
    }
    return i > (s[0] == '-' ? 1 : 0);
}

static void append_json_string(std::string& out, const char* s, int len) {
    out += '"';
    for (int i = 0; i < len; i++) {
        if (s[i] == '"') out += "\\\"";
        else if (s[i] == '\\') out += "\\\\";
        else if (s[i] == '\n') out += "\\n";
        else out += s[i];
    }
    out += '"';
}

// Forward declaration
static void parse_value(const char* s, int len, std::string& out);

static int find_unescaped(const char* s, int len, char sep) {
    int depth = 0;
    for (int i = 0; i < len; i++) {
        if (s[i] == '(' || s[i] == '[') depth++;
        else if (s[i] == ')' || s[i] == ']') depth--;
        else if (s[i] == sep && depth == 0) return i;
    }
    return -1;
}

static void parse_pairs(const char* s, int len, std::string& out) {
    out += '{';
    bool first = true;
    int pos = 0;
    while (pos < len) {
        int pipe = find_unescaped(s + pos, len - pos, '|');
        int pair_len = (pipe >= 0) ? pipe : (len - pos);
        const char* pair = s + pos;

        int colon = find_unescaped(pair, pair_len, ':');
        if (colon > 0) {
            if (!first) out += ',';
            first = false;
            // Key
            append_json_string(out, pair, colon);
            out += ':';
            // Value
            parse_value(pair + colon + 1, pair_len - colon - 1, out);
        }
        pos += pair_len + 1;
    }
    out += '}';
}

static void parse_value(const char* s, int len, std::string& out) {
    // Trim whitespace
    while (len > 0 && (s[0] == ' ' || s[0] == '\t')) { s++; len--; }
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\t')) len--;

    if (len == 0) { out += "\"\""; return; }

    // Nested object: (key:val|key:val)
    if (s[0] == '(' && s[len-1] == ')') {
        parse_pairs(s + 1, len - 2, out);
        return;
    }

    // Array: [val,val,val]
    if (s[0] == '[' && s[len-1] == ']') {
        out += '[';
        bool first = true;
        int pos = 1;
        int end = len - 1;
        while (pos < end) {
            int comma = find_unescaped(s + pos, end - pos, ',');
            int item_len = (comma >= 0) ? comma : (end - pos);
            if (!first) out += ',';
            first = false;
            parse_value(s + pos, item_len, out);
            pos += item_len + 1;
        }
        out += ']';
        return;
    }

    // Literals
    if (len == 4 && std::memcmp(s, "true", 4) == 0) { out += "true"; return; }
    if (len == 5 && std::memcmp(s, "false", 5) == 0) { out += "false"; return; }
    if (len == 4 && std::memcmp(s, "null", 4) == 0) { out += "null"; return; }

    // Number
    if (is_number(s, len)) { out.append(s, static_cast<size_t>(len)); return; }

    // String (no quotes in TOON)
    append_json_string(out, s, len);
}

std::string toon_to_json(const char* toon, int len) {
    if (toon == nullptr || len <= 0) return "{}";
    std::string out;
    out.reserve(static_cast<size_t>(len) * 2);
    parse_pairs(toon, len, out);
    return out;
}

// ── JSON → TOON converter ───────────────────────────────────────────

static void json_value_to_toon(const char* s, int& pos, int len, std::string& out);

static void skip_ws(const char* s, int& pos, int len) {
    while (pos < len && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\t' || s[pos] == '\r')) pos++;
}

static std::string read_json_string(const char* s, int& pos, int len) {
    if (pos >= len || s[pos] != '"') return "";
    pos++; // skip "
    std::string result;
    while (pos < len && s[pos] != '"') {
        if (s[pos] == '\\' && pos + 1 < len) {
            if (s[pos+1] == '"') { result += '"'; pos += 2; }
            else if (s[pos+1] == 'n') { result += '\n'; pos += 2; }
            else { result += s[pos+1]; pos += 2; }
        } else {
            result += s[pos++];
        }
    }
    if (pos < len) pos++; // skip closing "
    return result;
}

static void json_value_to_toon(const char* s, int& pos, int len, std::string& out) {
    skip_ws(s, pos, len);
    if (pos >= len) return;

    if (s[pos] == '{') {
        // Object → (key:val|key:val)
        pos++;
        bool is_root = out.empty();
        if (!is_root) out += '(';
        bool first = true;
        while (pos < len) {
            skip_ws(s, pos, len);
            if (s[pos] == '}') { pos++; break; }
            if (s[pos] == ',') { pos++; continue; }
            std::string key = read_json_string(s, pos, len);
            skip_ws(s, pos, len);
            if (pos < len && s[pos] == ':') pos++;
            if (!first) out += '|';
            first = false;
            out += key;
            out += ':';
            json_value_to_toon(s, pos, len, out);
        }
        if (!is_root) out += ')';
    } else if (s[pos] == '[') {
        pos++;
        out += '[';
        bool first = true;
        while (pos < len) {
            skip_ws(s, pos, len);
            if (s[pos] == ']') { pos++; break; }
            if (s[pos] == ',') { pos++; continue; }
            if (!first) out += ',';
            first = false;
            json_value_to_toon(s, pos, len, out);
        }
        out += ']';
    } else if (s[pos] == '"') {
        out += read_json_string(s, pos, len);
    } else {
        // Number, bool, null
        int start = pos;
        while (pos < len && s[pos] != ',' && s[pos] != '}' && s[pos] != ']'
               && s[pos] != ' ' && s[pos] != '\n') pos++;
        out.append(s + start, static_cast<size_t>(pos - start));
    }
}

std::string json_to_toon(const char* json_str, int len) {
    if (json_str == nullptr || len <= 0) return "";
    std::string out;
    out.reserve(static_cast<size_t>(len));
    int pos = 0;
    json_value_to_toon(json_str, pos, len, out);
    return out;
}

} // namespace infergo
