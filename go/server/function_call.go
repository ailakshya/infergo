package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

// ─── Grammar generation ─────────────────────────────────────────────────────

// GenerateFunctionCallGrammar builds a GBNF grammar from a list of tools.
// The grammar constrains LLM output to produce valid JSON matching:
//
//	{"name": "<tool_name>", "arguments": {<valid_args>}}
//
// If toolChoice is a specific function name, only that function is allowed.
func GenerateFunctionCallGrammar(tools []Tool, toolChoice string) string {
	if len(tools) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.Grow(2048)

	// Filter tools if a specific function is requested
	var activeFuncs []ToolFunction
	for _, t := range tools {
		if t.Type != "function" {
			continue
		}
		if toolChoice != "" && t.Function.Name != toolChoice {
			continue
		}
		activeFuncs = append(activeFuncs, t.Function)
	}
	if len(activeFuncs) == 0 {
		return ""
	}

	// Root rule: one of the function call alternatives
	if len(activeFuncs) == 1 {
		sb.WriteString("root ::= call-")
		sb.WriteString(sanitizeRuleName(activeFuncs[0].Name))
		sb.WriteByte('\n')
	} else {
		sb.WriteString("root ::= ")
		for i, f := range activeFuncs {
			if i > 0 {
				sb.WriteString(" | ")
			}
			sb.WriteString("call-")
			sb.WriteString(sanitizeRuleName(f.Name))
		}
		sb.WriteByte('\n')
	}

	// Each function gets its own call rule
	for _, f := range activeFuncs {
		name := sanitizeRuleName(f.Name)
		sb.WriteString("\ncall-")
		sb.WriteString(name)
		sb.WriteString(` ::= "{\"name\": \"`)
		sb.WriteString(f.Name)
		sb.WriteString(`\", \"arguments\": " args-`)
		sb.WriteString(name)
		sb.WriteString(` "}"`)
		sb.WriteByte('\n')

		// Generate argument rules from JSON Schema parameters
		argsGrammar := generateArgsGrammar(name, f.Parameters)
		sb.WriteString(argsGrammar)
	}

	// Common primitives
	sb.WriteString(functionCallPrimitives)

	return sb.String()
}

// generateArgsGrammar produces GBNF rules for function arguments based
// on the JSON Schema in the parameters field.
func generateArgsGrammar(funcName string, params interface{}) string {
	if params == nil {
		return fmt.Sprintf("args-%s ::= \"{}\"\n", funcName)
	}

	// Convert to map
	schema, ok := normalizeSchema(params)
	if !ok {
		return fmt.Sprintf("args-%s ::= object\n", funcName)
	}

	properties, _ := schema["properties"].(map[string]interface{})
	if properties == nil {
		return fmt.Sprintf("args-%s ::= object\n", funcName)
	}

	requiredSet := make(map[string]bool)
	if req, ok := schema["required"].([]interface{}); ok {
		for _, r := range req {
			if s, ok := r.(string); ok {
				requiredSet[s] = true
			}
		}
	}

	// Sort property names for deterministic grammar
	propNames := make([]string, 0, len(properties))
	for name := range properties {
		propNames = append(propNames, name)
	}
	sort.Strings(propNames)

	var sb strings.Builder

	// args-funcname rule: object with known properties
	sb.WriteString(fmt.Sprintf("args-%s ::= \"{\" ws ", funcName))

	first := true
	for _, pName := range propNames {
		ruleName := fmt.Sprintf("%s-%s", funcName, sanitizeRuleName(pName))

		if first {
			first = false
		} else {
			sb.WriteString(` "," ws `)
		}

		if requiredSet[pName] {
			// Required: always present
			sb.WriteString(fmt.Sprintf(`"\"%s\": " %s`, pName, ruleName))
		} else {
			// Optional: may or may not appear
			sb.WriteString(fmt.Sprintf(`("\"%s\": " %s)?`, pName, ruleName))
		}
	}

	sb.WriteString(` ws "}"`)
	sb.WriteByte('\n')

	// Generate type-specific rules for each property
	for _, pName := range propNames {
		propSchema, _ := properties[pName].(map[string]interface{})
		ruleName := fmt.Sprintf("%s-%s", funcName, sanitizeRuleName(pName))
		sb.WriteString(generateTypeRule(ruleName, propSchema))
	}

	return sb.String()
}

// generateTypeRule creates a GBNF rule for a single property based on its JSON Schema type.
func generateTypeRule(ruleName string, schema map[string]interface{}) string {
	if schema == nil {
		return fmt.Sprintf("%s ::= value\n", ruleName)
	}

	typ, _ := schema["type"].(string)

	// Handle enum constraint
	if enumVals, ok := schema["enum"].([]interface{}); ok && len(enumVals) > 0 {
		var parts []string
		for _, v := range enumVals {
			switch ev := v.(type) {
			case string:
				parts = append(parts, fmt.Sprintf(`"\"" "%s" "\""`, ev))
			case float64:
				parts = append(parts, fmt.Sprintf(`"%v"`, ev))
			default:
				b, _ := json.Marshal(v)
				parts = append(parts, fmt.Sprintf(`"%s"`, string(b)))
			}
		}
		return fmt.Sprintf("%s ::= %s\n", ruleName, strings.Join(parts, " | "))
	}

	switch typ {
	case "string":
		return fmt.Sprintf("%s ::= string\n", ruleName)
	case "number":
		return fmt.Sprintf("%s ::= number\n", ruleName)
	case "integer":
		return fmt.Sprintf("%s ::= integer\n", ruleName)
	case "boolean":
		return fmt.Sprintf("%s ::= boolean\n", ruleName)
	case "array":
		return fmt.Sprintf("%s ::= array\n", ruleName)
	case "object":
		return fmt.Sprintf("%s ::= object\n", ruleName)
	default:
		return fmt.Sprintf("%s ::= value\n", ruleName)
	}
}

// normalizeSchema converts the parameters field (which may be a raw
// map[string]interface{} or json.RawMessage) into a usable map.
func normalizeSchema(params interface{}) (map[string]interface{}, bool) {
	switch v := params.(type) {
	case map[string]interface{}:
		return v, true
	case json.RawMessage:
		var m map[string]interface{}
		if err := json.Unmarshal(v, &m); err != nil {
			return nil, false
		}
		return m, true
	default:
		// Try marshaling and unmarshaling (handles struct types)
		b, err := json.Marshal(v)
		if err != nil {
			return nil, false
		}
		var m map[string]interface{}
		if err := json.Unmarshal(b, &m); err != nil {
			return nil, false
		}
		return m, true
	}
}

// sanitizeRuleName converts a function/property name into a valid GBNF rule name.
// GBNF rule names allow [a-zA-Z0-9_-].
func sanitizeRuleName(name string) string {
	var sb strings.Builder
	sb.Grow(len(name))
	for _, c := range name {
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			sb.WriteRune(c)
		} else {
			sb.WriteByte('-')
		}
	}
	return sb.String()
}

// functionCallPrimitives are shared GBNF rules used by function call grammars.
const functionCallPrimitives = `
ws ::= ([ \t\n] ws)?

string ::=
  "\"" (
    [^\\"\x7F\x00-\x1F] |
    "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\""

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (([eE] [-+]? [0-9]+))?

integer ::= "-"? ([0-9] | [1-9] [0-9]*)

boolean ::= "true" | "false"

null ::= "null"

value ::= string | number | boolean | null | object | array

object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
`

// ─── Response parsing ────────────────────────────────────────────────────────

// FunctionCallJSON is the structure we expect the LLM to produce
// when constrained by a function call grammar.
type FunctionCallJSON struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ParseFunctionCallResponse attempts to parse LLM output as a function call.
// Returns the parsed ToolCall and true if successful, or zero value and false
// if the output is not a valid function call JSON.
func ParseFunctionCallResponse(text string) (ToolCall, bool) {
	text = strings.TrimSpace(text)
	if len(text) == 0 || text[0] != '{' {
		return ToolCall{}, false
	}

	var fc FunctionCallJSON
	if err := json.Unmarshal([]byte(text), &fc); err != nil {
		return ToolCall{}, false
	}

	// Must have a name field
	if fc.Name == "" {
		return ToolCall{}, false
	}

	// Normalize arguments: if absent, use "{}"
	argsStr := "{}"
	if len(fc.Arguments) > 0 {
		// Validate it's valid JSON
		if !json.Valid(fc.Arguments) {
			return ToolCall{}, false
		}
		argsStr = string(fc.Arguments)
	}

	return ToolCall{
		ID:   fastID("call"),
		Type: "function",
		Function: ToolCallFunction{
			Name:      fc.Name,
			Arguments: argsStr,
		},
	}, true
}

// ─── Tool choice resolution ─────────────────────────────────────────────────

// ResolvedToolChoice represents the parsed tool_choice value.
type ResolvedToolChoice struct {
	Mode         string // "none", "auto", "required", "function"
	FunctionName string // only set when Mode == "function"
}

// ResolveToolChoice parses the tool_choice field from a request.
// It accepts string values ("none", "auto", "required") or an object
// {"type": "function", "function": {"name": "..."}}.
func ResolveToolChoice(choice interface{}) ResolvedToolChoice {
	if choice == nil {
		return ResolvedToolChoice{Mode: "auto"}
	}

	switch v := choice.(type) {
	case string:
		switch v {
		case "none":
			return ResolvedToolChoice{Mode: "none"}
		case "required":
			return ResolvedToolChoice{Mode: "required"}
		default:
			return ResolvedToolChoice{Mode: "auto"}
		}
	case map[string]interface{}:
		// {"type": "function", "function": {"name": "get_weather"}}
		fn, _ := v["function"].(map[string]interface{})
		if fn != nil {
			if name, ok := fn["name"].(string); ok && name != "" {
				return ResolvedToolChoice{Mode: "function", FunctionName: name}
			}
		}
		return ResolvedToolChoice{Mode: "auto"}
	default:
		return ResolvedToolChoice{Mode: "auto"}
	}
}

// ─── Response writing ────────────────────────────────────────────────────────

// writeChatCompletionToolCall writes a ChatCompletionResponse with tool_calls
// instead of content, using the fast buffer pool.
func writeChatCompletionToolCall(w http.ResponseWriter, id string, model string, toolCalls []ToolCall, promptToks, genToks int) {
	bp := jsonBufPool.Get().(*[]byte)
	b := (*bp)[:0]

	b = append(b, `{"id":"`...)
	b = append(b, id...)
	b = append(b, `","object":"chat.completion","created":`...)
	b = strconv.AppendInt(b, time.Now().Unix(), 10)
	b = append(b, `,"model":"`...)
	b = append(b, model...)
	b = append(b, `","choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[`...)

	for i, tc := range toolCalls {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, `{"id":"`...)
		b = append(b, tc.ID...)
		b = append(b, `","type":"function","function":{"name":"`...)
		b = append(b, tc.Function.Name...)
		b = append(b, `","arguments":`...)
		b = appendJSONString(b, tc.Function.Arguments)
		b = append(b, `}}`...)
	}

	b = append(b, `]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":`...)
	b = strconv.AppendInt(b, int64(promptToks), 10)
	b = append(b, `,"completion_tokens":`...)
	b = strconv.AppendInt(b, int64(genToks), 10)
	b = append(b, `,"total_tokens":`...)
	b = strconv.AppendInt(b, int64(promptToks+genToks), 10)
	b = append(b, `}}`...)
	b = append(b, '\n')

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusOK)
	w.Write(b) //nolint:errcheck

	*bp = b
	jsonBufPool.Put(bp)
}
