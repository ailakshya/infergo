package server

import (
	"encoding/json"
	"net/http/httptest"
	"strings"
	"testing"
)

// ─── TestGenerateFunctionCallGrammar ─────────────────────────────────────────

func TestGenerateFunctionCallGrammar(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name:        "get_weather",
				Description: "Get current weather",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"location": map[string]interface{}{
							"type": "string",
						},
						"unit": map[string]interface{}{
							"type": "string",
							"enum": []interface{}{"celsius", "fahrenheit"},
						},
					},
					"required": []interface{}{"location"},
				},
			},
		},
	}

	grammar := GenerateFunctionCallGrammar(tools, "")

	// Must contain root rule
	if !strings.Contains(grammar, "root ::=") {
		t.Error("grammar missing root rule")
	}

	// Must reference the function by name
	if !strings.Contains(grammar, "call-get_weather") {
		t.Error("grammar missing call-get_weather rule")
	}

	// Must contain the function name as a literal string
	if !strings.Contains(grammar, `get_weather`) {
		t.Error("grammar missing function name literal")
	}

	// Must have the arguments rule
	if !strings.Contains(grammar, "args-get_weather") {
		t.Error("grammar missing args-get_weather rule")
	}

	// Must have primitive rules
	if !strings.Contains(grammar, "string ::=") {
		t.Error("grammar missing string primitive")
	}
	if !strings.Contains(grammar, "number ::=") {
		t.Error("grammar missing number primitive")
	}
	if !strings.Contains(grammar, "boolean ::=") {
		t.Error("grammar missing boolean primitive")
	}

	// Must have property-specific rules
	if !strings.Contains(grammar, "get_weather-location") {
		t.Error("grammar missing location property rule")
	}
	if !strings.Contains(grammar, "get_weather-unit") {
		t.Error("grammar missing unit property rule")
	}
}

func TestGenerateFunctionCallGrammar_MultipleTools(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name: "get_weather",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
		{
			Type: "function",
			Function: ToolFunction{
				Name: "search",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"query": map[string]interface{}{"type": "string"},
					},
					"required": []interface{}{"query"},
				},
			},
		},
	}

	grammar := GenerateFunctionCallGrammar(tools, "")

	// Root should have alternatives
	if !strings.Contains(grammar, "call-get_weather") {
		t.Error("grammar missing call-get_weather")
	}
	if !strings.Contains(grammar, "call-search") {
		t.Error("grammar missing call-search")
	}
	if !strings.Contains(grammar, "|") {
		t.Error("grammar root should have alternatives separated by |")
	}
}

func TestGenerateFunctionCallGrammar_SpecificFunction(t *testing.T) {
	tools := []Tool{
		{Type: "function", Function: ToolFunction{Name: "foo"}},
		{Type: "function", Function: ToolFunction{Name: "bar"}},
	}

	grammar := GenerateFunctionCallGrammar(tools, "foo")

	if !strings.Contains(grammar, "call-foo") {
		t.Error("grammar should contain call-foo")
	}
	if strings.Contains(grammar, "call-bar") {
		t.Error("grammar should NOT contain call-bar when specific function chosen")
	}
}

func TestGenerateFunctionCallGrammar_EmptyTools(t *testing.T) {
	grammar := GenerateFunctionCallGrammar(nil, "")
	if grammar != "" {
		t.Errorf("expected empty grammar for nil tools, got %q", grammar)
	}

	grammar = GenerateFunctionCallGrammar([]Tool{}, "")
	if grammar != "" {
		t.Errorf("expected empty grammar for empty tools, got %q", grammar)
	}
}

func TestGenerateFunctionCallGrammar_NoParameters(t *testing.T) {
	tools := []Tool{
		{
			Type:     "function",
			Function: ToolFunction{Name: "ping"},
		},
	}

	grammar := GenerateFunctionCallGrammar(tools, "")
	if !strings.Contains(grammar, "root ::=") {
		t.Error("grammar should still have root rule for no-param functions")
	}
	if !strings.Contains(grammar, `args-ping ::= "{}"`) {
		t.Error("no-param function should generate empty object args rule")
	}
}

// ─── TestParseFunctionCallResponse ──────────────────────────────────────────

func TestParseFunctionCallResponse(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantOK  bool
		wantFn  string
		wantArg string
	}{
		{
			name:    "valid function call",
			input:   `{"name": "get_weather", "arguments": {"location": "NYC"}}`,
			wantOK:  true,
			wantFn:  "get_weather",
			wantArg: `{"location": "NYC"}`,
		},
		{
			name:    "valid with no arguments",
			input:   `{"name": "ping"}`,
			wantOK:  true,
			wantFn:  "ping",
			wantArg: "{}",
		},
		{
			name:    "valid with empty arguments",
			input:   `{"name": "ping", "arguments": {}}`,
			wantOK:  true,
			wantFn:  "ping",
			wantArg: "{}",
		},
		{
			name:   "plain text",
			input:  "The weather is sunny today.",
			wantOK: false,
		},
		{
			name:   "empty string",
			input:  "",
			wantOK: false,
		},
		{
			name:   "JSON without name",
			input:  `{"foo": "bar"}`,
			wantOK: false,
		},
		{
			name:   "invalid JSON",
			input:  `{"name": "test"`,
			wantOK: false,
		},
		{
			name:    "whitespace around JSON",
			input:   `  {"name": "search", "arguments": {"q": "hello"}}  `,
			wantOK:  true,
			wantFn:  "search",
			wantArg: `{"q": "hello"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tc, ok := ParseFunctionCallResponse(tt.input)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if tc.Function.Name != tt.wantFn {
				t.Errorf("function name = %q, want %q", tc.Function.Name, tt.wantFn)
			}
			// Compare arguments as JSON to handle whitespace differences
			var gotArgs, wantArgs interface{}
			json.Unmarshal([]byte(tc.Function.Arguments), &gotArgs)
			json.Unmarshal([]byte(tt.wantArg), &wantArgs)
			gotJSON, _ := json.Marshal(gotArgs)
			wantJSON, _ := json.Marshal(wantArgs)
			if string(gotJSON) != string(wantJSON) {
				t.Errorf("arguments = %s, want %s", tc.Function.Arguments, tt.wantArg)
			}
			// Must have an ID
			if tc.ID == "" {
				t.Error("tool call ID is empty")
			}
			if tc.Type != "function" {
				t.Errorf("type = %q, want \"function\"", tc.Type)
			}
		})
	}
}

// ─── TestResolveToolChoice ──────────────────────────────────────────────────

func TestToolChoiceNone(t *testing.T) {
	tc := ResolveToolChoice("none")
	if tc.Mode != "none" {
		t.Errorf("mode = %q, want \"none\"", tc.Mode)
	}
}

func TestToolChoiceRequired(t *testing.T) {
	tc := ResolveToolChoice("required")
	if tc.Mode != "required" {
		t.Errorf("mode = %q, want \"required\"", tc.Mode)
	}
}

func TestToolChoiceAuto(t *testing.T) {
	tc := ResolveToolChoice("auto")
	if tc.Mode != "auto" {
		t.Errorf("mode = %q, want \"auto\"", tc.Mode)
	}

	// nil defaults to auto
	tc = ResolveToolChoice(nil)
	if tc.Mode != "auto" {
		t.Errorf("nil should default to auto, got %q", tc.Mode)
	}
}

func TestToolChoiceSpecificFunction(t *testing.T) {
	choice := map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": "get_weather",
		},
	}
	tc := ResolveToolChoice(choice)
	if tc.Mode != "function" {
		t.Errorf("mode = %q, want \"function\"", tc.Mode)
	}
	if tc.FunctionName != "get_weather" {
		t.Errorf("function name = %q, want \"get_weather\"", tc.FunctionName)
	}
}

// ─── TestWriteChatCompletionToolCall ─────────────────────────────────────────

func TestWriteChatCompletionToolCall(t *testing.T) {
	w := httptest.NewRecorder()
	toolCalls := []ToolCall{
		{
			ID:   "call-123",
			Type: "function",
			Function: ToolCallFunction{
				Name:      "get_weather",
				Arguments: `{"location": "NYC"}`,
			},
		},
	}

	writeChatCompletionToolCall(w, "chatcmpl-1", "test-model", toolCalls, 10, 5)

	if w.Code != 200 {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "application/json" {
		t.Errorf("content-type = %q, want application/json", ct)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("invalid JSON response: %v", err)
	}

	if resp["id"] != "chatcmpl-1" {
		t.Errorf("id = %v, want chatcmpl-1", resp["id"])
	}

	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) != 1 {
		t.Fatal("expected 1 choice")
	}
	choice := choices[0].(map[string]interface{})
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason = %v, want tool_calls", choice["finish_reason"])
	}

	msg := choice["message"].(map[string]interface{})
	if msg["content"] != nil {
		t.Errorf("content should be null, got %v", msg["content"])
	}

	tcs, ok := msg["tool_calls"].([]interface{})
	if !ok || len(tcs) != 1 {
		t.Fatal("expected 1 tool_call")
	}
	tc := tcs[0].(map[string]interface{})
	if tc["id"] != "call-123" {
		t.Errorf("tool_call id = %v, want call-123", tc["id"])
	}
	if tc["type"] != "function" {
		t.Errorf("tool_call type = %v, want function", tc["type"])
	}
	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "get_weather" {
		t.Errorf("function name = %v, want get_weather", fn["name"])
	}
}

// ─── TestSanitizeRuleName ───────────────────────────────────────────────────

func TestSanitizeRuleName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"get_weather", "get_weather"},
		{"search-web", "search-web"},
		{"my.function", "my-function"},
		{"a b c", "a-b-c"},
		{"CamelCase123", "CamelCase123"},
	}
	for _, tt := range tests {
		got := sanitizeRuleName(tt.input)
		if got != tt.want {
			t.Errorf("sanitizeRuleName(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// ─── TestGrammarWithEnumProperties ──────────────────────────────────────────

func TestGrammarWithEnumProperties(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name: "set_mode",
				Parameters: map[string]interface{}{
					"type": "object",
					"properties": map[string]interface{}{
						"mode": map[string]interface{}{
							"type": "string",
							"enum": []interface{}{"fast", "slow", "auto"},
						},
					},
					"required": []interface{}{"mode"},
				},
			},
		},
	}

	grammar := GenerateFunctionCallGrammar(tools, "")

	// Enum values should appear in grammar
	if !strings.Contains(grammar, "fast") {
		t.Error("grammar should contain enum value 'fast'")
	}
	if !strings.Contains(grammar, "slow") {
		t.Error("grammar should contain enum value 'slow'")
	}
}

// ─── TestToolChoiceNoneNoGrammar ────────────────────────────────────────────

func TestToolChoiceNoneNoGrammar(t *testing.T) {
	// When tool_choice is "none", no grammar should be generated.
	tools := []Tool{
		{
			Type:     "function",
			Function: ToolFunction{Name: "test_fn"},
		},
	}

	choice := ResolveToolChoice("none")
	if choice.Mode != "none" {
		t.Fatalf("expected mode none, got %q", choice.Mode)
	}

	// Grammar generation should not be called when mode is "none",
	// but verify our grammar function works correctly for edge cases
	grammar := GenerateFunctionCallGrammar(tools, "nonexistent_function")
	if grammar != "" {
		t.Error("should produce empty grammar when function name doesn't match any tool")
	}
}

// ─── TestToolChoiceRequiredGrammar ──────────────────────────────────────────

func TestToolChoiceRequiredGrammar(t *testing.T) {
	tools := []Tool{
		{
			Type: "function",
			Function: ToolFunction{
				Name: "alpha",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
		{
			Type: "function",
			Function: ToolFunction{
				Name: "beta",
				Parameters: map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{},
				},
			},
		},
	}

	choice := ResolveToolChoice("required")
	if choice.Mode != "required" {
		t.Fatalf("expected mode required, got %q", choice.Mode)
	}

	grammar := GenerateFunctionCallGrammar(tools, "")
	if grammar == "" {
		t.Fatal("required mode should produce a grammar")
	}
	// Should include all functions
	if !strings.Contains(grammar, "call-alpha") {
		t.Error("grammar missing call-alpha")
	}
	if !strings.Contains(grammar, "call-beta") {
		t.Error("grammar missing call-beta")
	}
}
