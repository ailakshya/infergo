package server

import (
	"context"
	"encoding/json"
	"testing"
)

func TestWithGrammarRoundTrip(t *testing.T) {
	ctx := context.Background()
	g, ok := GrammarFromContext(ctx)
	if ok || g != "" {
		t.Fatalf("expected no grammar, got %q", g)
	}

	ctx = WithGrammar(ctx, "root ::= \"hello\"")
	g, ok = GrammarFromContext(ctx)
	if !ok || g != "root ::= \"hello\"" {
		t.Fatalf("expected grammar 'root ::= \"hello\"', got %q (ok=%v)", g, ok)
	}
}

func TestJSONGrammarIsValid(t *testing.T) {
	// The built-in JSON grammar must not be empty.
	if JSONGrammar == "" {
		t.Fatal("JSONGrammar is empty")
	}
	// Sanity: it should contain "root ::="
	if !contains(JSONGrammar, "root") {
		t.Error("JSONGrammar does not contain 'root' rule")
	}
}

func contains(s, sub string) bool {
	return len(s) >= len(sub) && (s == sub || len(s) > 0 && containsHelper(s, sub))
}

func containsHelper(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestResponseFormatParsing(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantNil bool
		wantTyp string
	}{
		{"no format", `{"model":"m","messages":[{"role":"user","content":"hi"}]}`, true, ""},
		{"text", `{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}`, false, "text"},
		{"json_object", `{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}`, false, "json_object"},
		{"grammar", `{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"grammar","grammar":"root ::= \"x\""}}`, false, "grammar"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req ChatCompletionRequest
			if err := json.Unmarshal([]byte(tt.input), &req); err != nil {
				t.Fatalf("unmarshal: %v", err)
			}
			if tt.wantNil && req.ResponseFormat != nil {
				t.Fatal("expected nil ResponseFormat")
			}
			if !tt.wantNil {
				if req.ResponseFormat == nil {
					t.Fatal("expected non-nil ResponseFormat")
				}
				if req.ResponseFormat.Type != tt.wantTyp {
					t.Errorf("type = %q, want %q", req.ResponseFormat.Type, tt.wantTyp)
				}
			}
		})
	}
}
