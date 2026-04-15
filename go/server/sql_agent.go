package server

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// SQLAgentRequest is the body for POST /v1/agents/sql. OPT-72.
type SQLAgentRequest struct {
	Model string `json:"model"`
	Query string `json:"query"` // natural language query
	Schema string `json:"schema,omitempty"` // table schema for context
}

// SQLAgentResponse returns generated SQL and explanation.
type SQLAgentResponse struct {
	SQL         string `json:"sql"`
	Explanation string `json:"explanation"`
	ReadOnly    bool   `json:"read_only"`
}

func (s *Server) handleSQLAgent(w http.ResponseWriter, r *http.Request) {
	var req SQLAgentRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	ref, err := s.registry.Get(req.Model)
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	defer ref.Release()

	llm, ok := ref.Model.(LLMModel)
	if !ok {
		writeError(w, http.StatusBadRequest, "model does not support SQL generation")
		return
	}

	schemaCtx := ""
	if req.Schema != "" {
		schemaCtx = fmt.Sprintf("\nDatabase schema:\n%s\n", req.Schema)
	}

	prompt := fmt.Sprintf(`<|im_start|>system
You are a SQL expert. Generate a READ-ONLY SQL query for the user's question.
Rules:
- Only generate SELECT queries (no INSERT, UPDATE, DELETE, DROP, ALTER)
- Use standard SQL syntax
- Return JSON: {"sql": "<query>", "explanation": "<brief explanation>"}
%s<|im_end|>
<|im_start|>user
%s<|im_end|>
<|im_start|>assistant
`, schemaCtx, req.Query)

	ctx := WithGrammar(context.Background(), JSONGrammar)
	text, _, _, err := llm.Generate(ctx, prompt, 256, 0.1)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	var result SQLAgentResponse
	if json.Unmarshal([]byte(text), &result) != nil {
		result = SQLAgentResponse{SQL: text, Explanation: "Generated SQL"}
	}

	// Verify read-only
	result.ReadOnly = isReadOnly(result.SQL)
	if !result.ReadOnly {
		writeError(w, http.StatusBadRequest, "generated query is not read-only, blocked for safety")
		return
	}

	writeJSON(w, http.StatusOK, result)
}

func isReadOnly(sql string) bool {
	upper := fmt.Sprintf(" %s ", sql)
	for _, kw := range []string{"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "CREATE"} {
		if containsWord(upper, kw) {
			return false
		}
	}
	return true
}

func containsWord(text, word string) bool {
	// Simple word boundary check
	for i := 0; i <= len(text)-len(word); i++ {
		if text[i:i+len(word)] == word {
			if (i == 0 || !isAlpha(text[i-1])) && (i+len(word) >= len(text) || !isAlpha(text[i+len(word)])) {
				return true
			}
		}
	}
	return false
}

func isAlpha(b byte) bool {
	return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')
}
