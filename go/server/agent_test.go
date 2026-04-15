package server

import (
	"context"
	"encoding/json"
	"testing"
)

// ─── Expression evaluator tests ─────────────────────────────────────────────

func TestEvalExpr_Basic(t *testing.T) {
	tests := []struct {
		expr string
		want float64
	}{
		{"2 + 3", 5},
		{"10 - 4", 6},
		{"25 * 17", 425},
		{"100 / 4", 25},
		{"2 + 3 * 4", 14},       // precedence
		{"(2 + 3) * 4", 20},     // parentheses
		{"10 / 2 + 3", 8},       // left-to-right
		{"-5 + 10", 5},          // unary minus
		{"3.5 * 2", 7},          // floats
		{"(10 + 5) / 3", 5},     // combined
		{"100", 100},            // single number
		{"((2 + 3))", 5},        // nested parens
		{"2 * (3 + 4) * 5", 70}, // complex
	}

	for _, tt := range tests {
		got, err := evalExpr(tt.expr)
		if err != nil {
			t.Errorf("evalExpr(%q) error: %v", tt.expr, err)
			continue
		}
		if got != tt.want {
			t.Errorf("evalExpr(%q) = %v, want %v", tt.expr, got, tt.want)
		}
	}
}

func TestEvalExpr_DivisionByZero(t *testing.T) {
	_, err := evalExpr("10 / 0")
	if err == nil {
		t.Error("expected division by zero error")
	}
}

func TestEvalExpr_Invalid(t *testing.T) {
	invalids := []string{
		"",
		"abc",
		"2 +",
		"* 3",
		"(2 + 3",
	}
	for _, expr := range invalids {
		_, err := evalExpr(expr)
		if err == nil {
			t.Errorf("evalExpr(%q) should have returned error", expr)
		}
	}
}

// ─── Tool tests ─────────────────────────────────────────────────────────────

func TestToolCalculator(t *testing.T) {
	result, err := toolCalculator(map[string]interface{}{
		"expression": "25 * 17",
	})
	if err != nil {
		t.Fatalf("toolCalculator error: %v", err)
	}
	if result != "425" {
		t.Errorf("result = %q, want \"425\"", result)
	}
}

func TestToolCalculator_MissingExpression(t *testing.T) {
	_, err := toolCalculator(map[string]interface{}{})
	if err == nil {
		t.Error("expected error for missing expression")
	}
}

func TestToolCalculator_FloatResult(t *testing.T) {
	result, err := toolCalculator(map[string]interface{}{
		"expression": "10 / 3",
	})
	if err != nil {
		t.Fatalf("toolCalculator error: %v", err)
	}
	// Should be a decimal string, not integer
	if result == "" {
		t.Error("result is empty")
	}
}

func TestToolCurrentTime(t *testing.T) {
	result, err := toolCurrentTime(nil)
	if err != nil {
		t.Fatalf("toolCurrentTime error: %v", err)
	}
	if result == "" {
		t.Error("result is empty")
	}
	// Should be in RFC3339 format
	if len(result) < 20 {
		t.Errorf("result too short for RFC3339: %q", result)
	}
}

// ─── mockAgentLLM ───────────────────────────────────────────────────────────

// mockAgentLLM returns responses from a sequence. Each call to Generate
// returns the next response in the list.
type mockAgentLLM struct {
	responses []string
	callIdx   int
	prompts   []string // records prompts for inspection
}

func (m *mockAgentLLM) Close() {}
func (m *mockAgentLLM) Generate(_ context.Context, prompt string, _ int, _ float32) (string, int, int, error) {
	m.prompts = append(m.prompts, prompt)
	if m.callIdx >= len(m.responses) {
		return "No more responses", 5, 3, nil
	}
	resp := m.responses[m.callIdx]
	m.callIdx++
	return resp, 5, 3, nil
}

// ─── Agent unit tests ───────────────────────────────────────────────────────

func TestAgent_UsesCalculator(t *testing.T) {
	// Simulate: LLM calls calculator, then gives final answer
	mock := &mockAgentLLM{
		responses: []string{
			`{"name": "calculator", "arguments": {"expression": "25 * 17"}}`,
			"25 * 17 = 425",
		},
	}

	tools := map[string]AgentTool{
		"calculator": builtinTools()["calculator"],
	}
	agent := NewAgent(mock, tools, 5)
	resp, err := agent.Run(context.Background(), "What is 25 * 17?")
	if err != nil {
		t.Fatalf("agent.Run error: %v", err)
	}

	if resp.Answer != "25 * 17 = 425" {
		t.Errorf("answer = %q, want \"25 * 17 = 425\"", resp.Answer)
	}

	// Should have steps: thought, tool_call, answer
	if len(resp.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d: %+v", len(resp.Steps), resp.Steps)
	}
	if resp.Steps[0].Type != "thought" {
		t.Errorf("step 0 type = %q, want \"thought\"", resp.Steps[0].Type)
	}
	if resp.Steps[1].Type != "tool_call" {
		t.Errorf("step 1 type = %q, want \"tool_call\"", resp.Steps[1].Type)
	}
	if resp.Steps[1].Tool != "calculator" {
		t.Errorf("step 1 tool = %q, want \"calculator\"", resp.Steps[1].Tool)
	}
	if resp.Steps[1].Result != "425" {
		t.Errorf("step 1 result = %q, want \"425\"", resp.Steps[1].Result)
	}
	if resp.Steps[2].Type != "answer" {
		t.Errorf("step 2 type = %q, want \"answer\"", resp.Steps[2].Type)
	}
}

func TestAgent_MultiStep(t *testing.T) {
	// Simulate: LLM calls calculator twice, then gives final answer
	mock := &mockAgentLLM{
		responses: []string{
			`{"name": "calculator", "arguments": {"expression": "10 + 20"}}`,
			`{"name": "calculator", "arguments": {"expression": "30 * 2"}}`,
			"The result is 60",
		},
	}

	tools := map[string]AgentTool{
		"calculator": builtinTools()["calculator"],
	}
	agent := NewAgent(mock, tools, 10)
	resp, err := agent.Run(context.Background(), "Calculate (10+20)*2 step by step")
	if err != nil {
		t.Fatalf("agent.Run error: %v", err)
	}

	if resp.Answer != "The result is 60" {
		t.Errorf("answer = %q, want \"The result is 60\"", resp.Answer)
	}

	// Should have: thought, tool_call, thought, tool_call, answer = 5 steps
	if len(resp.Steps) != 5 {
		t.Fatalf("expected 5 steps, got %d: %+v", len(resp.Steps), resp.Steps)
	}

	// Verify first tool call result
	if resp.Steps[1].Result != "30" {
		t.Errorf("step 1 result = %q, want \"30\"", resp.Steps[1].Result)
	}
	// Verify second tool call result
	if resp.Steps[3].Result != "60" {
		t.Errorf("step 3 result = %q, want \"60\"", resp.Steps[3].Result)
	}
}

func TestAgent_MaxIterations(t *testing.T) {
	// LLM always calls a tool, never gives a final answer
	mock := &mockAgentLLM{
		responses: []string{
			`{"name": "calculator", "arguments": {"expression": "1 + 1"}}`,
			`{"name": "calculator", "arguments": {"expression": "2 + 2"}}`,
			`{"name": "calculator", "arguments": {"expression": "3 + 3"}}`,
			`{"name": "calculator", "arguments": {"expression": "4 + 4"}}`,
			`{"name": "calculator", "arguments": {"expression": "5 + 5"}}`,
		},
	}

	tools := map[string]AgentTool{
		"calculator": builtinTools()["calculator"],
	}
	agent := NewAgent(mock, tools, 3) // limit to 3 iterations
	resp, err := agent.Run(context.Background(), "Keep calculating")
	if err != nil {
		t.Fatalf("agent.Run error: %v", err)
	}

	// Should stop after 3 iterations, answer should contain the last tool result
	if resp.Answer == "" {
		t.Error("answer should not be empty")
	}
	// Should have exactly 3 iterations worth of steps (thought + tool_call per iter = 6)
	if len(resp.Steps) != 6 {
		t.Errorf("expected 6 steps for 3 iterations, got %d", len(resp.Steps))
	}
	// LLM should have been called exactly 3 times
	if mock.callIdx != 3 {
		t.Errorf("LLM called %d times, expected 3", mock.callIdx)
	}
}

func TestAgent_NoToolsNeeded(t *testing.T) {
	// LLM answers directly without calling any tools
	mock := &mockAgentLLM{
		responses: []string{
			"The capital of France is Paris.",
		},
	}

	tools := builtinTools()
	agent := NewAgent(mock, tools, 5)
	resp, err := agent.Run(context.Background(), "What is the capital of France?")
	if err != nil {
		t.Fatalf("agent.Run error: %v", err)
	}

	if resp.Answer != "The capital of France is Paris." {
		t.Errorf("answer = %q, want \"The capital of France is Paris.\"", resp.Answer)
	}
	// Only one step: the final answer
	if len(resp.Steps) != 1 {
		t.Fatalf("expected 1 step, got %d: %+v", len(resp.Steps), resp.Steps)
	}
	if resp.Steps[0].Type != "answer" {
		t.Errorf("step type = %q, want \"answer\"", resp.Steps[0].Type)
	}
}

func TestAgent_UnknownTool(t *testing.T) {
	// LLM tries to call a tool that doesn't exist, then answers
	mock := &mockAgentLLM{
		responses: []string{
			`{"name": "nonexistent", "arguments": {}}`,
			"I could not find that tool, but the answer is 42.",
		},
	}

	tools := builtinTools()
	agent := NewAgent(mock, tools, 5)
	resp, err := agent.Run(context.Background(), "Use nonexistent tool")
	if err != nil {
		t.Fatalf("agent.Run error: %v", err)
	}

	if resp.Answer == "" {
		t.Error("answer should not be empty")
	}
	// LLM should have been called twice
	if mock.callIdx != 2 {
		t.Errorf("LLM called %d times, expected 2", mock.callIdx)
	}
}

func TestAgent_EmptyTools(t *testing.T) {
	mock := &mockAgentLLM{
		responses: []string{
			"I have no tools, so here is my answer.",
		},
	}

	agent := NewAgent(mock, map[string]AgentTool{}, 5)
	resp, err := agent.Run(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("agent.Run error: %v", err)
	}
	if resp.Answer != "I have no tools, so here is my answer." {
		t.Errorf("unexpected answer: %q", resp.Answer)
	}
}

// ─── AgentRunResponse JSON marshaling ───────────────────────────────────────

func TestAgentRunResponse_JSON(t *testing.T) {
	resp := AgentRunResponse{
		Answer: "425",
		Steps: []AgentStep{
			{Type: "thought", Content: "I need to calculate 25*17"},
			{Type: "tool_call", Tool: "calculator", Args: map[string]interface{}{"expression": "25*17"}, Result: "425"},
			{Type: "answer", Content: "25 * 17 = 425"},
		},
	}

	data, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	var decoded AgentRunResponse
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if decoded.Answer != "425" {
		t.Errorf("answer = %q, want \"425\"", decoded.Answer)
	}
	if len(decoded.Steps) != 3 {
		t.Fatalf("expected 3 steps, got %d", len(decoded.Steps))
	}
	if decoded.Steps[1].Tool != "calculator" {
		t.Errorf("step 1 tool = %q, want \"calculator\"", decoded.Steps[1].Tool)
	}
}

// ─── Built-in tools set ─────────────────────────────────────────────────────

func TestBuiltinTools(t *testing.T) {
	tools := builtinTools()
	if _, ok := tools["calculator"]; !ok {
		t.Error("missing calculator tool")
	}
	if _, ok := tools["current_time"]; !ok {
		t.Error("missing current_time tool")
	}
	// Each tool must have a name and description
	for name, tool := range tools {
		if tool.Name == "" {
			t.Errorf("tool %q has empty Name", name)
		}
		if tool.Description == "" {
			t.Errorf("tool %q has empty Description", name)
		}
		if tool.Execute == nil {
			t.Errorf("tool %q has nil Execute", name)
		}
	}
}
