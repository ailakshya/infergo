package server

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ─── OPT-51: Agent Framework ─────────────────────────────────────────────────

// Agent implements a ReAct-style agent loop: plan, execute tools, observe, repeat.
type Agent struct {
	model        LLMModel
	tools        map[string]AgentTool
	maxIter      int
	systemPrompt string
}

// AgentTool is a tool the agent can invoke during its reasoning loop.
type AgentTool struct {
	Name        string
	Description string
	Parameters  string // human-readable parameter description for the prompt
	Execute     func(args map[string]interface{}) (string, error)
}

// AgentStep records one step of the agent's reasoning.
type AgentStep struct {
	Type    string                 `json:"type"`              // "thought", "tool_call", "answer"
	Content string                 `json:"content,omitempty"` // thought or answer text
	Tool    string                 `json:"tool,omitempty"`    // tool name (tool_call only)
	Args    map[string]interface{} `json:"args,omitempty"`    // tool arguments (tool_call only)
	Result  string                 `json:"result,omitempty"`  // tool result (tool_call only)
}

// AgentRunRequest is the body for POST /v1/agents/run.
type AgentRunRequest struct {
	Model         string   `json:"model"`
	Query         string   `json:"query"`
	Tools         []string `json:"tools,omitempty"`    // tool names to enable (empty = all built-ins)
	MaxIterations int      `json:"max_iterations,omitempty"` // default 5
}

// AgentRunResponse is returned by POST /v1/agents/run.
type AgentRunResponse struct {
	Answer string      `json:"answer"`
	Steps  []AgentStep `json:"steps"`
}

// ─── Built-in tools ──────────────────────────────────────────────────────────

// builtinTools returns the default set of agent tools.
func builtinTools() map[string]AgentTool {
	return map[string]AgentTool{
		"calculator": {
			Name:        "calculator",
			Description: "Evaluate a math expression and return the numeric result.",
			Parameters:  `{"expression": "string (math expression, e.g. '25 * 17')"}`,
			Execute:     toolCalculator,
		},
		"current_time": {
			Name:        "current_time",
			Description: "Return the current date and time in UTC.",
			Parameters:  `{}`,
			Execute:     toolCurrentTime,
		},
		"code_executor": codeExecutorTool(),
	}
}

// toolCalculator evaluates simple arithmetic expressions.
// Supports +, -, *, /, parentheses, and floating-point numbers.
func toolCalculator(args map[string]interface{}) (string, error) {
	expr, _ := args["expression"].(string)
	if expr == "" {
		return "", fmt.Errorf("missing 'expression' argument")
	}
	result, err := evalExpr(expr)
	if err != nil {
		return "", fmt.Errorf("calculator: %w", err)
	}
	// Format: drop trailing zeros for clean output
	if result == math.Trunc(result) {
		return strconv.FormatInt(int64(result), 10), nil
	}
	return strconv.FormatFloat(result, 'f', -1, 64), nil
}

// toolCurrentTime returns the current UTC time.
func toolCurrentTime(args map[string]interface{}) (string, error) {
	return time.Now().UTC().Format(time.RFC3339), nil
}

// ─── Expression evaluator ────────────────────────────────────────────────────

// evalExpr parses and evaluates a simple math expression supporting
// +, -, *, /, parentheses, and unary minus.
func evalExpr(expr string) (float64, error) {
	p := &exprParser{input: expr}
	result := p.parseExpr()
	if p.err != nil {
		return 0, p.err
	}
	p.skipSpaces()
	if p.pos < len(p.input) {
		return 0, fmt.Errorf("unexpected character at position %d: %q", p.pos, string(p.input[p.pos]))
	}
	return result, nil
}

type exprParser struct {
	input string
	pos   int
	err   error
}

func (p *exprParser) skipSpaces() {
	for p.pos < len(p.input) && (p.input[p.pos] == ' ' || p.input[p.pos] == '\t') {
		p.pos++
	}
}

func (p *exprParser) parseExpr() float64 {
	result := p.parseTerm()
	for p.err == nil {
		p.skipSpaces()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '+' && op != '-' {
			break
		}
		p.pos++
		right := p.parseTerm()
		if op == '+' {
			result += right
		} else {
			result -= right
		}
	}
	return result
}

func (p *exprParser) parseTerm() float64 {
	result := p.parseFactor()
	for p.err == nil {
		p.skipSpaces()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '*' && op != '/' {
			break
		}
		p.pos++
		right := p.parseFactor()
		if op == '*' {
			result *= right
		} else {
			if right == 0 {
				p.err = fmt.Errorf("division by zero")
				return 0
			}
			result /= right
		}
	}
	return result
}

func (p *exprParser) parseFactor() float64 {
	if p.err != nil {
		return 0
	}
	p.skipSpaces()
	if p.pos >= len(p.input) {
		p.err = fmt.Errorf("unexpected end of expression")
		return 0
	}

	// Unary minus
	if p.input[p.pos] == '-' {
		p.pos++
		return -p.parseFactor()
	}

	// Parenthesized sub-expression
	if p.input[p.pos] == '(' {
		p.pos++
		result := p.parseExpr()
		p.skipSpaces()
		if p.pos < len(p.input) && p.input[p.pos] == ')' {
			p.pos++
		} else {
			p.err = fmt.Errorf("missing closing parenthesis")
		}
		return result
	}

	// Number
	return p.parseNumber()
}

func (p *exprParser) parseNumber() float64 {
	p.skipSpaces()
	start := p.pos
	// Integer part
	for p.pos < len(p.input) && p.input[p.pos] >= '0' && p.input[p.pos] <= '9' {
		p.pos++
	}
	// Decimal part
	if p.pos < len(p.input) && p.input[p.pos] == '.' {
		p.pos++
		for p.pos < len(p.input) && p.input[p.pos] >= '0' && p.input[p.pos] <= '9' {
			p.pos++
		}
	}
	if p.pos == start {
		p.err = fmt.Errorf("expected number at position %d", p.pos)
		return 0
	}
	val, err := strconv.ParseFloat(p.input[start:p.pos], 64)
	if err != nil {
		p.err = err
		return 0
	}
	return val
}

// ─── Agent execution ─────────────────────────────────────────────────────────

// NewAgent creates an agent backed by the given LLM with the specified tools.
func NewAgent(model LLMModel, tools map[string]AgentTool, maxIter int) *Agent {
	if maxIter <= 0 {
		maxIter = 5
	}
	return &Agent{
		model:   model,
		tools:   tools,
		maxIter: maxIter,
		systemPrompt: `You are a helpful assistant with access to tools.

When you need to use a tool, respond with EXACTLY this JSON format on a single line:
{"name": "<tool_name>", "arguments": {<args>}}

When you have the final answer, respond with plain text (no JSON).

Do NOT wrap your answer in JSON when you have the final result. Just state the answer directly.`,
	}
}

// Run executes the ReAct agent loop for the given query.
func (a *Agent) Run(ctx context.Context, query string) (*AgentRunResponse, error) {
	var steps []AgentStep
	messages := []ChatMessage{
		{Role: "system", Content: a.buildSystemPrompt()},
		{Role: "user", Content: query},
	}

	for iter := 0; iter < a.maxIter; iter++ {
		prompt := buildPrompt(messages)
		text, _, _, err := a.model.Generate(ctx, prompt, 512, 0.1)
		if err != nil {
			return nil, fmt.Errorf("agent iteration %d: %w", iter, err)
		}
		text = strings.TrimSpace(text)

		// Try parsing as a tool call
		fc, isToolCall := ParseFunctionCallResponse(text)
		if !isToolCall {
			// No tool call — this is the final answer
			steps = append(steps, AgentStep{
				Type:    "answer",
				Content: text,
			})
			return &AgentRunResponse{
				Answer: text,
				Steps:  steps,
			}, nil
		}

		// It's a tool call — execute it
		toolName := fc.Function.Name
		tool, exists := a.tools[toolName]
		if !exists {
			// Unknown tool — add error observation and continue
			errMsg := fmt.Sprintf("Unknown tool: %s", toolName)
			steps = append(steps, AgentStep{
				Type:    "tool_call",
				Tool:    toolName,
				Result:  errMsg,
			})
			messages = append(messages,
				ChatMessage{Role: "assistant", Content: text},
				ChatMessage{Role: "user", Content: fmt.Sprintf("Tool error: %s. Available tools: %s", errMsg, a.toolNames())},
			)
			continue
		}

		// Parse arguments
		var args map[string]interface{}
		if err := json.Unmarshal([]byte(fc.Function.Arguments), &args); err != nil {
			args = make(map[string]interface{})
		}

		// Record thought (the LLM chose to call a tool)
		steps = append(steps, AgentStep{
			Type:    "thought",
			Content: fmt.Sprintf("I need to use the %s tool", toolName),
		})

		// Execute tool
		result, err := tool.Execute(args)
		if err != nil {
			result = fmt.Sprintf("Error: %s", err.Error())
		}

		steps = append(steps, AgentStep{
			Type:   "tool_call",
			Tool:   toolName,
			Args:   args,
			Result: result,
		})

		// Add tool call and result to conversation
		messages = append(messages,
			ChatMessage{Role: "assistant", Content: text},
			ChatMessage{Role: "user", Content: fmt.Sprintf("Tool result from %s: %s\n\nNow provide the final answer based on this result. Respond with plain text only.", toolName, result)},
		)
	}

	// Max iterations reached — return the last step content as answer
	answer := "Max iterations reached without final answer."
	if len(steps) > 0 {
		last := steps[len(steps)-1]
		if last.Result != "" {
			answer = last.Result
		} else if last.Content != "" {
			answer = last.Content
		}
	}

	return &AgentRunResponse{
		Answer: answer,
		Steps:  steps,
	}, nil
}

// buildSystemPrompt constructs the system prompt with tool descriptions.
func (a *Agent) buildSystemPrompt() string {
	if len(a.tools) == 0 {
		return "You are a helpful assistant. Answer the user's question directly."
	}

	var sb strings.Builder
	sb.WriteString(a.systemPrompt)
	sb.WriteString("\n\nAvailable tools:\n")
	for _, tool := range a.tools {
		sb.WriteString(fmt.Sprintf("- %s: %s\n  Parameters: %s\n", tool.Name, tool.Description, tool.Parameters))
	}
	return sb.String()
}

// toolNames returns a comma-separated list of available tool names.
func (a *Agent) toolNames() string {
	names := make([]string, 0, len(a.tools))
	for name := range a.tools {
		names = append(names, name)
	}
	return strings.Join(names, ", ")
}

// ─── HTTP handler ────────────────────────────────────────────────────────────

func (s *Server) handleAgentRun(w http.ResponseWriter, r *http.Request) {
	var req AgentRunRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if req.Query == "" {
		writeError(w, http.StatusBadRequest, "query is required")
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
		writeError(w, http.StatusBadRequest, "model does not support agent execution")
		return
	}

	// Build tool set
	allTools := builtinTools()
	activeTools := make(map[string]AgentTool)

	if len(req.Tools) == 0 {
		// No filter — use all built-in tools
		activeTools = allTools
	} else {
		for _, name := range req.Tools {
			if tool, exists := allTools[name]; exists {
				activeTools[name] = tool
			}
		}
	}

	maxIter := req.MaxIterations
	if maxIter <= 0 {
		maxIter = 5
	}

	agent := NewAgent(llm, activeTools, maxIter)
	resp, err := agent.Run(r.Context(), req.Query)
	if err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}

	writeJSON(w, http.StatusOK, resp)
}
