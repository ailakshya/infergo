package server

// Function calling types — OpenAI-compatible tool use.

// Tool describes a function the model can call.
type Tool struct {
	Type     string       `json:"type"` // always "function"
	Function ToolFunction `json:"function"`
}

// ToolFunction describes a callable function.
type ToolFunction struct {
	Name        string      `json:"name"`
	Description string      `json:"description,omitempty"`
	Parameters  interface{} `json:"parameters,omitempty"` // JSON Schema
}

// ToolCall is a function call requested by the model.
type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

// ToolCallFunction contains the function name and arguments.
type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // JSON string
}

// ToolChoice controls how the model selects tools.
// Can be "none", "auto", or {"type":"function","function":{"name":"..."}}.
type ToolChoice struct {
	Type     string        `json:"type,omitempty"`
	Function *ToolFunction `json:"function,omitempty"`
}
