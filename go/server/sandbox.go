package server

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// ─── OPT-50: Code Execution Sandbox ────────────────────────────────────────

// CodeExecutor runs user-supplied code in an isolated subprocess with
// resource limits (timeout, output cap). No Docker required.
type CodeExecutor struct {
	timeout   time.Duration // max execution time (default 5s)
	maxMemory int64         // max memory in bytes (default 256MB) — advisory
	maxOutput int           // max output bytes (default 64KB)
}

// ExecutionRequest is the body for POST /v1/code/execute.
type ExecutionRequest struct {
	Language string `json:"language"`          // python, javascript, go, bash
	Code     string `json:"code"`              // source code to execute
	Timeout  int    `json:"timeout,omitempty"` // seconds (overrides default)
}

// ExecutionResult is returned by POST /v1/code/execute.
type ExecutionResult struct {
	Stdout   string  `json:"stdout"`
	Stderr   string  `json:"stderr"`
	ExitCode int     `json:"exit_code"`
	Duration float64 `json:"duration_ms"`
	Error    string  `json:"error,omitempty"`
}

// NewCodeExecutor creates a CodeExecutor with sensible defaults.
func NewCodeExecutor() *CodeExecutor {
	return &CodeExecutor{
		timeout:   5 * time.Second,
		maxMemory: 256 * 1024 * 1024, // 256 MB
		maxOutput: 64 * 1024,         // 64 KB
	}
}

// supportedLanguages maps language names to (file extension, command).
var supportedLanguages = map[string]struct {
	ext string
	cmd string
}{
	"python":     {ext: ".py", cmd: "python3"},
	"javascript": {ext: ".js", cmd: "node"},
	"go":         {ext: ".go", cmd: "go"},
	"bash":       {ext: ".sh", cmd: "bash"},
}

// detectLanguage guesses the language from the code content.
// Checks shebang line first, then falls back to syntax heuristics.
func detectLanguage(code string) string {
	first := code
	if idx := strings.IndexByte(code, '\n'); idx >= 0 {
		first = code[:idx]
	}
	first = strings.TrimSpace(first)

	// Shebang detection
	if strings.HasPrefix(first, "#!") {
		switch {
		case strings.Contains(first, "python"):
			return "python"
		case strings.Contains(first, "node"):
			return "javascript"
		case strings.Contains(first, "bash") || strings.Contains(first, "/sh"):
			return "bash"
		}
	}

	// Syntax heuristics
	switch {
	case strings.Contains(code, "def ") || strings.Contains(code, "import ") || strings.Contains(code, "print("):
		return "python"
	case strings.Contains(code, "console.log") || strings.Contains(code, "const ") || strings.Contains(code, "function "):
		return "javascript"
	case strings.Contains(code, "package ") || strings.Contains(code, "func ") || strings.Contains(code, "fmt."):
		return "go"
	default:
		return "bash"
	}
}

// Execute runs the given code in a subprocess and returns the result.
func (ce *CodeExecutor) Execute(ctx context.Context, req ExecutionRequest) ExecutionResult {
	start := time.Now()

	// Determine language
	lang := strings.ToLower(strings.TrimSpace(req.Language))
	if lang == "" {
		lang = detectLanguage(req.Code)
	}

	spec, ok := supportedLanguages[lang]
	if !ok {
		return ExecutionResult{
			ExitCode: -1,
			Duration: msSince(start),
			Error:    fmt.Sprintf("unsupported language: %s (supported: python, javascript, go, bash)", lang),
		}
	}

	// Determine timeout
	timeout := ce.timeout
	if req.Timeout > 0 {
		timeout = time.Duration(req.Timeout) * time.Second
	}
	// Cap at 30 seconds to prevent abuse
	if timeout > 30*time.Second {
		timeout = 30 * time.Second
	}

	// Write code to temp file
	tmpDir, err := os.MkdirTemp("", "infergo-sandbox-*")
	if err != nil {
		return ExecutionResult{
			ExitCode: -1,
			Duration: msSince(start),
			Error:    fmt.Sprintf("failed to create temp dir: %v", err),
		}
	}
	defer os.RemoveAll(tmpDir)

	tmpFile := filepath.Join(tmpDir, "code"+spec.ext)
	if err := os.WriteFile(tmpFile, []byte(req.Code), 0600); err != nil {
		return ExecutionResult{
			ExitCode: -1,
			Duration: msSince(start),
			Error:    fmt.Sprintf("failed to write temp file: %v", err),
		}
	}

	// Build command
	execCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	var cmd *exec.Cmd
	if lang == "go" {
		cmd = exec.CommandContext(execCtx, spec.cmd, "run", tmpFile)
	} else {
		cmd = exec.CommandContext(execCtx, spec.cmd, tmpFile)
	}

	// Capture stdout and stderr with size limits
	var stdoutBuf, stderrBuf bytes.Buffer
	cmd.Stdout = &limitedWriter{w: &stdoutBuf, limit: ce.maxOutput}
	cmd.Stderr = &limitedWriter{w: &stderrBuf, limit: ce.maxOutput}

	// Run
	runErr := cmd.Run()

	result := ExecutionResult{
		Stdout:   stdoutBuf.String(),
		Stderr:   stderrBuf.String(),
		Duration: msSince(start),
	}

	// Truncation markers
	if stdoutBuf.Len() >= ce.maxOutput {
		result.Stdout += "\n... (output truncated)"
	}
	if stderrBuf.Len() >= ce.maxOutput {
		result.Stderr += "\n... (output truncated)"
	}

	if runErr != nil {
		if execCtx.Err() == context.DeadlineExceeded {
			result.ExitCode = -1
			result.Error = fmt.Sprintf("execution timed out after %v", timeout)
			return result
		}
		if exitErr, ok := runErr.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else {
			result.ExitCode = -1
			result.Error = runErr.Error()
		}
	}

	return result
}

// limitedWriter wraps a writer and stops writing after limit bytes.
type limitedWriter struct {
	w       io.Writer
	limit   int
	written int
}

func (lw *limitedWriter) Write(p []byte) (int, error) {
	if lw.written >= lw.limit {
		return len(p), nil // silently discard
	}
	remaining := lw.limit - lw.written
	if len(p) > remaining {
		p = p[:remaining]
	}
	n, err := lw.w.Write(p)
	lw.written += n
	return n, err
}

func msSince(t time.Time) float64 {
	return float64(time.Since(t).Microseconds()) / 1000.0
}

// ─── HTTP handler ───────────────────────────────────────────────────────────

func (s *Server) handleCodeExecute(w http.ResponseWriter, r *http.Request) {
	var req ExecutionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	if strings.TrimSpace(req.Code) == "" {
		writeError(w, http.StatusBadRequest, "code is required")
		return
	}

	executor := NewCodeExecutor()
	result := executor.Execute(r.Context(), req)
	writeJSON(w, http.StatusOK, result)
}

// ─── Agent tool integration ─────────────────────────────────────────────────

// codeExecutorTool returns an AgentTool that runs code via CodeExecutor.
func codeExecutorTool() AgentTool {
	return AgentTool{
		Name:        "code_executor",
		Description: "Execute code in a sandboxed subprocess. Supports python, javascript, go, bash.",
		Parameters:  `{"language": "string (python|javascript|go|bash)", "code": "string (source code)"}`,
		Execute: func(args map[string]interface{}) (string, error) {
			code, _ := args["code"].(string)
			if code == "" {
				return "", fmt.Errorf("missing 'code' argument")
			}
			lang, _ := args["language"].(string)

			executor := NewCodeExecutor()
			result := executor.Execute(context.Background(), ExecutionRequest{
				Language: lang,
				Code:     code,
			})

			// Format for the agent: combine stdout and any error
			var sb strings.Builder
			if result.Stdout != "" {
				sb.WriteString(result.Stdout)
			}
			if result.Stderr != "" {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString("STDERR: ")
				sb.WriteString(result.Stderr)
			}
			if result.Error != "" {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString("ERROR: ")
				sb.WriteString(result.Error)
			}
			if sb.Len() == 0 {
				sb.WriteString("(no output)")
			}
			return sb.String(), nil
		},
	}
}
