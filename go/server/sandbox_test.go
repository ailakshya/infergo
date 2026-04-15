package server

import (
	"context"
	"os/exec"
	"strings"
	"testing"
	"time"
)

// ─── CodeExecutor tests ────────────────────────────────────────────────────

func TestPythonExecution(t *testing.T) {
	if _, err := findExecutable("python3"); err != nil {
		t.Skip("python3 not available")
	}

	ce := NewCodeExecutor()
	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "python",
		Code:     "print(2+2)",
	})

	if result.ExitCode != 0 {
		t.Fatalf("exit code = %d, stderr = %q, error = %q", result.ExitCode, result.Stderr, result.Error)
	}
	if strings.TrimSpace(result.Stdout) != "4" {
		t.Errorf("stdout = %q, want %q", result.Stdout, "4\n")
	}
	if result.Duration <= 0 {
		t.Error("duration should be positive")
	}
}

func TestTimeout(t *testing.T) {
	if _, err := findExecutable("python3"); err != nil {
		t.Skip("python3 not available")
	}

	ce := NewCodeExecutor()
	ce.timeout = 1 * time.Second // short timeout

	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "python",
		Code:     "import time\nwhile True:\n    time.sleep(0.1)",
		Timeout:  1,
	})

	if result.Error == "" {
		t.Error("expected timeout error")
	}
	if !strings.Contains(result.Error, "timed out") {
		t.Errorf("error = %q, want to contain 'timed out'", result.Error)
	}
}

func TestStderr(t *testing.T) {
	if _, err := findExecutable("python3"); err != nil {
		t.Skip("python3 not available")
	}

	ce := NewCodeExecutor()
	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "python",
		Code:     "import sys\nsys.stderr.write('error message\\n')",
	})

	if result.ExitCode != 0 {
		t.Fatalf("exit code = %d, error = %q", result.ExitCode, result.Error)
	}
	if !strings.Contains(result.Stderr, "error message") {
		t.Errorf("stderr = %q, want to contain 'error message'", result.Stderr)
	}
}

func TestOutputLimit(t *testing.T) {
	if _, err := findExecutable("python3"); err != nil {
		t.Skip("python3 not available")
	}

	ce := NewCodeExecutor()
	ce.maxOutput = 100 // very small limit for testing

	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "python",
		Code:     "print('A' * 10000)",
	})

	// Output should be truncated around the limit
	if len(result.Stdout) > 200 { // some slack for truncation marker
		t.Errorf("stdout length = %d, expected truncated to ~100", len(result.Stdout))
	}
	if !strings.Contains(result.Stdout, "truncated") {
		t.Error("expected truncation marker in stdout")
	}
}

func TestBashExecution(t *testing.T) {
	if _, err := findExecutable("bash"); err != nil {
		t.Skip("bash not available")
	}

	ce := NewCodeExecutor()
	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "bash",
		Code:     "echo hello",
	})

	if result.ExitCode != 0 {
		t.Fatalf("exit code = %d, stderr = %q, error = %q", result.ExitCode, result.Stderr, result.Error)
	}
	if strings.TrimSpace(result.Stdout) != "hello" {
		t.Errorf("stdout = %q, want %q", result.Stdout, "hello\n")
	}
}

func TestExitCode(t *testing.T) {
	if _, err := findExecutable("python3"); err != nil {
		t.Skip("python3 not available")
	}

	ce := NewCodeExecutor()
	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "python",
		Code:     "exit(1)",
	})

	if result.ExitCode != 1 {
		t.Errorf("exit code = %d, want 1", result.ExitCode)
	}
}

// ─── Additional unit tests ──────────────────────────────────────────────────

func TestUnsupportedLanguage(t *testing.T) {
	ce := NewCodeExecutor()
	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "ruby",
		Code:     "puts 42",
	})

	if result.ExitCode != -1 {
		t.Errorf("exit code = %d, want -1", result.ExitCode)
	}
	if !strings.Contains(result.Error, "unsupported language") {
		t.Errorf("error = %q, want to contain 'unsupported language'", result.Error)
	}
}

func TestDetectLanguage(t *testing.T) {
	tests := []struct {
		code string
		want string
	}{
		{"#!/usr/bin/env python3\nprint(1)", "python"},
		{"#!/usr/bin/env node\nconsole.log(1)", "javascript"},
		{"#!/bin/bash\necho hi", "bash"},
		{"#!/bin/sh\necho hi", "bash"},
		{"print('hello')", "python"},
		{"console.log('hello')", "javascript"},
		{"package main\nfunc main() {}", "go"},
		{"ls -la", "bash"},
	}

	for _, tt := range tests {
		got := detectLanguage(tt.code)
		if got != tt.want {
			t.Errorf("detectLanguage(%q) = %q, want %q", tt.code[:min(30, len(tt.code))], got, tt.want)
		}
	}
}

func TestLimitedWriter(t *testing.T) {
	var buf strings.Builder
	lw := &limitedWriter{w: &buf, limit: 10}

	// Write exactly at limit
	n, err := lw.Write([]byte("0123456789"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 10 {
		t.Errorf("n = %d, want 10", n)
	}
	if buf.String() != "0123456789" {
		t.Errorf("buf = %q, want %q", buf.String(), "0123456789")
	}

	// Write beyond limit — should be silently discarded
	n, err = lw.Write([]byte("extra"))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// We return len(p) to avoid breaking io.Writer contract
	if n != 5 {
		t.Errorf("n = %d, want 5", n)
	}
	if buf.String() != "0123456789" {
		t.Errorf("buf = %q, want %q (no extra data)", buf.String(), "0123456789")
	}
}

func TestCodeExecutorTool(t *testing.T) {
	if _, err := findExecutable("python3"); err != nil {
		t.Skip("python3 not available")
	}

	tool := codeExecutorTool()
	if tool.Name != "code_executor" {
		t.Errorf("name = %q, want %q", tool.Name, "code_executor")
	}

	result, err := tool.Execute(map[string]interface{}{
		"language": "python",
		"code":     "print('agent test')",
	})
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if !strings.Contains(result, "agent test") {
		t.Errorf("result = %q, want to contain 'agent test'", result)
	}
}

func TestCodeExecutorTool_MissingCode(t *testing.T) {
	tool := codeExecutorTool()
	_, err := tool.Execute(map[string]interface{}{})
	if err == nil {
		t.Error("expected error for missing code")
	}
}

func TestTimeoutCap(t *testing.T) {
	if _, err := findExecutable("bash"); err != nil {
		t.Skip("bash not available")
	}

	ce := NewCodeExecutor()
	// Request 60s but cap should enforce 30s
	result := ce.Execute(context.Background(), ExecutionRequest{
		Language: "bash",
		Code:     "echo ok",
		Timeout:  60,
	})

	if result.ExitCode != 0 {
		t.Fatalf("exit code = %d, error = %q", result.ExitCode, result.Error)
	}
	if strings.TrimSpace(result.Stdout) != "ok" {
		t.Errorf("stdout = %q, want %q", result.Stdout, "ok\n")
	}
}

// findExecutable checks if a binary is available on PATH.
func findExecutable(name string) (string, error) {
	return exec.LookPath(name)
}
