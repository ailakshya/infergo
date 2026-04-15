package server

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"sync"
	"time"
)

// AuditLogger writes structured audit entries to a JSONL file.
// Thread-safe. Supports hash-only mode for privacy.
type AuditLogger struct {
	mu       sync.Mutex
	file     *os.File
	hashOnly bool // if true, store prompt hash instead of text
}

// AuditEntry is one audit log line.
type AuditEntry struct {
	Timestamp  string `json:"timestamp"`
	APIKey     string `json:"api_key,omitempty"` // masked: last 4 chars
	Model      string `json:"model"`
	PromptHash string `json:"prompt_hash"`
	Prompt     string `json:"prompt,omitempty"` // only if !hashOnly
	Tokens     int    `json:"tokens"`
	LatencyMs  int    `json:"latency_ms"`
	Status     int    `json:"status"`
}

// NewAuditLogger creates an audit logger. Path is the JSONL file path.
func NewAuditLogger(path string, hashOnly bool) (*AuditLogger, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0600)
	if err != nil {
		return nil, fmt.Errorf("audit: open %s: %w", path, err)
	}
	return &AuditLogger{file: f, hashOnly: hashOnly}, nil
}

// Log writes an audit entry.
func (a *AuditLogger) Log(entry AuditEntry) {
	entry.Timestamp = time.Now().UTC().Format(time.RFC3339)

	// Hash the prompt
	h := sha256.Sum256([]byte(entry.Prompt))
	entry.PromptHash = hex.EncodeToString(h[:8]) // first 8 bytes = 16 hex chars

	if a.hashOnly {
		entry.Prompt = "" // don't store prompt text
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	json.NewEncoder(a.file).Encode(entry)
}

// Close closes the audit file.
func (a *AuditLogger) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.file.Close()
}
