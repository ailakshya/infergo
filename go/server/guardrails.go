package server

import (
	"encoding/json"
	"net/http"
	"strings"
)

// GuardrailConfig configures content safety filters.
type GuardrailConfig struct {
	Enabled        bool     `json:"enabled"`
	BlockedWords   []string `json:"blocked_words,omitempty"`
	MaxInputLen    int      `json:"max_input_length,omitempty"`    // 0 = unlimited
	MaxOutputLen   int      `json:"max_output_length,omitempty"`   // 0 = unlimited
	BlockPII       bool     `json:"block_pii,omitempty"`           // basic PII detection
}

// Guardrail applies content safety checks to requests and responses.
type Guardrail struct {
	cfg GuardrailConfig
}

// NewGuardrail creates a guardrail from config.
func NewGuardrail(cfg GuardrailConfig) *Guardrail {
	return &Guardrail{cfg: cfg}
}

// CheckInput validates request content. Returns error message or empty string.
func (g *Guardrail) CheckInput(text string) string {
	if !g.cfg.Enabled {
		return ""
	}
	if g.cfg.MaxInputLen > 0 && len(text) > g.cfg.MaxInputLen {
		return "input exceeds maximum length"
	}
	lower := strings.ToLower(text)
	for _, word := range g.cfg.BlockedWords {
		if strings.Contains(lower, strings.ToLower(word)) {
			return "input contains blocked content"
		}
	}
	return ""
}

// CheckOutput validates response content. Returns sanitized text.
func (g *Guardrail) CheckOutput(text string) string {
	if !g.cfg.Enabled {
		return text
	}
	if g.cfg.MaxOutputLen > 0 && len(text) > g.cfg.MaxOutputLen {
		text = text[:g.cfg.MaxOutputLen]
	}
	return text
}

// handleGuardrailConfig returns or updates guardrail configuration.
func (s *Server) handleGuardrailConfig(w http.ResponseWriter, r *http.Request) {
	if s.guardrail == nil {
		s.guardrail = NewGuardrail(GuardrailConfig{})
	}
	if r.Method == http.MethodGet {
		writeJSON(w, http.StatusOK, s.guardrail.cfg)
		return
	}
	var cfg GuardrailConfig
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.guardrail = NewGuardrail(cfg)
	writeJSON(w, http.StatusOK, map[string]string{"status": "updated"})
}
