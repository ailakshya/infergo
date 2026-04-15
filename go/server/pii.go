package server

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"regexp"
	"strings"
)

// PIIType identifies the category of detected personal data.
type PIIType string

const (
	PIIEmail      PIIType = "EMAIL"
	PIIPhone      PIIType = "PHONE"
	PIISSN        PIIType = "SSN"
	PIICreditCard PIIType = "CREDIT_CARD"
	PIIIPAddress  PIIType = "IP_ADDRESS"
)

// PIIDetector scans text for personal identifiable information and applies
// the configured policy (block, redact, or log-only).
type PIIDetector struct {
	mode     string // "block", "redact", "log"
	patterns map[PIIType]*regexp.Regexp
}

// NewPIIDetector creates a detector with the given mode.
// Valid modes: "block" (reject request), "redact" (replace PII), "log" (warn only).
// An empty or unrecognised mode defaults to "log".
func NewPIIDetector(mode string) *PIIDetector {
	switch mode {
	case "block", "redact", "log":
	default:
		mode = "log"
	}

	return &PIIDetector{
		mode: mode,
		patterns: map[PIIType]*regexp.Regexp{
			PIIEmail:      regexp.MustCompile(`\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b`),
			PIIPhone:      regexp.MustCompile(`\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b`),
			PIISSN:        regexp.MustCompile(`\b\d{3}-\d{2}-\d{4}\b`),
			PIICreditCard: regexp.MustCompile(`\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b`),
			PIIIPAddress:  regexp.MustCompile(`\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`),
		},
	}
}

// PIIMatch records a single occurrence of detected PII.
type PIIMatch struct {
	Type  PIIType
	Value string
}

// Detect scans text and returns all PII matches found.
func (d *PIIDetector) Detect(text string) []PIIMatch {
	var matches []PIIMatch
	for piiType, re := range d.patterns {
		for _, m := range re.FindAllString(text, -1) {
			matches = append(matches, PIIMatch{Type: piiType, Value: m})
		}
	}
	return matches
}

// ContainsPII returns true if any PII pattern matches the text.
func (d *PIIDetector) ContainsPII(text string) bool {
	for _, re := range d.patterns {
		if re.MatchString(text) {
			return true
		}
	}
	return false
}

// Redact replaces all PII occurrences with [REDACTED_<TYPE>] placeholders.
func (d *PIIDetector) Redact(text string) string {
	for piiType, re := range d.patterns {
		placeholder := fmt.Sprintf("[REDACTED_%s]", piiType)
		text = re.ReplaceAllString(text, placeholder)
	}
	return text
}

// Mode returns the detector's current operating mode.
func (d *PIIDetector) Mode() string {
	return d.mode
}

// DetectedTypes returns deduplicated PII type names from a match list.
func DetectedTypes(matches []PIIMatch) []string {
	seen := make(map[PIIType]bool)
	var types []string
	for _, m := range matches {
		if !seen[m.Type] {
			seen[m.Type] = true
			types = append(types, string(m.Type))
		}
	}
	return types
}

// PIIMiddleware returns an HTTP middleware that scans JSON request bodies for
// PII before forwarding to the next handler.
//
// Behaviour depends on the detector mode:
//   - "block":  respond 400 if PII detected
//   - "redact": replace PII in the body before forwarding
//   - "log":    log a warning but forward unchanged
//
// The middleware sets the X-PII-Detected response header to "true" whenever
// PII is found (in redact and log modes the request still proceeds).
// Non-JSON and bodyless requests pass through untouched.
func PIIMiddleware(detector *PIIDetector) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Only scan methods that carry a body
			if r.Body == nil || r.ContentLength == 0 {
				next.ServeHTTP(w, r)
				return
			}

			// Read body
			body, err := io.ReadAll(r.Body)
			r.Body.Close()
			if err != nil {
				writeError(w, http.StatusBadRequest, "failed to read request body")
				return
			}

			text := string(body)

			matches := detector.Detect(text)
			if len(matches) == 0 {
				// No PII — restore body and proceed
				r.Body = io.NopCloser(bytes.NewReader(body))
				next.ServeHTTP(w, r)
				return
			}

			types := DetectedTypes(matches)
			typeStr := strings.Join(types, ", ")

			switch detector.mode {
			case "block":
				w.Header().Set("X-PII-Detected", "true")
				writeError(w, http.StatusBadRequest,
					fmt.Sprintf("request blocked: PII detected (%s)", typeStr))
				return

			case "redact":
				w.Header().Set("X-PII-Detected", "true")
				redacted := detector.Redact(text)
				r.Body = io.NopCloser(strings.NewReader(redacted))
				r.ContentLength = int64(len(redacted))
				next.ServeHTTP(w, r)
				return

			default: // "log"
				w.Header().Set("X-PII-Detected", "true")
				log.Printf("[pii] WARNING: PII detected in request (%s), allowing through (mode=log)", typeStr)
				r.Body = io.NopCloser(bytes.NewReader(body))
				next.ServeHTTP(w, r)
				return
			}
		})
	}
}

// handlePIIConfig returns or updates PII detector configuration via JSON API.
func (s *Server) handlePIIConfig(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		mode := "disabled"
		if s.pii != nil {
			mode = s.pii.Mode()
		}
		writeJSON(w, http.StatusOK, map[string]string{"mode": mode})
		return
	}
	var cfg struct {
		Mode string `json:"mode"`
	}
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	s.pii = NewPIIDetector(cfg.Mode)
	writeJSON(w, http.StatusOK, map[string]string{"status": "updated", "mode": s.pii.Mode()})
}
