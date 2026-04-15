package server

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ─── Detection tests ─────────────────────────────────────────────────────────

func TestDetectEmail(t *testing.T) {
	d := NewPIIDetector("log")

	tests := []struct {
		input string
		found bool
	}{
		{"contact me at user@example.com", true},
		{"user.name+tag@sub.domain.co", true},
		{"ALICE@EXAMPLE.ORG is her email", true},
		{"no email here", false},
		{"user@", false},
		{"@domain.com", false},
	}

	for _, tt := range tests {
		matches := d.Detect(tt.input)
		hasEmail := false
		for _, m := range matches {
			if m.Type == PIIEmail {
				hasEmail = true
			}
		}
		if hasEmail != tt.found {
			t.Errorf("Detect(%q) email=%v, want %v", tt.input, hasEmail, tt.found)
		}
	}
}

func TestDetectPhone(t *testing.T) {
	d := NewPIIDetector("log")

	tests := []struct {
		input string
		found bool
	}{
		{"call 555-123-4567", true},
		{"(555) 123-4567 office", true},
		{"+1-555-123-4567", true},
		{"1.555.123.4567", true},
		{"555 123 4567", true},
		{"12345", false},
		{"no phone", false},
	}

	for _, tt := range tests {
		matches := d.Detect(tt.input)
		hasPhone := false
		for _, m := range matches {
			if m.Type == PIIPhone {
				hasPhone = true
			}
		}
		if hasPhone != tt.found {
			t.Errorf("Detect(%q) phone=%v, want %v", tt.input, hasPhone, tt.found)
		}
	}
}

func TestDetectSSN(t *testing.T) {
	d := NewPIIDetector("log")

	tests := []struct {
		input string
		found bool
	}{
		{"SSN: 123-45-6789", true},
		{"my ssn is 999-88-7777", true},
		{"123456789", false},
		{"123-456-789", false},
		{"no ssn here", false},
	}

	for _, tt := range tests {
		matches := d.Detect(tt.input)
		hasSSN := false
		for _, m := range matches {
			if m.Type == PIISSN {
				hasSSN = true
			}
		}
		if hasSSN != tt.found {
			t.Errorf("Detect(%q) ssn=%v, want %v", tt.input, hasSSN, tt.found)
		}
	}
}

func TestDetectCreditCard(t *testing.T) {
	d := NewPIIDetector("log")

	tests := []struct {
		input string
		found bool
	}{
		{"card: 4111-1111-1111-1111", true},
		{"4111 1111 1111 1111", true},
		{"4111111111111111", true},
		{"411111111111", false}, // too short
		{"no card", false},
	}

	for _, tt := range tests {
		matches := d.Detect(tt.input)
		hasCC := false
		for _, m := range matches {
			if m.Type == PIICreditCard {
				hasCC = true
			}
		}
		if hasCC != tt.found {
			t.Errorf("Detect(%q) cc=%v, want %v", tt.input, hasCC, tt.found)
		}
	}
}

func TestDetectIP(t *testing.T) {
	d := NewPIIDetector("log")

	tests := []struct {
		input string
		found bool
	}{
		{"server at 192.168.1.1", true},
		{"IP: 10.0.0.255", true},
		{"from 255.255.255.0 subnet", true},
		{"no ip here", false},
		{"1.2.3", false},
	}

	for _, tt := range tests {
		matches := d.Detect(tt.input)
		hasIP := false
		for _, m := range matches {
			if m.Type == PIIIPAddress {
				hasIP = true
			}
		}
		if hasIP != tt.found {
			t.Errorf("Detect(%q) ip=%v, want %v", tt.input, hasIP, tt.found)
		}
	}
}

// ─── Mode tests ──────────────────────────────────────────────────────────────

func TestRedactMode(t *testing.T) {
	d := NewPIIDetector("redact")

	input := "Email user@example.com, phone 555-123-4567, SSN 123-45-6789"
	redacted := d.Redact(input)

	if strings.Contains(redacted, "user@example.com") {
		t.Error("email not redacted")
	}
	if !strings.Contains(redacted, "[REDACTED_EMAIL]") {
		t.Error("missing [REDACTED_EMAIL] placeholder")
	}
	if strings.Contains(redacted, "555-123-4567") {
		t.Error("phone not redacted")
	}
	if !strings.Contains(redacted, "[REDACTED_PHONE]") {
		t.Error("missing [REDACTED_PHONE] placeholder")
	}
	if strings.Contains(redacted, "123-45-6789") {
		t.Error("SSN not redacted")
	}
	if !strings.Contains(redacted, "[REDACTED_SSN]") {
		t.Error("missing [REDACTED_SSN] placeholder")
	}
}

func TestRedactCreditCard(t *testing.T) {
	d := NewPIIDetector("redact")
	input := "pay with 4111-1111-1111-1111 please"
	redacted := d.Redact(input)

	if strings.Contains(redacted, "4111") {
		t.Error("credit card not redacted")
	}
	if !strings.Contains(redacted, "[REDACTED_CREDIT_CARD]") {
		t.Error("missing [REDACTED_CREDIT_CARD] placeholder")
	}
}

func TestRedactIP(t *testing.T) {
	d := NewPIIDetector("redact")
	input := "connect to 192.168.1.100 now"
	redacted := d.Redact(input)

	if strings.Contains(redacted, "192.168.1.100") {
		t.Error("IP not redacted")
	}
	if !strings.Contains(redacted, "[REDACTED_IP_ADDRESS]") {
		t.Error("missing [REDACTED_IP_ADDRESS] placeholder")
	}
}

func TestBlockMode(t *testing.T) {
	d := NewPIIDetector("block")

	if !d.ContainsPII("contact user@example.com") {
		t.Error("should detect email as PII")
	}
	if d.Mode() != "block" {
		t.Errorf("mode = %q, want block", d.Mode())
	}
}

// ─── False-positive test ─────────────────────────────────────────────────────

func TestCleanTextPasses(t *testing.T) {
	d := NewPIIDetector("block")

	cleanTexts := []string{
		"Hello, how are you today?",
		"The weather is nice in Paris.",
		"Please summarize this document for me.",
		"What is the capital of France?",
		"Generate a haiku about mountains.",
		"Translate 'hello' to Spanish.",
	}

	for _, text := range cleanTexts {
		if d.ContainsPII(text) {
			t.Errorf("false positive on clean text: %q", text)
		}
	}
}

// ─── Multiple PII types in one text ─────────────────────────────────────────

func TestMultiplePII(t *testing.T) {
	d := NewPIIDetector("log")

	input := "Email: admin@corp.com, Phone: (800) 555-1234, SSN: 111-22-3333, " +
		"Card: 5500 0000 0000 0004, IP: 10.0.0.1"

	matches := d.Detect(input)

	typeSet := make(map[PIIType]bool)
	for _, m := range matches {
		typeSet[m.Type] = true
	}

	expected := []PIIType{PIIEmail, PIIPhone, PIISSN, PIICreditCard, PIIIPAddress}
	for _, piiType := range expected {
		if !typeSet[piiType] {
			t.Errorf("missing PII type %s in multi-PII text", piiType)
		}
	}

	// Redact should replace all
	redacted := d.Redact(input)
	if strings.Contains(redacted, "admin@corp.com") {
		t.Error("email still present after redact")
	}
	if strings.Contains(redacted, "111-22-3333") {
		t.Error("SSN still present after redact")
	}
	if strings.Contains(redacted, "10.0.0.1") {
		t.Error("IP still present after redact")
	}
}

// ─── Middleware integration tests ────────────────────────────────────────────

// echoHandler reads the (possibly redacted) body and writes it back.
var echoHandler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
	body, _ := io.ReadAll(r.Body)
	w.WriteHeader(http.StatusOK)
	w.Write(body)
})

func TestMiddlewareBlock(t *testing.T) {
	d := NewPIIDetector("block")
	handler := PIIMiddleware(d)(echoHandler)

	body := `{"prompt": "email me at test@example.com"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", rec.Code)
	}
	if rec.Header().Get("X-PII-Detected") != "true" {
		t.Error("missing X-PII-Detected header")
	}
	respBody := rec.Body.String()
	if !strings.Contains(respBody, "PII detected") {
		t.Errorf("response should mention PII detected, got: %s", respBody)
	}
}

func TestMiddlewareRedact(t *testing.T) {
	d := NewPIIDetector("redact")
	handler := PIIMiddleware(d)(echoHandler)

	body := `{"prompt": "my email is alice@test.org and ssn 222-33-4444"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	if rec.Header().Get("X-PII-Detected") != "true" {
		t.Error("missing X-PII-Detected header")
	}
	respBody := rec.Body.String()
	if strings.Contains(respBody, "alice@test.org") {
		t.Error("email not redacted in forwarded body")
	}
	if !strings.Contains(respBody, "[REDACTED_EMAIL]") {
		t.Error("missing [REDACTED_EMAIL] in forwarded body")
	}
	if strings.Contains(respBody, "222-33-4444") {
		t.Error("SSN not redacted in forwarded body")
	}
	if !strings.Contains(respBody, "[REDACTED_SSN]") {
		t.Error("missing [REDACTED_SSN] in forwarded body")
	}
}

func TestMiddlewareLog(t *testing.T) {
	d := NewPIIDetector("log")
	handler := PIIMiddleware(d)(echoHandler)

	body := `{"prompt": "call 555-123-4567"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", rec.Code)
	}
	if rec.Header().Get("X-PII-Detected") != "true" {
		t.Error("missing X-PII-Detected header in log mode")
	}
	// Body should pass through unchanged
	respBody := rec.Body.String()
	if !strings.Contains(respBody, "555-123-4567") {
		t.Error("log mode should not modify body")
	}
}

func TestMiddlewareCleanPassthrough(t *testing.T) {
	d := NewPIIDetector("block")
	handler := PIIMiddleware(d)(echoHandler)

	body := `{"prompt": "What is the meaning of life?"}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("clean request blocked: status = %d", rec.Code)
	}
	if rec.Header().Get("X-PII-Detected") != "" {
		t.Error("X-PII-Detected should not be set for clean request")
	}
}

func TestMiddlewareNoBody(t *testing.T) {
	d := NewPIIDetector("block")
	handler := PIIMiddleware(d)(echoHandler)

	req := httptest.NewRequest("GET", "/v1/models", nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("bodyless request blocked: status = %d", rec.Code)
	}
}

// ─── Default mode test ───────────────────────────────────────────────────────

func TestDefaultMode(t *testing.T) {
	d := NewPIIDetector("")
	if d.Mode() != "log" {
		t.Errorf("default mode = %q, want log", d.Mode())
	}

	d2 := NewPIIDetector("invalid")
	if d2.Mode() != "log" {
		t.Errorf("invalid mode should default to log, got %q", d2.Mode())
	}
}

// ─── DetectedTypes helper ────────────────────────────────────────────────────

func TestDetectedTypes(t *testing.T) {
	matches := []PIIMatch{
		{Type: PIIEmail, Value: "a@b.com"},
		{Type: PIIEmail, Value: "c@d.com"},
		{Type: PIIPhone, Value: "555-1234"},
	}
	types := DetectedTypes(matches)
	if len(types) != 2 {
		t.Errorf("DetectedTypes returned %d types, want 2 (deduplicated)", len(types))
	}
}
