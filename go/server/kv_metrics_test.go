package server_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ailakshya/infergo/server"
)

// TestKVPageMetricsExported verifies that the KV page metrics are exported in /metrics output.
// T1: infergo_kv_pages_free appears in metrics output
// T2: infergo_kv_pages_total appears in metrics output
// T3: UpdateKVPages sets the correct values visible in /metrics output
func TestKVPageMetricsExported(t *testing.T) {
	m := server.NewMetrics()

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from /metrics, got %d", rr.Code)
	}

	body := rr.Body.String()

	// T1: infergo_kv_pages_free must appear
	if !strings.Contains(body, "infergo_kv_pages_free") {
		t.Error("T1 FAIL: infergo_kv_pages_free not found in /metrics output")
	}

	// T2: infergo_kv_pages_total must appear
	if !strings.Contains(body, "infergo_kv_pages_total") {
		t.Error("T2 FAIL: infergo_kv_pages_total not found in /metrics output")
	}
}

// TestUpdateKVPages verifies that UpdateKVPages sets gauge values correctly.
// T3: calling UpdateKVPages with (model, free, total) results in correct label+value in /metrics
func TestUpdateKVPages(t *testing.T) {
	m := server.NewMetrics()
	m.UpdateKVPages("llama3-8b-q4", 42, 128)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from /metrics, got %d", rr.Code)
	}

	body := rr.Body.String()

	// T3a: free pages value should appear with the model label
	if !strings.Contains(body, `infergo_kv_pages_free{model="llama3-8b-q4"} 42`) {
		t.Errorf("T3a FAIL: expected infergo_kv_pages_free{model=\"llama3-8b-q4\"} 42 in /metrics output\ngot:\n%s", body)
	}

	// T3b: total pages value should appear with the model label
	if !strings.Contains(body, `infergo_kv_pages_total{model="llama3-8b-q4"} 128`) {
		t.Errorf("T3b FAIL: expected infergo_kv_pages_total{model=\"llama3-8b-q4\"} 128 in /metrics output\ngot:\n%s", body)
	}
}
