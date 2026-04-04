package server_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/ailakshya/infergo/server"
)

// TestKEDAMetricsExported verifies the KEDA-relevant metrics are exported in /metrics output.
// T1: infergo_queue_depth appears in metrics output
// T2: infergo_active_sequences appears in metrics output
func TestKEDAMetricsExported(t *testing.T) {
	m := server.NewMetrics()

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200 from /metrics, got %d", rr.Code)
	}

	body := rr.Body.String()

	// T1: infergo_queue_depth must appear
	if !strings.Contains(body, "infergo_queue_depth") {
		t.Error("T1 FAIL: infergo_queue_depth not found in /metrics output")
	}

	// T2: infergo_active_sequences must appear
	if !strings.Contains(body, "infergo_active_sequences") {
		t.Error("T2 FAIL: infergo_active_sequences not found in /metrics output")
	}
}
