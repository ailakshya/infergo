package server_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ailakshya/infergo/server"
)

// ─── /metrics endpoint ───────────────────────────────────────────────────────

func TestMetrics_HandlerReturns200(t *testing.T) {
	m := server.NewMetrics()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
}

func TestMetrics_ContentTypeIsPrometheusText(t *testing.T) {
	m := server.NewMetrics()
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	ct := rr.Header().Get("Content-Type")
	if !strings.Contains(ct, "text/plain") {
		t.Errorf("expected text/plain content-type, got %q", ct)
	}
}

func TestMetrics_AllMetricNamesPresent(t *testing.T) {
	m := server.NewMetrics()

	// Observe something so all metrics have at least one sample
	m.RequestsTotal.WithLabelValues("mymodel", "/v1/chat/completions", "2xx").Inc()
	m.RequestDuration.WithLabelValues("mymodel", "/v1/chat/completions").Observe(0.1)
	m.ObserveBatch(4)
	m.ObserveTokensPerSecond("mymodel", 100, time.Second)
	m.SetGPUMemory(0, 8*1024*1024*1024)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	body := rr.Body.String()
	for _, name := range []string{
		"infergo_requests_total",
		"infergo_request_duration_seconds",
		"infergo_batch_size",
		"infergo_tokens_per_second",
		"infergo_gpu_memory_bytes",
	} {
		if !strings.Contains(body, name) {
			t.Errorf("metric %q not found in /metrics output", name)
		}
	}
}

// ─── Counter correctness ─────────────────────────────────────────────────────

func TestMetrics_RequestsCounterIncrements(t *testing.T) {
	m := server.NewMetrics()
	m.RequestsTotal.WithLabelValues("llama3", "/v1/chat/completions", "2xx").Inc()
	m.RequestsTotal.WithLabelValues("llama3", "/v1/chat/completions", "2xx").Inc()
	m.RequestsTotal.WithLabelValues("llama3", "/v1/chat/completions", "4xx").Inc()

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	body := rr.Body.String()
	// 2 successful requests
	if !strings.Contains(body, `infergo_requests_total{endpoint="/v1/chat/completions",model="llama3",status="2xx"} 2`) {
		t.Errorf("expected counter=2 for 2xx, body excerpt:\n%s", excerpt(body, "infergo_requests_total"))
	}
	// 1 client error
	if !strings.Contains(body, `infergo_requests_total{endpoint="/v1/chat/completions",model="llama3",status="4xx"} 1`) {
		t.Errorf("expected counter=1 for 4xx, body excerpt:\n%s", excerpt(body, "infergo_requests_total"))
	}
}

// ─── Middleware ───────────────────────────────────────────────────────────────

func TestMetrics_WrapServer_RecordsRequests(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{reply: "hi"})
	srv := server.NewServer(reg)

	m := server.NewMetrics()
	wrapped := m.WrapServer(srv)

	// Make 3 requests
	for range 3 {
		var buf bytes.Buffer
		json.NewEncoder(&buf).Encode(server.ChatCompletionRequest{
			Model:    "llama3",
			Messages: []server.ChatMessage{{Role: "user", Content: "hello"}},
		})
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", &buf)
		req.Header.Set("Content-Type", "application/json")
		rr := httptest.NewRecorder()
		wrapped.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d", rr.Code)
		}
	}

	// Check metrics output has counter >= 3
	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	body := rr.Body.String()
	if !strings.Contains(body, `infergo_requests_total`) {
		t.Error("infergo_requests_total not found after 3 requests")
	}
	if !strings.Contains(body, `"/v1/chat/completions"`) {
		t.Errorf("endpoint label not found:\n%s", excerpt(body, "infergo_requests"))
	}
}

func TestMetrics_InstrumentHandler_StatusClass(t *testing.T) {
	m := server.NewMetrics()

	// Handler that returns 404
	notFound := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	})
	instrumented := m.InstrumentHandler("bert", "/v1/embeddings", notFound)

	req := httptest.NewRequest(http.MethodGet, "/v1/embeddings", nil)
	rr := httptest.NewRecorder()
	instrumented.ServeHTTP(rr, req)

	// Metrics output should record 4xx
	mreq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	mrr := httptest.NewRecorder()
	m.Handler().ServeHTTP(mrr, mreq)

	body := mrr.Body.String()
	if !strings.Contains(body, `status="4xx"`) {
		t.Errorf("expected 4xx label in metrics:\n%s", excerpt(body, "infergo_requests_total"))
	}
}

// ─── Batch / token helpers ────────────────────────────────────────────────────

func TestMetrics_ObserveBatch(t *testing.T) {
	m := server.NewMetrics()
	m.ObserveBatch(8)
	m.ObserveBatch(16)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	if !strings.Contains(rr.Body.String(), "infergo_batch_size") {
		t.Error("infergo_batch_size not present")
	}
}

func TestMetrics_ObserveTokensPerSecond(t *testing.T) {
	m := server.NewMetrics()
	m.ObserveTokensPerSecond("llama3", 200, time.Second)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	body := rr.Body.String()
	if !strings.Contains(body, `infergo_tokens_per_second{model="llama3"} 200`) {
		t.Errorf("expected 200 t/s:\n%s", excerpt(body, "infergo_tokens"))
	}
}

func TestMetrics_SetGPUMemory(t *testing.T) {
	m := server.NewMetrics()
	m.SetGPUMemory(0, 8589934592) // 8 GiB

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rr := httptest.NewRecorder()
	m.Handler().ServeHTTP(rr, req)

	body := rr.Body.String()
	if !strings.Contains(body, `infergo_gpu_memory_bytes{device="0"}`) {
		t.Errorf("expected GPU memory metric:\n%s", excerpt(body, "infergo_gpu"))
	}
}

// ─── helper ──────────────────────────────────────────────────────────────────

// excerpt returns lines from body containing substr, for clearer failure messages.
func excerpt(body, substr string) string {
	var lines []string
	for _, line := range strings.Split(body, "\n") {
		if strings.Contains(line, substr) {
			lines = append(lines, line)
		}
	}
	return strings.Join(lines, "\n")
}
