package server_test

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ailakshya/infergo/server"
)

func newHealthRequest(path string) *http.Request {
	return httptest.NewRequest(http.MethodGet, path, nil)
}

// ─── /health/live ─────────────────────────────────────────────────────────────

func TestHealth_Live_Returns200(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)

	rr := httptest.NewRecorder()
	h.LiveHandler(rr, newHealthRequest("/health/live"))

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHealth_Live_ResponseJSON(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)

	rr := httptest.NewRecorder()
	h.LiveHandler(rr, newHealthRequest("/health/live"))

	var status server.HealthStatus
	if err := json.NewDecoder(rr.Body).Decode(&status); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if status.Status != "ok" {
		t.Errorf("expected status=ok, got %q", status.Status)
	}
	if status.Timestamp == "" {
		t.Error("expected non-empty timestamp")
	}
}

func TestHealth_Live_ForcedUnhealthy(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)
	h.SetLive(false)

	rr := httptest.NewRecorder()
	h.LiveHandler(rr, newHealthRequest("/health/live"))

	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503, got %d", rr.Code)
	}
	var status server.HealthStatus
	json.NewDecoder(rr.Body).Decode(&status)
	if status.Status != "degraded" {
		t.Errorf("expected status=degraded, got %q", status.Status)
	}
}

func TestHealth_Live_RecoverAfterForce(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)
	h.SetLive(false)
	h.SetLive(true)

	rr := httptest.NewRecorder()
	h.LiveHandler(rr, newHealthRequest("/health/live"))

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 after recovery, got %d", rr.Code)
	}
}

// ─── /health/ready ────────────────────────────────────────────────────────────

func TestHealth_Ready_NoModelsRequired(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)

	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHealth_Ready_NotReadyWithoutModels(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 1) // require at least 1 model

	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))

	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when no models loaded, got %d", rr.Code)
	}
	var status server.HealthStatus
	json.NewDecoder(rr.Body).Decode(&status)
	if status.Status != "not_ready" {
		t.Errorf("expected status=not_ready, got %q", status.Status)
	}
}

func TestHealth_Ready_ReadyAfterModelLoad(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 1)

	// Not ready yet
	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))
	if rr.Code != http.StatusServiceUnavailable {
		t.Fatalf("expected 503 before load, got %d", rr.Code)
	}

	// Load a model
	reg.Load("llama3", &mockLLM{})

	// Now ready
	rr = httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))
	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 after load, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHealth_Ready_NotReadyAfterUnload(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 1)
	reg.Load("llama3", &mockLLM{})

	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}

	reg.Unload("llama3")

	rr = httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))
	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 after unload, got %d", rr.Code)
	}
}

func TestHealth_Ready_ExtraCheckBlocks(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{})
	h := server.NewHealthChecker(reg, 1)
	h.AddReadyCheck("gpu", func() error {
		return errors.New("GPU not available")
	})

	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))

	if rr.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when extra check fails, got %d", rr.Code)
	}
	var status server.HealthStatus
	json.NewDecoder(rr.Body).Decode(&status)
	if status.Details["check"] == "" {
		t.Error("expected check detail in response")
	}
}

func TestHealth_Ready_ExtraCheckPasses(t *testing.T) {
	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{})
	h := server.NewHealthChecker(reg, 1)
	h.AddReadyCheck("gpu", func() error { return nil })

	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))

	if rr.Code != http.StatusOK {
		t.Errorf("expected 200 when all checks pass, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestHealth_Ready_ContentTypeJSON(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)

	rr := httptest.NewRecorder()
	h.ReadyHandler(rr, newHealthRequest("/health/ready"))

	ct := rr.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("expected application/json, got %q", ct)
	}
}

// ─── Route registration ───────────────────────────────────────────────────────

func TestHealth_RegisterRoutes(t *testing.T) {
	reg := server.NewRegistry()
	h := server.NewHealthChecker(reg, 0)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	for _, path := range []string{"/health/live", "/health/ready"} {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, req)
		if rr.Code != http.StatusOK {
			t.Errorf("%s: expected 200, got %d", path, rr.Code)
		}
	}
}
