package server

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

// ─── Auth Middleware Tests (OPT-11) ──────────────────────────────────────────

func okHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
}

// TestAuthMiddleware_NoKey — empty apiKey, all requests pass (OPT-11-T5)
func TestAuthMiddleware_NoKey(t *testing.T) {
	mw := AuthMiddleware("")
	handler := mw(okHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

// TestAuthMiddleware_ValidKey — correct Bearer token, request passes (OPT-11-T1)
func TestAuthMiddleware_ValidKey(t *testing.T) {
	mw := AuthMiddleware("secret-key")
	handler := mw(okHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer secret-key")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", rec.Code)
	}
}

// TestAuthMiddleware_MissingHeader — no header → 401 (OPT-11-T2)
func TestAuthMiddleware_MissingHeader(t *testing.T) {
	mw := AuthMiddleware("secret-key")
	handler := mw(okHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}

// TestAuthMiddleware_WrongKey — wrong token → 401 (OPT-11-T3)
func TestAuthMiddleware_WrongKey(t *testing.T) {
	mw := AuthMiddleware("secret-key")
	handler := mw(okHandler())

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", rec.Code)
	}
}

// TestAuthMiddleware_HealthExempt — /health/live with wrong key → 200 (OPT-11-T4)
func TestAuthMiddleware_HealthExempt(t *testing.T) {
	mw := AuthMiddleware("secret-key")
	handler := mw(okHandler())

	req := httptest.NewRequest(http.MethodGet, "/health/live", nil)
	// No Authorization header — would normally be 401
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for exempt /health/live, got %d", rec.Code)
	}
}

// TestAuthMiddleware_MetricsExempt — /metrics with wrong key → 200
func TestAuthMiddleware_MetricsExempt(t *testing.T) {
	mw := AuthMiddleware("secret-key")
	handler := mw(okHandler())

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("expected 200 for exempt /metrics, got %d", rec.Code)
	}
}

// ─── Rate Limiter Tests (OPT-12) ─────────────────────────────────────────────

// TestRateLimiter_AllowsUnderLimit — rps=10, 5 requests: all pass (OPT-12-T1 partial)
func TestRateLimiter_AllowsUnderLimit(t *testing.T) {
	rl := NewRateLimiter(10)
	handler := rl.Middleware()(okHandler())

	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.RemoteAddr = "10.0.0.1:1234"
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("request %d: expected 200, got %d", i+1, rec.Code)
		}
	}
}

// TestRateLimiter_Blocks429 — rps=1, fire 5 requests instantly: some get 429 (OPT-12-T1)
func TestRateLimiter_Blocks429(t *testing.T) {
	rl := NewRateLimiter(1)
	handler := rl.Middleware()(okHandler())

	got429 := 0
	got200 := 0
	for i := 0; i < 5; i++ {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.RemoteAddr = "10.0.0.2:5678"
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		switch rec.Code {
		case http.StatusOK:
			got200++
		case http.StatusTooManyRequests:
			got429++
		default:
			t.Errorf("unexpected status %d", rec.Code)
		}
	}

	if got429 == 0 {
		t.Errorf("expected some 429 responses, got none (200=%d, 429=%d)", got200, got429)
	}
}

// TestRateLimiter_RetryAfterHeader — 429 response includes Retry-After: 1 (OPT-12-T2)
func TestRateLimiter_RetryAfterHeader(t *testing.T) {
	rl := NewRateLimiter(1)
	handler := rl.Middleware()(okHandler())

	// First request consumes the token
	req1 := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req1.RemoteAddr = "10.0.0.3:1111"
	httptest.NewRecorder()
	handler.ServeHTTP(httptest.NewRecorder(), req1)

	// Second request should be rate limited
	req2 := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req2.RemoteAddr = "10.0.0.3:1111"
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, req2)

	if rec.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", rec.Code)
	}
	if ra := rec.Header().Get("Retry-After"); ra != "1" {
		t.Errorf("expected Retry-After: 1, got %q", ra)
	}
}

// TestRateLimiter_PerIPIsolation — different IPs have independent buckets (OPT-12-T3)
func TestRateLimiter_PerIPIsolation(t *testing.T) {
	rl := NewRateLimiter(1)
	handler := rl.Middleware()(okHandler())

	// Exhaust IP A
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.RemoteAddr = "192.168.1.1:9000"
	handler.ServeHTTP(httptest.NewRecorder(), req) // consume token

	req2 := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req2.RemoteAddr = "192.168.1.1:9000"
	rec2 := httptest.NewRecorder()
	handler.ServeHTTP(rec2, req2)
	if rec2.Code != http.StatusTooManyRequests {
		t.Errorf("IP A second request: expected 429, got %d", rec2.Code)
	}

	// IP B should still be allowed
	reqB := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	reqB.RemoteAddr = "192.168.1.2:9000"
	recB := httptest.NewRecorder()
	handler.ServeHTTP(recB, reqB)
	if recB.Code != http.StatusOK {
		t.Errorf("IP B: expected 200 (isolated), got %d", recB.Code)
	}
}

// TestRateLimiter_Disabled — rps=0, all requests pass
func TestRateLimiter_Disabled(t *testing.T) {
	rl := NewRateLimiter(0)
	handler := rl.Middleware()(okHandler())

	for i := 0; i < 20; i++ {
		req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
		req.RemoteAddr = "10.0.0.4:1234"
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("request %d: expected 200, got %d (limiter should be disabled)", i+1, rec.Code)
		}
	}
}

// TestRateLimiter_HealthExempt — /health/ paths bypass rate limiting
func TestRateLimiter_HealthExempt(t *testing.T) {
	rl := NewRateLimiter(1)
	handler := rl.Middleware()(okHandler())

	// Fire many requests to /health/live — all should pass regardless of rps=1
	for i := 0; i < 10; i++ {
		req := httptest.NewRequest(http.MethodGet, "/health/live", nil)
		req.RemoteAddr = "10.0.0.5:1234"
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Errorf("health request %d: expected 200 (exempt), got %d", i+1, rec.Code)
		}
	}
}
