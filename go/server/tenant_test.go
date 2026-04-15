package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestTenantStore_UpsertAndGet(t *testing.T) {
	store := NewTenantStore()

	cfg := TenantConfig{
		ID:            "t1",
		APIKey:        "key-abc",
		AllowedModels: []string{"llama3"},
		MaxTokens:     1000,
		MonthlyQuota:  100000,
	}
	store.Upsert(cfg)

	got := store.Get("t1")
	if got == nil {
		t.Fatal("expected tenant, got nil")
	}
	if got.APIKey != "key-abc" {
		t.Errorf("expected api_key=key-abc, got %q", got.APIKey)
	}

	// Lookup by key.
	byKey := store.GetByKey("key-abc")
	if byKey == nil || byKey.ID != "t1" {
		t.Error("GetByKey failed")
	}

	// Unknown key.
	if store.GetByKey("unknown") != nil {
		t.Error("expected nil for unknown key")
	}
}

func TestTenantStore_QuotaEnforcement(t *testing.T) {
	store := NewTenantStore()
	store.Upsert(TenantConfig{
		ID:           "t1",
		APIKey:       "key-1",
		MonthlyQuota: 100,
	})

	// Record 90 tokens — should succeed.
	if !store.RecordUsage("t1", 90) {
		t.Fatal("expected usage recording to succeed")
	}

	// Record 20 more — exceeds quota.
	if store.RecordUsage("t1", 20) {
		t.Fatal("expected usage recording to fail (quota exceeded)")
	}

	// Check usage.
	usage := store.GetUsage("t1")
	if usage == nil {
		t.Fatal("expected usage, got nil")
	}
	if usage.TotalTokens != 90 {
		t.Errorf("expected 90 tokens, got %d", usage.TotalTokens)
	}
	if usage.QuotaUsedPct != 90.0 {
		t.Errorf("expected 90%% quota used, got %.1f%%", usage.QuotaUsedPct)
	}
}

func TestTenantStore_ModelRestriction(t *testing.T) {
	store := NewTenantStore()
	store.Upsert(TenantConfig{
		ID:            "t1",
		APIKey:        "key-1",
		AllowedModels: []string{"llama3", "bert"},
	})

	if !store.CheckModel("t1", "llama3") {
		t.Error("llama3 should be allowed")
	}
	if !store.CheckModel("t1", "bert") {
		t.Error("bert should be allowed")
	}
	if store.CheckModel("t1", "gpt4") {
		t.Error("gpt4 should not be allowed")
	}

	// No restriction for unknown tenant.
	if !store.CheckModel("unknown", "anything") {
		t.Error("unknown tenant should have no model restriction")
	}
}

func TestTenantStore_MaxTokens(t *testing.T) {
	store := NewTenantStore()
	store.Upsert(TenantConfig{
		ID:        "t1",
		APIKey:    "key-1",
		MaxTokens: 500,
	})

	// Request within limit.
	if got := store.CheckMaxTokens("t1", 100); got != 100 {
		t.Errorf("expected 100, got %d", got)
	}

	// Request exceeding limit.
	if got := store.CheckMaxTokens("t1", 1000); got != 500 {
		t.Errorf("expected capped to 500, got %d", got)
	}

	// Zero request.
	if got := store.CheckMaxTokens("t1", 0); got != 500 {
		t.Errorf("expected capped to 500, got %d", got)
	}

	// Unknown tenant.
	if got := store.CheckMaxTokens("unknown", 1000); got != 1000 {
		t.Errorf("expected 1000 for unknown tenant, got %d", got)
	}
}

func TestTenantStore_Delete(t *testing.T) {
	store := NewTenantStore()
	store.Upsert(TenantConfig{ID: "t1", APIKey: "k1"})

	if !store.Delete("t1") {
		t.Error("delete should return true for existing tenant")
	}
	if store.Delete("t1") {
		t.Error("delete should return false for already-deleted tenant")
	}
	if store.Get("t1") != nil {
		t.Error("tenant should be gone after delete")
	}
}

func TestHandleTenantCreate(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	store := NewTenantStore()
	srv.tenants = store

	body := `{"id":"tenant-1","api_key":"sk-abc123","allowed_models":["llama3"],"monthly_quota":50000}`
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/tenants", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	// Verify tenant was created.
	cfg := store.Get("tenant-1")
	if cfg == nil {
		t.Fatal("tenant not found after create")
	}
	if cfg.APIKey != "sk-abc123" {
		t.Errorf("expected api_key=sk-abc123, got %q", cfg.APIKey)
	}
}

func TestHandleTenantUsage(t *testing.T) {
	reg := NewRegistry()
	srv := NewServer(reg)
	store := NewTenantStore()
	srv.tenants = store

	store.Upsert(TenantConfig{
		ID:           "tenant-1",
		APIKey:       "key-1",
		MonthlyQuota: 100000,
	})
	store.RecordUsage("tenant-1", 5000)

	req := httptest.NewRequest(http.MethodGet, "/v1/admin/tenants/tenant-1/usage", nil)
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var usage TenantUsageJSON
	json.NewDecoder(rr.Body).Decode(&usage)
	if usage.TotalTokens != 5000 {
		t.Errorf("expected 5000 tokens, got %d", usage.TotalTokens)
	}
	if usage.QuotaUsedPct != 5.0 {
		t.Errorf("expected 5%%, got %.1f%%", usage.QuotaUsedPct)
	}
}

func TestTenantMiddleware_QuotaExceeded(t *testing.T) {
	store := NewTenantStore()
	store.Upsert(TenantConfig{
		ID:           "t1",
		APIKey:       "key-1",
		MonthlyQuota: 10,
	})
	// Exhaust quota.
	store.RecordUsage("t1", 10)

	handler := TenantMiddleware(store)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Authorization", "Bearer key-1")
	rr := httptest.NewRecorder()
	handler.ServeHTTP(rr, req)

	if rr.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", rr.Code)
	}
}
