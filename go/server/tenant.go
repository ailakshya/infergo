package server

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// TenantConfig defines per-API-key resource limits and allowed models.
type TenantConfig struct {
	ID              string   `json:"id"`
	APIKey          string   `json:"api_key"`
	AllowedModels   []string `json:"allowed_models,omitempty"`    // empty = all models allowed
	RateLimit       float64  `json:"rate_limit,omitempty"`        // requests per second (0 = unlimited)
	MaxTokens       int      `json:"max_tokens,omitempty"`        // max tokens per request (0 = unlimited)
	MonthlyQuota    int64    `json:"monthly_quota,omitempty"`     // total tokens per month (0 = unlimited)
}

// TenantUsage tracks token usage for a tenant.
type TenantUsage struct {
	TotalTokens   atomic.Int64
	RequestCount  atomic.Int64
	MonthStart    time.Time
}

// TenantUsageJSON is the JSON-serializable usage report.
type TenantUsageJSON struct {
	TenantID     string `json:"tenant_id"`
	TotalTokens  int64  `json:"total_tokens"`
	RequestCount int64  `json:"request_count"`
	MonthStart   string `json:"month_start"`
	QuotaUsedPct float64 `json:"quota_used_pct,omitempty"`
}

// TenantStore manages tenant configurations and tracks usage.
type TenantStore struct {
	mu      sync.RWMutex
	tenants map[string]*TenantConfig // id -> config
	byKey   map[string]string        // api_key -> tenant id
	usage   map[string]*TenantUsage  // id -> usage
}

// NewTenantStore creates an empty tenant store.
func NewTenantStore() *TenantStore {
	return &TenantStore{
		tenants: make(map[string]*TenantConfig),
		byKey:   make(map[string]string),
		usage:   make(map[string]*TenantUsage),
	}
}

// Upsert creates or updates a tenant configuration.
func (ts *TenantStore) Upsert(cfg TenantConfig) {
	ts.mu.Lock()
	defer ts.mu.Unlock()

	// Remove old key mapping if updating.
	if old, ok := ts.tenants[cfg.ID]; ok {
		delete(ts.byKey, old.APIKey)
	}

	ts.tenants[cfg.ID] = &cfg
	ts.byKey[cfg.APIKey] = cfg.ID

	// Initialize usage if not present.
	if _, ok := ts.usage[cfg.ID]; !ok {
		ts.usage[cfg.ID] = &TenantUsage{
			MonthStart: time.Now().Truncate(24 * time.Hour),
		}
	}
}

// Get returns the tenant config by ID, or nil if not found.
func (ts *TenantStore) Get(id string) *TenantConfig {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	return ts.tenants[id]
}

// GetByKey returns the tenant config for the given API key, or nil if not found.
func (ts *TenantStore) GetByKey(apiKey string) *TenantConfig {
	ts.mu.RLock()
	defer ts.mu.RUnlock()
	id, ok := ts.byKey[apiKey]
	if !ok {
		return nil
	}
	return ts.tenants[id]
}

// Delete removes a tenant by ID.
func (ts *TenantStore) Delete(id string) bool {
	ts.mu.Lock()
	defer ts.mu.Unlock()
	cfg, ok := ts.tenants[id]
	if !ok {
		return false
	}
	delete(ts.byKey, cfg.APIKey)
	delete(ts.tenants, id)
	delete(ts.usage, id)
	return true
}

// RecordUsage adds token usage for a tenant. Returns false if quota exceeded.
func (ts *TenantStore) RecordUsage(id string, tokens int64) bool {
	ts.mu.RLock()
	cfg, ok := ts.tenants[id]
	usage := ts.usage[id]
	ts.mu.RUnlock()

	if !ok || usage == nil {
		return true // unknown tenant, no restriction
	}

	// Reset monthly counter if we've rolled into a new month.
	now := time.Now()
	monthStart := usage.MonthStart
	if now.Year() != monthStart.Year() || now.Month() != monthStart.Month() {
		usage.TotalTokens.Store(0)
		usage.RequestCount.Store(0)
		ts.mu.Lock()
		usage.MonthStart = now.Truncate(24 * time.Hour)
		ts.mu.Unlock()
	}

	// Check quota before recording.
	if cfg.MonthlyQuota > 0 {
		current := usage.TotalTokens.Load()
		if current+tokens > cfg.MonthlyQuota {
			return false
		}
	}

	usage.TotalTokens.Add(tokens)
	usage.RequestCount.Add(1)
	return true
}

// GetUsage returns usage stats for a tenant.
func (ts *TenantStore) GetUsage(id string) *TenantUsageJSON {
	ts.mu.RLock()
	cfg := ts.tenants[id]
	usage := ts.usage[id]
	ts.mu.RUnlock()

	if cfg == nil || usage == nil {
		return nil
	}

	result := &TenantUsageJSON{
		TenantID:     id,
		TotalTokens:  usage.TotalTokens.Load(),
		RequestCount: usage.RequestCount.Load(),
		MonthStart:   usage.MonthStart.Format(time.RFC3339),
	}
	if cfg.MonthlyQuota > 0 {
		result.QuotaUsedPct = float64(result.TotalTokens) / float64(cfg.MonthlyQuota) * 100
	}
	return result
}

// CheckModel returns true if the tenant is allowed to use the given model.
func (ts *TenantStore) CheckModel(id, model string) bool {
	ts.mu.RLock()
	cfg := ts.tenants[id]
	ts.mu.RUnlock()

	if cfg == nil || len(cfg.AllowedModels) == 0 {
		return true // no restriction
	}
	for _, m := range cfg.AllowedModels {
		if m == model {
			return true
		}
	}
	return false
}

// CheckMaxTokens returns the effective max tokens for a tenant request.
// If the tenant has a limit and the requested amount exceeds it, returns the limit.
func (ts *TenantStore) CheckMaxTokens(id string, requested int) int {
	ts.mu.RLock()
	cfg := ts.tenants[id]
	ts.mu.RUnlock()

	if cfg == nil || cfg.MaxTokens <= 0 {
		return requested
	}
	if requested <= 0 || requested > cfg.MaxTokens {
		return cfg.MaxTokens
	}
	return requested
}

// TenantMiddleware returns HTTP middleware that enforces tenant limits.
// It extracts the API key from the Authorization header, looks up the tenant,
// and checks quotas. If quota is exceeded, it returns 429 Too Many Requests.
func TenantMiddleware(store *TenantStore) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if store == nil {
				next.ServeHTTP(w, r)
				return
			}

			// Extract API key from Authorization header.
			authHeader := r.Header.Get("Authorization")
			if !strings.HasPrefix(authHeader, "Bearer ") {
				next.ServeHTTP(w, r)
				return
			}
			apiKey := strings.TrimPrefix(authHeader, "Bearer ")

			cfg := store.GetByKey(apiKey)
			if cfg == nil {
				// Unknown key — let other auth middleware handle it.
				next.ServeHTTP(w, r)
				return
			}

			// Check monthly quota (pre-check with estimate of 100 tokens).
			usage := store.GetUsage(cfg.ID)
			if usage != nil && cfg.MonthlyQuota > 0 {
				if usage.TotalTokens >= cfg.MonthlyQuota {
					writeTenantQuotaError(w)
					return
				}
			}

			// Set tenant ID in header for downstream handlers.
			r.Header.Set("X-Tenant-ID", cfg.ID)

			next.ServeHTTP(w, r)
		})
	}
}

func writeTenantQuotaError(w http.ResponseWriter) {
	var resp errorResponse
	resp.Error.Message = "monthly token quota exceeded"
	resp.Error.Type = "quota_exceeded"
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "3600")
	w.WriteHeader(http.StatusTooManyRequests)
	json.NewEncoder(w).Encode(resp)
}

// handleTenantCreate handles POST /v1/admin/tenants — create or update a tenant.
func (s *Server) handleTenantCreate(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	if err != nil {
		writeError(w, http.StatusBadRequest, "read body: "+err.Error())
		return
	}

	var cfg TenantConfig
	if err := json.Unmarshal(body, &cfg); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request: "+err.Error())
		return
	}
	if cfg.ID == "" {
		writeError(w, http.StatusBadRequest, "id field is required")
		return
	}
	if cfg.APIKey == "" {
		writeError(w, http.StatusBadRequest, "api_key field is required")
		return
	}

	if s.tenants == nil {
		writeError(w, http.StatusInternalServerError, "tenant store not configured")
		return
	}

	s.tenants.Upsert(cfg)
	writeJSON(w, http.StatusOK, map[string]string{
		"status": "ok",
		"id":     cfg.ID,
	})
}

// handleTenantUsage handles GET /v1/admin/tenants/{id}/usage.
func (s *Server) handleTenantUsage(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeError(w, http.StatusBadRequest, "tenant id is required")
		return
	}

	if s.tenants == nil {
		writeError(w, http.StatusInternalServerError, "tenant store not configured")
		return
	}

	usage := s.tenants.GetUsage(id)
	if usage == nil {
		writeError(w, http.StatusNotFound, "tenant not found")
		return
	}

	writeJSON(w, http.StatusOK, usage)
}
