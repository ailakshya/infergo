package server

import (
	"encoding/json"
	"net/http"
	"strings"
	"sync"
	"time"
)

// ─── Auth Middleware ──────────────────────────────────────────────────────────

// AuthMiddleware returns middleware that checks Authorization: Bearer <key>.
// If apiKey is empty, all requests pass through (open server).
// /health and /metrics paths are always exempt.
func AuthMiddleware(apiKey string) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Exempt health and metrics endpoints
			if strings.HasPrefix(r.URL.Path, "/health") || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			// Open server: no key configured
			if apiKey == "" {
				next.ServeHTTP(w, r)
				return
			}

			// Extract Bearer token
			authHeader := r.Header.Get("Authorization")
			if !strings.HasPrefix(authHeader, "Bearer ") {
				writeAuthError(w)
				return
			}
			token := strings.TrimPrefix(authHeader, "Bearer ")
			if token != apiKey {
				writeAuthError(w)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

func writeAuthError(w http.ResponseWriter) {
	type errBody struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	var body errBody
	body.Error.Message = "unauthorized"
	body.Error.Type = "auth_error"
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusUnauthorized)
	json.NewEncoder(w).Encode(body)
}

// ─── Rate Limiter ─────────────────────────────────────────────────────────────

// tokenBucket is a per-IP token bucket for rate limiting.
type tokenBucket struct {
	mu         sync.Mutex
	tokens     float64
	lastRefill time.Time
	lastSeen   time.Time
}

// RateLimiter implements a per-IP token bucket rate limiter.
type RateLimiter struct {
	rps     float64
	mu      sync.Mutex
	buckets map[string]*tokenBucket
}

// NewRateLimiter creates a rate limiter allowing rps requests/second per client IP.
// If rps <= 0, the limiter is disabled and all requests pass through.
func NewRateLimiter(rps float64) *RateLimiter {
	return &RateLimiter{
		rps:     rps,
		buckets: make(map[string]*tokenBucket),
	}
}

// Middleware returns the http.Handler middleware for rate limiting.
// /health and /metrics paths are always exempt.
func (rl *RateLimiter) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Exempt health and metrics endpoints
			if strings.HasPrefix(r.URL.Path, "/health") || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			// Disabled limiter
			if rl.rps <= 0 {
				next.ServeHTTP(w, r)
				return
			}

			ip := clientIP(r.RemoteAddr)
			now := time.Now()

			bucket := rl.getOrCreateBucket(ip, now)

			bucket.mu.Lock()
			elapsed := now.Sub(bucket.lastRefill).Seconds()
			bucket.tokens += rl.rps * elapsed
			if bucket.tokens > rl.rps {
				bucket.tokens = rl.rps
			}
			bucket.lastRefill = now
			bucket.lastSeen = now

			if bucket.tokens >= 1 {
				bucket.tokens--
				bucket.mu.Unlock()
				next.ServeHTTP(w, r)
			} else {
				bucket.mu.Unlock()
				writeRateLimitError(w)
			}

			// Periodically clean stale entries (entries not seen in 60s)
			rl.cleanup(now)
		})
	}
}

func (rl *RateLimiter) getOrCreateBucket(ip string, now time.Time) *tokenBucket {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	b, ok := rl.buckets[ip]
	if !ok {
		b = &tokenBucket{
			tokens:     rl.rps,
			lastRefill: now,
			lastSeen:   now,
		}
		rl.buckets[ip] = b
	}
	return b
}

func (rl *RateLimiter) cleanup(now time.Time) {
	rl.mu.Lock()
	defer rl.mu.Unlock()
	for ip, b := range rl.buckets {
		b.mu.Lock()
		stale := now.Sub(b.lastSeen) > 60*time.Second
		b.mu.Unlock()
		if stale {
			delete(rl.buckets, ip)
		}
	}
}

func writeRateLimitError(w http.ResponseWriter) {
	type errBody struct {
		Error struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		} `json:"error"`
	}
	var body errBody
	body.Error.Message = "rate limit exceeded"
	body.Error.Type = "rate_limit_error"
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", "1")
	w.WriteHeader(http.StatusTooManyRequests)
	json.NewEncoder(w).Encode(body)
}

// clientIP extracts the IP portion from "IP:port" or returns the whole string.
func clientIP(remoteAddr string) string {
	if idx := strings.LastIndex(remoteAddr, ":"); idx != -1 {
		return remoteAddr[:idx]
	}
	return remoteAddr
}
