package server

import (
	"net/http"
	"strings"
)

// Role defines permission levels.
type Role string

const (
	RoleAdmin    Role = "admin"    // all endpoints
	RoleUser     Role = "user"     // inference only
	RoleReadonly Role = "readonly" // read-only (models, health)
)

// RBACConfig maps API keys to roles.
type RBACConfig struct {
	KeyRoles map[string]Role // api_key → role
}

// NewRBACConfig creates RBAC config from key:role pairs.
func NewRBACConfig(pairs map[string]string) *RBACConfig {
	kr := make(map[string]Role)
	for k, v := range pairs {
		switch Role(v) {
		case RoleAdmin, RoleUser, RoleReadonly:
			kr[k] = Role(v)
		}
	}
	return &RBACConfig{KeyRoles: kr}
}

// CheckAccess verifies if the API key has permission for the path.
func (rc *RBACConfig) CheckAccess(apiKey, path string) bool {
	role, ok := rc.KeyRoles[apiKey]
	if !ok {
		return false // unknown key
	}

	switch role {
	case RoleAdmin:
		return true // admin can do everything
	case RoleUser:
		// Can use inference endpoints, not admin
		return !strings.HasPrefix(path, "/v1/admin")
	case RoleReadonly:
		// Can only read models and health
		return path == "/v1/models" ||
			strings.HasPrefix(path, "/health") ||
			path == "/metrics" ||
			path == "/v1/openapi.json" ||
			strings.HasPrefix(path, "/ui")
	}
	return false
}

// RBACMiddleware enforces role-based access control.
func RBACMiddleware(rbac *RBACConfig) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Health/metrics always allowed
			if strings.HasPrefix(r.URL.Path, "/health") || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			// Extract API key
			auth := r.Header.Get("Authorization")
			key := strings.TrimPrefix(auth, "Bearer ")

			if !rbac.CheckAccess(key, r.URL.Path) {
				http.Error(w, `{"error":"forbidden"}`, http.StatusForbidden)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}
