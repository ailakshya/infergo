package server

import (
	"net"
	"net/http"
	"strings"
)

// IPFilter restricts access by IP address or CIDR range.
type IPFilter struct {
	allowNets []*net.IPNet
	enabled   bool
}

// NewIPFilter creates an IP filter from a comma-separated list of CIDR ranges.
// Example: "10.0.0.0/8,192.168.1.0/24,127.0.0.1/32"
// If allowList is empty, all IPs are allowed.
func NewIPFilter(allowList string) *IPFilter {
	if allowList == "" {
		return &IPFilter{enabled: false}
	}

	var nets []*net.IPNet
	for _, cidr := range strings.Split(allowList, ",") {
		cidr = strings.TrimSpace(cidr)
		if cidr == "" {
			continue
		}
		// If no mask, treat as single IP
		if !strings.Contains(cidr, "/") {
			cidr += "/32"
		}
		_, ipNet, err := net.ParseCIDR(cidr)
		if err == nil {
			nets = append(nets, ipNet)
		}
	}

	return &IPFilter{
		allowNets: nets,
		enabled:   len(nets) > 0,
	}
}

// Allowed returns true if the IP is in the allowlist (or filter is disabled).
func (f *IPFilter) Allowed(remoteAddr string) bool {
	if !f.enabled {
		return true
	}

	// Extract IP from addr (may include port)
	host, _, err := net.SplitHostPort(remoteAddr)
	if err != nil {
		host = remoteAddr
	}
	ip := net.ParseIP(host)
	if ip == nil {
		return false
	}

	for _, n := range f.allowNets {
		if n.Contains(ip) {
			return true
		}
	}
	return false
}

// IPFilterMiddleware returns an HTTP middleware that blocks non-allowed IPs.
// Health and metrics endpoints are always allowed.
func IPFilterMiddleware(filter *IPFilter) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// Always allow health/metrics
			if strings.HasPrefix(r.URL.Path, "/health") || r.URL.Path == "/metrics" {
				next.ServeHTTP(w, r)
				return
			}

			if !filter.Allowed(r.RemoteAddr) {
				http.Error(w, "Forbidden", http.StatusForbidden)
				return
			}
			next.ServeHTTP(w, r)
		})
	}
}
