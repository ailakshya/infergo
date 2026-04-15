package server

import (
	"io"
	"net/http"
	"strings"
)

// ProxyConfig configures OpenAI/Anthropic fallback.
type ProxyConfig struct {
	URL    string // upstream API URL (e.g. https://api.openai.com/v1)
	APIKey string // upstream API key
}

// ProxyHandler forwards requests to an upstream OpenAI-compatible API.
// Used when: model not loaded locally, or X-Force-Remote header set.
func (s *Server) proxyToUpstream(w http.ResponseWriter, r *http.Request, cfg *ProxyConfig) {
	// Build upstream URL
	upstreamURL := strings.TrimRight(cfg.URL, "/") + r.URL.Path

	// Read body
	body, err := io.ReadAll(r.Body)
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to read request body")
		return
	}

	// Create upstream request
	req, err := http.NewRequestWithContext(r.Context(), r.Method, upstreamURL, strings.NewReader(string(body)))
	if err != nil {
		writeError(w, http.StatusBadGateway, "failed to create upstream request")
		return
	}

	// Copy headers
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+cfg.APIKey)

	// Forward
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream request failed: "+err.Error())
		return
	}
	defer resp.Body.Close()

	// Copy response headers
	for k, v := range resp.Header {
		for _, vv := range v {
			w.Header().Add(k, vv)
		}
	}
	w.Header().Set("X-Served-By", "remote")
	w.WriteHeader(resp.StatusCode)
	io.Copy(w, resp.Body)
}
