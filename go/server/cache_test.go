package server_test

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/ailakshya/infergo/server"
)

// ─── Unit tests for ResponseCache ────────────────────────────────────────────

func TestCacheHit(t *testing.T) {
	// Same request sent twice: second response should come from cache.
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	srv.SetCache(server.NewResponseCache(100))
	reg.Load("test-llm", &mockLLM{reply: "hello world"})

	body := `{"model":"test-llm","messages":[{"role":"user","content":"hi"}],"max_tokens":10,"temperature":0.0}`

	// First request: cache MISS
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req1.Header.Set("Content-Type", "application/json")
	rr1 := httptest.NewRecorder()
	srv.ServeHTTP(rr1, req1)

	if rr1.Code != http.StatusOK {
		t.Fatalf("first request: expected 200, got %d: %s", rr1.Code, rr1.Body.String())
	}
	if got := rr1.Header().Get("X-Cache"); got != "MISS" {
		t.Errorf("first request: expected X-Cache=MISS, got %q", got)
	}

	// Second request: cache HIT
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req2.Header.Set("Content-Type", "application/json")
	rr2 := httptest.NewRecorder()
	srv.ServeHTTP(rr2, req2)

	if rr2.Code != http.StatusOK {
		t.Fatalf("second request: expected 200, got %d: %s", rr2.Code, rr2.Body.String())
	}
	if got := rr2.Header().Get("X-Cache"); got != "HIT" {
		t.Errorf("second request: expected X-Cache=HIT, got %q", got)
	}

	// Verify the cached response contains the same content.
	var resp1, resp2 server.ChatCompletionResponse
	json.Unmarshal(rr1.Body.Bytes(), &resp1)
	json.Unmarshal(rr2.Body.Bytes(), &resp2)
	if resp1.Choices[0].Message.Content != resp2.Choices[0].Message.Content {
		t.Errorf("content mismatch: %q vs %q",
			resp1.Choices[0].Message.Content, resp2.Choices[0].Message.Content)
	}
}

func TestCacheMiss(t *testing.T) {
	// Different request content should be a cache miss.
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	srv.SetCache(server.NewResponseCache(100))
	reg.Load("test-llm", &mockLLM{reply: "response"})

	body1 := `{"model":"test-llm","messages":[{"role":"user","content":"hello"}],"max_tokens":10}`
	body2 := `{"model":"test-llm","messages":[{"role":"user","content":"goodbye"}],"max_tokens":10}`

	// First request
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body1))
	req1.Header.Set("Content-Type", "application/json")
	rr1 := httptest.NewRecorder()
	srv.ServeHTTP(rr1, req1)
	if rr1.Header().Get("X-Cache") != "MISS" {
		t.Errorf("first request should be MISS, got %q", rr1.Header().Get("X-Cache"))
	}

	// Second request with different content
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body2))
	req2.Header.Set("Content-Type", "application/json")
	rr2 := httptest.NewRecorder()
	srv.ServeHTTP(rr2, req2)
	if rr2.Header().Get("X-Cache") != "MISS" {
		t.Errorf("different prompt should be MISS, got %q", rr2.Header().Get("X-Cache"))
	}
}

func TestCacheBypass(t *testing.T) {
	// X-No-Cache: true should skip cache entirely.
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	srv.SetCache(server.NewResponseCache(100))
	reg.Load("test-llm", &mockLLM{reply: "cached?"})

	body := `{"model":"test-llm","messages":[{"role":"user","content":"test"}],"max_tokens":10}`

	// First request: populate cache
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req1.Header.Set("Content-Type", "application/json")
	rr1 := httptest.NewRecorder()
	srv.ServeHTTP(rr1, req1)
	if rr1.Header().Get("X-Cache") != "MISS" {
		t.Fatal("first request should be MISS")
	}

	// Second request with X-No-Cache: should NOT hit cache
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("X-No-Cache", "true")
	rr2 := httptest.NewRecorder()
	srv.ServeHTTP(rr2, req2)
	if got := rr2.Header().Get("X-Cache"); got != "" {
		t.Errorf("bypass request should have no X-Cache header, got %q", got)
	}
}

func TestCacheLRUEviction(t *testing.T) {
	// Cache size=2: inserting 3 entries should evict the least recently used.
	cache := server.NewResponseCache(2)

	cache.Put(1, []byte("first"))
	cache.Put(2, []byte("second"))

	// Access key 1 to make it recently used.
	if _, ok := cache.Get(1); !ok {
		t.Fatal("key 1 should be present")
	}

	// Insert key 3: should evict key 2 (least recently used).
	cache.Put(3, []byte("third"))

	if cache.Len() != 2 {
		t.Fatalf("expected 2 entries, got %d", cache.Len())
	}

	// Key 2 should be evicted.
	if _, ok := cache.Get(2); ok {
		t.Error("key 2 should have been evicted")
	}

	// Key 1 and 3 should still be present.
	if _, ok := cache.Get(1); !ok {
		t.Error("key 1 should still be present")
	}
	if _, ok := cache.Get(3); !ok {
		t.Error("key 3 should still be present")
	}
}

func TestCacheDifferentParams(t *testing.T) {
	// Same messages but different temperature should produce different cache keys.
	req1 := &server.ChatCompletionRequest{
		Model:    "test",
		Messages: []server.ChatMessage{{Role: "user", Content: "hello"}},
		Temp:     0.0,
	}
	req2 := &server.ChatCompletionRequest{
		Model:    "test",
		Messages: []server.ChatMessage{{Role: "user", Content: "hello"}},
		Temp:     0.7,
	}
	key1 := server.CacheKey(req1)
	key2 := server.CacheKey(req2)
	if key1 == key2 {
		t.Errorf("same prompt with different temp should produce different keys: %d == %d", key1, key2)
	}

	// Same messages but different max_tokens.
	req3 := &server.ChatCompletionRequest{
		Model:     "test",
		Messages:  []server.ChatMessage{{Role: "user", Content: "hello"}},
		MaxTokens: 100,
	}
	req4 := &server.ChatCompletionRequest{
		Model:     "test",
		Messages:  []server.ChatMessage{{Role: "user", Content: "hello"}},
		MaxTokens: 200,
	}
	key3 := server.CacheKey(req3)
	key4 := server.CacheKey(req4)
	if key3 == key4 {
		t.Errorf("same prompt with different max_tokens should produce different keys: %d == %d", key3, key4)
	}

	// Same everything should produce the same key.
	req5 := &server.ChatCompletionRequest{
		Model:     "test",
		Messages:  []server.ChatMessage{{Role: "user", Content: "hello"}},
		Temp:      0.5,
		MaxTokens: 100,
	}
	req6 := &server.ChatCompletionRequest{
		Model:     "test",
		Messages:  []server.ChatMessage{{Role: "user", Content: "hello"}},
		Temp:      0.5,
		MaxTokens: 100,
	}
	key5 := server.CacheKey(req5)
	key6 := server.CacheKey(req6)
	if key5 != key6 {
		t.Errorf("identical requests should produce same key: %d != %d", key5, key6)
	}
}

func TestCacheHeader(t *testing.T) {
	// Verify X-Cache header is present in all cases.
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	srv.SetCache(server.NewResponseCache(100))
	reg.Load("test-llm", &mockLLM{reply: "ok"})

	body := `{"model":"test-llm","messages":[{"role":"user","content":"header test"}],"max_tokens":5}`

	// MISS case
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req1.Header.Set("Content-Type", "application/json")
	rr1 := httptest.NewRecorder()
	srv.ServeHTTP(rr1, req1)

	cacheHeader := rr1.Header().Get("X-Cache")
	if cacheHeader != "MISS" {
		t.Errorf("expected X-Cache=MISS, got %q", cacheHeader)
	}

	// HIT case
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req2.Header.Set("Content-Type", "application/json")
	rr2 := httptest.NewRecorder()
	srv.ServeHTTP(rr2, req2)

	cacheHeader = rr2.Header().Get("X-Cache")
	if cacheHeader != "HIT" {
		t.Errorf("expected X-Cache=HIT, got %q", cacheHeader)
	}

	// Verify response is valid JSON on HIT
	var resp server.ChatCompletionResponse
	if err := json.NewDecoder(rr2.Body).Decode(&resp); err != nil {
		t.Fatalf("cached response is not valid JSON: %v", err)
	}
	if resp.Choices[0].Message.Content != "ok" {
		t.Errorf("expected content 'ok', got %q", resp.Choices[0].Message.Content)
	}
}

func TestCacheNilSafe(t *testing.T) {
	// Server without cache should work normally (no X-Cache header).
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	// No SetCache call.
	reg.Load("test-llm", &mockLLM{reply: "no cache"})

	body := `{"model":"test-llm","messages":[{"role":"user","content":"test"}],"max_tokens":5}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", rr.Code)
	}
	if got := rr.Header().Get("X-Cache"); got != "" {
		t.Errorf("no cache configured, expected no X-Cache header, got %q", got)
	}
}

func TestCacheKeyDeterministic(t *testing.T) {
	// CacheKey should be deterministic for the same input.
	req := &server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "explain caching"}},
		Temp:     0.3,
		MaxTokens: 50,
		ResponseFormat: &server.ResponseFormat{Type: "json_object"},
	}
	key1 := server.CacheKey(req)
	key2 := server.CacheKey(req)
	if key1 != key2 {
		t.Errorf("CacheKey not deterministic: %d != %d", key1, key2)
	}
}

func TestCacheResponseFormatAffectsKey(t *testing.T) {
	// Different response_format types should produce different keys.
	req1 := &server.ChatCompletionRequest{
		Model:    "test",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
		ResponseFormat: &server.ResponseFormat{Type: "text"},
	}
	req2 := &server.ChatCompletionRequest{
		Model:    "test",
		Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
		ResponseFormat: &server.ResponseFormat{Type: "json_object"},
	}
	key1 := server.CacheKey(req1)
	key2 := server.CacheKey(req2)
	if key1 == key2 {
		t.Errorf("different response_format types should produce different keys")
	}
}

// ─── ResponseCache unit tests ────────────────────────────────────────────────

func TestResponseCacheGetPut(t *testing.T) {
	c := server.NewResponseCache(10)

	// Miss on empty cache.
	if _, ok := c.Get(42); ok {
		t.Fatal("expected miss on empty cache")
	}

	// Put + Get.
	c.Put(42, []byte("answer"))
	val, ok := c.Get(42)
	if !ok {
		t.Fatal("expected hit after Put")
	}
	if string(val) != "answer" {
		t.Errorf("expected 'answer', got %q", string(val))
	}
}

func TestResponseCacheOverwrite(t *testing.T) {
	c := server.NewResponseCache(10)
	c.Put(1, []byte("v1"))
	c.Put(1, []byte("v2"))

	val, ok := c.Get(1)
	if !ok {
		t.Fatal("expected hit")
	}
	if string(val) != "v2" {
		t.Errorf("expected 'v2' after overwrite, got %q", string(val))
	}
	if c.Len() != 1 {
		t.Errorf("expected 1 entry after overwrite, got %d", c.Len())
	}
}

func TestResponseCacheDisabledAtZero(t *testing.T) {
	c := server.NewResponseCache(0)
	c.Put(1, []byte("data"))
	if _, ok := c.Get(1); ok {
		t.Error("cache with size 0 should always miss")
	}
}

// mockLLMCounted tracks how many times Generate is called.
type mockLLMCounted struct {
	reply string
	calls int
}

func (m *mockLLMCounted) Close() {}
func (m *mockLLMCounted) Generate(_ context.Context, _ string, _ int, _ float32) (string, int, int, error) {
	m.calls++
	return m.reply, 5, 3, nil
}

func TestCacheSkipsGeneration(t *testing.T) {
	// Verify that a cache hit actually skips the LLM call.
	reg := server.NewRegistry()
	srv := server.NewServer(reg)
	srv.SetCache(server.NewResponseCache(100))
	llm := &mockLLMCounted{reply: "cached result"}
	reg.Load("test-llm", llm)

	body := `{"model":"test-llm","messages":[{"role":"user","content":"count me"}],"max_tokens":10}`

	// First call: generates.
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req1.Header.Set("Content-Type", "application/json")
	rr1 := httptest.NewRecorder()
	srv.ServeHTTP(rr1, req1)
	if llm.calls != 1 {
		t.Fatalf("expected 1 Generate call, got %d", llm.calls)
	}

	// Second call: should NOT call Generate again.
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req2.Header.Set("Content-Type", "application/json")
	rr2 := httptest.NewRecorder()
	srv.ServeHTTP(rr2, req2)
	if llm.calls != 1 {
		t.Errorf("expected Generate to still be 1 (cached), got %d", llm.calls)
	}
	if rr2.Header().Get("X-Cache") != "HIT" {
		t.Errorf("expected X-Cache=HIT, got %q", rr2.Header().Get("X-Cache"))
	}
}
