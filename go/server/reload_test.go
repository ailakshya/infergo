package server_test

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/ailakshya/infergo/server"
)

// ─── POST /v1/admin/reload ────────────────────────────────────────────────────

func TestAdminReload_NilReloader_501(t *testing.T) {
	// Server with no reloader set → 501 Not Implemented.
	srv, _ := newTestServer(t)
	rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
		Model: "llama3",
		Path:  "/some/path.gguf",
	})
	if rr.Code != http.StatusNotImplemented {
		t.Fatalf("expected 501, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAdminReload_ValidReloader_200(t *testing.T) {
	// Create a temporary file so the path exists (os.Stat check passes).
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("fake"), 0600); err != nil {
		t.Fatal(err)
	}

	srv, _ := newTestServer(t)

	var calledName, calledPath string
	srv.SetReloader(func(name, path string) error {
		calledName = name
		calledPath = path
		return nil
	})

	rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
		Model: "llama3",
		Path:  modelPath,
	})
	if rr.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", rr.Code, rr.Body.String())
	}

	var resp server.ReloadResponse
	if err := json.NewDecoder(rr.Body).Decode(&resp); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if resp.Status != "ok" {
		t.Errorf("expected status=ok, got %q", resp.Status)
	}
	if resp.Model != "llama3" {
		t.Errorf("expected model=llama3, got %q", resp.Model)
	}
	if resp.Path != modelPath {
		t.Errorf("unexpected path: %q", resp.Path)
	}
	if calledName != "llama3" || calledPath != modelPath {
		t.Errorf("reloader called with (%q, %q)", calledName, calledPath)
	}
}

func TestAdminReload_BadJSON_400(t *testing.T) {
	srv, _ := newTestServer(t)
	srv.SetReloader(func(_, _ string) error { return nil })

	// Send raw bad JSON.
	req := httptest.NewRequest(http.MethodPost, "/v1/admin/reload", bytes.NewBufferString("not-json"))
	req.Header.Set("Content-Type", "application/json")
	rr := httptest.NewRecorder()
	srv.ServeHTTP(rr, req)
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAdminReload_NonexistentPath_400(t *testing.T) {
	// T3: bad path must be rejected with 400; old model must still work.
	srv, reg := newTestServer(t)
	reg.Load("llama3", &mockLLM{reply: "still alive"})
	srv.SetReloader(func(name, path string) error {
		// This should never be reached — os.Stat check comes first.
		return nil
	})

	rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
		Model: "llama3",
		Path:  "/does/not/exist/model.gguf",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}

	// Old model must still respond.
	rr2 := doRequest(t, srv, http.MethodPost, "/v1/chat/completions", server.ChatCompletionRequest{
		Model:    "llama3",
		Messages: []server.ChatMessage{{Role: "user", Content: "ping"}},
	})
	if rr2.Code != http.StatusOK {
		t.Fatalf("old model should still work after bad reload, got %d", rr2.Code)
	}
}

func TestAdminReload_ReloaderError_500(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("fake"), 0600); err != nil {
		t.Fatal(err)
	}

	srv, _ := newTestServer(t)
	srv.SetReloader(func(_, _ string) error {
		return errors.New("GPU OOM")
	})

	rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
		Model: "llama3",
		Path:  modelPath,
	})
	if rr.Code != http.StatusInternalServerError {
		t.Fatalf("expected 500, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAdminReload_MissingModelField_400(t *testing.T) {
	srv, _ := newTestServer(t)
	srv.SetReloader(func(_, _ string) error { return nil })

	rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
		Path: "/some/path.gguf",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
}

func TestAdminReload_MissingPathField_400(t *testing.T) {
	srv, _ := newTestServer(t)
	srv.SetReloader(func(_, _ string) error { return nil })

	rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
		Model: "llama3",
	})
	if rr.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", rr.Code, rr.Body.String())
	}
}

// T1: 10 in-flight requests + concurrent reload — all complete, no panics.
func TestAdminReload_ConcurrentRequests_T1(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("fake"), 0600); err != nil {
		t.Fatal(err)
	}

	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{reply: "hello"})
	srv := server.NewServer(reg)
	srv.SetReloader(func(name, path string) error {
		return reg.Load(name, &mockLLM{reply: "reloaded"})
	})

	var wg sync.WaitGroup
	var failures int64

	// Launch 10 in-flight chat requests.
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rr := doRequest(t, srv, http.MethodPost, "/v1/chat/completions",
				server.ChatCompletionRequest{
					Model:    "llama3",
					Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
				})
			if rr.Code != http.StatusOK {
				atomic.AddInt64(&failures, 1)
			}
		}()
	}

	// Simultaneously trigger a reload.
	wg.Add(1)
	go func() {
		defer wg.Done()
		rr := doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
			Model: "llama3",
			Path:  modelPath,
		})
		if rr.Code != http.StatusOK {
			atomic.AddInt64(&failures, 1)
		}
	}()

	wg.Wait()

	if failures != 0 {
		t.Errorf("T1: %d requests failed during concurrent reload", failures)
	}
}

// T4: race detector — run with go test -race ./go/server/...
func TestAdminReload_Race_T4(t *testing.T) {
	tmp := t.TempDir()
	modelPath := filepath.Join(tmp, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("fake"), 0600); err != nil {
		t.Fatal(err)
	}

	reg := server.NewRegistry()
	reg.Load("llama3", &mockLLM{reply: "hello"})
	srv := server.NewServer(reg)
	srv.SetReloader(func(name, path string) error {
		return reg.Load(name, &mockLLM{reply: "new"})
	})

	var wg sync.WaitGroup
	for i := 0; i < 20; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			if n%5 == 0 {
				doRequest(t, srv, http.MethodPost, "/v1/admin/reload", server.ReloadRequest{
					Model: "llama3",
					Path:  modelPath,
				})
			} else {
				doRequest(t, srv, http.MethodPost, "/v1/chat/completions",
					server.ChatCompletionRequest{
						Model:    "llama3",
						Messages: []server.ChatMessage{{Role: "user", Content: "hi"}},
					})
			}
		}(i)
	}
	wg.Wait()
}
