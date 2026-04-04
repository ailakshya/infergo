package hub_test

import (
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/ailakshya/infergo/hub"
)

// mockHFServer creates an httptest.Server that emulates the HuggingFace API.
//
// Endpoints:
//   - GET /api/models/{owner}/{repo}        → returns sibling list JSON
//   - GET /{owner}/{repo}/resolve/main/{file} → returns file bytes
//
// The server supports Range requests (resume) and sets X-Linked-Etag.
func mockHFServer(t *testing.T, siblings []string, fileContent []byte, requireToken string) *httptest.Server {
	t.Helper()

	// Pre-compute the SHA256 of the file content.
	h := sha256.New()
	h.Write(fileContent)
	sha256hex := fmt.Sprintf("%x", h.Sum(nil))

	mux := http.NewServeMux()

	// API endpoint — list repo files.
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		if requireToken != "" {
			auth := r.Header.Get("Authorization")
			if auth != "Bearer "+requireToken {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
		}
		var sb strings.Builder
		sb.WriteString(`{"siblings":[`)
		for i, name := range siblings {
			if i > 0 {
				sb.WriteByte(',')
			}
			fmt.Fprintf(&sb, `{"rfilename":%q}`, name)
		}
		sb.WriteString(`]}`)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		io.WriteString(w, sb.String())
	})

	// File download endpoint.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if requireToken != "" {
			auth := r.Header.Get("Authorization")
			if auth != "Bearer "+requireToken {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
		}

		rangeHeader := r.Header.Get("Range")
		if rangeHeader != "" {
			var start int64
			fmt.Sscanf(rangeHeader, "bytes=%d-", &start)
			if start >= int64(len(fileContent)) {
				w.WriteHeader(http.StatusRequestedRangeNotSatisfiable)
				return
			}
			w.Header().Set("Content-Range",
				fmt.Sprintf("bytes %d-%d/%d", start, len(fileContent)-1, len(fileContent)))
			w.Header().Set("X-Linked-Etag", "sha256:"+sha256hex)
			w.WriteHeader(http.StatusPartialContent)
			w.Write(fileContent[start:])
			return
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(fileContent)))
		w.Header().Set("X-Linked-Etag", "sha256:"+sha256hex)
		w.WriteHeader(http.StatusOK)
		w.Write(fileContent)
	})

	return httptest.NewServer(mux)
}

// TestT1_FileSelectionByQuant verifies that quant filter picks the correct
// file from the sibling list (OPT-16-T1).
func TestT1_FileSelectionByQuant(t *testing.T) {
	siblings := []hub.Sibling{
		{Rfilename: "model.Q2_K.gguf"},
		{Rfilename: "model.Q4_K_M.gguf"},
		{Rfilename: "model.Q8_0.gguf"},
		{Rfilename: "README.md"},
	}

	got, err := hub.SelectFile(siblings, "Q4_K_M", "")
	if err != nil {
		t.Fatalf("SelectFile returned error: %v", err)
	}
	if got != "model.Q4_K_M.gguf" {
		t.Fatalf("expected model.Q4_K_M.gguf, got %q", got)
	}
}

// TestT1_FileSelectionByQuantNoMatch verifies a helpful error when the quant
// is not found.
func TestT1_FileSelectionByQuantNoMatch(t *testing.T) {
	siblings := []hub.Sibling{
		{Rfilename: "model.Q2_K.gguf"},
	}
	_, err := hub.SelectFile(siblings, "Q4_K_M", "")
	if err == nil {
		t.Fatal("expected error for missing quant, got nil")
	}
}

// TestT2_ResumePartialDownload verifies that a partial file is extended rather
// than re-downloaded from scratch (OPT-16-T2 / OPT-16-T3).
func TestT2_ResumePartialDownload(t *testing.T) {
	content := []byte("GGUF model data — this is a fake model file for testing resume logic.")
	splitAt := 20 // bytes already on disk

	srv := mockHFServer(t, []string{"model.Q4_K_M.gguf"}, content, "")
	defer srv.Close()

	orig := hub.BaseURL
	hub.BaseURL = srv.URL
	defer func() { hub.BaseURL = orig }()

	dir := t.TempDir()
	destPath := filepath.Join(dir, "model.Q4_K_M.gguf")

	// Write the partial file.
	if err := os.WriteFile(destPath, content[:splitAt], 0o644); err != nil {
		t.Fatalf("setup: %v", err)
	}

	url := fmt.Sprintf("%s/owner/repo/resolve/main/model.Q4_K_M.gguf", srv.URL)
	_, err := hub.Download(url, destPath, "")
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	got, err := os.ReadFile(destPath)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("resume produced wrong content:\n  want %q\n  got  %q", content, got)
	}
}

// TestT3_SHA256Verified checks that a correct hash passes and that a corrupt
// file produces a different hash (OPT-16-T4).
func TestT3_SHA256Verified(t *testing.T) {
	content := []byte("correct model content for sha256 test")

	h := sha256.New()
	h.Write(content)
	correctHash := fmt.Sprintf("%x", h.Sum(nil))

	dir := t.TempDir()
	goodPath := filepath.Join(dir, "good.gguf")
	if err := os.WriteFile(goodPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	// Correct hash → matches.
	got, err := hub.FileSHA256(goodPath)
	if err != nil {
		t.Fatalf("FileSHA256: %v", err)
	}
	if got != correctHash {
		t.Fatalf("SHA256 mismatch: want %s, got %s", correctHash, got)
	}

	// Corrupt file → hash differs.
	corruptPath := filepath.Join(dir, "corrupt.gguf")
	if err := os.WriteFile(corruptPath, []byte("this is corrupted data"), 0o644); err != nil {
		t.Fatal(err)
	}
	corruptHash, err := hub.FileSHA256(corruptPath)
	if err != nil {
		t.Fatalf("FileSHA256 corrupt: %v", err)
	}
	if corruptHash == correctHash {
		t.Fatal("corrupt file should not match correct SHA256")
	}
}

// TestT4_PrivateRepoWithoutToken verifies that a 401 response produces the
// correct human-readable error message (OPT-16-T4).
func TestT4_PrivateRepoWithoutToken(t *testing.T) {
	content := []byte("private model bytes")
	srv := mockHFServer(t, []string{"model.gguf"}, content, "secret-token")
	defer srv.Close()

	orig := hub.BaseURL
	hub.BaseURL = srv.URL
	defer func() { hub.BaseURL = orig }()

	// Call ListFiles WITHOUT a token — should get 401.
	_, err := hub.ListFiles("owner/repo", "")
	if err == nil {
		t.Fatal("expected error for private repo without token, got nil")
	}
	if !strings.Contains(err.Error(), "private repo") {
		t.Fatalf("expected 'private repo' in error, got: %v", err)
	}
}

// TestT5_FileFlagDownloadsSpecificFile verifies that Download retrieves a
// specific file directly by URL (models --file flag path) (OPT-16-T5).
func TestT5_FileFlagDownloadsSpecificFile(t *testing.T) {
	content := []byte("specific file content for --file test")
	srv := mockHFServer(t, nil, content, "")
	defer srv.Close()

	orig := hub.BaseURL
	hub.BaseURL = srv.URL
	defer func() { hub.BaseURL = orig }()

	dir := t.TempDir()
	destPath := filepath.Join(dir, "specific-model.gguf")

	url := fmt.Sprintf("%s/owner/repo/resolve/main/specific-model.gguf", srv.URL)
	_, err := hub.Download(url, destPath, "")
	if err != nil {
		t.Fatalf("Download: %v", err)
	}

	got, err := os.ReadFile(destPath)
	if err != nil {
		t.Fatalf("read result: %v", err)
	}
	if string(got) != string(content) {
		t.Fatalf("content mismatch: want %q, got %q", content, got)
	}
}

// TestONNXFormatSelection verifies that --format onnx selects model.onnx when
// multiple .onnx files are present.
func TestONNXFormatSelection(t *testing.T) {
	siblings := []hub.Sibling{
		{Rfilename: "encoder.onnx"},
		{Rfilename: "model.onnx"},
		{Rfilename: "README.md"},
	}
	got, err := hub.SelectFile(siblings, "", "onnx")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "model.onnx" {
		t.Fatalf("expected model.onnx, got %q", got)
	}
}

// TestPrivateRepoWithToken verifies that ListFiles with a valid token succeeds
// against a token-protected endpoint.
func TestPrivateRepoWithToken(t *testing.T) {
	content := []byte("private model content")
	srv := mockHFServer(t, []string{"private.gguf"}, content, "mytoken")
	defer srv.Close()

	orig := hub.BaseURL
	hub.BaseURL = srv.URL
	defer func() { hub.BaseURL = orig }()

	siblings, err := hub.ListFiles("owner/repo", "mytoken")
	if err != nil {
		t.Fatalf("ListFiles with token: %v", err)
	}
	if len(siblings) == 0 || siblings[0].Rfilename != "private.gguf" {
		t.Fatalf("unexpected siblings: %v", siblings)
	}
}
