package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestResolveModelPath_LocalPath(t *testing.T) {
	// A plain path should be returned as-is.
	path, err := ResolveModelPath("/tmp/model.gguf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != "/tmp/model.gguf" {
		t.Errorf("expected /tmp/model.gguf, got %q", path)
	}
}

func TestResolveModelPath_InvalidSpec(t *testing.T) {
	tests := []string{
		"hf:",
		"hf:repo:",
		"hf::quant",
		"hf:noslash:q4_k_m",
	}
	for _, spec := range tests {
		_, err := ResolveModelPath(spec)
		if err == nil {
			t.Errorf("expected error for spec %q, got nil", spec)
		}
	}
}

func TestResolveModelPath_Download(t *testing.T) {
	// Set up a fake HuggingFace API server.
	fileContent := []byte("fake-gguf-model-data")

	mux := http.NewServeMux()

	// Tree endpoint.
	mux.HandleFunc("/api/models/testorg/testrepo/tree/main", func(w http.ResponseWriter, r *http.Request) {
		entries := []hfFileEntry{
			{Type: "file", Path: "README.md", Size: 100},
			{Type: "file", Path: "testrepo-q4_k_m.gguf", Size: int64(len(fileContent))},
			{Type: "file", Path: "testrepo-q8_0.gguf", Size: 2000},
		}
		json.NewEncoder(w).Encode(entries)
	})

	// Download endpoint.
	mux.HandleFunc("/testorg/testrepo/resolve/main/testrepo-q4_k_m.gguf", func(w http.ResponseWriter, r *http.Request) {
		w.Write(fileContent)
	})

	ts := httptest.NewServer(mux)
	defer ts.Close()

	// Override the base URL.
	oldBase := hfBaseURL
	hfBaseURL = ts.URL
	defer func() { hfBaseURL = oldBase }()

	// Use a temp directory for the download.
	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", oldHome)

	path, err := ResolveModelPath("hf:testorg/testrepo:q4_k_m")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expected := filepath.Join(tmpDir, ".infergo", "models", "testorg", "testrepo", "testrepo-q4_k_m.gguf")
	if path != expected {
		t.Errorf("expected path %q, got %q", expected, path)
	}

	// Verify file exists and has correct content.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read downloaded file: %v", err)
	}
	if string(data) != string(fileContent) {
		t.Errorf("file content mismatch")
	}
}

func TestResolveModelPath_SkipExisting(t *testing.T) {
	fileContent := []byte("fake-gguf-model-data")

	mux := http.NewServeMux()
	downloadCalled := false

	mux.HandleFunc("/api/models/testorg/testrepo/tree/main", func(w http.ResponseWriter, r *http.Request) {
		entries := []hfFileEntry{
			{Type: "file", Path: "model-q4_k_m.gguf", Size: int64(len(fileContent))},
		}
		json.NewEncoder(w).Encode(entries)
	})

	mux.HandleFunc("/testorg/testrepo/resolve/main/model-q4_k_m.gguf", func(w http.ResponseWriter, r *http.Request) {
		downloadCalled = true
		w.Write(fileContent)
	})

	ts := httptest.NewServer(mux)
	defer ts.Close()

	oldBase := hfBaseURL
	hfBaseURL = ts.URL
	defer func() { hfBaseURL = oldBase }()

	tmpDir := t.TempDir()
	oldHome := os.Getenv("HOME")
	os.Setenv("HOME", tmpDir)
	defer os.Setenv("HOME", oldHome)

	// Pre-create the file with matching size.
	destDir := filepath.Join(tmpDir, ".infergo", "models", "testorg", "testrepo")
	os.MkdirAll(destDir, 0o755)
	destPath := filepath.Join(destDir, "model-q4_k_m.gguf")
	os.WriteFile(destPath, fileContent, 0o644)

	path, err := ResolveModelPath("hf:testorg/testrepo:q4_k_m")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if path != destPath {
		t.Errorf("expected %q, got %q", destPath, path)
	}
	if downloadCalled {
		t.Error("download should have been skipped for existing file with matching size")
	}
}

func TestResolveModelPath_NoMatch(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/api/models/testorg/testrepo/tree/main", func(w http.ResponseWriter, r *http.Request) {
		entries := []hfFileEntry{
			{Type: "file", Path: "model-q8_0.gguf", Size: 1000},
		}
		json.NewEncoder(w).Encode(entries)
	})

	ts := httptest.NewServer(mux)
	defer ts.Close()

	oldBase := hfBaseURL
	hfBaseURL = ts.URL
	defer func() { hfBaseURL = oldBase }()

	_, err := ResolveModelPath("hf:testorg/testrepo:q4_k_m")
	if err == nil {
		t.Fatal("expected error for no matching quant, got nil")
	}
}
