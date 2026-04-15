package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// hfBaseURL is the HuggingFace base URL. Tests can override this.
var hfBaseURL = "https://huggingface.co"

// hfFileEntry represents a file in a HuggingFace repository tree listing.
type hfFileEntry struct {
	Type string `json:"type"`
	Path string `json:"path"`
	Size int64  `json:"size"`
}

// ResolveModelPath parses a model specification and returns a local file path,
// downloading from HuggingFace if needed.
//
// Supported formats:
//
//	hf:org/repo:quant   — download GGUF matching quant from HF Hub
//	/path/to/model.gguf — local path (returned as-is)
//
// Downloaded models are stored in ~/.infergo/models/<org>/<repo>/.
func ResolveModelPath(spec string) (string, error) {
	if !strings.HasPrefix(spec, "hf:") {
		// Local path — return as-is.
		return spec, nil
	}

	// Parse hf:org/repo:quant
	rest := strings.TrimPrefix(spec, "hf:")
	parts := strings.SplitN(rest, ":", 2)
	if len(parts) < 2 || parts[0] == "" || parts[1] == "" {
		return "", fmt.Errorf("invalid hf spec %q: expected hf:org/repo:quant", spec)
	}
	repo := parts[0]
	quant := strings.ToLower(parts[1])

	// Validate repo format.
	if !strings.Contains(repo, "/") {
		return "", fmt.Errorf("invalid repo %q: expected org/repo format", repo)
	}

	// Resolve destination directory.
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", err)
	}
	destDir := filepath.Join(home, ".infergo", "models", repo)

	// Fetch repository tree from HuggingFace API.
	treeURL := fmt.Sprintf("%s/api/models/%s/tree/main", hfBaseURL, repo)
	entries, err := fetchRepoTree(treeURL)
	if err != nil {
		return "", fmt.Errorf("fetch repo tree: %w", err)
	}

	// Find matching GGUF file.
	var match *hfFileEntry
	for i := range entries {
		e := &entries[i]
		if e.Type != "file" {
			continue
		}
		lower := strings.ToLower(e.Path)
		if !strings.HasSuffix(lower, ".gguf") {
			continue
		}
		if strings.Contains(lower, quant) {
			match = e
			break
		}
	}
	if match == nil {
		return "", fmt.Errorf("no GGUF file matching quant %q in %s", quant, repo)
	}

	// Determine local path.
	destPath := filepath.Join(destDir, filepath.Base(match.Path))

	// Skip if already downloaded (check file exists + size matches).
	if fi, err := os.Stat(destPath); err == nil {
		if fi.Size() == match.Size {
			fmt.Printf("Model already downloaded: %s\n", destPath)
			return destPath, nil
		}
		// Size mismatch — re-download.
		fmt.Printf("Partial download detected, re-downloading %s\n", filepath.Base(match.Path))
	}

	// Create destination directory.
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return "", fmt.Errorf("create directory: %w", err)
	}

	// Download the file.
	downloadURL := fmt.Sprintf("%s/%s/resolve/main/%s", hfBaseURL, repo, match.Path)
	fmt.Printf("Downloading %s from %s\n", filepath.Base(match.Path), repo)
	if err := downloadWithProgress(downloadURL, destPath, match.Size); err != nil {
		return "", fmt.Errorf("download: %w", err)
	}

	fmt.Printf("Saved to %s\n", destPath)
	return destPath, nil
}

// fetchRepoTree fetches the file list from the HuggingFace API tree endpoint.
func fetchRepoTree(url string) ([]hfFileEntry, error) {
	token := os.Getenv("HF_TOKEN")

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var entries []hfFileEntry
	if err := json.NewDecoder(resp.Body).Decode(&entries); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	return entries, nil
}

// downloadWithProgress downloads a file showing progress to stdout.
func downloadWithProgress(url, destPath string, totalSize int64) error {
	token := os.Getenv("HF_TOKEN")

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download returned status %d", resp.StatusCode)
	}

	f, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	buf := make([]byte, 32*1024)
	var written int64
	for {
		nr, readErr := resp.Body.Read(buf)
		if nr > 0 {
			nw, writeErr := f.Write(buf[:nr])
			written += int64(nw)
			if writeErr != nil {
				return writeErr
			}
			if totalSize > 0 {
				pct := float64(written) / float64(totalSize) * 100
				fmt.Printf("\r  %.1f MB / %.1f MB (%.0f%%)",
					float64(written)/(1<<20),
					float64(totalSize)/(1<<20),
					pct)
			} else {
				fmt.Printf("\r  %.1f MB downloaded", float64(written)/(1<<20))
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return readErr
		}
	}
	fmt.Println()
	return nil
}
