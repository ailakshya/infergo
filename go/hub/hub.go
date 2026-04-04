// Package hub provides HuggingFace model-hub download functionality for infergo.
//
// It is deliberately pure Go (no CGo) so it can be imported and tested on any
// platform, including the Mac development machine where the C++ inference
// libraries are not available.
package hub

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// BaseURL is the root URL for HuggingFace API calls.
// Tests override this with an httptest.Server address.
var BaseURL = "https://huggingface.co"

// Sibling represents a single file entry from the HuggingFace model API.
type Sibling struct {
	Rfilename string `json:"rfilename"`
}

// ModelInfo is the response from GET /api/models/{owner}/{repo}.
type ModelInfo struct {
	Siblings []Sibling `json:"siblings"`
}

// ListFiles fetches the list of files in a HuggingFace repository.
// If token is non-empty it is sent as a Bearer token.
func ListFiles(repo, token string) ([]Sibling, error) {
	url := fmt.Sprintf("%s/api/models/%s", BaseURL, repo)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("list files: %w", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("list files: request failed: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
		// handled below
	case http.StatusUnauthorized:
		return nil, fmt.Errorf("private repo — set --hf-token or HF_TOKEN env var")
	case http.StatusNotFound:
		return nil, fmt.Errorf("repo or file not found: %s", repo)
	default:
		return nil, fmt.Errorf("list files: unexpected status %s", resp.Status)
	}

	var info ModelInfo
	if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
		return nil, fmt.Errorf("list files: decode response: %w", err)
	}
	return info.Siblings, nil
}

// SelectFile picks the best matching file from a list of siblings.
//
// Priority:
//  1. If quant is set: look for *.gguf files containing the quant string (case-insensitive).
//  2. If format is "onnx": look for *.onnx files, prefer "model.onnx".
//  3. If format is "gguf": return any *.gguf file.
//  4. Auto-detect: prefer *.gguf, then *.onnx.
func SelectFile(siblings []Sibling, quant, format string) (string, error) {
	lower := strings.ToLower

	if quant != "" {
		for _, s := range siblings {
			if strings.HasSuffix(lower(s.Rfilename), ".gguf") &&
				strings.Contains(lower(s.Rfilename), lower(quant)) {
				return s.Rfilename, nil
			}
		}
		return "", fmt.Errorf("no .gguf file matching quant %q found", quant)
	}

	if lower(format) == "onnx" {
		var candidates []string
		for _, s := range siblings {
			if strings.HasSuffix(lower(s.Rfilename), ".onnx") {
				candidates = append(candidates, s.Rfilename)
			}
		}
		if len(candidates) == 0 {
			return "", fmt.Errorf("no .onnx files found in repo")
		}
		// Prefer model.onnx if present.
		for _, c := range candidates {
			if filepath.Base(c) == "model.onnx" {
				return c, nil
			}
		}
		return candidates[0], nil
	}

	if lower(format) == "gguf" {
		for _, s := range siblings {
			if strings.HasSuffix(lower(s.Rfilename), ".gguf") {
				return s.Rfilename, nil
			}
		}
		return "", fmt.Errorf("no .gguf files found in repo")
	}

	// Auto-detect: gguf first, then onnx.
	for _, s := range siblings {
		if strings.HasSuffix(lower(s.Rfilename), ".gguf") {
			return s.Rfilename, nil
		}
	}
	for _, s := range siblings {
		if strings.HasSuffix(lower(s.Rfilename), ".onnx") {
			return s.Rfilename, nil
		}
	}
	return "", fmt.Errorf("no .gguf or .onnx files found in repo")
}

// Download downloads a file from url to destPath, resuming if a partial file
// already exists.
//
// It returns the expected SHA256 hex string if the server provided one via the
// X-Linked-Etag response header (format "sha256:<hex>"), or an empty string if
// no hash was provided. The caller is responsible for verifying the hash.
func Download(url, destPath, token string) (expectedSHA256 string, err error) {
	// Check for an existing partial file.
	var offset int64
	if fi, statErr := os.Stat(destPath); statErr == nil {
		offset = fi.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", fmt.Errorf("download: %w", err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	if offset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	}

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("download: request failed: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK, http.StatusPartialContent:
		// good
	case http.StatusUnauthorized:
		return "", fmt.Errorf("private repo — set --hf-token or HF_TOKEN env var")
	case http.StatusNotFound:
		return "", fmt.Errorf("repo or file not found")
	case http.StatusRequestedRangeNotSatisfiable:
		// File is already complete.
		fmt.Println("File already downloaded.")
		return "", nil
	default:
		return "", fmt.Errorf("download: unexpected status %s", resp.Status)
	}

	// Extract SHA256 from X-Linked-Etag if available.
	etag := resp.Header.Get("X-Linked-Etag")
	if strings.HasPrefix(etag, "sha256:") {
		expectedSHA256 = strings.TrimPrefix(etag, "sha256:")
	}

	// Determine total size for progress display.
	var totalBytes int64
	if resp.StatusCode == http.StatusPartialContent {
		cr := resp.Header.Get("Content-Range")
		if cr != "" {
			var start, end, total int64
			if n, _ := fmt.Sscanf(cr, "bytes %d-%d/%d", &start, &end, &total); n == 3 {
				totalBytes = total
			}
		}
	} else {
		totalBytes = resp.ContentLength
		if totalBytes < 0 {
			totalBytes = 0
		}
	}

	// Open destination file — append if resuming, create/truncate if new.
	flags := os.O_CREATE | os.O_WRONLY
	if resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		offset = 0
	}
	f, err := os.OpenFile(destPath, flags, 0o644)
	if err != nil {
		return "", fmt.Errorf("download: open destination file: %w", err)
	}
	defer f.Close()

	// Copy with progress reporting.
	written, copyErr := CopyProgress(f, resp.Body, offset, totalBytes, filepath.Base(destPath))
	if copyErr != nil {
		return "", fmt.Errorf("download interrupted at %.1f MB, re-run to resume: %w",
			float64(offset+written)/(1<<20), copyErr)
	}

	fmt.Println() // newline after progress line
	return expectedSHA256, nil
}

// CopyProgress copies src to dst while printing download progress to stdout.
// alreadyBytes is the number of bytes already on disk (for resume display).
// totalBytes is the full file size (0 = unknown).
func CopyProgress(dst io.Writer, src io.Reader, alreadyBytes, totalBytes int64, name string) (int64, error) {
	buf := make([]byte, 32*1024)
	var written int64
	for {
		nr, readErr := src.Read(buf)
		if nr > 0 {
			nw, writeErr := dst.Write(buf[:nr])
			written += int64(nw)
			if writeErr != nil {
				return written, writeErr
			}
			downloaded := alreadyBytes + written
			if totalBytes > 0 {
				pct := float64(downloaded) / float64(totalBytes) * 100
				fmt.Printf("\rDownloading %s... %.1f MB / %.1f MB (%.0f%%)",
					name,
					float64(downloaded)/(1<<20),
					float64(totalBytes)/(1<<20),
					pct,
				)
			} else {
				fmt.Printf("\rDownloading %s... %.1f MB", name, float64(downloaded)/(1<<20))
			}
		}
		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return written, readErr
		}
	}
	return written, nil
}

// FileSHA256 computes the hex-encoded SHA256 digest of the file at path.
func FileSHA256(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return fmt.Sprintf("%x", h.Sum(nil)), nil
}
