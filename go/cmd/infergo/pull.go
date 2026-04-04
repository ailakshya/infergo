package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ailakshya/infergo/hub"
)

// runPull implements the `infergo pull` subcommand.
func runPull(args []string) {
	fs := flag.NewFlagSet("pull", flag.ExitOnError)
	quant   := fs.String("quant", "", "quantization filter, e.g. Q4_K_M (GGUF only)")
	format  := fs.String("format", "", "file format filter: gguf or onnx")
	file    := fs.String("file", "", "download a specific file by name (overrides --quant / --format)")
	hfToken := fs.String("hf-token", "", "HuggingFace API token for private repos")
	dir     := fs.String("dir", "", "destination directory (default: ~/.infergo/models/<repo>)")
	fs.Parse(args)

	if fs.NArg() < 1 {
		fmt.Fprintln(os.Stderr, "pull: repo argument required")
		fmt.Fprintln(os.Stderr, "  usage: infergo pull owner/repo [--quant Q4_K_M] [--format gguf|onnx] [--file name]")
		os.Exit(1)
	}
	repo := fs.Arg(0)

	// Resolve token: flag takes priority over env var.
	token := *hfToken
	if token == "" {
		token = os.Getenv("HF_TOKEN")
	}

	// Resolve destination directory.
	destDir := *dir
	if destDir == "" {
		home, err := os.UserHomeDir()
		if err != nil {
			fmt.Fprintf(os.Stderr, "pull: cannot determine home directory: %v\n", err)
			os.Exit(1)
		}
		destDir = filepath.Join(home, ".infergo", "models", repo)
	}

	// Step 1: determine which file to download.
	var filename string
	if *file != "" {
		filename = *file
	} else {
		// Fetch the repo metadata from the HuggingFace API.
		siblings, err := hub.ListFiles(repo, token)
		if err != nil {
			fmt.Fprintf(os.Stderr, "pull: %v\n", err)
			os.Exit(1)
		}
		filename, err = hub.SelectFile(siblings, *quant, *format)
		if err != nil {
			fmt.Fprintf(os.Stderr, "pull: %v\n", err)
			fmt.Fprintln(os.Stderr, "Available files:")
			for _, s := range siblings {
				fmt.Fprintf(os.Stderr, "  %s\n", s.Rfilename)
			}
			os.Exit(1)
		}
	}

	// Step 2: create destination directory and download.
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "pull: cannot create destination directory %s: %v\n", destDir, err)
		os.Exit(1)
	}

	destPath := filepath.Join(destDir, filepath.Base(filename))
	downloadURL := fmt.Sprintf("%s/%s/resolve/main/%s", hub.BaseURL, repo, filename)

	fmt.Printf("Pulling %s from %s\n", filename, repo)
	expectedSHA256, err := hub.Download(downloadURL, destPath, token)
	if err != nil {
		fmt.Fprintf(os.Stderr, "pull: %v\n", err)
		os.Exit(1)
	}

	// Step 3: SHA256 verification if the server provided a hash.
	if expectedSHA256 != "" {
		fmt.Printf("Verifying SHA256... ")
		actual, err := hub.FileSHA256(destPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "\npull: cannot read file for verification: %v\n", err)
			os.Exit(1)
		}
		if !strings.EqualFold(actual, expectedSHA256) {
			// Remove corrupt file so next run re-downloads from scratch.
			os.Remove(destPath)
			fmt.Fprintf(os.Stderr, "\npull: SHA256 mismatch — file may be corrupt and has been removed\n")
			fmt.Fprintf(os.Stderr, "  expected: %s\n  got:      %s\n", expectedSHA256, actual)
			os.Exit(1)
		}
		fmt.Println("OK")
	}

	fmt.Printf("Saved to %s\n", destPath)
}
