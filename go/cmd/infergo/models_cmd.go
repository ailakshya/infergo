package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// runModels implements "infergo models" subcommand.
// Usage:
//   infergo models list              — list local models
//   infergo models info <name>       — show model details
//   infergo models delete <name>     — remove a model
func runModels(args []string) {
	if len(args) == 0 {
		fmt.Fprintln(os.Stderr, "usage: infergo models [list|info|delete] [name]")
		os.Exit(1)
	}

	switch args[0] {
	case "list":
		listModels()
	case "info":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: infergo models info <name>")
			os.Exit(1)
		}
		modelInfo(args[1])
	case "delete":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "usage: infergo models delete <name>")
			os.Exit(1)
		}
		deleteModel(args[1])
	default:
		fmt.Fprintf(os.Stderr, "unknown models subcommand: %q\n", args[0])
		os.Exit(1)
	}
}

func listModels() {
	dir := modelsDir()
	entries, err := os.ReadDir(dir)
	if err != nil {
		fmt.Fprintf(os.Stderr, "no models directory: %s\n", dir)
		return
	}

	fmt.Printf("%-30s %-10s %10s\n", "NAME", "FORMAT", "SIZE")
	fmt.Println(strings.Repeat("-", 52))

	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		info, _ := e.Info()
		ext := strings.ToLower(filepath.Ext(e.Name()))
		format := "unknown"
		switch ext {
		case ".gguf":
			format = "GGUF"
		case ".onnx":
			format = "ONNX"
		case ".pt", ".pth":
			format = "TorchScript"
		case ".engine":
			format = "TensorRT"
		case ".safetensors":
			format = "SafeTensors"
		}
		size := ""
		if info != nil {
			mb := float64(info.Size()) / (1024 * 1024)
			if mb > 1024 {
				size = fmt.Sprintf("%.1f GB", mb/1024)
			} else {
				size = fmt.Sprintf("%.0f MB", mb)
			}
		}
		name := strings.TrimSuffix(e.Name(), ext)
		fmt.Printf("%-30s %-10s %10s\n", name, format, size)
	}
}

func modelInfo(name string) {
	dir := modelsDir()
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), name) {
			info, _ := e.Info()
			data := map[string]interface{}{
				"name":   name,
				"file":   e.Name(),
				"path":   filepath.Join(dir, e.Name()),
				"format": filepath.Ext(e.Name()),
				"size":   info.Size(),
			}
			b, _ := json.MarshalIndent(data, "", "  ")
			fmt.Println(string(b))
			return
		}
	}
	fmt.Fprintf(os.Stderr, "model %q not found in %s\n", name, dir)
	os.Exit(1)
}

func deleteModel(name string) {
	dir := modelsDir()
	fs := flag.NewFlagSet("delete", flag.ExitOnError)
	force := fs.Bool("force", false, "skip confirmation")
	fs.Parse(os.Args[3:])

	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), name) {
			path := filepath.Join(dir, e.Name())
			if !*force {
				fmt.Printf("delete %s? [y/N] ", path)
				var ans string
				fmt.Scanln(&ans)
				if strings.ToLower(ans) != "y" {
					fmt.Println("cancelled")
					return
				}
			}
			os.Remove(path)
			fmt.Printf("deleted %s\n", path)
			return
		}
	}
	fmt.Fprintf(os.Stderr, "model %q not found\n", name)
}

func modelsDir() string {
	if d := os.Getenv("INFERGO_MODELS_DIR"); d != "" {
		return d
	}
	return "models"
}
