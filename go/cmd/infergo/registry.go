package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// ModelRegistryEntry records metadata about a converted model.
type ModelRegistryEntry struct {
	Name       string            `json:"name"`
	Source     string            `json:"source"`
	Export     string            `json:"export"`
	Format     string            `json:"format"`
	ExportedAt time.Time         `json:"exported_at"`
	ImgSize    int               `json:"imgsz"`
	Validation *ValidationResult `json:"validation,omitempty"`
}

// ValidationResult records the outcome of comparing a source model against
// its exported counterpart on random inputs.
type ValidationResult struct {
	Samples int     `json:"samples"`
	MaxDiff float64 `json:"max_diff"`
	Passed  bool    `json:"passed"`
}

// defaultRegistryPath returns the default path for the model registry file.
func defaultRegistryPath() string {
	return filepath.Join("models", "registry.json")
}

// LoadRegistry reads the model registry from a JSON file on disk.
// If the file does not exist an empty slice is returned (not an error).
func LoadRegistry(path string) ([]ModelRegistryEntry, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("load registry: %w", err)
	}
	var entries []ModelRegistryEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("load registry: %w", err)
	}
	return entries, nil
}

// SaveRegistry writes the model registry entries to a JSON file.
// Parent directories are created if they do not exist.
func SaveRegistry(path string, entries []ModelRegistryEntry) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("save registry: %w", err)
	}
	data, err := json.MarshalIndent(entries, "", "  ")
	if err != nil {
		return fmt.Errorf("save registry: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("save registry: %w", err)
	}
	return nil
}

// updateRegistry adds or updates an entry in the model registry for a
// newly exported model. If an entry with the same export path already
// exists it is replaced.
func updateRegistry(exportPath, sourcePath, format string, imgsz int) {
	regPath := defaultRegistryPath()
	entries, err := LoadRegistry(regPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not load registry: %v\n", err)
		entries = nil
	}

	name := filepath.Base(exportPath)

	// Replace existing entry with same export path, or append.
	found := false
	for i, e := range entries {
		if e.Export == exportPath {
			entries[i] = ModelRegistryEntry{
				Name:       name,
				Source:     sourcePath,
				Export:     exportPath,
				Format:     format,
				ExportedAt: time.Now().UTC(),
				ImgSize:    imgsz,
			}
			found = true
			break
		}
	}
	if !found {
		entries = append(entries, ModelRegistryEntry{
			Name:       name,
			Source:     sourcePath,
			Export:     exportPath,
			Format:     format,
			ExportedAt: time.Now().UTC(),
			ImgSize:    imgsz,
		})
	}

	if err := SaveRegistry(regPath, entries); err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not save registry: %v\n", err)
	}
}

// updateRegistryValidation attaches validation results to the registry
// entry matching exportPath.
func updateRegistryValidation(exportPath string, result *ValidationResult) {
	regPath := defaultRegistryPath()
	entries, err := LoadRegistry(regPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "warning: could not load registry: %v\n", err)
		return
	}

	for i, e := range entries {
		if e.Export == exportPath {
			entries[i].Validation = result
			if err := SaveRegistry(regPath, entries); err != nil {
				fmt.Fprintf(os.Stderr, "warning: could not save registry: %v\n", err)
			}
			return
		}
	}
	fmt.Fprintf(os.Stderr, "warning: no registry entry found for %s\n", exportPath)
}
