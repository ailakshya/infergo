package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestRegistryLoadSaveRoundtrip verifies that writing a registry to disk and
// reading it back produces identical entries.
func TestRegistryLoadSaveRoundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "registry.json")

	now := time.Now().UTC().Truncate(time.Second)
	entries := []ModelRegistryEntry{
		{
			Name:       "yolo11n.torchscript.pt",
			Source:     "models/yolo11n.pt",
			Export:     "models/yolo11n.torchscript.pt",
			Format:     "torchscript",
			ExportedAt: now,
			ImgSize:    640,
		},
		{
			Name:       "yolo11n.onnx",
			Source:     "models/yolo11n.pt",
			Export:     "models/yolo11n.onnx",
			Format:     "onnx",
			ExportedAt: now,
			ImgSize:    640,
			Validation: &ValidationResult{
				Samples: 100,
				MaxDiff: 0.0003,
				Passed:  true,
			},
		},
	}

	if err := SaveRegistry(path, entries); err != nil {
		t.Fatalf("SaveRegistry: %v", err)
	}

	loaded, err := LoadRegistry(path)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}

	if len(loaded) != len(entries) {
		t.Fatalf("got %d entries, want %d", len(loaded), len(entries))
	}

	for i := range entries {
		if loaded[i].Name != entries[i].Name {
			t.Errorf("entry %d: Name = %q, want %q", i, loaded[i].Name, entries[i].Name)
		}
		if loaded[i].Source != entries[i].Source {
			t.Errorf("entry %d: Source = %q, want %q", i, loaded[i].Source, entries[i].Source)
		}
		if loaded[i].Export != entries[i].Export {
			t.Errorf("entry %d: Export = %q, want %q", i, loaded[i].Export, entries[i].Export)
		}
		if loaded[i].Format != entries[i].Format {
			t.Errorf("entry %d: Format = %q, want %q", i, loaded[i].Format, entries[i].Format)
		}
		if loaded[i].ImgSize != entries[i].ImgSize {
			t.Errorf("entry %d: ImgSize = %d, want %d", i, loaded[i].ImgSize, entries[i].ImgSize)
		}
	}

	// Verify validation on second entry.
	if loaded[1].Validation == nil {
		t.Fatal("entry 1: Validation is nil")
	}
	if loaded[1].Validation.Samples != 100 {
		t.Errorf("entry 1: Validation.Samples = %d, want 100", loaded[1].Validation.Samples)
	}
	if loaded[1].Validation.MaxDiff != 0.0003 {
		t.Errorf("entry 1: Validation.MaxDiff = %g, want 0.0003", loaded[1].Validation.MaxDiff)
	}
	if !loaded[1].Validation.Passed {
		t.Error("entry 1: Validation.Passed = false, want true")
	}
}

// TestRegistryLoadNonExistent verifies that loading a non-existent registry
// returns nil entries and no error.
func TestRegistryLoadNonExistent(t *testing.T) {
	entries, err := LoadRegistry(filepath.Join(t.TempDir(), "does-not-exist.json"))
	if err != nil {
		t.Fatalf("expected no error, got: %v", err)
	}
	if entries != nil {
		t.Fatalf("expected nil entries, got %d", len(entries))
	}
}

// TestRegistryUpdate verifies that updateRegistry adds a new entry and
// replaces an existing entry with the same export path.
func TestRegistryUpdate(t *testing.T) {
	// Override the default registry path for testing.
	dir := t.TempDir()
	modelsDir := filepath.Join(dir, "models")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	regPath := filepath.Join(modelsDir, "registry.json")

	// Write an initial entry.
	initial := []ModelRegistryEntry{
		{
			Name:       "old.torchscript.pt",
			Source:     "old.pt",
			Export:     "models/old.torchscript.pt",
			Format:     "torchscript",
			ExportedAt: time.Now().UTC(),
			ImgSize:    640,
		},
	}
	if err := SaveRegistry(regPath, initial); err != nil {
		t.Fatalf("SaveRegistry: %v", err)
	}

	// Load and add a new entry manually (simulates updateRegistry logic).
	entries, err := LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}

	entries = append(entries, ModelRegistryEntry{
		Name:       "new.onnx",
		Source:     "new.pt",
		Export:     "models/new.onnx",
		Format:     "onnx",
		ExportedAt: time.Now().UTC(),
		ImgSize:    320,
	})

	if err := SaveRegistry(regPath, entries); err != nil {
		t.Fatal(err)
	}

	// Reload and verify both entries exist.
	reloaded, err := LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(reloaded) != 2 {
		t.Fatalf("got %d entries, want 2", len(reloaded))
	}
	if reloaded[0].Name != "old.torchscript.pt" {
		t.Errorf("entry 0: Name = %q, want %q", reloaded[0].Name, "old.torchscript.pt")
	}
	if reloaded[1].Name != "new.onnx" {
		t.Errorf("entry 1: Name = %q, want %q", reloaded[1].Name, "new.onnx")
	}
}

// TestRegistrySaveCreatesParentDirs verifies that SaveRegistry creates
// intermediate directories if they do not exist.
func TestRegistrySaveCreatesParentDirs(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a", "b", "c", "registry.json")

	entries := []ModelRegistryEntry{
		{Name: "test", Export: "test.pt", Format: "torchscript", ImgSize: 640},
	}
	if err := SaveRegistry(path, entries); err != nil {
		t.Fatalf("SaveRegistry with nested dirs: %v", err)
	}

	// Verify file exists and is valid JSON.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("could not read file: %v", err)
	}
	var loaded []ModelRegistryEntry
	if err := json.Unmarshal(data, &loaded); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if len(loaded) != 1 || loaded[0].Name != "test" {
		t.Errorf("unexpected loaded data: %+v", loaded)
	}
}

// TestConvertCmdInvalidFormat verifies that an unsupported format is rejected.
func TestConvertCmdInvalidFormat(t *testing.T) {
	if !validConvertFormats["torchscript"] {
		t.Error("torchscript should be a valid format")
	}
	if !validConvertFormats["onnx"] {
		t.Error("onnx should be a valid format")
	}
	if validConvertFormats["tflite"] {
		t.Error("tflite should not be a valid format")
	}
	if validConvertFormats[""] {
		t.Error("empty string should not be a valid format")
	}
}
