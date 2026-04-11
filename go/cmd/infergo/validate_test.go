package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestValidationResultJSON verifies that ValidationResult serializes to and
// from JSON correctly, matching the format the Python tool produces.
func TestValidationResultJSON(t *testing.T) {
	result := ValidationResult{
		Samples: 100,
		MaxDiff: 0.0003,
		Passed:  true,
	}

	data, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	var decoded ValidationResult
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if decoded.Samples != 100 {
		t.Errorf("Samples = %d, want 100", decoded.Samples)
	}
	if decoded.MaxDiff != 0.0003 {
		t.Errorf("MaxDiff = %g, want 0.0003", decoded.MaxDiff)
	}
	if !decoded.Passed {
		t.Error("Passed = false, want true")
	}
}

// TestValidationResultJSONFromPython verifies we can parse the exact JSON
// format that the Python validate_export.py tool produces.
func TestValidationResultJSONFromPython(t *testing.T) {
	// This is the exact format the Python tool outputs.
	raw := `{"samples": 50, "max_diff": 1.2e-05, "passed": true}`

	var result ValidationResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("unmarshal Python JSON: %v", err)
	}

	if result.Samples != 50 {
		t.Errorf("Samples = %d, want 50", result.Samples)
	}
	if result.MaxDiff != 1.2e-05 {
		t.Errorf("MaxDiff = %g, want 1.2e-05", result.MaxDiff)
	}
	if !result.Passed {
		t.Error("Passed = false, want true")
	}
}

// TestValidationResultJSONFailed verifies parsing a failed validation result.
func TestValidationResultJSONFailed(t *testing.T) {
	raw := `{"samples": 100, "max_diff": 0.5, "passed": false}`

	var result ValidationResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if result.Passed {
		t.Error("Passed = true, want false")
	}
	if result.MaxDiff != 0.5 {
		t.Errorf("MaxDiff = %g, want 0.5", result.MaxDiff)
	}
}

// TestUpdateRegistryValidation verifies that validation results are correctly
// attached to an existing registry entry.
func TestUpdateRegistryValidation(t *testing.T) {
	dir := t.TempDir()
	modelsDir := filepath.Join(dir, "models")
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		t.Fatal(err)
	}

	regPath := filepath.Join(modelsDir, "registry.json")
	exportPath := "models/yolo11n.torchscript.pt"

	// Create an initial registry entry without validation.
	entries := []ModelRegistryEntry{
		{
			Name:       "yolo11n.torchscript.pt",
			Source:     "models/yolo11n.pt",
			Export:     exportPath,
			Format:     "torchscript",
			ExportedAt: time.Now().UTC(),
			ImgSize:    640,
		},
	}
	if err := SaveRegistry(regPath, entries); err != nil {
		t.Fatal(err)
	}

	// Now attach validation results.
	vr := &ValidationResult{Samples: 50, MaxDiff: 0.00012, Passed: true}

	// Read, update, save (simulating updateRegistryValidation but with explicit path).
	loaded, err := LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range loaded {
		if e.Export == exportPath {
			loaded[i].Validation = vr
		}
	}
	if err := SaveRegistry(regPath, loaded); err != nil {
		t.Fatal(err)
	}

	// Reload and verify.
	final, err := LoadRegistry(regPath)
	if err != nil {
		t.Fatal(err)
	}
	if len(final) != 1 {
		t.Fatalf("got %d entries, want 1", len(final))
	}
	if final[0].Validation == nil {
		t.Fatal("Validation is nil after update")
	}
	if final[0].Validation.Samples != 50 {
		t.Errorf("Validation.Samples = %d, want 50", final[0].Validation.Samples)
	}
	if final[0].Validation.MaxDiff != 0.00012 {
		t.Errorf("Validation.MaxDiff = %g, want 0.00012", final[0].Validation.MaxDiff)
	}
	if !final[0].Validation.Passed {
		t.Error("Validation.Passed = false, want true")
	}
}
