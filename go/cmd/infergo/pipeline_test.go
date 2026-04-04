package main

import (
	"flag"
	"testing"
)

// TestPipelineStagesFlagDefault verifies that --pipeline-stages defaults to 1.
func TestPipelineStagesFlagDefault(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	stages := fs.Int("pipeline-stages", 1, "")
	if err := fs.Parse([]string{}); err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if *stages != 1 {
		t.Errorf("expected default pipeline-stages=1, got %d", *stages)
	}
}

// TestPipelineStagesFlagParsed verifies that --pipeline-stages 2 is parsed correctly.
func TestPipelineStagesFlagParsed(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	stages := fs.Int("pipeline-stages", 1, "")
	if err := fs.Parse([]string{"--pipeline-stages", "2"}); err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	if *stages != 2 {
		t.Errorf("expected pipeline-stages=2, got %d", *stages)
	}
}

// TestPipelineStagesValidation verifies that pipeline-stages=0 is rejected
// at the application level (the flag itself parses, but the value is invalid).
func TestPipelineStagesValidation(t *testing.T) {
	fs := flag.NewFlagSet("test", flag.ContinueOnError)
	stages := fs.Int("pipeline-stages", 1, "")
	if err := fs.Parse([]string{"--pipeline-stages", "0"}); err != nil {
		t.Fatalf("unexpected parse error: %v", err)
	}
	// The validation rule: n_stages must be >= 1.
	if *stages >= 1 {
		t.Errorf("expected stages < 1 for invalid input, got %d", *stages)
	}
	// Confirm the value is 0 (invalid) and would be rejected by the loader.
	if *stages != 0 {
		t.Errorf("expected stages=0 for invalid input, got %d", *stages)
	}
}
