package main

import "testing"

// TestModeFlagParsing verifies that validateMode accepts valid modes and
// rejects invalid ones.
func TestModeFlagParsing(t *testing.T) {
	validModes := []string{"combined", "prefill", "decode"}
	for _, m := range validModes {
		if err := validateMode(m); err != nil {
			t.Errorf("mode %q should be valid, got error: %v", m, err)
		}
	}

	invalidModes := []string{"invalid", "", "COMBINED", "Prefill", "decode2"}
	for _, m := range invalidModes {
		if err := validateMode(m); err == nil {
			t.Errorf("mode %q should be rejected, but validateMode returned nil", m)
		}
	}
}
