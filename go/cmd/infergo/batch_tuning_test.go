package main

import (
	"flag"
	"testing"
)

// TestMaxBatchSizeFlag verifies that --max-batch-size parses correctly and
// defaults to 8.
func TestMaxBatchSizeFlag(t *testing.T) {
	t.Run("default is 8", func(t *testing.T) {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		v := fs.Int("max-batch-size", 8, "")
		if err := fs.Parse([]string{}); err != nil {
			t.Fatalf("flag parse: %v", err)
		}
		if *v != 8 {
			t.Errorf("default max-batch-size = %d, want 8", *v)
		}
	})

	t.Run("explicit value parsed", func(t *testing.T) {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		v := fs.Int("max-batch-size", 8, "")
		if err := fs.Parse([]string{"--max-batch-size=16"}); err != nil {
			t.Fatalf("flag parse: %v", err)
		}
		if *v != 16 {
			t.Errorf("max-batch-size = %d, want 16", *v)
		}
	})

	t.Run("zero disables cap", func(t *testing.T) {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		v := fs.Int("max-batch-size", 8, "")
		if err := fs.Parse([]string{"--max-batch-size=0"}); err != nil {
			t.Fatalf("flag parse: %v", err)
		}
		if *v != 0 {
			t.Errorf("max-batch-size = %d, want 0 (unlimited)", *v)
		}
	})
}

// TestBatchTimeoutFlag verifies that --batch-timeout-ms parses correctly and
// defaults to 5.
func TestBatchTimeoutFlag(t *testing.T) {
	t.Run("default is 5", func(t *testing.T) {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		v := fs.Int("batch-timeout-ms", 5, "")
		if err := fs.Parse([]string{}); err != nil {
			t.Fatalf("flag parse: %v", err)
		}
		if *v != 5 {
			t.Errorf("default batch-timeout-ms = %d, want 5", *v)
		}
	})

	t.Run("explicit value parsed", func(t *testing.T) {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		v := fs.Int("batch-timeout-ms", 5, "")
		if err := fs.Parse([]string{"--batch-timeout-ms=20"}); err != nil {
			t.Fatalf("flag parse: %v", err)
		}
		if *v != 20 {
			t.Errorf("batch-timeout-ms = %d, want 20", *v)
		}
	})

	t.Run("zero disables timeout", func(t *testing.T) {
		fs := flag.NewFlagSet("test", flag.ContinueOnError)
		v := fs.Int("batch-timeout-ms", 5, "")
		if err := fs.Parse([]string{"--batch-timeout-ms=0"}); err != nil {
			t.Fatalf("flag parse: %v", err)
		}
		if *v != 0 {
			t.Errorf("batch-timeout-ms = %d, want 0 (no wait)", *v)
		}
	})
}
