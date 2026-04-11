package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strconv"
)

// runValidate implements the "infergo validate" subcommand.
// It shells out to a Python tool that compares the original PyTorch model
// against an exported model (ONNX or TorchScript) on random inputs,
// reporting the maximum numerical difference.
func runValidate(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	source := fs.String("source", "", "original PyTorch model (.pt)")
	export := fs.String("export", "", "exported model (.onnx or .torchscript.pt)")
	samples := fs.Int("samples", 100, "number of random inputs to test")
	tolerance := fs.Float64("tolerance", 1e-4, "max allowed output difference")
	fs.Parse(args)

	if *source == "" {
		fmt.Fprintln(os.Stderr, "validate: --source is required")
		fmt.Fprintln(os.Stderr, "usage: infergo validate --source <original.pt> --export <exported.pt> [--samples 100] [--tolerance 1e-4]")
		os.Exit(1)
	}
	if *export == "" {
		fmt.Fprintln(os.Stderr, "validate: --export is required")
		fmt.Fprintln(os.Stderr, "usage: infergo validate --source <original.pt> --export <exported.pt> [--samples 100] [--tolerance 1e-4]")
		os.Exit(1)
	}

	log.Printf("[validate] source:    %s", *source)
	log.Printf("[validate] export:    %s", *export)
	log.Printf("[validate] samples:   %d", *samples)
	log.Printf("[validate] tolerance: %g", *tolerance)

	// Shell out to the Python validation tool.
	cmd := exec.Command("python3", "tools/validate_export.py",
		"--source", *source,
		"--export", *export,
		"--samples", strconv.Itoa(*samples),
		"--tolerance", fmt.Sprintf("%g", *tolerance))

	cmd.Stderr = os.Stderr

	output, err := cmd.Output()
	if err != nil {
		// The Python script exits 1 if validation fails (max_diff > tolerance).
		// Try to parse the JSON output even on non-zero exit.
		if len(output) == 0 {
			fmt.Fprintf(os.Stderr, "validate: validation tool failed: %v\n", err)
			os.Exit(1)
		}
	}

	// Parse JSON result from the Python tool.
	var result ValidationResult
	if err := json.Unmarshal(output, &result); err != nil {
		fmt.Fprintf(os.Stderr, "validate: could not parse validation output: %v\nraw output: %s\n", err, string(output))
		os.Exit(1)
	}

	// Print results.
	if result.Passed {
		log.Printf("[validate] PASSED — max_diff=%.6g across %d samples (tolerance=%g)",
			result.MaxDiff, result.Samples, *tolerance)
	} else {
		log.Printf("[validate] FAILED — max_diff=%.6g exceeds tolerance %g (%d samples)",
			result.MaxDiff, *tolerance, result.Samples)
	}

	// Update registry with validation results.
	updateRegistryValidation(*export, &result)

	if !result.Passed {
		os.Exit(1)
	}
}
