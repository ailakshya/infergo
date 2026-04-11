package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// validConvertFormats lists the export formats supported by the convert command.
var validConvertFormats = map[string]bool{
	"torchscript": true,
	"onnx":        true,
}

// runConvert implements the "infergo convert" subcommand.
// It shells out to Python tools to export a PyTorch model to TorchScript or ONNX,
// then records the result in the local model registry.
func runConvert(args []string) {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	input := fs.String("input", "", "source model path (.pt PyTorch checkpoint or model name)")
	format := fs.String("format", "torchscript", "output format: onnx|torchscript")
	output := fs.String("output", "", "output path (auto-generated if empty)")
	imgsz := fs.Int("imgsz", 640, "input image size")
	fs.Parse(args)

	if *input == "" {
		fmt.Fprintln(os.Stderr, "convert: --input is required")
		fmt.Fprintln(os.Stderr, "usage: infergo convert --input <model.pt> [--format torchscript|onnx] [--output <path>] [--imgsz 640]")
		os.Exit(1)
	}

	if !validConvertFormats[*format] {
		fmt.Fprintf(os.Stderr, "convert: unsupported format %q (supported: torchscript, onnx)\n", *format)
		os.Exit(1)
	}

	// Auto-generate output path if not provided.
	if *output == "" {
		base := strings.TrimSuffix(filepath.Base(*input), filepath.Ext(*input))
		switch *format {
		case "torchscript":
			*output = filepath.Join("models", base+".torchscript.pt")
		case "onnx":
			*output = filepath.Join("models", base+".onnx")
		}
	}

	log.Printf("[convert] input:  %s", *input)
	log.Printf("[convert] format: %s", *format)
	log.Printf("[convert] output: %s", *output)
	log.Printf("[convert] imgsz:  %d", *imgsz)

	// Shell out to the Python conversion tool.
	// This is the ONLY place Python is used — for model export.
	var cmd *exec.Cmd
	switch *format {
	case "torchscript":
		cmd = exec.Command("python3", "tools/convert_to_torchscript.py",
			"--source", *input,
			"--output", *output,
			"--imgsz", strconv.Itoa(*imgsz))
	case "onnx":
		cmd = exec.Command("python3", "tools/convert_to_torchscript.py",
			"--source", *input,
			"--output", *output,
			"--imgsz", strconv.Itoa(*imgsz),
			"--format", "onnx")
	}

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "convert: conversion failed: %v\n", err)
		os.Exit(1)
	}

	// Verify the output file exists.
	info, err := os.Stat(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "convert: output file not found after conversion: %v\n", err)
		os.Exit(1)
	}

	log.Printf("[convert] success: %s (%.1f MB)", *output, float64(info.Size())/(1024*1024))

	// Update model registry.
	updateRegistry(*output, *input, *format, *imgsz)
	log.Printf("[convert] registry updated: %s", defaultRegistryPath())
}
