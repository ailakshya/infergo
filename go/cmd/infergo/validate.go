package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// runValidate implements the "infergo validate" subcommand.
//
// Two modes of operation:
//
//  1. Quick validation (single model):
//     infergo validate model.gguf
//     infergo validate model.onnx --type embed
//     Loads the model, runs one inference, reports model info and success/failure.
//
//  2. Comparison validation (source vs export):
//     infergo validate --source original.pt --export exported.onnx
//     Shells out to Python to compare outputs on random inputs.
func runValidate(args []string) {
	fs := flag.NewFlagSet("validate", flag.ExitOnError)
	source := fs.String("source", "", "original PyTorch model (.pt) for comparison mode")
	export := fs.String("export", "", "exported model (.onnx or .torchscript.pt) for comparison mode")
	samples := fs.Int("samples", 100, "number of random inputs to test (comparison mode)")
	tolerance := fs.Float64("tolerance", 1e-4, "max allowed output difference (comparison mode)")
	modelType := fs.String("type", "", "model type hint: llm|embed|detect (quick validation mode)")
	verbose := fs.Bool("verbose", false, "print detailed model information")
	fs.Parse(args)

	// Determine which mode to use.
	// If --source and --export are set, use comparison mode.
	// If positional arg is given, use quick validation mode.
	if *source != "" && *export != "" {
		runCompareValidation(*source, *export, *samples, *tolerance)
		return
	}

	// Quick validation mode: positional argument is the model path.
	if fs.NArg() < 1 && *source == "" {
		fmt.Fprintln(os.Stderr, `validate: model path required

Quick validation (single model):
  infergo validate model.gguf
  infergo validate model.onnx --type embed

Comparison validation (source vs export):
  infergo validate --source original.pt --export exported.onnx [--samples 100] [--tolerance 1e-4]`)
		os.Exit(1)
	}

	// If only --source is given (no --export), treat it as a quick validation path.
	modelPath := fs.Arg(0)
	if modelPath == "" && *source != "" {
		modelPath = *source
	}

	runQuickValidation(modelPath, *modelType, *verbose)
}

// runQuickValidation loads a model, runs a test inference, and reports results.
func runQuickValidation(modelPath, modelType string, verbose bool) {
	// Verify file exists.
	info, err := os.Stat(modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "validate: cannot stat %q: %v\n", modelPath, err)
		os.Exit(1)
	}

	ext := strings.ToLower(filepath.Ext(modelPath))

	// Auto-detect model type if not specified.
	if modelType == "" {
		modelType = detectModelType(modelPath, ext)
	}

	// Print model info header.
	sizeMB := float64(info.Size()) / (1024 * 1024)
	sizeStr := fmt.Sprintf("%.1f MB", sizeMB)
	if sizeMB > 1024 {
		sizeStr = fmt.Sprintf("%.2f GB", sizeMB/1024)
	}

	fmt.Println("Model Validation Report")
	fmt.Println(strings.Repeat("=", 50))
	fmt.Printf("  File:       %s\n", modelPath)
	fmt.Printf("  Format:     %s\n", formatName(ext))
	fmt.Printf("  Type:       %s\n", modelType)
	fmt.Printf("  Size:       %s\n", sizeStr)

	// Format-specific metadata.
	printModelMetadata(modelPath, ext, modelType, verbose)

	// Run a test inference based on the model type and format.
	fmt.Println()
	fmt.Println("Inference Test")
	fmt.Println(strings.Repeat("-", 50))

	switch ext {
	case ".gguf":
		validateGGUF(modelPath, verbose)
	case ".onnx":
		validateONNX(modelPath, modelType, verbose)
	case ".pt", ".pth":
		validateTorchScript(modelPath, modelType, verbose)
	default:
		fmt.Printf("  SKIP: no test inference for %s format\n", ext)
		fmt.Println()
		fmt.Println("Result: SKIP (format not directly loadable)")
	}
}

// detectModelType auto-detects the model type from the filename and extension.
func detectModelType(path, ext string) string {
	lname := strings.ToLower(filepath.Base(path))

	// Check filename patterns.
	if strings.Contains(lname, "yolo") {
		return "detect"
	}
	if strings.Contains(lname, "embed") || strings.Contains(lname, "minilm") ||
		strings.Contains(lname, "bge") || strings.Contains(lname, "e5-") ||
		strings.Contains(lname, "gte-") {
		return "embed"
	}

	// Check for tokenizer.json nearby (indicates embedding model for ONNX).
	if ext == ".onnx" {
		if tok := findTokenizerJSON(filepath.Dir(path), 2); tok != "" {
			return "embed"
		}
	}

	// Default by extension.
	switch ext {
	case ".gguf":
		return "llm"
	default:
		return "unknown"
	}
}

// formatName returns a human-readable format name for a file extension.
func formatName(ext string) string {
	switch ext {
	case ".gguf":
		return "GGUF (llama.cpp)"
	case ".onnx":
		return "ONNX"
	case ".pt", ".pth":
		return "TorchScript"
	case ".engine":
		return "TensorRT"
	case ".safetensors":
		return "SafeTensors"
	default:
		return ext
	}
}

// printModelMetadata prints format-specific metadata about the model.
func printModelMetadata(path, ext, modelType string, verbose bool) {
	switch ext {
	case ".gguf":
		printGGUFMetadata(path, verbose)
	case ".onnx":
		printONNXMetadata(path, verbose)
	case ".pt", ".pth":
		printTorchMetadata(path, verbose)
	}
}

// printGGUFMetadata extracts and prints GGUF model metadata.
// Uses a small Python script since GGUF metadata parsing is complex.
func printGGUFMetadata(path string, verbose bool) {
	script := `
import sys, struct, json

def read_gguf_metadata(path):
    info = {}
    try:
        with open(path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                info['error'] = 'not a valid GGUF file'
                return info
            version = struct.unpack('<I', f.read(4))[0]
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_kv_count = struct.unpack('<Q', f.read(8))[0]
            info['version'] = version
            info['tensor_count'] = tensor_count
            info['metadata_kv_count'] = metadata_kv_count

            # Read metadata key-value pairs (simplified).
            kv = {}
            for _ in range(min(metadata_kv_count, 50)):  # cap to avoid huge reads
                try:
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    if key_len > 256:
                        break
                    key = f.read(key_len).decode('utf-8', errors='replace')
                    val_type = struct.unpack('<I', f.read(4))[0]
                    # Type 8 = string
                    if val_type == 8:
                        str_len = struct.unpack('<Q', f.read(8))[0]
                        if str_len > 1024:
                            f.seek(str_len, 1)
                            continue
                        val = f.read(str_len).decode('utf-8', errors='replace')
                        kv[key] = val
                    # Type 4 = uint32
                    elif val_type == 4:
                        val = struct.unpack('<I', f.read(4))[0]
                        kv[key] = val
                    # Type 6 = float32
                    elif val_type == 6:
                        val = struct.unpack('<f', f.read(4))[0]
                        kv[key] = round(val, 6)
                    # Type 10 = uint64
                    elif val_type == 10:
                        val = struct.unpack('<Q', f.read(8))[0]
                        kv[key] = val
                    # Type 5 = int32
                    elif val_type == 5:
                        val = struct.unpack('<i', f.read(4))[0]
                        kv[key] = val
                    # Type 0 = uint8
                    elif val_type == 0:
                        val = struct.unpack('<B', f.read(1))[0]
                        kv[key] = val
                    # Type 1 = int8
                    elif val_type == 1:
                        val = struct.unpack('<b', f.read(1))[0]
                        kv[key] = val
                    # Type 7 = bool
                    elif val_type == 7:
                        val = struct.unpack('<B', f.read(1))[0]
                        kv[key] = bool(val)
                    else:
                        break  # skip complex types
                except:
                    break
            info['metadata'] = kv
    except Exception as e:
        info['error'] = str(e)
    return info

info = read_gguf_metadata(sys.argv[1])
print(json.dumps(info))
`
	cmd := exec.Command("python3", "-c", script, path)
	out, err := cmd.Output()
	if err != nil {
		// Not critical -- just skip metadata.
		return
	}

	var info map[string]interface{}
	if err := json.Unmarshal(out, &info); err != nil {
		return
	}

	if errMsg, ok := info["error"]; ok {
		fmt.Printf("  Metadata:   error: %v\n", errMsg)
		return
	}

	if tc, ok := info["tensor_count"]; ok {
		fmt.Printf("  Tensors:    %.0f\n", tc)
	}
	if v, ok := info["version"]; ok {
		fmt.Printf("  GGUF ver:   %.0f\n", v)
	}

	if kv, ok := info["metadata"].(map[string]interface{}); ok {
		if arch, ok := kv["general.architecture"]; ok {
			fmt.Printf("  Arch:       %v\n", arch)
		}
		if name, ok := kv["general.name"]; ok {
			fmt.Printf("  Name:       %v\n", name)
		}
		if qt, ok := kv["general.file_type"]; ok {
			fmt.Printf("  File type:  %v\n", qt)
		}
		if cs, ok := kv["llama.context_length"]; ok {
			fmt.Printf("  Context:    %v\n", cs)
		}
		if es, ok := kv["llama.embedding_length"]; ok {
			fmt.Printf("  Embed dim:  %v\n", es)
		}

		if verbose {
			fmt.Println()
			fmt.Println("  All metadata:")
			for k, v := range kv {
				fmt.Printf("    %s = %v\n", k, v)
			}
		}
	}
}

// printONNXMetadata prints ONNX model metadata using Python.
func printONNXMetadata(path string, verbose bool) {
	script := `
import sys, json
try:
    import onnx
    model = onnx.load(sys.argv[1], load_external_data=False)
    info = {
        'opset': model.opset_import[0].version if model.opset_import else 0,
        'ir_version': model.ir_version,
        'producer': model.producer_name,
        'inputs': [],
        'outputs': [],
    }
    # Count parameters.
    param_count = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        param_count += n
    info['parameters'] = param_count

    for inp in model.graph.input:
        shape = []
        for d in inp.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else -1)
        info['inputs'].append({'name': inp.name, 'shape': shape})
    for out in model.graph.output:
        shape = []
        for d in out.type.tensor_type.shape.dim:
            shape.append(d.dim_value if d.dim_value > 0 else -1)
        info['outputs'].append({'name': out.name, 'shape': shape})
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({'error': str(e)}))
`
	cmd := exec.Command("python3", "-c", script, path)
	out, err := cmd.Output()
	if err != nil {
		return
	}

	var info map[string]interface{}
	if err := json.Unmarshal(out, &info); err != nil {
		return
	}

	if errMsg, ok := info["error"]; ok {
		fmt.Printf("  Metadata:   error: %v\n", errMsg)
		return
	}

	if opset, ok := info["opset"]; ok {
		fmt.Printf("  Opset:      %.0f\n", opset)
	}
	if params, ok := info["parameters"]; ok {
		p := params.(float64)
		if p > 1e9 {
			fmt.Printf("  Parameters: %.2fB\n", p/1e9)
		} else if p > 1e6 {
			fmt.Printf("  Parameters: %.1fM\n", p/1e6)
		} else if p > 0 {
			fmt.Printf("  Parameters: %.0f\n", p)
		}
	}
	if producer, ok := info["producer"]; ok && producer != "" {
		fmt.Printf("  Producer:   %v\n", producer)
	}

	if inputs, ok := info["inputs"].([]interface{}); ok && len(inputs) > 0 {
		for _, inp := range inputs {
			m := inp.(map[string]interface{})
			fmt.Printf("  Input:      %s %v\n", m["name"], m["shape"])
		}
	}
	if outputs, ok := info["outputs"].([]interface{}); ok && len(outputs) > 0 {
		for _, out := range outputs {
			m := out.(map[string]interface{})
			fmt.Printf("  Output:     %s %v\n", m["name"], m["shape"])
		}
	}
}

// printTorchMetadata prints TorchScript model metadata using Python.
func printTorchMetadata(path string, verbose bool) {
	script := `
import sys, json
try:
    import torch
    model = torch.jit.load(sys.argv[1], map_location='cpu')
    info = {}
    param_count = sum(p.numel() for p in model.parameters())
    info['parameters'] = param_count
    info['training'] = model.training
    print(json.dumps(info))
except Exception as e:
    print(json.dumps({'error': str(e)}))
`
	cmd := exec.Command("python3", "-c", script, path)
	out, err := cmd.Output()
	if err != nil {
		return
	}

	var info map[string]interface{}
	if err := json.Unmarshal(out, &info); err != nil {
		return
	}

	if errMsg, ok := info["error"]; ok {
		fmt.Printf("  Metadata:   error: %v\n", errMsg)
		return
	}

	if params, ok := info["parameters"]; ok {
		p := params.(float64)
		if p > 1e9 {
			fmt.Printf("  Parameters: %.2fB\n", p/1e9)
		} else if p > 1e6 {
			fmt.Printf("  Parameters: %.1fM\n", p/1e6)
		} else if p > 0 {
			fmt.Printf("  Parameters: %.0f\n", p)
		}
	}
}

// validateGGUF loads a GGUF model through infergo's LLM backend and runs a test inference.
func validateGGUF(path string, verbose bool) {
	// Use a Python script with llama-cpp-python for quick validation,
	// since loading through infergo's C bindings requires the full server setup.
	script := `
import sys, json, time
info = {}
try:
    from llama_cpp import Llama
    t0 = time.time()
    llm = Llama(model_path=sys.argv[1], n_ctx=512, n_gpu_layers=0, verbose=False)
    load_time = time.time() - t0
    info['load_time_ms'] = round(load_time * 1000, 1)

    # Run a simple completion.
    t0 = time.time()
    output = llm("Hello", max_tokens=16, echo=False)
    infer_time = time.time() - t0
    info['infer_time_ms'] = round(infer_time * 1000, 1)

    text = output['choices'][0]['text'].strip()
    tokens = output['usage']['completion_tokens']
    info['output'] = text[:200]
    info['tokens'] = tokens
    info['passed'] = True
    del llm
except ImportError:
    # Fall back: just verify GGUF magic bytes.
    try:
        with open(sys.argv[1], 'rb') as f:
            magic = f.read(4)
        if magic == b'GGUF':
            info['passed'] = True
            info['note'] = 'GGUF header valid (install llama-cpp-python for full test)'
        else:
            info['passed'] = False
            info['error'] = f'invalid GGUF magic: {magic!r}'
    except Exception as e:
        info['passed'] = False
        info['error'] = str(e)
except Exception as e:
    info['passed'] = False
    info['error'] = str(e)

print(json.dumps(info))
`
	cmd := exec.Command("python3", "-c", script, path)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil && len(out) == 0 {
		fmt.Printf("  FAIL: could not run validation: %v\n", err)
		fmt.Println()
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		fmt.Printf("  FAIL: could not parse validation output: %v\n", err)
		fmt.Println()
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}

	if errMsg, ok := result["error"]; ok {
		fmt.Printf("  Error:      %v\n", errMsg)
	}
	if note, ok := result["note"]; ok {
		fmt.Printf("  Note:       %v\n", note)
	}
	if lt, ok := result["load_time_ms"]; ok {
		fmt.Printf("  Load time:  %.1f ms\n", lt)
	}
	if it, ok := result["infer_time_ms"]; ok {
		fmt.Printf("  Infer time: %.1f ms\n", it)
	}
	if tokens, ok := result["tokens"]; ok {
		fmt.Printf("  Tokens:     %.0f\n", tokens)
	}
	if output, ok := result["output"]; ok && verbose {
		fmt.Printf("  Output:     %v\n", output)
	}

	passed, _ := result["passed"].(bool)
	fmt.Println()
	if passed {
		fmt.Println("Result: PASS")
	} else {
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}
}

// validateONNX loads an ONNX model and runs a test inference.
func validateONNX(path, modelType string, verbose bool) {
	script := `
import sys, json, time
import numpy as np
model_path = sys.argv[1]
model_type = sys.argv[2]
info = {}
try:
    import onnxruntime as ort
    t0 = time.time()
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    load_time = time.time() - t0
    info['load_time_ms'] = round(load_time * 1000, 1)

    # Get input/output info.
    inputs = sess.get_inputs()
    outputs = sess.get_outputs()
    info['input_count'] = len(inputs)
    info['output_count'] = len(outputs)

    # Build dummy input.
    feed = {}
    for inp in inputs:
        shape = []
        for d in inp.shape:
            shape.append(d if isinstance(d, int) and d > 0 else 1)
        dtype = np.float32
        if 'int64' in inp.type:
            dtype = np.int64
        elif 'int32' in inp.type:
            dtype = np.int32
        feed[inp.name] = np.random.randn(*shape).astype(dtype)

    # Run inference.
    t0 = time.time()
    results = sess.run(None, feed)
    infer_time = time.time() - t0
    info['infer_time_ms'] = round(infer_time * 1000, 1)

    # Report output shapes.
    output_shapes = [list(r.shape) for r in results]
    info['output_shapes'] = output_shapes

    if model_type == 'embed' and len(results) > 0:
        # For embedding models, show the embedding dimension.
        if len(results[0].shape) == 3:
            info['embed_dim'] = int(results[0].shape[2])
        elif len(results[0].shape) == 2:
            info['embed_dim'] = int(results[0].shape[1])
        # Show sample embedding values.
        flat = results[0].flatten()[:5]
        info['sample_output'] = [round(float(v), 6) for v in flat]

    elif model_type == 'detect' and len(results) > 0:
        info['detection_shape'] = list(results[0].shape)
        info['sample_output'] = [round(float(v), 6) for v in results[0].flatten()[:5]]

    info['passed'] = True

except Exception as e:
    info['passed'] = False
    info['error'] = str(e)

print(json.dumps(info))
`
	cmd := exec.Command("python3", "-c", script, path, modelType)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil && len(out) == 0 {
		fmt.Printf("  FAIL: could not run validation: %v\n", err)
		fmt.Println()
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		fmt.Printf("  FAIL: could not parse validation output: %v\n", err)
		fmt.Println()
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}

	printValidateResult(result, verbose)
}

// validateTorchScript loads a TorchScript model and runs a test inference.
func validateTorchScript(path, modelType string, verbose bool) {
	script := `
import sys, json, time
import numpy as np
model_path = sys.argv[1]
model_type = sys.argv[2]
info = {}
try:
    import torch
    t0 = time.time()
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    load_time = time.time() - t0
    info['load_time_ms'] = round(load_time * 1000, 1)

    param_count = sum(p.numel() for p in model.parameters())
    info['parameters'] = param_count

    # Determine input shape.
    if model_type == 'detect':
        dummy = torch.randn(1, 3, 640, 640)
    elif model_type == 'embed':
        dummy = torch.randint(0, 1000, (1, 128)).long()
    else:
        dummy = torch.randn(1, 3, 640, 640)

    # Run inference.
    t0 = time.time()
    with torch.no_grad():
        out = model(dummy)
    infer_time = time.time() - t0
    info['infer_time_ms'] = round(infer_time * 1000, 1)

    if isinstance(out, (tuple, list)):
        out = out[0]
    info['output_shape'] = list(out.shape)

    # Sample output values.
    flat = out.flatten()[:5]
    info['sample_output'] = [round(float(v), 6) for v in flat]
    info['passed'] = True

except Exception as e:
    info['passed'] = False
    info['error'] = str(e)

print(json.dumps(info))
`
	cmd := exec.Command("python3", "-c", script, path, modelType)
	cmd.Stderr = os.Stderr
	out, err := cmd.Output()
	if err != nil && len(out) == 0 {
		fmt.Printf("  FAIL: could not run validation: %v\n", err)
		fmt.Println()
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		fmt.Printf("  FAIL: could not parse validation output: %v\n", err)
		fmt.Println()
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}

	printValidateResult(result, verbose)
}

// printValidateResult prints the common validation result fields.
func printValidateResult(result map[string]interface{}, verbose bool) {
	if errMsg, ok := result["error"]; ok {
		fmt.Printf("  Error:         %v\n", errMsg)
	}
	if lt, ok := result["load_time_ms"]; ok {
		fmt.Printf("  Load time:     %.1f ms\n", lt)
	}
	if it, ok := result["infer_time_ms"]; ok {
		fmt.Printf("  Infer time:    %.1f ms\n", it)
	}
	if params, ok := result["parameters"]; ok {
		p := params.(float64)
		if p > 1e9 {
			fmt.Printf("  Parameters:    %.2fB\n", p/1e9)
		} else if p > 1e6 {
			fmt.Printf("  Parameters:    %.1fM\n", p/1e6)
		} else if p > 0 {
			fmt.Printf("  Parameters:    %.0f\n", p)
		}
	}
	if shapes, ok := result["output_shapes"]; ok {
		fmt.Printf("  Output shapes: %v\n", shapes)
	}
	if shape, ok := result["output_shape"]; ok {
		fmt.Printf("  Output shape:  %v\n", shape)
	}
	if dim, ok := result["embed_dim"]; ok {
		fmt.Printf("  Embed dim:     %.0f\n", dim)
	}
	if ds, ok := result["detection_shape"]; ok {
		fmt.Printf("  Detect shape:  %v\n", ds)
	}
	if sample, ok := result["sample_output"]; ok && verbose {
		fmt.Printf("  Sample output: %v\n", sample)
	}

	passed, _ := result["passed"].(bool)
	fmt.Println()
	if passed {
		fmt.Println("Result: PASS")
	} else {
		fmt.Println("Result: FAIL")
		os.Exit(1)
	}
}

// runCompareValidation compares a source PyTorch model against an exported model.
// This is the original validation mode that shells out to tools/validate_export.py.
func runCompareValidation(source, export string, samples int, tolerance float64) {
	log.Printf("[validate] source:    %s", source)
	log.Printf("[validate] export:    %s", export)
	log.Printf("[validate] samples:   %d", samples)
	log.Printf("[validate] tolerance: %g", tolerance)

	// Shell out to the Python validation tool.
	cmd := exec.Command("python3", "tools/validate_export.py",
		"--source", source,
		"--export", export,
		"--samples", strconv.Itoa(samples),
		"--tolerance", fmt.Sprintf("%g", tolerance))

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
			result.MaxDiff, result.Samples, tolerance)
	} else {
		log.Printf("[validate] FAILED — max_diff=%.6g exceeds tolerance %g (%d samples)",
			result.MaxDiff, tolerance, result.Samples)
	}

	// Update registry with validation results.
	updateRegistryValidation(export, &result)

	if !result.Passed {
		os.Exit(1)
	}
}
