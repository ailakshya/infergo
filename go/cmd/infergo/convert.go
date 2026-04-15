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
	"tensorrt":    true,
	"gguf":        true,
}

// runConvert implements the "infergo convert" subcommand.
//
// Supported conversions:
//
//	infergo convert --input model.pt  --output model.onnx           --format onnx
//	infergo convert --input model.pt  --output model.torchscript.pt --format torchscript
//	infergo convert --input model.onnx --output model.trt           --format tensorrt
//	infergo convert --input model.safetensors --output model.gguf   --format gguf --quant q4_k_m
//
// ONNX and TorchScript conversions shell out to Python torch/onnx tools.
// GGUF conversion shells out to llama.cpp's convert_hf_to_gguf.py.
// TensorRT conversion exports to ONNX first (if needed), then builds a TRT engine.
func runConvert(args []string) {
	fs := flag.NewFlagSet("convert", flag.ExitOnError)
	input := fs.String("input", "", "source model path (.pt, .pth, .onnx, .safetensors, or model name)")
	format := fs.String("format", "torchscript", "output format: onnx|torchscript|tensorrt|gguf")
	output := fs.String("output", "", "output path (auto-generated if empty)")
	imgsz := fs.Int("imgsz", 640, "input image size (for detection models)")
	quantize := fs.String("quantize", "", "quantization: int8|uint8 (ONNX dynamic quantization)")
	quant := fs.String("quant", "", "GGUF quantization type: q4_0|q4_k_m|q5_k_m|q8_0|f16 (GGUF only)")
	llamaCppDir := fs.String("llama-cpp-dir", "", "path to llama.cpp directory (for GGUF conversion; default: auto-detect)")
	hfModelDir := fs.String("hf-model-dir", "", "path to HuggingFace model directory (for GGUF conversion; default: parent of --input)")
	fs.Parse(args)

	if *input == "" {
		fmt.Fprintln(os.Stderr, "convert: --input is required")
		fmt.Fprintln(os.Stderr, `usage: infergo convert --input <model> [--format onnx|torchscript|tensorrt|gguf]
       [--output <path>] [--imgsz 640] [--quant q4_k_m]`)
		os.Exit(1)
	}

	if !validConvertFormats[*format] {
		fmt.Fprintf(os.Stderr, "convert: unsupported format %q\n", *format)
		fmt.Fprintln(os.Stderr, "supported formats: torchscript, onnx, tensorrt, gguf")
		os.Exit(1)
	}

	// Validate format/input combinations.
	inputExt := strings.ToLower(filepath.Ext(*input))
	if *format == "gguf" && inputExt != ".safetensors" && inputExt != "" {
		// For GGUF conversion, the input should be a directory containing
		// safetensors files, or a path to a .safetensors file.
		if info, err := os.Stat(*input); err == nil && !info.IsDir() && inputExt != ".safetensors" {
			fmt.Fprintf(os.Stderr, "convert: GGUF conversion expects a HuggingFace model directory or .safetensors file, got %q\n", *input)
			fmt.Fprintln(os.Stderr, "hint: point --input at the directory containing config.json + model.safetensors")
			os.Exit(1)
		}
	}
	if *format == "tensorrt" && inputExt != ".onnx" && inputExt != ".pt" && inputExt != ".pth" {
		// Accept .pt/.pth (will export to ONNX first) and .onnx (direct TRT build).
		if inputExt != "" {
			fmt.Fprintf(os.Stderr, "convert: TensorRT conversion expects .pt, .pth, or .onnx input, got %q\n", inputExt)
			os.Exit(1)
		}
	}

	// Auto-generate output path if not provided.
	if *output == "" {
		base := strings.TrimSuffix(filepath.Base(*input), filepath.Ext(*input))
		switch *format {
		case "torchscript":
			*output = filepath.Join("models", base+".torchscript.pt")
		case "onnx":
			*output = filepath.Join("models", base+".onnx")
		case "tensorrt":
			*output = filepath.Join("models", base+".engine")
		case "gguf":
			suffix := "f16"
			if *quant != "" {
				suffix = strings.ToLower(*quant)
			}
			*output = filepath.Join("models", base+"-"+suffix+".gguf")
		}
	}

	// Ensure output directory exists.
	outDir := filepath.Dir(*output)
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "convert: cannot create output directory %s: %v\n", outDir, err)
		os.Exit(1)
	}

	log.Printf("[convert] input:  %s", *input)
	log.Printf("[convert] format: %s", *format)
	log.Printf("[convert] output: %s", *output)
	if *format != "gguf" {
		log.Printf("[convert] imgsz:  %d", *imgsz)
	}
	if *quant != "" {
		log.Printf("[convert] quant:  %s", *quant)
	}

	// Dispatch to the appropriate conversion pipeline.
	switch *format {
	case "torchscript":
		convertTorchScript(*input, *output, *imgsz)
	case "onnx":
		convertONNX(*input, *output, *imgsz, *quantize)
	case "tensorrt":
		convertTensorRT(*input, *output, *imgsz)
	case "gguf":
		convertGGUF(*input, *output, *quant, *llamaCppDir, *hfModelDir)
	}

	// Verify the output file exists.
	info, err := os.Stat(*output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "convert: output file not found after conversion: %v\n", err)
		os.Exit(1)
	}

	log.Printf("[convert] success: %s (%.1f MB)", *output, float64(info.Size())/(1024*1024))

	// Update model registry.
	imgszVal := *imgsz
	if *format == "gguf" {
		imgszVal = 0 // not applicable for LLMs
	}
	updateRegistry(*output, *input, *format, imgszVal)
	log.Printf("[convert] registry updated: %s", defaultRegistryPath())
}

// convertTorchScript exports a PyTorch model to TorchScript format.
func convertTorchScript(input, output string, imgsz int) {
	cmd := exec.Command("python3", "tools/convert_to_torchscript.py",
		"--source", input,
		"--output", output,
		"--imgsz", strconv.Itoa(imgsz))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "convert: TorchScript conversion failed: %v\n", err)
		os.Exit(1)
	}
}

// convertONNX exports a PyTorch model to ONNX format.
func convertONNX(input, output string, imgsz int, quantize string) {
	cmd := exec.Command("python3", "tools/convert_to_torchscript.py",
		"--source", input,
		"--output", output,
		"--imgsz", strconv.Itoa(imgsz),
		"--format", "onnx")
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "convert: ONNX conversion failed: %v\n", err)
		os.Exit(1)
	}

	// Optional: dynamic quantization of the ONNX model.
	if quantize != "" {
		quantizeONNX(output, quantize)
	}
}

// quantizeONNX applies dynamic quantization to an ONNX model.
func quantizeONNX(modelPath, quantType string) {
	info, err := os.Stat(modelPath)
	if err != nil {
		log.Printf("[convert] WARNING: cannot stat model for quantization: %v", err)
		return
	}

	quantOut := strings.TrimSuffix(modelPath, ".onnx") + "-" + quantType + ".onnx"
	log.Printf("[convert] quantizing to %s (%s)...", quantOut, quantType)

	qcmd := exec.Command("python3", "-c", fmt.Sprintf(`
from onnxruntime.quantization import quantize_dynamic, QuantType
qt = QuantType.QInt8 if "%s" == "int8" else QuantType.QUInt8
quantize_dynamic("%s", "%s", weight_type=qt)
print("quantization complete")
`, quantType, modelPath, quantOut))
	qcmd.Stdout = os.Stdout
	qcmd.Stderr = os.Stderr

	if err := qcmd.Run(); err != nil {
		log.Printf("[convert] WARNING: quantization failed: %v", err)
	} else {
		qinfo, _ := os.Stat(quantOut)
		if qinfo != nil {
			ratio := float64(info.Size()) / float64(qinfo.Size())
			log.Printf("[convert] quantized: %s (%.1f MB, %.1fx smaller)",
				quantOut, float64(qinfo.Size())/(1024*1024), ratio)
		}
	}
}

// convertTensorRT builds a TensorRT engine from a model.
// If the input is not already ONNX, it first exports to ONNX.
func convertTensorRT(input, output string, imgsz int) {
	onnxPath := input

	// If input is not ONNX, export to ONNX first.
	inputExt := strings.ToLower(filepath.Ext(input))
	if inputExt != ".onnx" {
		trtBase := strings.TrimSuffix(filepath.Base(input), filepath.Ext(input))
		onnxPath = filepath.Join("models", trtBase+".onnx")

		if _, err := os.Stat(onnxPath); err != nil {
			log.Printf("[convert] step 1: exporting to ONNX first...")
			pre := exec.Command("python3", "tools/convert_to_torchscript.py",
				"--source", input,
				"--output", onnxPath,
				"--imgsz", strconv.Itoa(imgsz),
				"--format", "onnx")
			pre.Stdout = os.Stdout
			pre.Stderr = os.Stderr
			if err := pre.Run(); err != nil {
				fmt.Fprintf(os.Stderr, "convert: ONNX export (for TensorRT) failed: %v\n", err)
				os.Exit(1)
			}
		} else {
			log.Printf("[convert] step 1: using existing ONNX file %s", onnxPath)
		}
	}

	log.Printf("[convert] step 2: building TensorRT engine from %s...", onnxPath)
	cmd := exec.Command("python3", "-c", fmt.Sprintf(`
import tensorrt as trt
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open("%s", "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
config.set_flag(trt.BuilderFlag.FP16)
engine = builder.build_serialized_network(network, config)
with open("%s", "wb") as f:
    f.write(engine)
print("TensorRT engine built successfully")
`, onnxPath, output))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "convert: TensorRT engine build failed: %v\n", err)
		os.Exit(1)
	}
}

// convertGGUF converts a HuggingFace model (safetensors) to GGUF format.
// Uses llama.cpp's convert_hf_to_gguf.py for the conversion, then optionally
// quantizes with llama-quantize.
func convertGGUF(input, output, quant, llamaCppDir, hfModelDir string) {
	// Determine the HuggingFace model directory.
	modelDir := hfModelDir
	if modelDir == "" {
		info, err := os.Stat(input)
		if err != nil {
			fmt.Fprintf(os.Stderr, "convert: cannot stat input %q: %v\n", input, err)
			os.Exit(1)
		}
		if info.IsDir() {
			modelDir = input
		} else {
			modelDir = filepath.Dir(input)
		}
	}

	// Verify the model directory contains the expected files.
	configPath := filepath.Join(modelDir, "config.json")
	if _, err := os.Stat(configPath); err != nil {
		fmt.Fprintf(os.Stderr, "convert: config.json not found in %s\n", modelDir)
		fmt.Fprintln(os.Stderr, "hint: --input should point to a HuggingFace model directory containing config.json + model.safetensors")
		os.Exit(1)
	}

	// Find llama.cpp's convert script.
	convertScript := findConvertScript(llamaCppDir)
	if convertScript == "" {
		fmt.Fprintln(os.Stderr, "convert: cannot find llama.cpp's convert_hf_to_gguf.py")
		fmt.Fprintln(os.Stderr, "hint: set --llama-cpp-dir to your llama.cpp directory, or ensure convert_hf_to_gguf.py is in PATH")
		os.Exit(1)
	}
	log.Printf("[convert] using convert script: %s", convertScript)

	// Determine output type for the initial conversion.
	// If quantization is requested, first convert to f16, then quantize.
	// If no quantization, convert to f16 directly.
	needsQuantize := quant != "" && strings.ToLower(quant) != "f16"
	initialOutput := output
	if needsQuantize {
		// Convert to f16 first, then quantize to the desired type.
		initialOutput = strings.TrimSuffix(output, ".gguf") + "-f16.gguf"
	}

	// Step 1: Convert HF model to GGUF (f16 or f32 base).
	log.Printf("[convert] step 1: converting HuggingFace model to GGUF...")
	convertArgs := []string{convertScript, modelDir, "--outfile", initialOutput, "--outtype", "f16"}
	cmd := exec.Command("python3", convertArgs...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		fmt.Fprintf(os.Stderr, "convert: GGUF conversion failed: %v\n", err)
		os.Exit(1)
	}

	if _, err := os.Stat(initialOutput); err != nil {
		fmt.Fprintf(os.Stderr, "convert: GGUF output not created at %s\n", initialOutput)
		os.Exit(1)
	}

	// Step 2: Quantize if requested.
	if needsQuantize {
		log.Printf("[convert] step 2: quantizing to %s...", quant)
		quantizeBin := findLlamaQuantize(llamaCppDir)
		if quantizeBin == "" {
			fmt.Fprintln(os.Stderr, "convert: cannot find llama-quantize binary")
			fmt.Fprintln(os.Stderr, "hint: build llama.cpp first, or set --llama-cpp-dir")
			// Fall back: keep the f16 version as the output.
			log.Printf("[convert] WARNING: quantization skipped, using f16 output")
			if err := os.Rename(initialOutput, output); err != nil {
				fmt.Fprintf(os.Stderr, "convert: cannot rename %s to %s: %v\n", initialOutput, output, err)
			}
			return
		}

		qcmd := exec.Command(quantizeBin, initialOutput, output, strings.ToUpper(quant))
		qcmd.Stdout = os.Stdout
		qcmd.Stderr = os.Stderr

		if err := qcmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "convert: GGUF quantization failed: %v\n", err)
			os.Exit(1)
		}

		// Clean up the intermediate f16 file.
		if initialOutput != output {
			os.Remove(initialOutput)
			log.Printf("[convert] cleaned up intermediate file: %s", initialOutput)
		}
	}
}

// findConvertScript searches for llama.cpp's convert_hf_to_gguf.py in common locations.
func findConvertScript(llamaCppDir string) string {
	candidates := []string{}

	if llamaCppDir != "" {
		candidates = append(candidates,
			filepath.Join(llamaCppDir, "convert_hf_to_gguf.py"),
			filepath.Join(llamaCppDir, "convert-hf-to-gguf.py"),
		)
	}

	// Check common locations.
	home, _ := os.UserHomeDir()
	if home != "" {
		candidates = append(candidates,
			filepath.Join(home, "llama.cpp", "convert_hf_to_gguf.py"),
			filepath.Join(home, "llama.cpp", "convert-hf-to-gguf.py"),
		)
	}
	candidates = append(candidates,
		"/usr/local/share/llama.cpp/convert_hf_to_gguf.py",
		"/opt/llama.cpp/convert_hf_to_gguf.py",
	)

	// Also check if it's available in PATH.
	if p, err := exec.LookPath("convert_hf_to_gguf.py"); err == nil {
		return p
	}
	if p, err := exec.LookPath("convert-hf-to-gguf.py"); err == nil {
		return p
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// findLlamaQuantize searches for the llama-quantize (or llama.cpp quantize) binary.
func findLlamaQuantize(llamaCppDir string) string {
	candidates := []string{}

	if llamaCppDir != "" {
		candidates = append(candidates,
			filepath.Join(llamaCppDir, "build", "bin", "llama-quantize"),
			filepath.Join(llamaCppDir, "build", "llama-quantize"),
			filepath.Join(llamaCppDir, "llama-quantize"),
			filepath.Join(llamaCppDir, "quantize"),
		)
	}

	// Check PATH.
	if p, err := exec.LookPath("llama-quantize"); err == nil {
		return p
	}
	if p, err := exec.LookPath("quantize"); err == nil {
		return p
	}

	home, _ := os.UserHomeDir()
	if home != "" {
		candidates = append(candidates,
			filepath.Join(home, "llama.cpp", "build", "bin", "llama-quantize"),
			filepath.Join(home, "llama.cpp", "llama-quantize"),
		)
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}
