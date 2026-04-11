// Command infergo is the CLI for the infergo inference server.
//
// Usage:
//
//	infergo serve     --model <path> [--provider cpu|cuda] [--backend auto|onnx|torch] [--port 9090]
//	infergo list-models [--addr http://localhost:9090]
//	infergo benchmark --model <path> --requests 1000 [--concurrency 8]
//	infergo convert   --input <model.pt> [--format torchscript|onnx] [--output <path>]
//	infergo validate  --source <original.pt> --export <exported.pt> [--samples 100]
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	log.SetPrefix("[infergo] ")
	log.SetFlags(log.LstdFlags | log.Lmsgprefix)

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "serve":
		runServe(os.Args[2:])
	case "list-models":
		runListModels(os.Args[2:])
	case "benchmark":
		runBenchmark(os.Args[2:])
	case "pull":
		runPull(os.Args[2:])
	case "detect":
		runDetect(os.Args[2:])
	case "convert":
		runConvert(os.Args[2:])
	case "validate":
		runValidate(os.Args[2:])
	case "-h", "--help", "help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand: %q\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Fprint(os.Stderr, `infergo — production AI inference runtime

Usage:
  infergo serve       --model <path> [--provider cpu|cuda|tensorrt|coreml]
                      [--backend auto|onnx|tensorrt|torch]
                      [--port 9090] [--gpu-layers 999] [--ctx-size 4096]
  infergo list-models [--addr http://localhost:9090]
  infergo benchmark   --addr http://localhost:9090 --requests 1000
                      [--concurrency 8] [--prompt "..."]
  infergo detect      --model <path> --source rtsp://cam1,cam2
                      [--zones zones.yaml] [--webhook http://...] [--output rtsp://...]
  infergo pull        owner/repo [--quant Q4_K_M] [--format gguf|onnx]
                      [--file model.gguf] [--hf-token <token>] [--dir <path>]
  infergo convert     --input <model.pt> [--format torchscript|onnx]
                      [--output <path>] [--imgsz 640]
  infergo validate    --source <original.pt> --export <exported.pt>
                      [--samples 100] [--tolerance 1e-4]

Subcommands:
  serve         Load a model and start the HTTP server.
  detect        Run the detection control center (video → detect → track → alert).
  list-models   List models loaded on a running infergo server.
  benchmark     Stress-test a running infergo server.
  pull          Download a model from HuggingFace Hub.
  convert       Export a PyTorch model to TorchScript or ONNX format.
  validate      Compare a source model against its export on random inputs.

`)
	flag.PrintDefaults()
}
