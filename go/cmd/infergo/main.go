// Command infergo is the CLI for the infergo inference server.
//
// Usage:
//
//	infergo serve    --model <path> [--provider cpu|cuda] [--port 9090]
//	infergo list-models [--addr http://localhost:9090]
//	infergo benchmark --model <path> --requests 1000 [--concurrency 8]
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
                      [--port 9090] [--gpu-layers 999] [--ctx-size 4096]
  infergo list-models [--addr http://localhost:9090]
  infergo benchmark   --addr http://localhost:9090 --requests 1000
                      [--concurrency 8] [--prompt "..."]

Subcommands:
  serve         Load a model and start the HTTP server.
  list-models   List models loaded on a running infergo server.
  benchmark     Stress-test a running infergo server.

`)
	flag.PrintDefaults()
}
