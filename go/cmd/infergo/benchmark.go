package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"
)

func runBenchmark(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	addr        := fs.String("addr", "http://localhost:8080", "address of a running infergo server")
	model       := fs.String("model", "", "model name to benchmark (required)")
	requests    := fs.Int("requests", 100, "total number of requests to send")
	concurrency := fs.Int("concurrency", 4, "number of parallel workers")
	prompt      := fs.String("prompt", "Hello, world!", "prompt to send in each request")
	maxTokens   := fs.Int("max-tokens", 64, "max tokens per response")
	fs.Parse(args)

	if *model == "" {
		fmt.Fprintln(os.Stderr, "benchmark: --model is required")
		fs.Usage()
		os.Exit(1)
	}

	type result struct {
		latency time.Duration
		err     error
	}

	results := make([]result, *requests)
	work := make(chan int, *requests)
	for i := 0; i < *requests; i++ {
		work <- i
	}
	close(work)

	payload, _ := json.Marshal(map[string]any{
		"model":      *model,
		"max_tokens": *maxTokens,
		"messages":   []map[string]string{{"role": "user", "content": *prompt}},
	})

	var wg sync.WaitGroup
	start := time.Now()

	for w := 0; w < *concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			client := &http.Client{Timeout: 120 * time.Second}
			for idx := range work {
				t0 := time.Now()
				resp, err := client.Post(
					*addr+"/v1/chat/completions",
					"application/json",
					bytes.NewReader(payload),
				)
				if err != nil {
					results[idx] = result{err: err}
					continue
				}
				resp.Body.Close()
				if resp.StatusCode != http.StatusOK {
					results[idx] = result{err: fmt.Errorf("HTTP %s", resp.Status)}
					continue
				}
				results[idx] = result{latency: time.Since(t0)}
			}
		}()
	}

	wg.Wait()
	total := time.Since(start)

	// Collect latencies
	var latencies []float64
	var errCount int
	for _, r := range results {
		if r.err != nil {
			errCount++
		} else {
			latencies = append(latencies, float64(r.latency.Milliseconds()))
		}
	}
	sort.Float64s(latencies)

	success := len(latencies)
	if success == 0 {
		fmt.Fprintf(os.Stderr, "benchmark: all %d requests failed\n", errCount)
		os.Exit(1)
	}

	p50 := percentile(latencies, 50)
	p99 := percentile(latencies, 99)
	mean := mean(latencies)
	rps := float64(success) / total.Seconds()

	fmt.Printf("\n── Benchmark Results ─────────────────────────────\n")
	fmt.Printf("  Requests:     %d total, %d ok, %d errors\n", *requests, success, errCount)
	fmt.Printf("  Concurrency:  %d workers\n", *concurrency)
	fmt.Printf("  Duration:     %s\n", total.Round(time.Millisecond))
	fmt.Printf("  Throughput:   %.1f req/s\n", rps)
	fmt.Printf("  Latency mean: %.0f ms\n", mean)
	fmt.Printf("  Latency P50:  %.0f ms\n", p50)
	fmt.Printf("  Latency P99:  %.0f ms\n", p99)
	fmt.Printf("──────────────────────────────────────────────────\n\n")
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(len(sorted))*p/100)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func mean(vals []float64) float64 {
	if len(vals) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	return sum / float64(len(vals))
}
