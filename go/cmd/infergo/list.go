package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"net/http"
	"os"
)

func runListModels(args []string) {
	fs := flag.NewFlagSet("list-models", flag.ExitOnError)
	addr := fs.String("addr", "http://localhost:9090", "address of a running infergo server")
	fs.Parse(args)

	resp, err := http.Get(*addr + "/v1/models")
	if err != nil {
		fmt.Fprintf(os.Stderr, "list-models: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Fprintf(os.Stderr, "list-models: server returned %s\n", resp.Status)
		os.Exit(1)
	}

	var result struct {
		Object string `json:"object"`
		Data   []struct {
			ID      string `json:"id"`
			Object  string `json:"object"`
			Created int64  `json:"created"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		fmt.Fprintf(os.Stderr, "list-models: decode: %v\n", err)
		os.Exit(1)
	}

	if len(result.Data) == 0 {
		fmt.Println("(no models loaded)")
		return
	}
	fmt.Printf("%-30s  %s\n", "MODEL", "OBJECT")
	fmt.Printf("%-30s  %s\n", "-----", "------")
	for _, m := range result.Data {
		fmt.Printf("%-30s  %s\n", m.ID, m.Object)
	}
}
